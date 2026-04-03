"""Module for reading from and writing to a fews webservice."""

import asyncio
import concurrent.futures
import io
import tempfile
import zipfile
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Self, TypeVar

import requests
import xarray as xr

from dpyverification.api.fewswebservice import DocumentFormat, FewsWebserviceClient, TimeseriesType
from dpyverification.configuration.default.datasources import (
    ArchiveKind,
    FewsNetCDFConfig,
    FewsWebserviceConfig,
    ForecastRetrievalMethod,
)
from dpyverification.constants import DataSourceKind, DataType
from dpyverification.datasources.base import BaseDatasource
from dpyverification.datasources.fewsnetcdf import (
    FewsNetCDF,
    FewsNetCDFKind,
)

T = TypeVar("T")

FORECAST_COUNT_WHEN_SEARCHING_FOR_FORECAST_REFERENCE_TIMES = 1000000


def run_async_in_compatible_environment(coro: Awaitable[T]) -> T:
    """Run an async coroutine in a way that works in both normal Python and Jupyter environments.

    This function detects if there's already an event loop running (like in Jupyter)
    and handles the execution appropriately.
    """
    try:
        # Try to get the current event loop
        _ = asyncio.get_running_loop()
        # If we get here, there's already a running event loop (e.g., in Jupyter)
        # Create a new event loop in a separate thread

        def run_in_new_loop() -> T:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result: T = new_loop.run_until_complete(coro)
                return result
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()
    except RuntimeError:
        # No event loop is running, we can use asyncio.run() safely
        result: T = asyncio.run(coro)  # type:ignore[arg-type]
        return result


class FewsWebservice(BaseDatasource):
    """For downloading data using a Delft-FEWS webservice."""

    # TODO(AU): Fix and document timezone information in fewswebservice requests # noqa: FIX002
    #   A hardcoded Z is added at the end, that cannot be right? See the issue for details:
    #   https://github.com/Deltares-research/DPyVerification/issues/43
    #

    kind = "fewswebservice"
    config_class = FewsWebserviceConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.observed_historical,
        DataType.simulated_forecast_ensemble,
        DataType.simulated_forecast_single,
        DataType.simulated_forecast_probabilistic,
    }

    # Annotate the correct type, otherwise mypy will infer from baseclass
    config: FewsWebserviceConfig
    # The datetime format that is used to pass datetimes to the fewswebservice
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    timeout = 30

    def __init__(self, config: FewsWebserviceConfig) -> None:
        self.config = config
        self.data_type = config.data_type
        self.dataset = xr.Dataset()

        # Initialize the webservice client
        self.client = FewsWebserviceClient(
            url=self.config.auth_config.url.unicode_string(),
            username=self.config.auth_config.username.get_secret_value(),
            password=self.config.auth_config.password.get_secret_value(),
        )

    @staticmethod
    def write_netcdf_response_to_dir(
        response: requests.Response,
        write_dir: Path,
        unique_prefix: str | None = None,
    ) -> Path:
        """Unzip a file and write the NetCDF file(s) to a directory.

        Optional parameter unique_prefix is used only when using the leadTime
        parameter in the request. In this case, the Delft-FEWS webservice response
        does not contain any meta data on the exact leadTime that was used. That's why
        we need to store it in the filename, so that we can internally assign it later
        as a proper coordinate on the internal xr.DataArray.
        """
        zip_bytes = io.BytesIO(response.content)  # type:ignore[misc]

        if not write_dir.is_dir():
            msg = "Provided path is not a directory."
            raise ValueError(msg)

        # Open the zipfile in memory
        with zipfile.ZipFile(zip_bytes) as zf:
            n_files = len(zf.namelist())
            if n_files == 0:
                msg = f"No NetCDF file present in webservice response. Request URL: {response.url}"
                raise ValueError(msg)

            # NetCDF responses from the Delft-FEWS Webservice come zipped. The zipped file
            #   contains one unique NetCDF file for each requested parameter. For example,
            #   when requesting 'waterlevel' and 'discharge', we get one zip file with two
            #   NetCDF files.
            for netcdf_file_name in (name for name in zf.namelist() if name.endswith(".nc")):
                # Extract that file in memory
                with zf.open(netcdf_file_name) as netcdf_file:
                    netcdf_data = netcdf_file.read()  # bytes of the .nc file

                # Write the NetCDF file(s) to the write dir
                if unique_prefix is not None:
                    netcdf_path = write_dir / f"{unique_prefix}_{netcdf_file_name}"
                else:
                    netcdf_path = write_dir / netcdf_file_name
                netcdf_path.write_bytes(netcdf_data)
        return write_dir

    def fetch_data(self) -> Self:
        """Retrieve :py::class`~xarray.Dataset` from Delft-FEWS Webservice."""
        # Get observations
        if self.config.data_type == DataType.observed_historical:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Download data to temporary folder
                response = self.client.get_timeseries(
                    location_ids=self.config.location_ids,
                    parameter_ids=self.config.parameter_ids,
                    module_instance_ids=self.config.module_instance_id,
                    start_time=self.config.verification_period_on_time.start,
                    end_time=self.config.verification_period_on_time.end,
                    export_id_map=self.config.export_id_map,
                    timeseries_type=TimeseriesType.EXTERNAL_HISTORICAL,
                )

                # Unzip and write
                self.write_netcdf_response_to_dir(
                    response,
                    write_dir=Path(tmpdir),
                )

                # Load all downloaded data into one object
                datasource = FewsNetCDF(
                    FewsNetCDFConfig(
                        data_type=DataType.observed_historical,
                        directory=tmpdir,
                        filename_glob="*.nc",
                        import_adapter=DataSourceKind.FEWSNETCDF,
                        general=self.config.general,
                        netcdf_kind=FewsNetCDFKind.observation,
                        id_mapping=self.config.id_mapping,
                        source=self.config.source,
                        parameter_ids=self.config.parameter_ids,
                    ),
                )

                # Call get_data directly, to immediately cache the xr.DataArray and break links to
                #   to tmpdir. This prevents os.PermissionErrors upon __exit__ of the current
                #   context manager.
                datasource.get_data()

                # After this, the context manager will be closed, and tmpdir deleted
                self.data_array = datasource.data_array

                return self

        # Get forecasts
        if (
            self.config.data_type
            in [
                DataType.simulated_forecast_ensemble,
                DataType.simulated_forecast_single,
                DataType.simulated_forecast_probabilistic,
            ]
            and self.config.forecast_retrieval_method
            == ForecastRetrievalMethod.retrieve_all_forecast_data
        ):
            # Get all relevant forecast reference times, based on the configured verification period
            if self.config.archive_kind == ArchiveKind.external_storage_archive:
                forecast_reference_times = self.client.get_netcdf_storage_forecast_reference_times(
                    start_time=self.config.verification_period_on_frt.start,
                    end_time=self.config.verification_period_on_frt.end,
                    module_instance_ids=self.config.module_instance_id,
                )

            # Standard Open Archive
            else:
                response = self.client.get_timeseries(
                    location_ids=self.config.location_ids,
                    parameter_ids=self.config.parameter_ids,
                    module_instance_ids=self.config.module_instance_id,
                    ensemble_id=self.config.ensemble_id,
                    start_forecast_time=self.config.verification_period_on_frt.start,
                    end_forecast_time=self.config.verification_period_on_frt.end,
                    document_format=DocumentFormat.PI_JSON,
                    forecast_count=FORECAST_COUNT_WHEN_SEARCHING_FOR_FORECAST_REFERENCE_TIMES,
                )
                forecast_reference_times = (
                    self.client.parse_forecast_reference_times_from_json_headers(
                        response.json(),  # type:ignore[misc]
                        module_instance_id=self.config.module_instance_id,
                    )
                )

            if len(forecast_reference_times) == 0:
                msg = (
                    f"No forecasts found between {self.config.verification_period_on_frt.start} "
                    f"and {self.config.verification_period_on_frt.end} and module instance ids "
                    f"{self.config.module_instance_id}."
                )
                raise ValueError(msg)

            # Asynchronously download data, create one dataset and append to self.dataset
            with tempfile.TemporaryDirectory() as tmpdir:

                async def fetch_and_write(  # noqa: PLR0913
                    executor: ThreadPoolExecutor,
                    loop: asyncio.AbstractEventLoop,
                    client: FewsWebserviceClient,
                    forecast_reference_time: datetime,
                    config: FewsWebserviceConfig,
                    write_dir: Path,
                ) -> None:
                    """Run get_timeseries in loop and write responses to NetCDF."""
                    # In the open archive, we get one forecast, by querying the exact forecast
                    #   reference time for both start_forecast_time and end_forecast_time.
                    if config.archive_kind == ArchiveKind.open_archive:
                        response = await loop.run_in_executor(
                            executor,
                            lambda: client.get_timeseries(
                                location_ids=config.location_ids,
                                parameter_ids=config.parameter_ids,
                                module_instance_ids=config.module_instance_id,
                                ensemble_id=config.ensemble_id,
                                start_forecast_time=forecast_reference_time,
                                end_forecast_time=forecast_reference_time,
                                export_id_map=config.export_id_map,
                            ),
                        )

                    # External storage archive
                    else:
                        response = await loop.run_in_executor(
                            executor,
                            lambda: client.get_timeseries(
                                location_ids=config.location_ids,
                                parameter_ids=config.parameter_ids,
                                module_instance_ids=config.module_instance_id,
                                ensemble_id=config.ensemble_id,
                                start_forecast_time=min(forecast_reference_times),
                                end_forecast_time=max(forecast_reference_times),
                                external_forecast_times=[forecast_reference_time],
                                export_id_map=config.export_id_map,
                                timeseries_type=TimeseriesType.EXTERNAL_FORECASTING,
                            ),
                        )

                    response.raise_for_status()

                    # Write NetCDF response to disk
                    unique_prefix = forecast_reference_time.strftime(
                        "%Y%m%d_%H%M%S",
                    )
                    self.write_netcdf_response_to_dir(
                        response,
                        write_dir=write_dir,
                        unique_prefix=unique_prefix,
                    )

                async def download_all_timeseries_async(
                    client: FewsWebserviceClient,
                    forecast_reference_times: list[datetime],
                    config: FewsWebserviceConfig,
                ) -> None:
                    """Asynchronously download all timeseries."""
                    loop = asyncio.get_event_loop()

                    with ThreadPoolExecutor(
                        max_workers=self.config.max_workers_in_thread_pool,
                    ) as executor:
                        # Create async tasks for each forecast_reference_time
                        tasks = [
                            fetch_and_write(
                                executor,
                                loop,
                                client,
                                frt,
                                config,
                                Path(tmpdir),
                            )
                            for frt in forecast_reference_times
                        ]
                        await asyncio.gather(*tasks, return_exceptions=True)

                # Use the compatible async runner instead of asyncio.run()
                run_async_in_compatible_environment(
                    download_all_timeseries_async(
                        self.client,
                        forecast_reference_times,
                        self.config,
                    ),
                )

                # Load all downloaded data into one object
                datasource = FewsNetCDF(
                    FewsNetCDFConfig(
                        data_type=self.config.data_type,
                        directory=tmpdir,
                        filename_glob="*.nc",
                        import_adapter=DataSourceKind.FEWSNETCDF,
                        general=self.config.general,
                        netcdf_kind=FewsNetCDFKind.simulated_forecast_per_forecast_reference_time,
                        id_mapping=self.config.id_mapping,
                        source=self.config.source,
                    ),
                )

                # Call get_data directly, to immediately cache the xr.DataArray and break links to
                #   to tmpdir. This prevents os.PermissionErrors upon __exit__ of the current
                #   context manager.
                datasource.get_data()

                # After this, the context manager will be closed and tmpdir deleted
                self.data_array = datasource.data_array

                return self

        elif (
            self.config.data_type
            in [
                DataType.simulated_forecast_ensemble,
                DataType.simulated_forecast_single,
                DataType.simulated_forecast_probabilistic,
            ]
            and self.config.forecast_retrieval_method
            == ForecastRetrievalMethod.retrieve_forecast_data_per_lead_time
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                for fp in self.config.general.forecast_periods.stdlib_timedelta:
                    response = self.client.get_timeseries(
                        location_ids=self.config.location_ids,
                        parameter_ids=self.config.parameter_ids,
                        module_instance_ids=self.config.module_instance_id,
                        ensemble_id=self.config.ensemble_id,
                        start_time=self.config.verification_period_on_time.start,
                        end_time=self.config.verification_period_on_time.end,
                        lead_time=fp,
                        export_id_map=self.config.export_id_map,
                        timeseries_type=TimeseriesType.EXTERNAL_FORECASTING
                        if self.config.archive_kind == ArchiveKind.external_storage_archive
                        else None,
                    )

                    # Write NetCDF response to disk, prefix with the forecast period
                    #   (lead time) in milliseconds.
                    unique_prefix = str(int(fp.total_seconds() * 1000))
                    self.write_netcdf_response_to_dir(
                        response,
                        write_dir=tmpdir_path,
                        unique_prefix=unique_prefix,
                    )

                # After this, the context manager will be closed and tmpdir deleted
                datasource = FewsNetCDF(
                    FewsNetCDFConfig(
                        import_adapter=DataSourceKind.FEWSNETCDF,
                        data_type=self.config.data_type,
                        directory=tmpdir,
                        filename_glob="*.nc",
                        general=self.config.general,
                        netcdf_kind=FewsNetCDFKind.simulated_forecast_per_forecast_period,
                        id_mapping=self.config.id_mapping,
                        source=self.config.source,
                    ),
                )

                # Call get_data directly, to immediately cache the xr.DataArray and break links to
                #   to tmpdir. This prevents os.PermissionErrors upon __exit__ of the current
                #   context manager.
                datasource.get_data()

                # Assign to self
                self.data_array = datasource.data_array

                return self

        # Other simobs kinds are not supported (yet)
        msg3 = f"Data type {self.data_type} not implemented yet."
        raise NotImplementedError(msg3)
