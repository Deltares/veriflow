"""Module for reading from and writing to a fews webservice."""

import asyncio
import io
import tempfile
import zipfile
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Self, TypeVar

import requests
import xarray as xr

from dpyverification.api.fewswebservice import (
    FewsWebserviceClient,
    TimeseriesType,
)
from dpyverification.configuration import (
    FewsWebserviceInputConfig,
    FileInputFewsNetCDFConfig,
    SimulationRetrievalMethod,
)
from dpyverification.constants import DataSourceKind, SimObsKind
from dpyverification.datasources.base import BaseDatasource
from dpyverification.datasources.fewsnetcdf import FewsNetCDFFile, FewsNetCDFKind

T = TypeVar("T")


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
        import concurrent.futures

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
    config_class = FewsWebserviceInputConfig
    # Annotate the correct type, otherwise mypy will infer from baseclass
    config: FewsWebserviceInputConfig
    # The datetime format that is used to pass datetimes to the fewswebservice
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    timeout = 30

    def __init__(self, config: FewsWebserviceInputConfig) -> None:
        self.config = config
        self.simobskind = config.simobskind
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
        """Unzip a file and write the first NetCDF to a directory."""
        zip_bytes = io.BytesIO(response.content)

        if not write_dir.is_dir():
            msg = "Provided path is not a directory."
            raise ValueError(msg)

        # Open the zipfile in memory
        with zipfile.ZipFile(zip_bytes) as zf:
            n_files = len(zf.namelist())
            if n_files != 1:
                msg = f"Expected exactly one file in .zip, got {n_files}"
                raise ValueError(msg)

            netcdf_filename = next(name for name in zf.namelist() if name.endswith(".nc"))

            # Extract that file in memory
            with zf.open(netcdf_filename) as netcdf_file:
                netcdf_data = netcdf_file.read()  # bytes of the .nc file

            # Write the NetCDF file to the write dir
            if unique_prefix is not None:
                netcdf_path = write_dir / f"{unique_prefix}_{netcdf_filename}"
            else:
                netcdf_path = write_dir / netcdf_filename
            netcdf_path.write_bytes(netcdf_data)
        return write_dir

    def fetch_data(self) -> Self:
        """Retrieve :py::class`~xarray.Dataset` from Delft-FEWS Webservice."""
        # Get observations
        if self.config.simobskind == SimObsKind.obs:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Download data to temporary folder
                response = self.client.get_timeseries(
                    location_ids=self.config.location_ids,
                    parameter_ids=self.config.parameter_ids,
                    module_instance_ids=self.config.module_instance_ids,
                    start_time=self.config.verification_period.start,
                    end_time=self.config.verification_period.end,
                    timeseries_type=TimeseriesType.EXTERNAL_HISTORICAL,
                )

                # Unzip and write
                self.write_netcdf_response_to_dir(
                    response,
                    write_dir=Path(tmpdir),
                )

                # Load all downloaded data into one object
                datasource = FewsNetCDFFile(
                    FileInputFewsNetCDFConfig(
                        simobskind=SimObsKind.obs,
                        directory=tmpdir,
                        filename_glob="*.nc",
                        kind=DataSourceKind.FEWSNETCDF,
                        general=self.config.general,
                        netcdf_kind=FewsNetCDFKind.observation,
                    ),
                )
                datasource.fetch_data()

                # After this, the context manager will be closed, and tmpdir deleted
                self.dataset = datasource.data_array

                return self
        # Get simulations
        if (
            self.config.simobskind == SimObsKind.sim
            and self.config.simulation_retrieval_method
            == SimulationRetrievalMethod.retrieve_all_forecast_data
        ):
            # Get all forecast reference times relevant to the configured verification
            #   period. Start is located at verification period start minus the maximum
            #   forecast period. End is located at verification period end minus the minimum
            #   forecast period.
            forecast_reference_times = (
                self.client.get_netcdf_storage_forecasts_forecast_reference_times(
                    start_time=(
                        self.config.verification_period.start - self.config.forecast_periods.max
                    ),
                    end_time=(
                        self.config.verification_period.end - self.config.forecast_periods.min
                    ),
                    module_instance_ids=self.config.module_instance_ids,
                )
            )

            if len(forecast_reference_times) == 0:
                msg = (
                    f"No forecasts found between {self.config.verification_period.start}",
                    f"and {self.config.verification_period.end} and module instance ids",
                    f"{self.config.module_instance_ids}.",
                )
                raise ValueError(msg)

            # Asynchronously download data, create one dataset and append to self.dataset
            with tempfile.TemporaryDirectory() as tmpdir:

                async def fetch_and_write(  # noqa: PLR0913
                    executor: ThreadPoolExecutor,
                    loop: asyncio.AbstractEventLoop,
                    client: FewsWebserviceClient,
                    forecast_reference_time: datetime,
                    config: FewsWebserviceInputConfig,
                    write_dir: Path,
                ) -> None:
                    """Run get_timeseries in loop and write responses to NetCDF."""
                    response = await loop.run_in_executor(
                        executor,
                        lambda: client.get_timeseries(
                            location_ids=config.location_ids,
                            parameter_ids=config.parameter_ids,
                            module_instance_ids=config.module_instance_ids,
                            ensemble_id=config.ensemble_id,
                            start_forecast_time=min(forecast_reference_times),
                            end_forecast_time=max(forecast_reference_times),
                            external_forecast_times=[forecast_reference_time],
                            timeseries_type=TimeseriesType.EXTERNAL_FORECASTING,
                        ),
                    )

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
                    config: FewsWebserviceInputConfig,
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
                datasource = FewsNetCDFFile(
                    FileInputFewsNetCDFConfig(
                        simobskind=SimObsKind.sim,
                        directory=tmpdir,
                        filename_glob="*.nc",
                        kind=DataSourceKind.FEWSNETCDF,
                        general=self.config.general,
                        netcdf_kind=FewsNetCDFKind.simulation_per_forecast_reference_time,
                    ),
                )
                datasource.fetch_data()

                # After this, the context manager will be closed and tmpdir deleted
                self.dataset = datasource.data_array

                return self

        elif (
            self.config.simobskind == SimObsKind.sim
            and self.config.simulation_retrieval_method
            == SimulationRetrievalMethod.retrieve_forecast_data_per_lead_time
        ):
            # Implement forecast retrieval, once Delft-FEWS development is completed.
            msg2 = "Simulation retrieval per lead time is not yet supported."
            raise NotImplementedError(msg2)

        # Other simobs kinds are not supported (yet)
        msg3 = f"Simobskind {self.simobskind} not implemented yet. Only sim and obs are supported."
        raise NotImplementedError(msg3)
