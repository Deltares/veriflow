"""A module for default implementation of datasources."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

from veriflow.configuration.base import BaseDatasourceConfig
from veriflow.configuration.utils import (
    BaseZarrConfig,
    CRSString,
    FewsWebserviceAuthConfig,
    LocalFile,
    LocalFiles,
)
from veriflow.constants import DataSourceKind, DataType


class ArchiveKind(StrEnum):
    """Archive kind."""

    open_archive = "open_archive"
    external_storage_archive = "external_storage_archive"


class FewsNetCDFKind(StrEnum):
    """FEWS NetCDF kind."""

    external_historical = "external_historical"
    simulated_historical = "simulated_historical"
    simulated_forecast_per_forecast_reference_time = (
        "simulated_forecast_per_forecast_reference_time"
    )
    simulated_forecast_per_lead_time = "simulated_forecast_per_lead_time"


class ForecastRetrievalMethod(StrEnum):
    """Retrieval methods for simulations."""

    retrieve_all_forecast_data = "retrieve_all_forecast_data"
    retrieve_forecast_data_per_lead_time = "retrieve_forecast_data_per_lead_time"


FewsWebserviceVersionString = Annotated[
    str,
    StringConstraints(pattern=r"^\d{4}\.(0[1-2])$"),
    Field(description="Please specify the version as 'YYYY.01' or 'YYYY.02"),
]


class FewsWebserviceVersion(BaseModel):
    """Configuration of FEWS Webservice version."""

    year: Annotated[int, Field(gt=2012, lt=2100)]
    subversion: Literal[1, 2]

    @property
    def supports_lead_time(self) -> bool:
        """Return True if lead time parameter is available in the webservice."""
        year_of_implementation = 2025
        return self.year >= year_of_implementation


class FewsWebserviceConfig(BaseDatasourceConfig):
    """A fews webservice input config element."""

    import_adapter: Literal[DataSourceKind.FEWSWEBSERVICE]
    auth_config: FewsWebserviceAuthConfig = Field(
        default_factory=FewsWebserviceAuthConfig,  # type:ignore[misc]
    )
    location_ids: Annotated[list[str], Field(min_length=1)]
    parameter_ids: Annotated[list[str], Field(min_length=1)]
    module_instance_id: Annotated[str, Field(min_length=1)]
    ensemble_id: Annotated[str, Field(min_length=1)] | None = None
    qualifier_ids: Annotated[list[str], Field(min_length=1)] | None = None
    export_id_map: Annotated[str, Field(min_length=1)] | None = None
    webservice_version: FewsWebserviceVersionString
    archive_kind: Annotated[
        ArchiveKind,
        Field(
            description="Archive kind. Defaults to a Delft-FEWS Open Archive, "
            "which is the Delft-FEWS standard.",
        ),
    ] = ArchiveKind.open_archive
    forecast_retrieval_method: Annotated[
        ForecastRetrievalMethod,
        Field(
            description="Since Delft-FEWS 2025.01, the Delft-FEWS Webservice can"
            "retrieve forecasts for specific lead times (lead times). This avoid having "
            "to retrieve all forecast data outside of the configured lead times "
            "(lead times) for the verification pipeline. If not provided, the method will be "
            "automatically determined based on the configured webservice version.",
        ),
    ] = ForecastRetrievalMethod.retrieve_all_forecast_data
    max_workers_in_thread_pool: Annotated[
        int,
        Field(
            description="This datasource asynchronously retrieves data from the "
            "Delft-FEWS webservice. Define here the maximum workers it can use. "
            "Use 5-10 for gentle load on the server-side and keep below 30 "
            "to avoid instability and minimize the risk of internal server errors.",
        ),
    ] = 2

    @property
    def webservice_supports_lead_time_in_get_timeseries(self) -> bool:
        """Wether or not the leadTime parameter is supported.

        This determines the forecast retrieval method.
        """
        implementation_year = 2025
        webservice_version_year = int(self.webservice_version.split(".")[0])
        return webservice_version_year >= implementation_year

    @model_validator(mode="after")
    def validate_forecast_retrieval_method(self) -> "FewsWebserviceConfig":
        """Validate that the configures retrieval method is compatible with the webservice."""
        if (
            not self.webservice_supports_lead_time_in_get_timeseries
            and self.forecast_retrieval_method
            == ForecastRetrievalMethod.retrieve_forecast_data_per_lead_time
        ):
            msg = (
                f"Configured forecast retrieval method {self.forecast_retrieval_method} is not "
                f"compatible with the configured webservice version {self.webservice_version}. "
            )
            raise ValueError(msg)
        return self


class FewsNetCDFConfig(BaseDatasourceConfig, LocalFiles):
    """A FEWS NetCDF config element."""

    import_adapter: Literal[DataSourceKind.FEWSNETCDF]
    netcdf_kind: FewsNetCDFKind
    station_ids: Annotated[list[str], Field(min_length=1)] | None = None
    parameter_ids: Annotated[list[str], Field(min_length=1)] | None = None


class NetCDFConfig(BaseDatasourceConfig, LocalFiles):
    """A NetCDF config element."""

    import_adapter: Literal[DataSourceKind.NETCDF]
    stations: Annotated[list[str] | None, Field(min_length=1)] = None
    variables: Annotated[list[str] | None, Field(min_length=1)] = None


class ZarrConfig(BaseDatasourceConfig, BaseZarrConfig):
    """A Zarr config element.

    Reads a single Zarr store via :func:`xarray.open_zarr`. The ``path`` may point to a
    local directory or a remote location (e.g. ``s3://bucket/key/store.zarr``). When the
    path uses an ``s3://`` URL, credentials and connection details are taken from
    ``auth_config`` (an :class:`S3AuthConfig`), which is populated from environment
    variables prefixed with ``S3_``. Additional ``storage_options`` are merged on top of
    the ones derived from ``auth_config`` and forwarded to ``xr.open_zarr``.
    """

    import_adapter: Literal[DataSourceKind.ZARR]
    crs: Annotated[
        CRSString | None,
        Field(
            default=None,
            description="Coordinate reference system of the data in the store. Optional: when "
            "omitted, the CRS is read from the dataset's 'crs' attribute if present. When set, it "
            "takes precedence and is written onto the dataset's 'crs' attribute. Required for "
            "gridded data that does not already carry a 'crs' attribute.",
        ),
    ] = None

    variables: Annotated[list[str] | None, Field(min_length=1)] = None
    stations: Annotated[list[str] | None, Field(min_length=1)] = None


class CsvConfig(LocalFile, BaseDatasourceConfig):
    """A CSV input config element."""

    import_adapter: Literal[DataSourceKind.CSV]
    data_type: Literal[DataType.threshold]
    stations: Annotated[list[str], Field(min_length=1)]
    variables: Annotated[list[str], Field(min_length=1)]
    thresholds: Annotated[list[str], Field(min_length=1)]
