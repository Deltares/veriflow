"""A module for default implementation of datasources."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

from dpyverification.configuration.base import BaseDatasourceConfig
from dpyverification.configuration.utils import (
    FewsWebserviceAuthConfig,
    LocalFiles,
)
from dpyverification.constants import DataSourceKind


class ArchiveKind(StrEnum):
    """Archive kind."""

    open_archive = "open_archive"
    external_storage_archive = "external_storage_archive"


class FewsNetCDFKind(StrEnum):
    """FEWS NetCDF kind."""

    observation = "observation"
    simulated_forecast_per_forecast_reference_time = (
        "simulated_forecast_per_forecast_reference_time"
    )
    simulated_forecast_per_forecast_period = "simulated_forecast_per_forecast_period"


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

    year: Annotated[int, Field(gt=2012, lt=2100, type=int)]
    subversion: Literal[1, 2]

    @property
    def supports_lead_time(self) -> bool:
        """Return True if lead time parameter is available in the webservice."""
        year_of_implementation = 2025
        return self.year >= year_of_implementation


class FewsWebserviceConfig(BaseDatasourceConfig):
    """A fews webservice input config element."""

    kind: Literal[DataSourceKind.FEWSWEBSERVICE]
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
    forecast_retrieval_method: (
        Annotated[
            ForecastRetrievalMethod,
            Field(
                default=ForecastRetrievalMethod.retrieve_all_forecast_data,
                description="Since Delft-FEWS 2025.01, the Delft-FEWS Webservice can"
                "retrieve forecasts for specific forecast periods (lead times). This avoid having "
                "to retrieve all forecast data outside of the configured forecast periods "
                "(lead times) for the verification pipeline. If not provided, the method will be "
                "automatically determined based on the configured webservice version.",
            ),
        ]
        | None
    ) = None
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
    def set_forecast_retrieval_method_based_on_version_if_none(self) -> "FewsWebserviceConfig":
        """Automatically set the forecast retrieval method based on the webservice version."""
        if (
            self.webservice_supports_lead_time_in_get_timeseries
            and self.forecast_retrieval_method is None
        ):
            self.forecast_retrieval_method = (
                ForecastRetrievalMethod.retrieve_forecast_data_per_lead_time
            )
        elif (
            self.forecast_retrieval_method is None
            and not self.webservice_supports_lead_time_in_get_timeseries
        ):
            self.forecast_retrieval_method = ForecastRetrievalMethod.retrieve_all_forecast_data
        return self


class FewsNetCDFConfig(BaseDatasourceConfig, LocalFiles):
    """A file input fewsnetcdf config element."""

    kind: Literal[DataSourceKind.FEWSNETCDF]
    netcdf_kind: FewsNetCDFKind
    station_ids: Annotated[list[str], Field(min_length=1)] | None = None
    parameter_ids: Annotated[list[str], Field(min_length=1)] | None = None
