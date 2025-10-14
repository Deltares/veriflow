"""A module for default implementation of datasources."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from dpyverification.configuration.base import BaseDatasourceConfig
from dpyverification.configuration.utils import (
    FewsWebserviceAuthConfig,
    LocalFiles,
    Source,
)
from dpyverification.constants import DataSourceKind


class ArchiveKind(StrEnum):
    """Archive kind."""

    open_archive = "open_archive"
    external_storage_archive = "external_storage_archive"


class FewsNetCDFKind(StrEnum):
    """List of kinds of FEWS NetCDFs."""

    observation = "observation"
    simulated_forecast_per_forecast_reference_time = (
        "simulated_forecast_per_forecast_reference_time"
    )
    simulated_forecast_per_forecast_period = "simulated_forecast_per_forecast_period"


class SimulationRetrievalMethod(StrEnum):
    """Retrieval methods for simulations."""

    retrieve_all_forecast_data = "retrieve_all_forecast_data"
    retrieve_forecast_data_per_lead_time = "retrieve_forecast_data_per_lead_time"


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
    archive_kind: Annotated[
        ArchiveKind,
        Field(
            description="Archive kind. Defaults to a Delft-FEWS Open Archive, "
            "which is the Delft-FEWS standard and is most used.",
        ),
    ] = ArchiveKind.open_archive
    webservice_version: FewsWebserviceVersion = FewsWebserviceVersion(year=2025, subversion=1)
    forecast_retrieval_method: (
        Annotated[
            SimulationRetrievalMethod,
            Field(
                default=SimulationRetrievalMethod.retrieve_all_forecast_data,
                description="Since Delft-FEWS 2025.01, the Delft-FEWS Webservice can"
                "retrieve forecasts for specific lead times. This avoid having to retrieve all "
                "forecast data outside of the configured lead times (forecast periods) for the "
                "verification pipeline. Note: this method is not yet implemented, so defaults to"
                "retrieving all forecast data.",
            ),
        ]
        | None
    ) = None
    source: Annotated[
        Source,
        Field(description="If not provided, source will be equal to module_instance_id."),
    ] = ""
    max_workers_in_thread_pool: Annotated[
        int,
        Field(
            description="This datasource asynchronously retrieves data from the "
            "Delft-FEWS webservice. Define here the maximum workers it can use. "
            "Use 5-10 for gentle load on the server-side and keep below 30 "
            "to avoid instability and minimize the risk of internal server errors.",
        ),
    ] = 2

    @model_validator(mode="after")
    def set_source_equal_to_module_instance_id_if_none(self) -> "FewsWebserviceConfig":
        """By default, set source equal to module instance id."""
        if self.source == "":
            self.source = self.module_instance_id
        return self


class FewsNetCDFConfig(BaseDatasourceConfig, LocalFiles):
    """A file input fewsnetcdf config element."""

    kind: Literal[DataSourceKind.FEWSNETCDF]
    netcdf_kind: FewsNetCDFKind
    station_ids: Annotated[list[str], Field(min_length=1)] | None = None
    parameter_ids: Annotated[list[str], Field(min_length=1)] | None = None
