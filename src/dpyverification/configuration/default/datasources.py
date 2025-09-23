"""A module for default implementation of datasources."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from dpyverification.configuration.base import BaseDatasourceConfig
from dpyverification.configuration.utils import (
    FewsWebserviceAuthConfig,
    LocalFiles,
    Source,
)
from dpyverification.constants import DataSourceKind


class FewsNetCDFKind(StrEnum):
    """List of kinds of FEWS NetCDFs."""

    observation = "observation"
    simulation_per_forecast_reference_time = "simulation_per_forecast_reference_time"
    simulation_per_forecast_period = "simulation_per_forecast_period"


class SimulationRetrievalMethod(StrEnum):
    """Retrieval methods for simulations."""

    retrieve_all_forecast_data = "retrieve_all_forecast_data"
    retrieve_forecast_data_per_lead_time = "retrieve_forecast_data_per_lead_time"


class FewsWebserviceInputConfig(BaseDatasourceConfig):
    """A fews webservice input config element."""

    kind: Literal[DataSourceKind.FEWSWEBSERVICE]
    auth_config: FewsWebserviceAuthConfig = Field(
        default_factory=FewsWebserviceAuthConfig,  # type:ignore[misc]
    )
    location_ids: Annotated[list[str], Field(min_length=1)]
    parameter_ids: Annotated[list[str], Field(min_length=1)]
    module_instance_ids: Annotated[list[str], Field(min_length=1)]
    ensemble_id: Annotated[str, Field(min_length=1)] | None = None
    qualifier_ids: Annotated[list[str], Field(min_length=1)] | None = None
    simulation_retrieval_method: (
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
    max_workers_in_thread_pool: Annotated[
        int,
        Field(
            description="This datasource asynchronously retrieves data from the "
            "Delft-FEWS webservice. Define here the maximum workers it can use. "
            "Use 5-10 for gentle load on the server-side and keep below 30 "
            "to avoid instability and minimize the risk of internal server errors.",
        ),
    ] = 2


class FewsWebserviceOutputConfig(FewsWebserviceInputConfig):
    """A fews webservice output config element."""


class FileInputFewsNetCDFConfig(BaseDatasourceConfig, LocalFiles):
    """A file input fewsnetcdf config element."""

    kind: Literal[DataSourceKind.FEWSNETCDF]

    netcdf_kind: FewsNetCDFKind
    source: Source
    station_ids: Annotated[list[str], Field(min_length=1)] | None = None
    parameter_ids: Annotated[list[str], Field(min_length=1)] | None = None
