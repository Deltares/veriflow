"""A module for default implementation of datasources."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from dpyverification.configuration.base import BaseDatasourceConfig
from dpyverification.configuration.utils import FewsWebserviceAuthConfig, LocalFiles
from dpyverification.constants import DataSourceKind


class FewsNetcdfKind(StrEnum):
    """List of kinds of FEWS NetCDFs."""

    observation = "observation"
    one_full_simulation = "one_full_simulation"
    simulation_for_one_forecast_period = "simulation_for_one_forecast_period"


class SimulationRetrievalMethod(StrEnum):
    """Retrieval methods for simulations."""

    retrieve_all_forecast_data = "retrieve_all_forecast_data"
    retrieve_forecast_data_per_lead_time = "retrieve_forecast_data_per_lead_time"


class FewsWebserviceInputConfig(BaseDatasourceConfig):
    """A fews webservice input config element."""

    kind: Literal[DataSourceKind.FEWSWEBSERVICE]
    auth_config: FewsWebserviceAuthConfig
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


class FewsWebserviceOutputConfig(FewsWebserviceInputConfig):
    """A fews webservice output config element."""


class FileInputFewsnetcdfConfig(BaseDatasourceConfig, LocalFiles):
    """A file input fewsnetcdf config element."""

    kind: Literal[DataSourceKind.FEWSNETCDF]
    netcdf_kind: FewsNetcdfKind
    station_ids: Annotated[list[str], Field(min_length=1)] | None = None
