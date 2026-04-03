"""The definition of the base configuration settings.

This definition is used both as the schema for the configuration yaml file, and as the content of
the dpyverification configuration object.

To generate a yaml / json file with the json representation of this schema:
    import pathlib
    import yaml
    from dpyverification.configuration import Config
    FILEPATH = pathlib.Path("YOUR_PATH_HERE")
    with FILEPATH.open("w") as myfile:
        yaml.dump(Config.model_json_schema(), myfile)
"""


# ruff: noqa: D102 Do not require class docstrings for the classes in this file

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Self

import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic.json_schema import SkipJsonSchema

from dpyverification.constants import DataType, StandardDim

from .utils import ForecastPeriods, Source, TimePeriod, VerificationPair, VerificationPeriod

if TYPE_CHECKING:
    from collections.abc import Generator


class GeneralInfoConfig(BaseModel):
    """General configuration information that is shared across the pipeline."""

    verification_period: Annotated[
        VerificationPeriod,
        Field(description="The start and end of the verification period."),
    ]
    verification_pairs: Annotated[
        list[VerificationPair],
        Field(
            description="Specify pairs for computation of verification metrics. This allows you to "
            "verify multiple variables and multiple sources. For example, by specifying two pairs: "
            "verify simulated discharge for ModelA and ModelB against observed discharge from"
            "source Observed.",
        ),
    ]
    forecast_periods: Annotated[
        ForecastPeriods,
        Field(
            "A set of forecast periods for which to evaluate of the verification scores. "
            "A forecast period is the timedelta between the forecast reference time of a forecast "
            "(t0, analysis_time, initialization time) and the valid time (time, observed time) "
            "and is also known as: lead time or forecast horizon)",
        ),
    ]
    cache_dir: Annotated[
        Path,
        Field(
            description=(
                "Path pointing to a cache directory. ",
                "Will be automatically created if it doesn't yet exist.",
            ),
        ),
    ] = ".verification_cache"  # type:ignore[assignment] # Allow Path type for default value, since it will be converted to Path during validation.

    def get_verification_pair(self, pair_id: str) -> VerificationPair:
        """Get one verification_pair by its id."""
        for pair in self.verification_pairs:
            if pair.id == pair_id:
                return pair
        # At runtime, the following statement should be unreachable, because
        #   we already validated all pair_ids are present during config initialization.
        msg = f"Pair with id '{pair_id}' not found in general verification_pairs configuration."
        raise ValueError(msg)

    @property
    def verification_period_on_time(self) -> TimePeriod:
        """The verification period along the time dimension."""
        if self.verification_period.dimension == "forecast_reference_time":
            start = self.verification_period.start + self.forecast_periods.min
            end = self.verification_period.end + self.forecast_periods.max
            return TimePeriod(start=start, end=end)
        return self.verification_period

    @property
    def verification_period_on_frt(self) -> TimePeriod:
        """The verification period along the forecast reference time dimension."""
        if self.verification_period.dimension == "time":
            start = self.verification_period.start - self.forecast_periods.max
            end = self.verification_period.end - self.forecast_periods.min
            return TimePeriod(start=start, end=end)
        return self.verification_period


class IdMap(RootModel[dict[str, dict[str, str]]]):
    """Mapping from internal IDs to external IDs per data source."""

    def get_external_to_internal_mapping(self, source: str) -> dict[str, str]:
        """Return external → internal mapping for this data source."""
        # Check that the source is defined in the IdMap
        if not any(source in inner for inner in self.root.values()):
            msg = f"No IdMapping found for source: {source}"
            raise ValueError(msg)

        return {v[source]: k for k, v in self.root.items()}


class IdMappingConfig(BaseModel):
    """Config for mapping external ids to their internal definition."""

    variable: Annotated[
        IdMap | None,
        Field(
            description="Mapping of internal to external definitions per source as a dictionary. "
            "The key corresponds to the internal definition and the value is another dictionary "
            "with keys corresponding to the source and the value to the external definition. ",
        ),
    ] = None
    station: Annotated[
        IdMap | None,
        Field(
            description="Mapping of internal to external definitions per source as a dictionary. "
            "The key corresponds to the internal definition and the value is another dictionary "
            "with keys corresponding to the source and the value to the external definition. ",
        ),
    ] = None

    def rename_data_array(self, data_array: xr.DataArray) -> xr.DataArray:
        source = str(data_array.name)

        # Re-assign variable coordinates, if mapping is provided for source
        if self.variable is not None:
            data_array = data_array.assign_coords(
                {  # type:ignore[misc]
                    StandardDim.variable: (  # type:ignore[misc]
                        StandardDim.variable,
                        data_array[StandardDim.variable]  # type:ignore[misc]
                        .to_series()
                        .replace(self.variable.get_external_to_internal_mapping(source))
                        .to_numpy(),
                    ),
                },
            )
        # Re-assign station coordinates, if mapping is provided for source
        if self.station is not None:
            data_array = data_array.assign_coords(
                {  # type:ignore[misc]
                    StandardDim.station: (  # type:ignore[misc]
                        StandardDim.station,
                        data_array[StandardDim.station]  # type:ignore[misc]
                        .to_series()
                        .replace(self.station.get_external_to_internal_mapping(source))
                        .to_numpy(),
                    ),
                },
            )
        return data_array


class BaseConfig(BaseModel):
    """A base config.

    Each element in the pipeline (datasource, score, datasink)
    inherits from this BaseConfig, so that each config has a 'kind' attribute.

    Based on a user-input to the configuration field 'kind', the pipeline
    will find the correct user-provided class for either a Datasource, score
    or a Datasink.
    """

    # Accept additional fields.
    # This is a requirement to make sure that all fields are
    # available after initializing the config instance when
    # the fields are created by external users and thus
    # not known upfront.
    model_config = ConfigDict(extra="allow")


class BaseDatasourceConfig(BaseConfig):
    """
    Base config for a datasource config.

    Specific config definitions should inherit from
    this base class.
    """

    import_adapter: str
    source: Source
    data_type: DataType
    general: SkipJsonSchema[GeneralInfoConfig]  # Do not serialize to json schema, since general
    # config is propagated from the general config section in the main config. This will prevent
    # users that use the json-schema for making config having to explicitly set a duplicate general
    # configuration section for each datasource.

    id_mapping: SkipJsonSchema[IdMappingConfig] | None = None

    @property
    def forecast_periods(self) -> ForecastPeriods:
        return self.general.forecast_periods

    @property
    def verification_period(self) -> TimePeriod:
        return self.general.verification_period

    @property
    def verification_period_on_frt(self) -> TimePeriod:
        return self.general.verification_period_on_frt

    @property
    def verification_period_on_time(self) -> TimePeriod:
        return self.general.verification_period_on_time


class BaseDatasinkConfig(BaseConfig):
    """
    Base config for a datasink config.

    Specific config definitions should inherit from
    this base class.
    """

    export_adapter: str

    force_overwrite: bool = True

    general: SkipJsonSchema[GeneralInfoConfig]  # Do not serialize to json schema, since general
    # config is propagated from the general config section in the main config. This will prevent
    # users that use the json-schema for making config having to explicitly set a duplicate general
    # configuration section for each datasource.

    @property
    def verification_period(self) -> TimePeriod:
        return self.general.verification_period


class BaseScoreConfig(BaseConfig):
    """
    Base config for a score config.

    Specific config definitions should inherit from
    this base class.
    """

    score_adapter: str
    general: SkipJsonSchema[GeneralInfoConfig]  # Do not serialize to json schema, since general
    # config is propagated from the general config section in the main config. This will prevent
    # users that use the json-schema for making config having to explicitly set a duplicate general
    # configuration section for each datasource.

    verification_pair_ids: Annotated[
        list[str],
        Field(
            description="Optional field to select verification_pairs from the general "
            "configuration, by providing a list of verification pair ids from the general config. "
            "Only these pair ids will be used in the computation of this score.",
        ),
    ] = []

    @property
    def verification_pairs(self) -> list[VerificationPair]:
        """The configured variable pairs.

        If the verification_pairs element is configured for the score, filter only these ids
        from the verification_pairs defined in general config.
        """
        if self.verification_pair_ids == []:
            return self.general.verification_pairs
        return [
            self.general.get_verification_pair(pair_id) for pair_id in self.verification_pair_ids
        ]

    @property
    def forecast_periods(self) -> ForecastPeriods:
        return self.general.forecast_periods

    @model_validator(mode="after")
    def verification_pair_ids_valid(self) -> Self:
        """Check provided filter for verification pairs contains valid ids."""
        valid_pair_ids: Generator[str, None, None] = (
            pair.id for pair in self.general.verification_pairs
        )

        for pair_id in self.verification_pair_ids:
            if pair_id not in valid_pair_ids:
                msg = (
                    f"Pair id '{pair_id}' in filter_verification_pairs is not present in "
                    "the general configuration for verification_pairs. "
                    "Please make sure ids match exactly."
                )
                raise ValueError(msg)
        return self


class BaseEvent(BaseModel):
    """Base class for event definitions."""


class BaseCategoricalScoreConfig(BaseScoreConfig):
    """
    Base config for a categorical score config.

    Specific config definitions should inherit from
    this base class.
    """

    events: Annotated[
        Iterable[
            BaseEvent
        ],  # we use Iterable instead of list to also allow subclasses of BaseEvent (see: https://docs.python.org/3/library/typing.html#generics)
        Field(
            description="A list of event definitions. For each event, a categorical score will be "
            "computed.",
        ),
    ]
