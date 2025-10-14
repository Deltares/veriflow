"""The definition of the configuration settings.

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

# TODO(AU): Add pydantic Field with description, and maybe title, to all attributes. # noqa: FIX002
#   https://github.com/Deltares-research/DPyVerification/issues/9
#   Add pydantic Field with description, and maybe title, to approximately every attribute. To both
#   have a descriptive json schema when the json schema is generated from the pydantic objects, and
#   to document what the fields are for in the code. Maybe only for Literal attributes, the
#   description can be skipped. Do also add the description to private attributes, to document
#   their use.

# ruff: noqa: D101 Do not require class docstrings for the classes in this file
# ruff: noqa: D102 Do not require class docstrings for the classes in this file

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Annotated, Self

import xarray as xr
import yaml
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic.json_schema import SkipJsonSchema

from dpyverification.constants import StandardDim, TimeseriesKind

from .utils import ForecastPeriods, Source, TimePeriod, VerificationPair


class GeneralInfoConfig(BaseModel):
    verification_period: Annotated[
        TimePeriod,
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
    ] = Path("./.verification_cache")

    def get_verification_pair(self, pair_id: str) -> VerificationPair:
        """Get one verification_pair by its id."""
        for pair in self.verification_pairs:
            if pair.id == pair_id:
                return pair
        # At runtime, the following statement should be unreachable, because
        #   we already validated all pair_ids are present during config initialization.
        msg = f"Pair with id '{pair_id}' not found in general verification_pairs configuration."
        raise ValueError(msg)


class IdMap(RootModel[dict[str, dict[str, str]]]):
    """Mapping from internal IDs to external IDs per data source."""

    def check_source_defined(self, source: str) -> None:
        """Check a source is defined in the IdMap."""
        if not any(source in inner for inner in self.root.values()):
            msg = f"No IdMapping found for source: {source}"
            raise ValueError(msg)

    def get_external_to_internal_mapping(self, source: str) -> dict[str, str]:
        """Return external → internal mapping for this data source."""
        self.check_source_defined(source)
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

    kind: str

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

    source: Source
    timeseries_kind: TimeseriesKind
    general: SkipJsonSchema[GeneralInfoConfig]  # Do not serialize to json schema, since general
    # config is propagated from the general config section in the main config. This will prevent
    # users that use the json-schema for making config having to explicitly set a duplicate general
    # configuration section for each datasource.

    id_mapping: SkipJsonSchema[IdMappingConfig] | None = None  # Do not serialize to json schema

    @property
    def forecast_periods(self) -> ForecastPeriods:
        return self.general.forecast_periods

    @property
    def verification_period(self) -> TimePeriod:
        return self.general.verification_period


class BaseDatasinkConfig(BaseConfig):
    """
    Base config for a datasink config.

    Specific config definitions should inherit from
    this base class.
    """

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

    general: SkipJsonSchema[GeneralInfoConfig]  # Do not serialize to json schema, since general
    # config is propagated from the general config section in the main config. This will prevent
    # users that use the json-schema for making config having to explicitly set a duplicate general
    # configuration section for each datasource.

    filter_verification_pairs: Annotated[
        list[str] | None,
        Field(
            description="Optional field to filter verification_pairs from the general "
            "configuration, by providing a list of verification pair ids from the general config. "
            "Only the pair ids will be used in the computation of "
            "this score.",
        ),
    ] = None

    @property
    def verification_pairs(self) -> list[VerificationPair]:
        """The configured variable pairs.

        If the verification_pairs element is configured for the score, filter only these ids
        from the verification_pairs defined in general config.
        """
        if self.filter_verification_pairs is None:
            return self.general.verification_pairs
        return [
            self.general.get_verification_pair(pair_id)
            for pair_id in self.filter_verification_pairs
        ]

    @property
    def forecast_periods(self) -> ForecastPeriods:
        return self.general.forecast_periods

    @model_validator(mode="after")
    def filter_verification_pairs_valid(self) -> Self:
        """Check provided filter for verification pairs contains valid ids."""
        valid_pair_ids: Generator[str] = (pair.id for pair in self.general.verification_pairs)
        if self.filter_verification_pairs is not None:
            for pair_id in self.filter_verification_pairs:
                if pair_id not in valid_pair_ids:
                    msg = (
                        f"Pair id '{pair_id}' in filter_verification_pairs is not present in "
                        "the general configuration for verification_pairs. "
                        "Please make sure ids match exactly."
                    )
                    raise ValueError(msg)
        return self


class Config(BaseModel):
    """Config object for running the verification pipeline."""

    fileversion: str
    general: GeneralInfoConfig
    datasources: Annotated[Sequence[BaseDatasourceConfig], Field(min_length=1)]
    scores: Annotated[Sequence[BaseScoreConfig], Field(min_length=1)]
    datasinks: Annotated[Sequence[BaseDatasinkConfig] | None, Field(min_length=1)] = None
    id_mapping: IdMappingConfig | None = None

    @staticmethod
    def write_yaml_schema(output_path: Path) -> None:
        """Generate a YAML schema from the Pydantic model."""
        schema = Config.model_json_schema()
        output_path.write_text(
            yaml.safe_dump(schema, sort_keys=False),
            encoding="utf-8",
        )
