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

import json
from collections.abc import Generator, Sequence
from functools import reduce
from pathlib import Path
from typing import Annotated, Self, TypeVar

import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic.json_schema import SkipJsonSchema

from dpyverification.constants import StandardDim, TimeseriesKind, TimeUnits

from .utils import ForecastPeriods, Source, TimePeriod, VerificationPair, VerificationPeriod


class GeneralInfoConfig(BaseModel):
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

    kind: str

    # Accept additional fields.
    # This is a requirement to make sure that all fields are
    # available after initializing the config instance when
    # the fields are created by external users and thus
    # not known upfront.
    model_config = ConfigDict(extra="allow")


class BaseTimeseriesDatasourceConfig(BaseConfig):
    """
    Base config for a datasource config.

    Specific config definitions should inherit from
    this base class.
    """

    source: Source
    timeseries_kind: TimeseriesKind
    time_step: TimeUnits = TimeUnits.HOUR
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


class BaseThresholdsDatasourceConfig(BaseConfig):
    """Base config for threshold datasources."""


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


TItem = TypeVar(
    "TItem",
    bound=BaseTimeseriesDatasourceConfig | BaseDatasinkConfig | BaseScoreConfig,
)


class Config(BaseModel):
    """Config object for running the verification pipeline."""

    fileversion: str
    general: GeneralInfoConfig
    datasources: Annotated[Sequence[BaseTimeseriesDatasourceConfig], Field(min_length=1)]
    scores: Annotated[Sequence[BaseScoreConfig], Field(min_length=1)]
    datasinks: Annotated[Sequence[BaseDatasinkConfig] | None, Field(min_length=1)] = None
    id_mapping: IdMappingConfig | None = None

    @staticmethod
    def write_schema(
        output_path: Path,
        user_datasources_config: list[type[BaseTimeseriesDatasourceConfig]] | None = None,
        users_scores_config: list[type[BaseScoreConfig]] | None = None,
        user_datasinks_config: list[type[BaseDatasinkConfig]] | None = None,
    ) -> None:
        """Generate a YAML schema from the Pydantic model.

        By default, will write the schema for all default implementations of datasources, scores and
        datasinks. If you're using user-implementations of any of these, you can provide these
        classes to the function. This function will then add your user-implementation to the schema,
        together with the default implementations. As a result, YAML language support tools will be
        able to support both default and user-implementations. For example, by using the YAML
        extension in VSCode, using CTRL+SPACE will automatically suggest fields, based on the
        schema for both the default and user implementations.


        Parameters
        ----------
        output_path : Path
            Where to write the schema.
        user_datasources_config : list[type[BaseDatasourceConfig]] | None, optional
            Option to provide user-implemented config classes, by default None
        users_scores_config : list[type[BaseScoreConfig]] | None, optional
            Option to provide user-implemented config classes, by default None
        user_datasink_config : list[type[BaseDatasinkConfig]] | None, optional
            Option to provide user-implemented config classes, by default None

        """
        # importing at the top of module leads to circular import
        from dpyverification.configuration.default.datasinks import (  # noqa: PLC0415
            CFCompliantNetCDFConfig,
        )
        from dpyverification.configuration.default.datasources import (  # noqa: PLC0415
            FewsNetCDFConfig,
            FewsWebserviceConfig,
        )
        from dpyverification.configuration.default.scores import (  # noqa: PLC0415
            ContinuousScoresConfig,
            CrpsCDFConfig,
            CrpsForEnsembleConfig,
            RankHistogramConfig,
        )

        default_datasources_config = [FewsNetCDFConfig, FewsWebserviceConfig]
        default_scores_config = [
            CrpsForEnsembleConfig,
            RankHistogramConfig,
            CrpsCDFConfig,
            ContinuousScoresConfig,
        ]
        default_datasinks_config = [CFCompliantNetCDFConfig]

        def create_config_union(
            models: list[type[TItem]],
        ) -> type:
            union_type = reduce(lambda a, b: a | b, models)  # type:ignore[misc, return-value, arg-type]
            return list[Annotated[union_type, Field(discriminator="kind")]]  # type:ignore[valid-type]

        merged_datasource_models = (
            default_datasources_config + user_datasources_config
            if user_datasources_config is not None
            else default_datasources_config
        )

        merged_scores_models = (
            default_scores_config + users_scores_config
            if users_scores_config is not None
            else default_scores_config
        )
        merged_datasinks_models = (
            default_datasinks_config + user_datasinks_config
            if user_datasinks_config is not None
            else default_datasinks_config
        )

        CombinedDataSourceConfig = create_config_union(  # noqa: N806
            merged_datasource_models,  # type:ignore[arg-type]
        )
        CombinedScoreConfig = create_config_union(  # noqa: N806
            merged_scores_models,  # type:ignore[arg-type]
        )
        CombinedDatasinkConfig = create_config_union(  # noqa: N806
            merged_datasinks_models,  # type:ignore[arg-type]
        )

        class ConfigSchema(Config):
            datasources: CombinedDataSourceConfig  # type:ignore[valid-type]
            scores: CombinedScoreConfig  # type:ignore[valid-type]
            datasinks: CombinedDatasinkConfig | None = None  # type:ignore[valid-type]

        schema = ConfigSchema.model_json_schema()  # type:ignore[misc]

        # Write the schema to file
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)  # type:ignore[misc]
