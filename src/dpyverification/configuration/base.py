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

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, RootModel
from pydantic.json_schema import SkipJsonSchema

from dpyverification.constants import StandardDim

from .utils import ForecastPeriods, TimePeriod, VerificationPair


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


class IdMap(RootModel[dict[str, dict[str, str]]]):
    """Mapping from internal IDs to external IDs per data source."""

    def get_external_to_internal_mapping(self, data_source: str) -> dict[str, str]:
        """Return external → internal mapping for this data source."""
        return {v[data_source]: k for k, v in self.root.items()}


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
        subsets = []

        # Iterate over sources. Each source may have its own unique mapping.
        for source in data_array[StandardDim.source].to_numpy():  # type:ignore[misc]
            subset = data_array.sel({StandardDim.source: source})  # type:ignore[misc]

            # Re-assign variable coordinates
            if self.variable is not None:
                subset = subset.assign_coords(
                    {  # type:ignore[misc]
                        StandardDim.variable: (  # type:ignore[misc]
                            StandardDim.variable,
                            subset[StandardDim.variable]  # type:ignore[misc]
                            .to_series()
                            .replace(self.variable.get_external_to_internal_mapping(source))  # type:ignore[misc]
                            .to_numpy(),
                        ),
                    },
                )
            # Re-assign station coordinates
            if self.station is not None:
                subset = subset.assign_coords(
                    {  # type:ignore[misc]
                        StandardDim.station: (  # type:ignore[misc]
                            StandardDim.station,
                            subset[StandardDim.station]  # type:ignore[misc]
                            .to_series()
                            .replace(self.station.get_external_to_internal_mapping(source))  # type:ignore[misc]
                            .to_numpy(),
                        ),
                    },
                )
            subsets.append(subset)

        # Return the fully renamed dataset
        return xr.concat(subsets, dim=StandardDim.source)


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

    simobskind: str
    general: SkipJsonSchema[GeneralInfoConfig]  # Do not serialize to json schema, since general
    # config is propagated from the general config section in the main config. This will prevent
    # users that use the json-schema for making config having to explicitly set a duplicate general
    # configuration section for each datasource.

    id_mapping: SkipJsonSchema[IdMappingConfig]  # Do not serialize to json schema, since general
    # config is propagated from the general config section in the main config. This will prevent
    # users that use the json-schema for making config having to explicitly set a duplicate general
    # configuration section for each datasource.

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

    @property
    def verification_pairs(self) -> list[VerificationPair]:
        """The configured variable pairs from general config."""
        return self.general.verification_pairs

    @property
    def forecast_periods(self) -> ForecastPeriods:
        return self.general.forecast_periods


class Config(BaseModel):
    """Config object for running the verification pipeline."""

    fileversion: str
    general: GeneralInfoConfig
    datasources: Annotated[Sequence[BaseDatasourceConfig], Field(min_length=1)]
    scores: Annotated[Sequence[BaseScoreConfig], Field(min_length=1)]
    datasinks: Annotated[Sequence[BaseDatasinkConfig] | None, Field(min_length=1)] = None
    id_mapping: IdMappingConfig | None = None
