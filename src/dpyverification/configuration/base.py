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

Side note: It is also possible to go the other way around and generate a pydantic schema from a
yaml/json file, see datamodel_code_generator, for example from
https://docs.pydantic.dev/latest/integrations/datamodel_code_generator/ . Note that this can
generate a pydantic model that is not up-to-date with the latest pydantic / python, and might
need some modifications.
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

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from .utils import ForecastPeriods, SimObsVariables, TimePeriod


class GeneralInfoConfig(BaseModel):
    verification_period: Annotated[
        TimePeriod,
        Field(description="The start and end of the verification period."),
    ]
    forecast_periods: Annotated[
        ForecastPeriods,
        Field(
            "A set of forecast periods for which to evaluate of the verification scores. "
            "A forecast period is the timedelta between the forecast reference time of a forecast"
            "(t0, analysis_time, initialization time) and the valid time (time, observed time) "
            "and is also known as: lead time or forecast horizon)",
        ),
    ]
    cache_dir: Annotated[
        Path,
        Field(
            description=(
                "Path pointing to a cache directory.",
                "Will be automatically created if it doesn't yet exist.",
            ),
        ),
    ] = Path("./.verification_cache")

    @field_validator("cache_dir")
    @classmethod
    def validate_is_dir_and_create_if_not_exists(cls, v: Path) -> Path:
        """Validate the projects cache directory."""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        elif not v.is_dir():
            msg = "Provided cache dir is not a directory."
            raise ValueError(msg)
        return v


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

    variable_pairs: Annotated[
        list[SimObsVariables],
        Field(
            description="Variable pairs to use for the computation.",
        ),
    ]

    @property
    def forecast_periods(self) -> ForecastPeriods:
        return self.general.forecast_periods


class Config(BaseModel):
    fileversion: str
    general: GeneralInfoConfig
    datasources: Annotated[list[BaseDatasourceConfig], Field(min_length=1)]
    scores: Annotated[list[BaseScoreConfig], Field(min_length=1)]
    datasinks: Annotated[list[BaseDatasinkConfig], Field(min_length=1)]
