"""A module for frequently used config elements in the context of verification."""

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import (
    AnyUrl,
    BaseModel,
    BeforeValidator,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from dpyverification.constants import TimeUnits

Source = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", min_length=1),
    Field(
        min_length=1,
        description="The source indicates the origin of the data. For simulated data this may, for "
        "example, refer to the model that produced the data. For observation this may, for "
        "example, be simply 'observed' or 'my_validated_observations_database'. The source should "
        "start with a letter, consists of only letters, digits and underscores. Spaces or "
        "punctuation are disallowed.",
    ),
]

Variable = Annotated[
    str,
    Field(
        min_length=1,
        description="The variable name references a physical variable to be verified. Must match "
        "the variable definition in the datasource. IdMapping can be set in the general config, to "
        "map external variables to an internal definition. This is needed when you want to verify "
        "data from different sources, where the variable definition is not equal.",
    ),
]


class Range(BaseModel):
    """A range."""

    start: int
    end: int
    step: int

    def to_list(self) -> list[int]:
        """Convert to list."""
        return list(range(self.start, self.end + 1, self.step))


class ForecastPeriods(BaseModel):
    """A forecast periods config element."""

    unit: TimeUnits
    values: Annotated[
        list[int] | Range,
        BeforeValidator(lambda v: v.to_list() if isinstance(v, Range) else v),
    ]

    @field_validator("values", mode="after")
    @classmethod
    def convert_range_to_list(cls, v: Range | list[int]) -> list[int]:
        """Convert range to list."""
        if isinstance(v, Range):
            return v.to_list()
        return v

    @property
    def timedelta64(self) -> list[np.timedelta64]:
        """As numpy timedelta64."""
        return [np.timedelta64(v, self.unit) for v in self.values]  # type:ignore[call-overload, misc] # BeforeValidator takes care of conversion to list

    @property
    def stdlib_timedelta(self) -> list[timedelta]:
        """As datetime timedelta."""

        def convert_to_timedelta(value: int) -> timedelta:
            return np.timedelta64(value, self.unit).astype(timedelta)  # type: ignore[no-any-return, misc, call-overload]

        return [convert_to_timedelta(v) for v in self.values]  # type:ignore[arg-type]

    @property
    def max(self) -> timedelta:
        """Get the maximum forecast period."""
        return max(self.stdlib_timedelta)

    @property
    def min(self) -> timedelta:
        """Get the minimum forecast period."""
        return min(self.stdlib_timedelta)


class TimePeriod(BaseModel):
    """A time period config element."""

    start: Annotated[
        datetime,
        Field(
            description=(
                "YYYY-MM-DD[T]HH:MM[:SS[.ffffff]][Z or [±]HH[:]MM], ",
                "see: https://docs.pydantic.dev/2.0/usage/types/datetime/#validation-of-datetime-types",
            ),
        ),
    ]
    end: Annotated[
        datetime,
        Field(
            description=(
                "YYYY-MM-DD[T]HH:MM[:SS[.ffffff]][Z or [±]HH[:]MM], ",
                "see: https://docs.pydantic.dev/2.0/usage/types/datetime/#validation-of-datetime-types",
            ),
        ),
    ]

    @field_validator("start", "end", mode="after")
    @classmethod
    def to_utc_naive_numpy_datetime(cls, v: datetime) -> datetime:
        """Convert timezone aware datetimes to naive UTC; leave naive untouched (assumed UTC)."""
        if v.tzinfo is not None:
            v = v.astimezone(timezone.utc).replace(tzinfo=None)
        return v

    @property
    def start_datetime64(self) -> np.datetime64:
        """Get start as numpy format."""
        return np.datetime64(self.start)

    @property
    def end_datetime64(self) -> np.datetime64:
        """Get start as numpy format."""
        return np.datetime64(self.end)


class VerificationPeriod(TimePeriod):
    """Definition of the verification period."""

    dimension: Annotated[
        Literal["forecast_reference_time", "time"],
        Field(
            description="The dimension along which the verification period is defined. Using "
            "'forecast_reference_time' allows for a forecast-centric verification run, "
            "whereas 'time' allows for an observation-centric verification run.",
        ),
    ] = "forecast_reference_time"


class VerificationPair(BaseModel):
    """
    Configuration for a verification pair.

    Should consist of an id and reference to a source for
    observations and simulations. The id can be any arbitrary string, and the obs and sim fields
    should contain an exact reference to a configured source in the datasource configuration.
    """

    id: str
    obs: Source
    sim: Source

    model_config = {
        "frozen": True,
    }

    def __eq__(self, other: object) -> bool:
        """Test equality of id's between pairs."""
        if not isinstance(other, VerificationPair):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Return the hashed id."""
        return hash(self.id)


class LocalFile(BaseModel):
    """Configuration pointing to a local file."""

    directory: Path
    filename: str


class LocalFiles(BaseModel):
    """Config for multiple local files using Path.glob()."""

    directory: str
    filename_glob: Annotated[
        str,
        Field(
            description="A valid filename glob, like '*.nc' for all netcdf files.",
        ),
    ]

    @property
    def paths(self) -> Generator[Path, None, None]:
        """Return all filepaths as Path objects."""
        return Path(self.directory).rglob(self.filename_glob)


class FewsWebserviceAuthConfig(BaseSettings):
    """
    Get url, username and password safely from environment variables.

    This config class inherits from :class:`pydantic_settings.BaseSettings`,
    that will try to infer field values from environment variables.

    Make sure to prefix each environment variable with FEWSWEBSERVICE_.

    For url: set the environment variable as: FEWSWEBSERVICE_URL.
    For username: set the environment variable as: FEWSWEBSERVICE_USERNAME.
    For password: set the environment variable as: FEWSWEBSERVICE_PASSWORD.

    see: https://docs.pydantic.dev/latest/concepts/pydantic_settings/#usage
    """

    model_config = SettingsConfigDict(env_prefix="FEWSWEBSERVICE_")

    url: AnyUrl
    username: SecretStr
    password: SecretStr
