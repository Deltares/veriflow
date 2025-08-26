"""A module for frequently used config elements in the context of verification."""

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import AnyUrl, BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dpyverification.constants import TimeUnits


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
    values: list[int] | Range

    @field_validator("values", mode="after")
    @classmethod
    def expand_range(cls, v: list[int] | Range) -> list[int]:
        """Make a list from provided range."""
        if isinstance(v, Range):
            return v.to_list()
        return v  # already a list[int]

    @property
    def timedelta64(self) -> list[np.timedelta64]:
        """As numpy timedelta64."""
        return [np.timedelta64(v, self.unit) for v in self.values]

    @property
    def py_timedelta(self) -> list[timedelta]:
        """As datetime timedelta."""

        def convert_to_timedelta(value: int) -> timedelta:
            return np.timedelta64(value, self.unit).astype(timedelta)  # type: ignore[no-any-return, misc]

        return [convert_to_timedelta(v) for v in self.values]

    @property
    def max(self) -> timedelta:
        """Get the maxium forecast period."""
        return max(self.py_timedelta)

    @property
    def min(self) -> timedelta:
        """Get the minimum forecast period."""
        return min(self.py_timedelta)


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


class SimObsVariables(BaseModel):
    """A simobs variables config element."""

    sim: str
    obs: str

    @property
    def sim_obs_string(self) -> str:
        """Return a string representation of sim and obs."""
        return f"{self.sim}_{self.obs}"

    @property
    def sim_variable_name(self) -> str:
        """The name for simulation in the internal dataset."""
        return f"sim_{self.sim}"

    @property
    def obs_variable_name(self) -> str:
        """The name for observation in the internal dataset."""
        return f"obs_{self.obs}"


class LocalFile(BaseModel):
    """A local file config element."""

    directory: str | Path
    filename: str


class LocalFiles(BaseModel):
    """Config for multiple local files using Path.glob()."""

    directory: str
    filename_pattern: Annotated[
        str,
        Field(
            description="Provide a valid filenamepattern, like '*.nc' for all netcdf files.",
        ),
    ]

    @property
    def paths(self) -> Generator[Path]:
        """Return all filepaths as Path objects."""
        return Path(self.directory).rglob(self.filename_pattern)


class FewsWebserviceAuthConfig(BaseSettings):  # type: ignore  # noqa: PGH003
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
