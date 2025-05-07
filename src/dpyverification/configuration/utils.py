"""A module for frequently used config elements in the context of verification."""

from datetime import datetime, timedelta
from typing import Annotated

import numpy as np
from pydantic import AnyUrl, BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from dpyverification.constants import TimeUnits


class LeadTimes(BaseModel):
    """A leadtimes config element."""

    unit: TimeUnits
    values: list[int]

    @property
    def timedelta64(self) -> list[np.timedelta64]:
        """As numpy timedelta64."""
        return [np.timedelta64(v, self.unit) for v in self.values]

    @property
    def timedelta(self) -> list[timedelta]:
        """As datetime timedelta."""

        def convert_to_timedelta(value: int) -> timedelta:
            return np.timedelta64(value, self.unit).astype(timedelta)  # type: ignore[no-any-return, misc]

        return [convert_to_timedelta(v) for v in self.values]


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


class SimObsVariables(BaseModel):
    """A simobs variables config element."""

    sim: str
    obs: str


class LocalFile(BaseModel):
    """A local file config element."""

    directory: str
    filename: str
