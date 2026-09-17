"""A module for frequently used config elements in the context of verification."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from os import R_OK, access
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import (
    AnyUrl,
    BaseModel,
    BeforeValidator,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from veriflow.constants import TimeUnits

__all__ = [
    # "FewsWebserviceAuthConfig",
    "BaseZarrConfig",
    "CRSString",
    "LeadTimes",
    "LocalFile",
    "LocalFiles",
    "Range",
    # "S3AuthConfig",
    "Source",
    "TimePeriod",
    "Variable",
    "VerificationPair",
    "VerificationPeriod",
]
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


CRSString = Annotated[
    str,
    Field(
        min_length=1,
        description="A coordinate reference system as a string compatible with "
        "pyproj.CRS.from_string, i.e. a PROJ string, a CRS WKT string, or an authority string "
        "such as 'EPSG:4326'. The string is not parsed at configuration time so that pyproj "
        "remains an optional dependency; it is only parsed when an actual coordinate "
        "transformation is required.",
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


class LeadTimes(BaseModel):
    """A lead times config element."""

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
            # ``np.timedelta64(...).astype(timedelta)`` returns an int (not a timedelta)
            # when the unit is finer than microsecond (e.g. nanosecond), since stdlib
            # ``timedelta`` has only microsecond precision. Coerce via microseconds to
            # always return a proper ``timedelta``.
            td_us = np.timedelta64(value, self.unit).astype("timedelta64[us]")  # type: ignore[call-overload, misc]
            return timedelta(microseconds=int(td_us.astype("int64")))  # type: ignore[misc]

        return [convert_to_timedelta(v) for v in self.values]  # type:ignore[arg-type]

    @property
    def max(self) -> timedelta:
        """Get the maximum lead time."""
        return max(self.stdlib_timedelta)

    @property
    def min(self) -> timedelta:
        """Get the minimum lead time."""
        return min(self.stdlib_timedelta)


class TimePeriod(BaseModel):
    """A time period config element."""

    start: Annotated[
        datetime,
        Field(
            description="YYYY-MM-DD[T]HH:MM[:SS[.ffffff]][Z or [±]HH[:]MM], "
            "see: https://docs.pydantic.dev/2.0/usage/types/datetime/#validation-of-datetime-types",
        ),
    ]
    end: Annotated[
        datetime,
        Field(
            description="YYYY-MM-DD[T]HH:MM[:SS[.ffffff]][Z or [±]HH[:]MM], "
            "see: https://docs.pydantic.dev/2.0/usage/types/datetime/#validation-of-datetime-types",
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
    """Configuration for a verification pair.

    The id uniquely identifies the verification pair.
    The observations_source_id and simulations_source_id fields reference the respective data
    sources and should match with the configured source_id in the datasource configuration.
    The variable field specifies the (internal; i.e. after id mapping) physical variable to be
    verified.
    """

    id: Annotated[
        str,
        Field(
            min_length=1,
            description="Unique identifier for the verification pair. All strings are allowed, but "
            "'input_data' is reserved. And can not be used.",
        ),
    ]
    observations_source_id: Annotated[Source, Field(description="Source ID for the observations.")]
    simulations_source_id: Annotated[Source, Field(description="Source ID for the simulations.")]
    variable: Annotated[
        Variable,
        Field(description="The (internal) physical variable to be verified."),
    ]

    model_config = {
        "frozen": True,
    }

    @field_validator("id", mode="after")
    @classmethod
    def id_not_reserved(cls, v: str) -> str:
        """Reject 'input_data', reserved as the DataTree node name for raw input datasets."""
        if v == "input_data":
            msg = "'input_data' is a reserved id and cannot be used as a verification pair id."
            raise ValueError(msg)
        return v

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
    """Config for multiple local files using pathlib.Path.glob."""

    directory: str
    filename_glob: Annotated[
        str,
        Field(
            description="A valid filename glob, like '*.nc' for all netcdf files.",
        ),
    ]

    @property
    def paths(self) -> Iterator[Path]:
        """Return all filepaths as Path objects in a deterministic sorted order."""
        return iter(sorted(Path(self.directory).rglob(self.filename_glob)))


class FewsWebserviceAuthConfig(BaseSettings):
    """
    Get url, username and password safely from environment variables.

    This config class inherits from :class:`pydantic_settings.BaseSettings`,
    that will try to infer field values from environment variables.

    Environment variables:

    - ``FEWSWEBSERVICE_URL``: URL of the FEWS webservice (required).
    - ``FEWSWEBSERVICE_USERNAME``: Username for the FEWS webservice (set to "" if not required).
    - ``FEWSWEBSERVICE_PASSWORD``: Password for the FEWS webservice (set to "" if not required).

    see: https://docs.pydantic.dev/latest/concepts/pydantic_settings/#usage

    Notes
    -----
    For local development, an easy and recommended way to set environment variables is to
    use a ``.env`` file in the project root with the required environment variables, and they
    will be automatically loaded when you instantiate this config class. For example:

    .. code-block:: python

        from dotenv import load_dotenv
        load_dotenv()  # Load environment variables from .env file

        # You can check that the variables are loaded correctly
        import os
        print(os.getenv("FEWSWEBSERVICE_URL"))
    """

    model_config = SettingsConfigDict(env_prefix="FEWSWEBSERVICE_")

    url: AnyUrl
    username: SecretStr
    password: SecretStr


class S3AuthConfig(BaseSettings):
    """
    Get S3 credentials and connection info safely from environment variables.

    This config class inherits from :class:`pydantic_settings.BaseSettings`,
    that will try to infer field values from environment variables.

    Environment variables (all optional):

    - ``S3_ENDPOINT_URL``: Custom S3 endpoint (e.g. for MinIO or non-AWS S3).
    - ``S3_REGION_NAME``: AWS region.
    - ``S3_ACCESS_KEY_ID``: Access key id.
    - ``S3_SECRET_ACCESS_KEY``: Secret access key.
    - ``S3_SESSION_TOKEN``: Session token (for temporary credentials).
    - ``S3_ANON``: Set to ``true`` for anonymous access to public buckets.

    Fields default to ``None`` (or ``False`` for ``anon``) so that callers can rely on
    ``s3fs`` / ``botocore`` falling back to standard AWS credential discovery (e.g.
    ``~/.aws/credentials``, instance metadata) when a value is not explicitly set.

    see: https://docs.pydantic.dev/latest/concepts/pydantic_settings/#usage
    """

    model_config = SettingsConfigDict(env_prefix="S3_")

    endpoint_url: AnyUrl | None = None
    region_name: str | None = None
    access_key_id: SecretStr | None = None
    secret_access_key: SecretStr | None = None
    session_token: SecretStr | None = None
    anon: bool = False

    def to_storage_options(self) -> dict[str, object]:
        """Build a ``storage_options`` dict for ``xr.open_zarr`` / ``s3fs``.

        Only keys with non-``None`` values are included. ``SecretStr`` values are
        unwrapped to their plain string form so that ``s3fs`` can use them.

        We currently assume a minio config, such as::

            storage_options = {
                "key": "****",
                "secret": "*****",
                "client_kwargs": {
                    "endpoint_url": "https://s3.deltares.nl",
                    "region_name": "eu-west-1",
                },
                "config_kwargs": {
                    "s3": {
                        "addressing_style": "path",
                    },
                },
            }

        """
        client_kwargs: dict[str, str] = {}
        if self.endpoint_url is not None:
            client_kwargs["endpoint_url"] = str(self.endpoint_url)
        if self.region_name is not None:
            client_kwargs["region_name"] = self.region_name

        options: dict[str, object] = {}
        if self.access_key_id is not None:
            options["key"] = self.access_key_id.get_secret_value()
        if self.secret_access_key is not None:
            options["secret"] = self.secret_access_key.get_secret_value()
        if self.session_token is not None:
            options["token"] = self.session_token.get_secret_value()
        if client_kwargs:
            options["client_kwargs"] = client_kwargs

        return options


class BaseZarrConfig(BaseModel):
    """Configuration for connecting to a single Zarr store.

    The store may live on the local filesystem or on remote object storage (e.g. S3), and is
    read via ``xr.open_zarr`` / written via ``xr.Dataset.to_zarr`` / ``xr.DataTree.to_zarr``.
    This is a shared base: it's used directly for the veriflow cache (:class:`ZarrCacheConfig`),
    for reading Zarr datasources (``ZarrConfig``), and for writing Zarr datasinks
    (:class:`CFCompliantZarrConfig`).
    """

    path: Annotated[
        str,
        Field(
            min_length=1,
            description="Path to a single Zarr store. Local filesystem path (absolute or "
            "relative) or a remote URL such as 's3://bucket/key/store.zarr'.",
        ),
    ]
    auth_config: Annotated[
        S3AuthConfig | None,
        Field(
            default=None,
            description="Authentication configuration for remote stores. Only consulted "
            "when 'path' points to an 's3://' location. When the path is remote and this is "
            "left unset, credentials are loaded automatically from S3_-prefixed environment "
            "variables, so configuring 'auth_config: {}' in YAML is not required.",
        ),
    ] = None
    storage_options: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description="Additional storage_options forwarded to xr.open_zarr. Merged on "
            "top of the options derived from 'auth_config'. Use this for advanced "
            "fsspec / s3fs settings not exposed by S3AuthConfig.",
        ),
    ] = None
    consolidated: Annotated[
        bool | None,
        Field(
            default=None,
            description="Whether to use consolidated metadata when opening the store. "
            "Forwarded to xr.open_zarr. Default ('None') lets xarray auto-detect.",
        ),
    ] = None

    def is_remote_path(self) -> bool:
        """Return True if ``path`` looks like a remote/fsspec URL (e.g. ``s3://``)."""
        return "://" in self.path

    @model_validator(mode="after")
    def validate_zarr_path_accessible(self) -> Self:
        """Check that a local cache dir exists, or initialize S3 auth for remote paths."""
        if self.is_remote_path():
            # Auto-load S3 credentials from S3_-prefixed environment variables so users do
            # not have to explicitly configure 'auth_config: {}' for remote stores.
            if self.auth_config is None:
                self.auth_config = S3AuthConfig()
        else:
            path = Path(self.path)
            if not path.exists():
                path.mkdir(parents=True)
            elif not path.is_dir() and access(path, R_OK):
                msg = "Cache directory is not an accessible directory."
                raise NotADirectoryError(msg)
        return self
