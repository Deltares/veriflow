"""Config for the cache."""

from enum import StrEnum

from pydantic import Field

from veriflow.configuration.utils import BaseZarrConfig


class CacheBackEnd(StrEnum):
    """The veriflow cache backends."""

    zarr = "zarr"


class CacheType(StrEnum):
    """The veriflow cache types."""

    local = "local"
    remote = "remote"


class ReadWriteMode(StrEnum):
    """The veriflow cache read/write modes."""

    read = "r"
    read_write = "rw"


class ZarrCacheConfig(BaseZarrConfig):
    """Configuration for the veriflow cache.

    The cache will be materialized as a Zarr store on the local filesystem or remote object storage
    (e.g. S3). When configured, the cache will be used to store and retrieve datasets from any
    datasource. For example: when requesting forecast data from a datasource, and part of the data
    is already cached, the cache will be used to retrieve the cached data and only the missing data
    will be fetched from the datasource.
    """

    read_write_mode: ReadWriteMode = Field(
        ReadWriteMode.read,
        description="The read/write mode for the cache.",
    )
