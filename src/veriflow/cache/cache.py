"""Zarr cache implementation."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import fsspec  # type:ignore[import-untyped]
import numpy as np
import xarray as xr
import zarr
from pydantic import BaseModel, model_validator
from zarr.errors import GroupNotFoundError

from veriflow.cache.config import ReadWriteMode, ZarrCacheConfig
from veriflow.configuration.utils import LeadTimes, TimePeriod
from veriflow.constants import FORECAST_DATA_TYPES, HISTORICAL_DATA_TYPES, StandardDim, TimeUnits

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from veriflow.datasources.base import BaseDatasource

DataRequestValue = set[str] | TimePeriod | LeadTimes | None

__all__ = [  # noqa: RUF022
    "ZarrCache",
    "ZarrCacheConfig",
    "CacheRequest",
    "DataRequest",
]


class DataRequest(BaseModel):
    """Helper class to track requested and cached data and what data is missing from the cache."""

    requested: DataRequestValue
    cached: DataRequestValue

    @model_validator(mode="after")
    def validate_requested_and_cached(self) -> Self:
        """Validate that the requested and cached values are of the same type."""
        if type(self.requested) != type(self.cached):  # noqa: E721
            msg = "Requested and cached values must be of the same type."
            raise ValueError(msg)
        return self

    @staticmethod
    def find_missing_and_available_time_period(
        requested: TimePeriod,
        cached: TimePeriod,
    ) -> tuple[TimePeriod | None, TimePeriod | None]:
        """Find the missing time period between the requested and cached time periods.

        This helper function is used to determine what time period needs to be fetched from the
        datasource given what is already available in the cache. The returned tuple contains the
        missing time period and the available time period, respectively.

        The requested period ``R`` is compared against the cached period ``C`` on the time
        axis. The five possible cases and their outcomes are shown below
        (``#`` marks the extent of each period):

        .. code-block:: text

            time --->

            (1) R inside C  ->  full hit
                R      ####
                C   ##########
                missing:   (none)          available: R

            (2) R and C disjoint
                R   ####
                C            ####
                missing:   R               available: (none)

            (3) C inside R
                R   ##########
                C      ####
                missing:   R               available: (none)

            (4) left overlap (R starts before C)
                R   ######
                C       ######
                missing:   R.start .. C.start
                available: C.start .. R.end

            (5) right overlap (R starts inside C)
                R       ######
                C   ######
                missing:   C.end .. R.end
                available: R.start .. C.end
        """
        # Requested fully contained in (or equal to) cached → nothing missing.
        if requested.start >= cached.start and requested.end <= cached.end:
            return None, requested
        #  RRRR
        #        CCCC
        if requested.end <= cached.start:
            return requested, None
        #        RRRR
        #  CCCC
        if requested.start >= cached.end:
            return requested, None
        #  RRRR
        #   CC      (cached strictly inside requested)
        if requested.start <= cached.start and requested.end >= cached.end:
            return requested, None
        # RRRR
        #   CCCC    (left overlap: requested starts before cached, ends inside cached)
        if requested.start < cached.start <= requested.end < cached.end:
            return TimePeriod(start=requested.start, end=cached.start), TimePeriod(
                start=cached.start,
                end=requested.end,
            )
        #   RRRR
        # CCCC      (right overlap: requested starts inside cached, ends after cached)
        if cached.start < requested.start <= cached.end < requested.end:
            return TimePeriod(start=cached.end, end=requested.end), TimePeriod(
                start=requested.start,
                end=cached.end,
            )
        msg = "Unexpected case of intersecting time periods."
        raise ValueError(msg)

    @staticmethod
    def _to_lead_times(values: list[np.timedelta64]) -> LeadTimes:
        """Convert a list of ``timedelta64`` values to a nanosecond ``LeadTimes``."""
        return LeadTimes(
            unit=TimeUnits.nanosecond,
            values=[int(td / np.timedelta64(1, "ns")) for td in values],
        )

    @property
    def get_missing_and_available(  # noqa: PLR0911
        self,
    ) -> tuple[DataRequestValue, DataRequestValue]:
        """Return the missing and available data given what is cached."""
        requested, cached = self.requested, self.cached
        if requested is None:
            return None, None
        if cached is None:
            return requested, None

        # Sets are used for variables and stations.
        if isinstance(requested, set) and isinstance(cached, set):
            missing = requested - cached
            return missing if len(missing) else None, (requested - missing) if len(
                requested - missing,
            ) else None

        # Time periods are used for forecast reference times and historical periods.
        if isinstance(requested, TimePeriod) and isinstance(cached, TimePeriod):
            return self.find_missing_and_available_time_period(requested, cached)

        # Lead times are used for forecast lead times.
        if isinstance(requested, LeadTimes) and isinstance(cached, LeadTimes):
            cached_values = cached.timedelta64
            missing_lts = [lt for lt in requested.timedelta64 if lt not in cached_values]
            available_lts = [lt for lt in requested.timedelta64 if lt in cached_values]
            if not len(missing_lts) > 0:
                return None, requested
            if not len(available_lts) > 0:
                return requested, None
            return self._to_lead_times(missing_lts) if len(
                missing_lts,
            ) > 0 else None, self._to_lead_times(
                available_lts,
            ) if len(available_lts) > 0 else None

        msg = "Unexpected case of missing data."
        raise ValueError(msg)


class CacheRequest(BaseModel):
    """Determine what requested data is missing from the cache.

    Handles both historical requests (``time_period``) and forecast requests
    (``forecast_reference_time_period`` + ``lead_times``); fields that do not apply stay ``None``.
    """

    variables: DataRequest
    stations: DataRequest
    time_period: DataRequest | None = None
    forecast_reference_time_period: DataRequest | None = None
    lead_times: DataRequest | None = None

    _DIM_MAP: ClassVar[dict[str, str]] = {
        "variables": "variable",
        "stations": StandardDim.station,
        "time_period": StandardDim.time,
        "forecast_reference_time_period": StandardDim.forecast_reference_time,
        "lead_times": StandardDim.lead_time,
    }

    @property
    def _requests(self) -> dict[str, DataRequest]:
        """Return the populated ``DataRequest`` fields keyed by field name."""
        candidates: dict[str, DataRequest | None] = {
            "variables": self.variables,
            "stations": self.stations,
            "time_period": self.time_period,
            "forecast_reference_time_period": self.forecast_reference_time_period,
            "lead_times": self.lead_times,
        }
        return {name: req for name, req in candidates.items() if req is not None}

    @property
    def _missing(self) -> dict[str, DataRequestValue]:
        """Return a mapping of field name to its missing data (``None`` if nothing missing)."""
        return {name: req.get_missing_and_available[0] for name, req in self._requests.items()}

    @property
    def missing_count(self) -> int:
        """Return the number of dimensions along which data is missing from the cache."""
        return sum(value is not None for value in self._missing.values())

    @property
    def missing_dims(self) -> list[str]:
        """Return the dimensions along which data is missing from the cache."""
        return [self._DIM_MAP[name] for name, value in self._missing.items() if value is not None]

    @staticmethod
    def _apply(datasource: "BaseDatasource", field: str, value: DataRequestValue) -> None:
        """Apply the missing/available ``value`` for ``field`` to ``datasource``."""
        if field in ("time_period", "forecast_reference_time_period") and isinstance(
            value,
            TimePeriod,
        ):
            datasource.config.general.verification_period.start = value.start
            datasource.config.general.verification_period.end = value.end
        elif field == "variables" and isinstance(value, set):
            datasource.configured_variables = value
        elif field == "stations" and isinstance(value, set):
            datasource.configured_stations = value
        elif field == "lead_times" and isinstance(value, LeadTimes):
            datasource.config.general.lead_times = value

    def split_config(
        self,
        datasource: "BaseDatasource",
    ) -> tuple["BaseDatasource", "BaseDatasource"] | None:
        """Split the datasource into a datasource-fetch and a cache-fetch datasource.

        Only splits when exactly one dimension is missing from the cache; otherwise
        returns ``None`` so that all data is fetched from the datasource.
        """
        if self.missing_count != 1:
            return None
        field = next(name for name, value in self._missing.items() if value is not None)
        missing, available = self._requests[field].get_missing_and_available
        if missing is None or available is None:
            return None
        fetch_from_datasource = type(datasource)(datasource.config.model_copy(deep=True))
        fetch_from_cache = type(datasource)(datasource.config.model_copy(deep=True))
        self._apply(fetch_from_datasource, field, missing)
        self._apply(fetch_from_cache, field, available)
        return fetch_from_datasource, fetch_from_cache


class ZarrCache:
    """The Zarr cache.

    In veriflow, you can cache datasets fetched from any datasource using a Zarr store. The cache
    enables incremental appends of new data along a single dimension
    (e.g. time, lead time, or station), and can be materialized on both a local filesystem and
    remote object storage (e.g. S3). An example use-case for using the cache is the operational
    application of veriflow, where new forecast data is generated and added to the cache on a daily
    basis, or when working with multiple people on a centralized verification project, where
    high-speed access to a shared cache is desired.

    For now, the cache is only used to store datasets fetched from other datasources, and is not
    used to cache computation results.

    The cache is configured in the :attr:`veriflow.configuration.base.GeneralInfoConfig.cache`
    field. All caching logic is handled in the
    :meth:`veriflow.datasources.base.BaseDatasource.get_data` method, which consults the cache
    configuration and uses this ZarrCache class to read/write cached datasets as needed.


    The current cache can handle the following scenarios:
    1. Historical data with missing time steps, variables, or stations.
    2. Forecast data with missing forecast reference times, lead times, variables, or stations

    The current implementation has the following limitations:
    1. When data is requested that is missing in the cache along more than one dimension, the cache
    will not retrieve data from the cache, but will instead fetch all data from the datasource.
    2. The current cache does not (yet) support caching of computation results.
    3. Large Zarr files on local filesystems may be slow to open due to the overhead of opening many
    small files. This is a known limitation of Zarr and is not specific to this implementation.
    However, using consolidated metadata (``consolidated=True``) can help mitigate this issue by
    reducing the number of files that need to be opened. However, datasets with a size of below
    1TB should be fine. See: https://github.com/zarr-developers/zarr-python/issues/86. Also note
    that caching is optional and can always be turned off.

    The layout of the Zarr cache is as follows, where each source has its own subgroup under the
    ``datasets`` group. The source name is identical to the ``source`` field in the datasource
    configuration.
    ::

        example_cache.zarr/
        ├── datasets/
        │   ├── observations/
        │   │   ├── temperature/
        │   │   ├── precipitation/
        │   │   └── discharge/
        │   └── forecasts_model_a
        │   │   ├── temperature/
        │   │   ├── precipitation/
        │   │   └── discharge/
        │   └── forecasts_model_b
        │   │   ├── temperature/
        │   │   ├── precipitation/
        │   │   └── discharge/

    A flowchart of the caching logic is shown below, which is implemented in the
    :meth:`veriflow.datasources.base.BaseDatasource.get_data` method.

    .. mermaid::

        flowchart TD
            A([get_data]) --> B{Cache configured and<br/>data type cacheable?}
            B -- no --> F[Fetch all data<br/>from datasource]
            B -- yes --> C[Open cached dataset<br/>from the Zarr store]
            C --> D{Any data cached<br/>for this source?}
            D -- no --> F
            D -- yes --> E[Build CacheRequest and<br/>determine missing dimensions]
            E --> G{How many dimensions<br/>are missing?}
            G -- none --> H["Load requested data from cache<br/>(full hit)"]
            G -- one --> I["Fetch missing, append to cache,<br/>read back dataset"]
            G -- many --> F
            F --> W[(Write to cache<br/>if writable)]
            I --> W
            H --> R([Return dataset])
            W --> R
    """

    def __init__(
        self,
        config: ZarrCacheConfig,
    ) -> None:
        self.config = config
        self.storage_options = self._build_storage_options()
        msg = f"Zarr cache initialized with path: {self.config.path}"
        logger.info(msg)

    def _build_storage_options(self) -> dict[str, object] | None:
        """Build storage_options for xr.open_zarr based on path and config.

        Returns ``None`` for non-remote (local) paths so that xarray opens the store
        directly from the local filesystem.
        """
        if not self.config.is_remote_path():
            return None

        options: dict[str, object] = {}
        if self.config.auth_config is not None:
            options.update(self.config.auth_config.to_storage_options())
        if self.config.storage_options is not None:
            options.update(self.config.storage_options)
        return options

    def get_dataset(self, source: str) -> xr.Dataset | None:
        """Open a dataset from the cache."""
        try:
            time_start = datetime.now()  # noqa: DTZ005
            ds: xr.Dataset = xr.open_zarr(
                self.config.path,
                group=f"datasets/{source}",
                storage_options=self.storage_options,
                consolidated=self.config.consolidated,
            )
            time_end = datetime.now()  # noqa: DTZ005
            msg = (
                f"Opened dataset for source '{source}' from {list(self.config.path)} "
                f"(took {(time_end - time_start).total_seconds():.2f} seconds)"
            )
            logger.info(msg)
        except (FileNotFoundError, KeyError):
            return None

        if not hasattr(ds, "data_type"):
            msg = (
                f"Dataset for source '{source}' in {list(self.config.path)} is missing the "
                "'data_type' attribute. This is required for proper sorting of the dataset."
            )
            raise ValueError(msg)

        # Incremental appends place new coordinate values at the end of the store, which can
        # leave the axis out of order; ``slice`` selection requires a monotonic index, so sort
        # first. Sorting only touches the (small) coordinate, keeping the data lazy.
        sort_dims: tuple = ()
        if ds.data_type in FORECAST_DATA_TYPES:  # type:ignore[misc]
            sort_dims = (
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
                StandardDim.station,
            )
        elif ds.data_type in HISTORICAL_DATA_TYPES:  # type:ignore[misc]
            sort_dims = (StandardDim.time, StandardDim.station)
        else:
            msg = (
                f"Dataset for source '{source}' in {self.config.path} has an unknown "
                f"data_type: {ds.data_type}"  # type:ignore[misc]
            )
            raise ValueError(msg)

        # Sort the proper dimensions, depending on the data_type.
        for dim in sort_dims:
            if dim in ds.coords:
                ds = ds.sortby(dim)

        return ds

    def append(
        self,
        new_dataset: xr.Dataset,
        source: str,
        append_dim: Literal[
            StandardDim.station,
            StandardDim.forecast_reference_time,
            StandardDim.lead_time,
            StandardDim.time,
        ]
        | None = None,
    ) -> None:
        """Incrementally add ``new_dataset`` to the cache along a single dimension.

        Only the coordinate index of the existing store is read to skip values that are already
        cached; the cached data itself is never loaded into memory, keeping the cache
        lazy-friendly. New coordinate values are appended along ``dim`` (``append_dim``), new
        variables are added in place (``mode="a"``), and an empty store is created from scratch.
        """
        cached = self.get_dataset(source)
        group = f"datasets/{source}"

        time_start = datetime.now()  # noqa: DTZ005
        # Dataset does not yet exist in store
        if cached is None or append_dim is None or append_dim not in cached.coords:
            to_add = new_dataset
            # Force append_dim to None if no dataset exists yet, since the first write creates the
            # store and cannot append along a non-existent dimension.
            append_dim = None

        # Append only the coordinate values not already cached.
        else:
            existing = cached[append_dim].to_numpy()  # type: ignore[misc]
            incoming = new_dataset[append_dim].to_numpy()  # type: ignore[misc]
            is_new = ~np.isin(incoming, existing)  # type: ignore[misc]
            if not bool(is_new.any()):
                return
            to_add = new_dataset.sel({append_dim: incoming[is_new]})  # type: ignore[misc]

        to_add.to_zarr(  # type: ignore[call-overload]
            self.config.path,
            group=group,
            mode="a",
            append_dim=append_dim,
            storage_options=self.storage_options,
            consolidated=self.config.consolidated,
        )
        time_end = datetime.now()  # noqa: DTZ005
        total_time_seconds = (time_end - time_start).total_seconds()
        msg = (
            f"Appended dataset for source '{source}' along dimension '{append_dim}' to "
            f"{self.config.path} (took {total_time_seconds:.2f} seconds)"
        )
        logger.info(msg)

    @property
    def is_remote(self) -> bool:
        """Return True if ``path`` looks like a remote/fsspec URL (e.g. ``s3://``)."""
        return "://" in self.config.path

    @property
    def is_writable(self) -> bool:
        """Return True if the cache is writable."""
        return self.config.read_write_mode == ReadWriteMode.read_write

    @property
    def sources(self) -> list[str]:
        """Return the list of sources (subgroups) cached in the ``datasets`` group."""
        try:
            group = zarr.open_group(  # type:ignore[misc]
                self.config.path,
                mode="r",
                path="datasets",
                storage_options=self.storage_options,
                use_consolidated=self.config.consolidated,
            )
        except (FileNotFoundError, KeyError, GroupNotFoundError):  # type:ignore[misc]
            return []

        return sorted(group.group_keys())  # type:ignore[misc]

    def clear(self, source: str | None = None) -> None:
        """Delete a Zarr archive from either a local filesystem or a remote MinIO server.

        If ``source`` is specified, only the corresponding subgroup is deleted.
        """
        options = self.config.storage_options or {}

        # Automatically infers protocol (e.g., 's3' or 'file') from the path
        fs, clean_path = fsspec.core.url_to_fs(self.config.path, **options)  # type:ignore[misc]

        if source is None:
            if fs.exists(clean_path):  # type:ignore[misc]
                fs.rm(clean_path, recursive=True)  # type:ignore[misc]
                msg = f"Cleared Zarr cache at {self.config.path}"
                logger.info(msg)
        else:
            group_path = f"{clean_path}/datasets/{source}"  # type:ignore[misc]
            if fs.exists(group_path):  # type:ignore[misc]
                fs.rm(group_path, recursive=True)  # type:ignore[misc]
                msg = f"Cleared Zarr cache for source '{source}' at {self.config.path}"
                logger.info(msg)
