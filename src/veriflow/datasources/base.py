"""Module with the base class that all datasources should inherit from."""

import logging
from abc import abstractmethod
from typing import ClassVar, Self, cast

import pandas as pd
import xarray as xr

from veriflow.base import Base
from veriflow.cache.cache import (
    CacheRequest,
    DataRequest,
    ZarrCache,
)
from veriflow.cache.utils import combine_cached_and_fetched_data
from veriflow.configuration.base import (
    BaseDatasourceConfig,
)
from veriflow.configuration.utils import LeadTimes, TimePeriod
from veriflow.constants import (
    FORECAST_DATA_TYPES,
    HISTORICAL_DATA_TYPES,
    DataType,
    SpatialType,
    StandardDim,
    TimeUnits,
)
from veriflow.datasources.inputschemas import validate_input_data
from veriflow.types import DataSpec

logger = logging.getLogger(__name__)


__all__ = [
    "BaseDatasource",
    "BaseDatasourceConfig",
]

CACHABLE_DATA_TYPES = FORECAST_DATA_TYPES + HISTORICAL_DATA_TYPES  # type: ignore[misc]


class BaseDatasource(Base):
    """Class to inherit from, defines the required methods and attributes."""

    kind: str = ""
    config_class: type[BaseDatasourceConfig] = BaseDatasourceConfig
    supported_data_specs: ClassVar[set[DataSpec]] = set()
    _cache: ZarrCache | None = None

    def __init__(self, config: BaseDatasourceConfig) -> None:
        self.config: BaseDatasourceConfig = config
        self.data_type = config.data_type
        self.spatial_type = config.spatial_type
        self.dataset: xr.Dataset = xr.Dataset()

    @property
    def data_type(self) -> DataType:
        """Whether the instance represents simulations or observations data."""
        return self.config.data_type

    @data_type.setter
    def data_type(self, new_data_type: DataType) -> None:
        supported_data_types = {dt for dt, _ in self.supported_data_specs}
        if new_data_type not in supported_data_types:
            msg = (
                f"Data type '{new_data_type}' is not supported ",
                f"by {self.__class__.__name__}",
            )
            raise NotImplementedError(msg)

        self._data_type = new_data_type

    @property
    def spatial_type(self) -> SpatialType:
        """Spatial structure (point/gridded) of the datasource's data."""
        return self.config.spatial_type

    @spatial_type.setter
    def spatial_type(self, new_spatial_type: SpatialType) -> None:
        if (self.data_type, new_spatial_type) not in self.supported_data_specs:
            msg = (
                f"Data type / spatial type combination "
                f"'({self.data_type}, {new_spatial_type})' is not supported "
                f"by {self.__class__.__name__}"
            )
            raise NotImplementedError(msg)

        self._spatial_type = new_spatial_type

    @property
    def cache(self) -> ZarrCache | None:
        """Return the cache instance if caching is enabled, otherwise None."""
        return self._cache

    @cache.setter
    def cache(self, new_cache: ZarrCache | None) -> None:
        """Set the cache instance for this datasource."""
        self._cache = new_cache

    @property
    @abstractmethod
    def configured_stations(self) -> set[str] | None:
        """Return the standardized internal station identifiers configured for this datasource.

        This standardized format is needed for caching across different datasources. If your
        datasource implementation does not have any configurable stations, return None.
        """
        raise NotImplementedError

    @configured_stations.setter
    @abstractmethod
    def configured_stations(self, new_stations: set[str]) -> None:
        """Set the standardized internal station identifiers configured for this datasource.

        This standardized format is needed for caching across different datasources.If your
        datasource implementation does not have any configurable stations, return None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def configured_variables(self) -> set[str] | None:
        """Return the standardized internal variable identifiers configured for this datasource.

        This standardized format is needed for caching across different datasources. If your
        datasource implementation does not have any configurable variables, return None.
        """
        raise NotImplementedError

    @configured_variables.setter
    @abstractmethod
    def configured_variables(self, new_variables: set[str]) -> None:
        """Set the standardized internal variable identifiers configured for this datasource.

        This standardized format is needed for caching across different datasources. If your
        datasource implementation does not have any configurable variables, return None.
        """
        raise NotImplementedError

    @property
    def configured_variables_internal(self) -> set[str] | None:
        """Return the standardized internal variable identifiers configured for this datasource.

        This standardized format is needed for caching across different datasources. If your
        datasource implementation does not have any configurable variables, return None.
        """
        if (
            self.config.id_mapping is not None
            and self.config.id_mapping.variable is not None
            and self.configured_variables is not None
        ):
            if self.config.source_id in self.config.id_mapping.variable.sources:
                return self.config.id_mapping.variable.rename_external_to_internal(
                    self.configured_variables,
                    self.config.source_id,
                )
        else:
            return self.configured_variables
        return self.configured_variables

    @abstractmethod
    def fetch_data(self) -> Self:
        """Fetch data from datasource."""

    def _validate_data_type(self) -> None:
        # Check that the datatype is defined, and consistent with the config
        if "data_type" not in self.dataset.attrs:  # type:ignore[misc]
            msg = "The fetched dataset does not have a 'data_type' attribute."
            raise ValueError(msg)

        if self.dataset.attrs["data_type"] != self.data_type:  # type:ignore[misc]
            msg = (
                f"The data type of the fetched dataset "
                f"({self.dataset.attrs['data_type']}) does not match the expected data "  # type:ignore[misc]
                f"type ({self.config.data_type})."
            )
            raise ValueError(msg)

    def _persist_configured_source_id_to_attrs(self) -> None:
        # Make sure the source attribute is set to the expected source
        self.dataset.attrs["source_id"] = self.config.source_id  # type:ignore[misc]

    def _persist_configured_spatial_type_to_attrs(self) -> None:
        # Always persist spatial_type so the full (temporal + spatial) data type is
        # retrievable downstream (e.g. for schema selection and score dispatch).
        self.dataset.attrs["spatial_type"] = self.config.spatial_type  # type:ignore[misc]

    def _validate_dataset_structure_against_schema(self) -> None:
        """Validate the fetched dataset against the expected schema."""
        validate_input_data(
            dataset=self.dataset,
        )

    def _validate_lead_times(self) -> None:
        """Check that lead times are provided for forecast data types."""
        if not self.config.lead_times and self.data_type in FORECAST_DATA_TYPES:
            msg = "Lead times must be provided in the config for forecast data types, but got None."
            raise ValueError(msg)

    @staticmethod
    def _filter_lead_times(
        dataset: xr.Dataset,
        lead_times: LeadTimes | None,
    ) -> xr.Dataset:
        """Filter forecast dataset on lead times."""
        if dataset.attrs["data_type"] in FORECAST_DATA_TYPES and lead_times is not None:  # type:ignore[misc]
            # Select only relevant lead times for simulations
            dataset = dataset.sel(
                lead_time=lead_times.timedelta64,
            )
        return dataset

    @staticmethod
    def _filter_times(dataset: xr.Dataset, verification_period_on_time: TimePeriod) -> xr.Dataset:
        """Filter the times outside the verification period and lead times."""
        data_type = dataset.attrs["data_type"]  # type:ignore[misc]

        if data_type in FORECAST_DATA_TYPES:  # type:ignore[misc]
            # Mask and drop time values outside of the configured vp
            filtered = dataset.where(
                (dataset[StandardDim.time] >= verification_period_on_time.start_datetime64)
                & (dataset[StandardDim.time] <= verification_period_on_time.end_datetime64),
            )
            # Drop NaN values along frt and fp dims, if all values are NaN
            return filtered.dropna(dim=StandardDim.forecast_reference_time, how="all").dropna(
                dim=StandardDim.lead_time,
                how="all",
            )
        if data_type in HISTORICAL_DATA_TYPES:  # type:ignore[misc]
            # Mask and drop time values outside of the configured vp
            # Historical data type
            dataset = dataset.sel(
                {
                    StandardDim.time: slice(  # type:ignore[misc]
                        verification_period_on_time.start,
                        verification_period_on_time.end,
                    ),
                },
            )
        return dataset

    def validate_fetched_data(self) -> None:
        """Validate that the dataset is consistent with the config."""
        self._validate_data_type()
        self._persist_configured_source_id_to_attrs()
        self._persist_configured_spatial_type_to_attrs()
        self._validate_dataset_structure_against_schema()
        self._validate_lead_times()

    def filter_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """Filter the dataset on lead times and times outside the verification period."""
        dataset = self._filter_lead_times(
            dataset,
            self.config.lead_times,
        )
        return self._filter_times(
            dataset,
            self.config.verification_period_on_time,
        )

    def fetch_validate_filter_cache(self, *, clear_cache: bool = False) -> Self:
        """High-level wrapper to fetch, validate, filter and apply id mapping to the dataset."""
        self.fetch_data()
        self.validate_fetched_data()
        self.dataset = self.filter_dataset(self.dataset)

        if self.config.id_mapping is not None:
            self.dataset = self.config.id_mapping.apply(self.dataset)

        if self.cache is not None and self.cache.is_writable:
            if clear_cache:
                msg = (
                    f"Clearing existing cache for source '{self.config.source_id}' before writing "
                    f"the full requested dataset."
                )
                self.cache.clear(source=self.config.source_id)
                logger.info(msg)
            msg = f"Writing fetched dataset to cache for source '{self.config.source_id}'."
            self.cache.append(
                self.dataset,
                source=self.config.source_id,
            )
            logger.info(msg)
        return self

    def create_cache_request(
        self,
        cached_dataset: xr.Dataset,
    ) -> CacheRequest:
        """Get the cache request based on the cached dataset and the config."""
        # Determine what data is missing from the cache
        if self.data_type in HISTORICAL_DATA_TYPES:
            return CacheRequest(
                variables=DataRequest(
                    requested=self.configured_variables_internal,
                    cached=cast("set[str]", set(cached_dataset.data_vars)),
                ),
                stations=DataRequest(
                    requested=self.configured_stations,
                    cached=set(cached_dataset[StandardDim.station].values),  # type: ignore[misc]
                ),
                time_period=DataRequest(
                    requested=TimePeriod(
                        start=self.config.verification_period_on_time.start,
                        end=self.config.verification_period_on_time.end,
                    ),
                    cached=TimePeriod(
                        start=pd.Timestamp(
                            cached_dataset[StandardDim.time].min().item(),  # type: ignore[misc]
                        ).to_pydatetime(),
                        end=pd.Timestamp(
                            cached_dataset[StandardDim.time].max().item(),  # type: ignore[misc]
                        ).to_pydatetime(),
                    ),
                ),
            )
        if self.data_type in FORECAST_DATA_TYPES:
            if self.config.lead_times is None:
                msg = "lead_times must be configured for forecast data types."
                raise ValueError(msg)
            cached_lead_times_array = cached_dataset[StandardDim.lead_time].to_numpy()  # type: ignore[misc]
            return CacheRequest(
                variables=DataRequest(
                    requested=self.configured_variables_internal,
                    cached=cast("set[str]", set(cached_dataset.data_vars)),
                ),
                stations=DataRequest(
                    requested=self.configured_stations,
                    cached=set(cached_dataset[StandardDim.station].values),  # type: ignore[misc]
                ),
                forecast_reference_time_period=DataRequest(
                    requested=TimePeriod(
                        start=self.config.verification_period_on_frt.start,
                        end=self.config.verification_period_on_frt.end,
                    ),
                    cached=TimePeriod(
                        start=pd.Timestamp(
                            cached_dataset[StandardDim.forecast_reference_time].min().item(),  # type: ignore[misc]
                        ).to_pydatetime(),
                        end=pd.Timestamp(
                            cached_dataset[StandardDim.forecast_reference_time].max().item(),  # type: ignore[misc]
                        ).to_pydatetime(),
                    ),
                ),
                lead_times=DataRequest(
                    requested=self.config.lead_times,
                    cached=LeadTimes(
                        unit=TimeUnits.nanosecond,
                        values=[  # type: ignore[misc]
                            int(td)  # type: ignore[misc]
                            for td in cached_lead_times_array.astype(  # type: ignore[misc]
                                "timedelta64[ns]",
                            ).astype("int64")
                        ],
                    ),
                ),
            )
        msg = f"Unsupported data type '{self.data_type}' for creating cache request."
        raise NotImplementedError(msg)

    @staticmethod
    def get_cached_data(
        cached_dataset: xr.Dataset,
        datasource: "BaseDatasource",
    ) -> xr.Dataset:
        """Get the dataset from the cache based on the datasource configuration."""
        config = datasource.config
        variables = datasource.configured_variables_internal
        stations = datasource.configured_stations
        subset = cached_dataset if variables is None else cached_dataset[sorted(variables)]

        if len(subset.data_vars) == 0:
            msg = (
                f"No data found in cache for source '{config.source_id}' with the requested "
                "variables."
            )
            raise ValueError(msg)

        # If all requested data is available in the cache, load directly.
        if config.data_type in FORECAST_DATA_TYPES:
            if config.lead_times is None:
                msg = "lead_times must be configured for forecast data types."
                raise ValueError(msg)
            frt = config.verification_period_on_frt

            selection: dict[str, object] = {
                StandardDim.forecast_reference_time: slice(frt.start, frt.end),  # type: ignore[misc]
                StandardDim.lead_time: config.lead_times.timedelta64,
            }
        elif config.data_type in HISTORICAL_DATA_TYPES:
            time_period = config.verification_period_on_time
            subset = subset.sortby(StandardDim.time)
            selection = {
                StandardDim.time: slice(time_period.start, time_period.end),  # type: ignore[misc]
            }
        else:
            msg = f"Unsupported data type '{config.data_type}' for loading from cache."
            raise NotImplementedError(msg)

        if stations is not None:
            selection[StandardDim.station] = sorted(stations)
        return subset.sel(selection)

    def _check_reason_to_skip_getting_data_from_cache(
        self,
    ) -> str | None:
        """Return why the cache cannot be used for this request, or None if it can."""
        if self.cache is None:
            return "no cache configured"
        if self.data_type not in CACHABLE_DATA_TYPES:  # type:ignore[misc]
            return f"data type '{self.data_type}' is not cacheable"
        if self.config.source_id not in self.cache.sources:
            return f"source '{self.config.source_id}' is not registered in the cache"
        if self.configured_stations is None:
            return "configured stations are None"
        if self.configured_variables is None:
            return "configured variables are None"

        return None

    @staticmethod
    def _validate_type(instance: object, expected_type: type) -> None:
        """Check that the instance is of the expected type."""
        if not isinstance(instance, expected_type):
            msg = (
                f"Expected instance of {expected_type.__name__}, got "
                f"{type(instance).__name__} instead."
            )
            raise TypeError(msg)

    def get_data(self) -> Self:
        """Get data and make use of cache if configured."""
        # If no cache is configured, or the data type is not cacheable, fetch and process the data
        # directly from the datasource, without using the cache.
        msg = f"Starting data fetch for {self.config.source_id} from {self.__class__.__name__}."
        logger.info(msg)

        # Check if we should skip fetching data from the cache, and if so, fetch and process the
        # data directly from the datasource.
        reason = self._check_reason_to_skip_getting_data_from_cache()

        if reason is not None:
            msg = (
                f"Skipped reading data from cache for datasource {self.__class__.__name__}: "
                f" {reason}."
            )
            logger.debug(msg)
            return self.fetch_validate_filter_cache()

        # Validate and cast type of the cache to ensure it is a ZarrCache instance.
        self._validate_type(self.cache, ZarrCache)
        self.cache = cast("ZarrCache", self.cache)

        # Get the cached dataset for the configured source.
        cached_dataset: xr.Dataset = self.cache.get_dataset(  # type:ignore[assignment]
            source=self.config.source_id,
        )

        # Validate and cast type of the cached dataset to ensure it is an xarray Dataset.
        self._validate_type(cached_dataset, xr.Dataset)  # type: ignore[misc]
        cached_dataset = cast("xr.Dataset", cached_dataset)

        cache_request = self.create_cache_request(cached_dataset)

        # No data is missing from the cache, so we can load it directly and skip fetching from the
        # datasource.
        if cache_request.missing_count == 0:
            msg = (
                "All requested data is available in the cache for source "
                f"'{self.config.source_id}', skipping fetch from datasource."
            )
            logger.info(msg)
            cached_dataset = self.get_cached_data(
                cached_dataset=cached_dataset,
                datasource=self,
            )
            self.dataset = cached_dataset
            return self

        # Some data is missing from the cache. Try to split the datasource so we only fetch the
        # missing data (works only when exactly one dim is missing). If split_config returns
        # None — either because multiple dims are missing or the missing data isn't expressible
        # by tweaking the datasource — fall back to fetching the full requested dataset.
        split_result = cache_request.split_config(self)
        if split_result is None:
            msg = (
                "Some requested data is missing from the cache for source "
                f"'{self.config.source_id}', but the datasource cannot be split to fetch only the "
                "missing data. Fetching the full requested dataset from the datasource."
            )
            logger.info(msg)
            return self.fetch_validate_filter_cache(clear_cache=True)

        missing_dim = cache_request.missing_dims[0]
        fetch_from_datasource, fetch_from_cache = split_result

        msg = (
            f"Some requested data is missing from the cache for source '{self.config.source_id}', "
            f"fetching only the missing slice along dim '{missing_dim}' from the datasource."
        )
        logger.info(msg)
        # Fetch only the missing slice from the datasource.
        fetch_from_datasource.fetch_validate_filter_cache()
        newly_fetched_dataset = fetch_from_datasource.dataset

        if self.cache.is_writable:
            # Incrementally append only the newly fetched slice to the store (no full-store
            # read/rewrite and no in-memory load), then read the full requested window back
            # lazily from the updated store.
            msg = (
                f"Writing newly fetched slice along dim '{missing_dim}' to cache for source "
                f"'{self.config.source_id}'."
            )
            logger.info(msg)

            # The cache append method expects None for the append_dim if the dimension is not
            # present in the dataset.
            append_dim = missing_dim if missing_dim != "variable" else None
            self.cache.append(
                new_dataset=newly_fetched_dataset,
                source=self.config.source_id,
                append_dim=append_dim,  # type: ignore[arg-type]
            )
            self.dataset: xr.Dataset = self.get_cached_data(  # type: ignore[no-redef]
                cached_dataset=self.cache.get_dataset(  # type:ignore[arg-type]
                    source=self.config.source_id,
                ),
                datasource=self,
            )
        else:
            msg = (
                f"Cache is read-only, skipping write of newly fetched slice along dim "
                f"'{missing_dim}' for source '{self.config.source_id}'."
            )
            logger.info(msg)
            # Read-only cache: combine the lazily-loaded cached slice with the freshly fetched
            # data. The store is not modified, so the lazy cached reference stays valid.
            cached_dataset = self.get_cached_data(
                cached_dataset=cached_dataset,
                datasource=fetch_from_cache,
            )
            self.dataset = combine_cached_and_fetched_data(
                cached_dataset=cached_dataset,
                fetched_dataset=newly_fetched_dataset,
                dim=missing_dim,
            )
        msg = f"Successfully got {self.config.source_id} data from {self.__class__.__name__}."
        logger.info(msg)
        return self
