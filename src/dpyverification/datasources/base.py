"""Module with the base class that all datasources should inherit from."""

import hashlib
from abc import abstractmethod
from os import R_OK, access
from typing import ClassVar, Self

import xarray
import xarray as xr

from dpyverification.base import Base
from dpyverification.configuration.base import BaseDatasourceConfig
from dpyverification.constants import FORECAST_TIMESERIES_KIND, TimeseriesKind


class BaseDatasource(Base):
    """Class to inherit from, defines the required methods and attributes."""

    kind: str = ""
    config_class: type[BaseDatasourceConfig] = BaseDatasourceConfig
    supported_timeseries_kinds: ClassVar[set[TimeseriesKind]] = set()

    def __init__(self, config: BaseDatasourceConfig) -> None:
        self.config: BaseDatasourceConfig = config
        self.timeseries_kind = config.timeseries_kind
        self.data_array = xarray.DataArray()

    @property
    def timeseries_kind(self) -> str:
        """Whether the instance represents sim or obs data."""
        return self.config.timeseries_kind

    @timeseries_kind.setter
    def timeseries_kind(self, new_timeseries_kind: TimeseriesKind) -> None:
        if new_timeseries_kind not in self.supported_timeseries_kinds:
            msg = (
                f"Timeseries kind '{new_timeseries_kind}' is not supported ",
                f"by {self.__class__.__name__}",
            )
            raise NotImplementedError(msg)

        self._timeseries_kind = new_timeseries_kind

    @abstractmethod
    def fetch_data(self) -> Self:
        """Fetch data from datasource."""

    def get_data(self) -> Self:
        """Get cached data, or fetch and cache."""
        config_json = self.config.model_dump_json().encode("utf-8")
        config_hash = hashlib.sha256(config_json).hexdigest()

        cache_dir = self.config.general.cache_dir

        # Create cache if not exists
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        # If it exists, check it's an accessible dir
        elif not cache_dir.is_dir() & access(cache_dir, R_OK):
            msg = "Cache directory is not an accessible directory."
            raise NotADirectoryError(msg)

        # Define file path for caching
        cached_dataset_path = cache_dir / f"{self.__class__.__name__}_{config_hash}.nc"

        if cached_dataset_path.exists():
            self.data_array = xr.open_dataarray(cached_dataset_path)
            return self

        # Go fetch and cache
        self.fetch_data()

        # Apply re-naming based on configured id mapping, if not None
        if self.config.id_mapping is not None:
            self.data_array = self.config.id_mapping.rename_data_array(self.data_array)

        # Make sure the name of the array is set to the configured source
        self.data_array.name = self.config.source

        # Select only relevant time stamps
        self.data_array = self.data_array.sel(
            time=slice(self.config.verification_period.start, self.config.verification_period.end),
        )

        # Select only relevant forecast periods for simulations
        if self.data_array.attrs["timeseries_kind"] in FORECAST_TIMESERIES_KIND:  # type:ignore[misc]
            self.data_array = self.data_array.sel(
                forecast_period=self.config.forecast_periods.timedelta64,
            )
        # Write to cache
        self.data_array.to_netcdf(cached_dataset_path)

        return self
