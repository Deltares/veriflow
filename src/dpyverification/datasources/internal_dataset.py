"""A datasource compliant with the internal data structure."""

from pathlib import Path
from typing import ClassVar, Self

import xarray as xr

from dpyverification.configuration import InternalDatasetConfig
from dpyverification.constants import TimeseriesKind
from dpyverification.datasources.base import BaseDatasource


class InternalDataset(BaseDatasource):
    """An internal dataset, loaded by xarray from files."""

    kind = "internal_dataset"
    config_class = InternalDatasetConfig
    supported_timeseries_kinds: ClassVar[set[TimeseriesKind]] = {
        TimeseriesKind.observed_historical,
        TimeseriesKind.simulated_forecast_ensemble,
        TimeseriesKind.simulated_forecast_probabilistic,
        TimeseriesKind.simulated_forecast_single,
    }

    def __init__(self, config: InternalDatasetConfig) -> None:
        self.config: InternalDatasetConfig = config

    def fetch_data(self) -> Self:
        """Load a local file using xarray."""
        file_path = Path(self.config.directory) / self.config.filename
        self.data_array = xr.open_dataarray(file_path)
        return self
