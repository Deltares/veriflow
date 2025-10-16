"""An abstract implementation of a calculation."""

from abc import abstractmethod
from typing import ClassVar

import xarray as xr

from dpyverification.base import Base
from dpyverification.configuration.base import BaseScoreConfig
from dpyverification.constants import TimeseriesKind
from dpyverification.datamodel import InputDataset


class BaseScore(Base):
    """An abstract calculation class."""

    kind = ""  # to be defined by subclasses
    config_class: type[BaseScoreConfig] = BaseScoreConfig  # to be defined by subclasses
    supported_timeseries_kinds: ClassVar[set[TimeseriesKind]] = set()

    def __init__(self, config: BaseScoreConfig) -> None:
        self.config: BaseScoreConfig = config

    @abstractmethod
    def compute(
        self,
        input_dataset: InputDataset,
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Abstract calculation."""

    def validate_and_compute(
        self,
        input_dataset: InputDataset,
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Validate and compute."""
        for pair in self.config.verification_pairs:
            kind = input_dataset.get_simulated_timeseries_kind_from_pair(pair)
            if kind not in self.supported_timeseries_kinds:
                msg = f"The timeseries kind '{kind}' in verification pair '{pair.id}' "
                f"is not supported by {self.__class__.__name__}. "
                f"Supported types: {sorted(self.supported_timeseries_kinds)}."
                raise ValueError(msg)
        return self.compute(input_dataset)
