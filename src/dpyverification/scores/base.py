"""An abstract implementation of a calculation."""

from abc import abstractmethod
from typing import ClassVar

import xarray as xr

from dpyverification.base import Base
from dpyverification.configuration.base import (
    BaseCategoricalScoreConfig,
    BaseEvent,
    BaseScoreConfig,
)
from dpyverification.constants import DataType

__all__ = ["BaseScore", "BaseScoreConfig"]


class BaseScore(Base):
    """An abstract calculation class."""

    kind = ""  # to be defined by subclasses
    config_class: type[BaseScoreConfig] = BaseScoreConfig  # to be defined by subclasses
    supported_data_types: ClassVar[set[DataType]] = set()

    def __init__(self, config: BaseScoreConfig) -> None:
        self.config: BaseScoreConfig = config

    @abstractmethod
    def compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> xr.DataArray | xr.Dataset:
        """Abstract calculation."""

    def validate_and_compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> xr.DataArray | xr.Dataset:
        """Validate and compute."""
        data_type: DataType = sim.verification.data_type  # type:ignore[misc]
        if data_type not in self.supported_data_types:
            msg = f"The data type '{data_type} is not supported by"
            f"{self.__class__.__name__}. Supported types: "
            f"{sorted(self.supported_data_types)}."
            raise ValueError(msg)
        result = self.compute(obs, sim)
        if isinstance(result, xr.DataArray) and result.name is None:  # type:ignore[misc]
            result.name = self.config.score_adapter
        return result


class BaseCategoricalScore(Base):
    """An abstract calculation class for categorical scores."""

    kind = ""  # to be defined by subclasses
    config_class: type[BaseCategoricalScoreConfig] = (
        BaseCategoricalScoreConfig  # to be defined by subclasses
    )
    supported_data_types: ClassVar[set[DataType]] = set()

    def __init__(self, config: BaseCategoricalScoreConfig) -> None:
        self.config: BaseCategoricalScoreConfig = config

    @abstractmethod
    def compute_score_for_single_event(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
        threshold_array: xr.DataArray,
        event: BaseEvent,
    ) -> xr.DataArray | xr.Dataset:
        """Abstract calculation."""

    def validate_and_compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
        thresholds: xr.DataArray,
    ) -> xr.DataArray | xr.Dataset:
        """Validate and compute."""
        data_type: DataType = sim.verification.data_type  # type:ignore[misc]
        if data_type not in self.supported_data_types:
            msg = f"The data type '{data_type} is not supported by"
            f"{self.__class__.__name__}. Supported types: "
            f"{sorted(self.supported_data_types)}."
            raise ValueError(msg)

        results: list[xr.DataArray | xr.Dataset] = []
        for event in self.config.events:
            if not isinstance(event, BaseEvent):
                msg = f"Unsupported event type: {type(event)}. Expected a BaseEvent."  # type:ignore[unreachable] # runtime check
                raise TypeError(msg)
            if event.threshold not in thresholds["threshold"]:  # type:ignore[unreachable] # runtime check
                msg = f"Threshold '{event.threshold}' not found in thresholds DataArray."  # type:ignore[unreachable] # runtime check
                raise ValueError(msg)
            threshold_array = thresholds.sel({"threshold": event.threshold})  # type:ignore[unreachable] # runtime check
            result_for_a_single_event = self.compute_score_for_single_event(
                obs,
                sim,
                threshold_array=threshold_array,
                event=event,
            )
            results.append(result_for_a_single_event)
        result = xr.combine_by_coords(results)

        if isinstance(result, xr.DataArray) and result.name is None:  # type:ignore[misc]
            result.name = self.config.score_adapter
        return result
