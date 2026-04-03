"""Module for continuous scores."""

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import xarray as xr
from scores.continuous import (  # type:ignore[import-untyped]
    additive_bias,
    kge,
    mae,
    mean_error,
    mse,
    rmse,
)

from dpyverification.configuration.default.scores import ContinuousScoresConfig
from dpyverification.constants import DataType, SupportedContinuousScore
from dpyverification.scores.base import BaseScore

if TYPE_CHECKING:
    from dpyverification.scores.utils import ScoreFunc

score_funcs: dict[SupportedContinuousScore, Callable] = {
    SupportedContinuousScore.additive_bias: additive_bias,  # type:ignore[misc]
    SupportedContinuousScore.kge: kge,  # type:ignore[misc]
    SupportedContinuousScore.mae: mae,  # type:ignore[misc]
    SupportedContinuousScore.mse: mse,  # type:ignore[misc]
    SupportedContinuousScore.rmse: rmse,  # type:ignore[misc]
    SupportedContinuousScore.mean_error: mean_error,  # type:ignore[misc]
}


class ContinuousScores(BaseScore):
    """Implementation for CRPS for probabilistic forecasts, expressed as cdf."""

    kind = "continuous_scores"
    config_class = ContinuousScoresConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.simulated_forecast_single,
    }

    def __init__(self, config: ContinuousScoresConfig) -> None:
        self.config: ContinuousScoresConfig = config

    def compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> xr.Dataset:
        """Compute any number of continous scores."""
        results: list[xr.DataArray | xr.Dataset] = []
        for score in self.config.scores:
            func: ScoreFunc = score_funcs[score]  # type:ignore[misc]
            result = func(fcst=sim, obs=obs, reduce_dims=self.config.reduce_dims)
            result.name = func.__qualname__  # type:ignore[attr-defined]
            results.append(result)
        return xr.merge(results)
