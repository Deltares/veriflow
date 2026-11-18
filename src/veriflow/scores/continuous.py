"""
Continuous verification scores.

For verification of non-probabilistic (deterministic) forecasts, and historical simulations of
continuous variables.

For reference, see: https://scores.readthedocs.io/en/stable/included.html#continuous.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import xarray as xr
from scores.continuous import (  # type:ignore[import-untyped]
    additive_bias,
    kge,
    mae,
    mean_error,
    mse,
    nse,
    rmse,
)

from veriflow.configuration.default.scores import ContinuousScoresConfig
from veriflow.constants import DataType, SpatialType, SupportedContinuousScore
from veriflow.scores.base import BaseScore
from veriflow.scores.utils import compute_reduce_and_preserve_dims
from veriflow.types import DataSpec

if TYPE_CHECKING:
    from veriflow.scores.utils import ScoreFunc

score_funcs: dict[SupportedContinuousScore, Callable] = {
    SupportedContinuousScore.additive_bias: additive_bias,  # type:ignore[misc]
    SupportedContinuousScore.kge: kge,  # type:ignore[misc]
    SupportedContinuousScore.nse: nse,  # type:ignore[misc]
    SupportedContinuousScore.mae: mae,  # type:ignore[misc]
    SupportedContinuousScore.mse: mse,  # type:ignore[misc]
    SupportedContinuousScore.rmse: rmse,  # type:ignore[misc]
    SupportedContinuousScore.mean_error: mean_error,  # type:ignore[misc]
}

__all__ = [
    "ContinuousScores",
    "ContinuousScoresConfig",
]


class ContinuousScores(BaseScore):
    """Implementation for CRPS for probabilistic forecasts, expressed as cdf."""

    kind = "continuous_scores"
    config_class = ContinuousScoresConfig
    supported_data_specs: ClassVar[set[DataSpec]] = {
        (DataType.simulated_forecast_single, SpatialType.point),
        (DataType.observed_historical, SpatialType.point),
        (DataType.simulated_historical, SpatialType.point),
        (DataType.simulated_forecast_ensemble, SpatialType.point),
    }

    def __init__(self, config: ContinuousScoresConfig) -> None:
        self.config: ContinuousScoresConfig = config

    def compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> xr.Dataset:
        """Compute any number of continous scores."""
        # Compute reduce_dims filtered to actual data dimensions
        reduce_dims, _ = compute_reduce_and_preserve_dims(self.config.reduce_dims, sim.dims)
        results: list[xr.DataArray | xr.Dataset] = []
        for score in self.config.scores:
            func: ScoreFunc = score_funcs[score]  # type:ignore[misc]
            result = func(fcst=sim, obs=obs, reduce_dims=reduce_dims)
            result.name = func.__qualname__  # type:ignore[attr-defined]
            results.append(result)
        return xr.merge(results)
