"""Module for continuous scores."""

from collections.abc import Callable
from typing import ClassVar

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
from dpyverification.constants import SupportedContinuousScore, TimeseriesKind
from dpyverification.datamodel.main import InputDataset
from dpyverification.scores.base import BaseScore
from dpyverification.scores.utils import ScoreFunc, loop_verification_pairs

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
    supported_timeseries_kinds: ClassVar[set[TimeseriesKind]] = {
        TimeseriesKind.simulated_forecast_single,
    }

    def __init__(self, config: ContinuousScoresConfig) -> None:
        self.config: ContinuousScoresConfig = config

    def compute(
        self,
        data: InputDataset,
    ) -> list[xr.DataArray]:
        """Compute the CRPS for an ensemble of forecasts and observations."""
        results = []
        for score in self.config.scores:
            typed_score_func: ScoreFunc = score_funcs[score]  # type:ignore[misc]

            results.append(
                loop_verification_pairs(typed_score_func)(
                    data,
                    self.config,
                    preserve_dims=self.config.reduce_dims.inverse,
                ),
            )
        return list(results)
