"""Compute the Continuous Ranked Probability Score (CRPS) for an ensemble of forecasts.

For documentation, see below:
https://scores.readthedocs.io/en/1.0.0/tutorials/CRPS_for_Ensembles.html
"""

from typing import ClassVar

import xarray as xr
from scores.probability import crps_cdf, crps_for_ensemble  # type: ignore[import-untyped]
from xskillscore import rank_histogram  # type: ignore[import-untyped]

from dpyverification.configuration import CrpsCDFConfig, CrpsForEnsembleConfig, RankHistogramConfig
from dpyverification.constants import StandardDim, TimeseriesKind
from dpyverification.datamodel import InputDataset
from dpyverification.scores.base import BaseScore
from dpyverification.scores.utils import (
    ScoreFunc,
    loop_verification_pairs,
    reassign_station_auxiliary_coords,
)


class CrpsForEnsemble(BaseScore):
    """Implementation for CRPS for an ensemble."""

    kind = "crps_for_ensemble"
    config_class = CrpsForEnsembleConfig
    supported_timeseries_kinds: ClassVar[set[TimeseriesKind]] = {
        TimeseriesKind.simulated_forecast_ensemble,
    }

    def __init__(self, config: CrpsForEnsembleConfig) -> None:
        self.config: CrpsForEnsembleConfig = config

    def compute(
        self,
        data: InputDataset,
    ) -> xr.DataArray:
        """Compute the CRPS for an ensemble of forecasts and observations."""
        typed_crps_for_ensemble: ScoreFunc = crps_for_ensemble
        return loop_verification_pairs(typed_crps_for_ensemble)(
            data,
            self.config,
            ensemble_member_dim=StandardDim.realization.value,
            preserve_dims=self.config.reduce_dims.inverse,
        )


class CrpsCDF(BaseScore):
    """Implementation for CRPS for probabilistic forecasts, expressed as cdf."""

    kind = "crps_cdf"
    config_class = CrpsCDFConfig
    supported_timeseries_kinds: ClassVar[set[TimeseriesKind]] = {
        TimeseriesKind.simulated_forecast_probabilistic,
    }

    def __init__(self, config: CrpsCDFConfig) -> None:
        self.config: CrpsCDFConfig = config

    def compute(
        self,
        data: InputDataset,
    ) -> xr.DataArray:
        """Compute the CRPS for an ensemble of forecasts and observations."""
        typed_crps_cdf: ScoreFunc = crps_cdf
        return loop_verification_pairs(typed_crps_cdf)(
            data,
            self.config,
            preserve_dims=self.config.reduce_dims.inverse,
        )


class RankHistogram(BaseScore):
    """Compute the rank histogram (Talagrand diagram) over the specified dimensions.

    For external documentation, see below:
    https://xskillscore.readthedocs.io/en/stable/api/xskillscore.rank_histogram.html?highlight=rank%20histogram#xskillscore.rank_histogram
    """

    kind = "rank_histogram"
    config_class = RankHistogramConfig
    supported_timeseries_kinds: ClassVar[set[TimeseriesKind]] = {
        TimeseriesKind.simulated_forecast_ensemble,
    }

    def __init__(self, config: RankHistogramConfig) -> None:
        self.config: RankHistogramConfig = config

    def compute(self, data: InputDataset) -> xr.DataArray:
        """Compute the histogram of ranks over the specified dimensions."""

        def _rank_histogram(
            sim: xr.DataArray,
            obs: xr.DataArray,
            **kwargs: object,
        ) -> xr.DataArray:
            """Call xskillscore.rank_histogram while preserving auxiliary coords."""
            dims = kwargs.get("dim") or kwargs.get("dims") or [StandardDim.time]
            result: xr.DataArray = rank_histogram(
                obs,
                sim,
                member_dim=StandardDim.realization.value,
                dim=dims,
            )
            return reassign_station_auxiliary_coords(result, sim)

        return loop_verification_pairs(_rank_histogram)(  # type: ignore[arg-type]
            data,
            self.config,
            dims=self.config.reduce_dims.values,
            member_dim=StandardDim.realization.value,
        )
