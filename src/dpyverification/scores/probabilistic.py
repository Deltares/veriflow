"""
Probabilistic verification scores.

For verification of probabilistic and ensemble forecasts, and probabilistic historical simulations
of continuous variables.

For reference, see: https://scores.readthedocs.io/en/stable/included.html#probability
"""

from typing import ClassVar

import xarray as xr
from scores.probability import crps_cdf, crps_for_ensemble  # type: ignore[import-untyped]
from xskillscore import rank_histogram  # type: ignore[import-untyped]

from dpyverification.configuration import CrpsCDFConfig, CrpsForEnsembleConfig, RankHistogramConfig
from dpyverification.constants import DataType, StandardDim
from dpyverification.scores.base import BaseScore

__all__ = [
    "CrpsCDF",
    "CrpsCDFConfig",
    "CrpsForEnsemble",
    "CrpsForEnsembleConfig",
    "RankHistogram",
    "RankHistogramConfig",
]


class CrpsForEnsemble(BaseScore):
    """Implementation for CRPS for an ensemble."""

    kind = "crps_for_ensemble"
    config_class = CrpsForEnsembleConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.simulated_forecast_ensemble,
    }

    def __init__(self, config: CrpsForEnsembleConfig) -> None:
        self.config: CrpsForEnsembleConfig = config

    def compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> xr.Dataset | xr.DataArray:
        """Compute the CRPS for an ensemble of forecasts and observations."""
        result: xr.DataArray | xr.Dataset = crps_for_ensemble(
            fcst=sim,
            obs=obs,
            ensemble_member_dim=StandardDim.realization.value,
            preserve_dims=self.config.preserve_dims,
        )
        return result


class CrpsCDF(BaseScore):
    """Implementation for CRPS for probabilistic forecasts, expressed as cdf."""

    kind = "crps_cdf"
    config_class = CrpsCDFConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.simulated_forecast_probabilistic,
    }

    def __init__(self, config: CrpsCDFConfig) -> None:
        self.config: CrpsCDFConfig = config

    def compute(self, obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray | xr.Dataset:
        """Compute the CRPS for an ensemble of forecasts and observations."""
        result: xr.DataArray | xr.Dataset = crps_cdf(
            fcst=sim,
            obs=obs,
            preserve_dims=self.config.preserve_dims,
        )

        # crps_cdf outputs a rather ambiguous variable 'total', hence rename to score kind.
        return result.rename_vars({"total": str(self.config.score_adapter)})  # type:ignore[misc]


class RankHistogram(BaseScore):
    """Compute the rank histogram (Talagrand diagram) over the specified dimensions.

    For external documentation, see below:
    https://xskillscore.readthedocs.io/en/stable/api/xskillscore.rank_histogram.html?highlight=rank%20histogram#xskillscore.rank_histogram
    """

    kind = "rank_histogram"
    config_class = RankHistogramConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.simulated_forecast_ensemble,
    }

    def __init__(self, config: RankHistogramConfig) -> None:
        self.config: RankHistogramConfig = config

    def compute(self, obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray | xr.Dataset:
        """Compute the histogram of ranks over the specified dimensions."""
        result: xr.DataArray | xr.Dataset = rank_histogram(
            observations=obs,
            forecasts=sim,
            dim=self.config.preserve_dims,
            member_dim=StandardDim.realization.value,
        )
        return result
