"""
Probabilistic verification scores.

For verification of probabilistic and ensemble forecasts, and probabilistic historical simulations
of continuous variables.

For reference, see: https://scores.readthedocs.io/en/stable/included.html#probability
"""

from typing import ClassVar

import xarray as xr
from scores.probability import crps_cdf, crps_for_ensemble, rank_histogram  # type: ignore[import-untyped]

from veriflow.configuration.default.scores import (
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    RankHistogramConfig,
)
from veriflow.constants import DataType, SpatialType, StandardDim
from veriflow.scores.base import BaseScore
from veriflow.scores.utils import compute_reduce_and_preserve_dims
from veriflow.types import DataSpec

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
    supported_data_specs: ClassVar[set[DataSpec]] = {
        (DataType.simulated_forecast_ensemble, SpatialType.point),
        (DataType.simulated_forecast_ensemble, SpatialType.gridded),
    }

    def __init__(self, config: CrpsForEnsembleConfig) -> None:
        self.config: CrpsForEnsembleConfig = config

    def compute(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> xr.Dataset | xr.DataArray:
        """Compute the CRPS for an ensemble of forecasts and observations."""
        # Compute preserve_dims filtered to actual data dimensions
        _, preserve_dims = compute_reduce_and_preserve_dims(self.config.reduce_dims, sim.dims)
        result: xr.Dataset | xr.DataArray = crps_for_ensemble(
            fcst=sim,
            obs=obs,
            ensemble_member_dim=StandardDim.realization.value,
            preserve_dims=preserve_dims or None,
        )
        return result


class CrpsCDF(BaseScore):
    """Implementation for CRPS for probabilistic forecasts, expressed as cdf."""

    kind = "crps_cdf"
    config_class = CrpsCDFConfig
    supported_data_specs: ClassVar[set[DataSpec]] = {
        (DataType.simulated_forecast_probabilistic, SpatialType.point),
    }

    def __init__(self, config: CrpsCDFConfig) -> None:
        self.config: CrpsCDFConfig = config

    def compute(self, obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray | xr.Dataset:
        """Compute the CRPS for an ensemble of forecasts and observations."""
        # Compute preserve_dims filtered to actual data dimensions
        _, preserve_dims = compute_reduce_and_preserve_dims(self.config.reduce_dims, sim.dims)
        result: xr.DataArray | xr.Dataset = crps_cdf(
            fcst=sim,
            obs=obs,
            preserve_dims=preserve_dims or None,
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
    supported_data_specs: ClassVar[set[DataSpec]] = {
        (DataType.simulated_forecast_ensemble, SpatialType.point),
        (DataType.simulated_forecast_ensemble, SpatialType.gridded),
    }

    def __init__(self, config: RankHistogramConfig) -> None:
        self.config: RankHistogramConfig = config

    def compute(self, obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray | xr.Dataset:
        """Compute the histogram of ranks over the specified dimensions."""
        # This current implementation requires aligned dimensions
        obs, sim = xr.align(obs, sim)

        # Compute reduce_dims filtered to actual data dimensions
        reduce_dims, _ = compute_reduce_and_preserve_dims(self.config.reduce_dims, sim.dims)
        # Pass empty list to preserve all dimensions, or the filtered reduce_dims
        result: xr.DataArray | xr.Dataset = rank_histogram(
            obs=obs,
            fcst=sim,
            reduce_dims=reduce_dims or [],
            ens_member_dim=StandardDim.realization.value,
        )
        return result
