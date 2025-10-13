"""Compute the Continuous Ranked Probability Score (CRPS) for an ensemble of forecasts.

For documentation, see below:
https://scores.readthedocs.io/en/1.0.0/tutorials/CRPS_for_Ensembles.html
"""

from typing import ClassVar, Protocol

import xarray as xr
from scores.probability import crps_for_ensemble  # type: ignore[import-untyped]
from xskillscore import rank_histogram  # type: ignore[import-untyped]

from dpyverification.configuration import CrpsForEnsembleConfig, RankHistogramConfig
from dpyverification.configuration.base import BaseScoreConfig
from dpyverification.constants import StandardCoord, StandardDim, TimeseriesKind
from dpyverification.datamodel import InputDataset
from dpyverification.scores.base import BaseScore
from dpyverification.scores.utils import set_data_array_attributes


def reassign_station_auxiliary_coords(
    result: xr.DataArray,
    sim: xr.DataArray,
) -> xr.DataArray:
    """Reassign auxiliary coordinates on dimension station.

    These typically include, station_id, station_name, lat, lon, x, y, z.
    """
    for coord in sim.coords:  # type:ignore[misc]
        # Reassign only coords with dim station
        if sim[coord].dims == (StandardDim.station,):
            result = result.assign_coords({coord: sim[coord]})  # type:ignore[misc]
    return result


class ScoreFunc(Protocol):
    """Callable score taking two DataArrays and returning a DataArray."""

    def __call__(  # noqa: D102
        self,
        first: xr.DataArray,
        second: xr.DataArray,
        **kwargs: object,
    ) -> xr.DataArray: ...


class WrappedScoreFunc(Protocol):
    """Callable that consumes dataset and config and returns a Dataset."""

    def __call__(  # noqa: D102
        self,
        data: InputDataset,
        config: BaseScoreConfig,
        **kwargs: object,
    ) -> xr.DataArray: ...


def loop_verification_pairs(func: ScoreFunc) -> WrappedScoreFunc:
    """Loop over verification pairs.

    A helper function that can be re-used for scores to avoid duplicate code.
    """

    def wrapper(data: InputDataset, config: BaseScoreConfig, **kwargs: object) -> xr.DataArray:
        results: list[xr.DataArray] = []
        for pair in config.verification_pairs:
            obs, sim = data.get_verification_pair(pair)

            # Broadcast obs like sim
            obs = obs.broadcast_like(sim.isel({StandardDim.realization: 0}))  # type:ignore[misc]

            # Function call
            result: xr.DataArray = func(sim, obs, **kwargs)

            # Set verification_pair dim
            result = result.expand_dims({"verification_pair": 1})

            # Assign auxiliary coords on dim, indicating the obs source and sim source
            result = result.assign_coords(
                {
                    "verification_pair": ("verification_pair", [pair.id]),  # type:ignore[misc]
                    "obs_source": ("verification_pair", [pair.obs]),
                    "sim_source": ("verification_pair", [pair.sim]),
                },
            )

            # Set variable name on xr.DataArray
            result.name = str(config.kind)

            # Set attributes on data array
            result = set_data_array_attributes(
                result,
                long_name=str(config.kind),
                units=sim[StandardCoord.units.name].to_numpy()[0],  # type:ignore[misc]
            )

            results.append(result)

        return xr.merge(results)[str(config.kind)]

    return wrapper


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
