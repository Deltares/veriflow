"""Compute the rank histogram (Talagrand diagram) over the specified dimensions.

For external documentation, see below:
https://xskillscore.readthedocs.io/en/stable/api/xskillscore.rank_histogram.html?highlight=rank%20histogram#xskillscore.rank_histogram
"""

import xarray as xr
from xskillscore import rank_histogram as _rank_histogram  # type: ignore[import-untyped]

from dpyverification.configuration import RankHistogramConfig
from dpyverification.constants import ScoreKind, StandardDim
from dpyverification.datamodel import SimObsDataset
from dpyverification.scores.base import BaseScore
from dpyverification.scores.utils import set_data_array_attributes


class RankHistogram(BaseScore):
    """Rank histogram implementation."""

    kind = "rank_histogram"
    config_class = RankHistogramConfig

    def __init__(self, config: RankHistogramConfig) -> None:
        self.config: RankHistogramConfig = config

    def compute(self, data: SimObsDataset) -> xr.Dataset:
        """Compute the histogram of ranks over the specified dimensions."""
        results = []
        for variable_pair in self.config.variable_pairs:
            obs: xr.DataArray = data.dataset[variable_pair.obs]
            sim: xr.DataArray = data.dataset[variable_pair.sim]

            # xskillscore.rank_histogram assumes equal dimensions between observations
            #   and simulations, so expand observations to have forecast_period dim.
            obs = obs.expand_dims({StandardDim.forecast_period: sim[StandardDim.forecast_period]})

            # Compute
            result: xr.DataArray = _rank_histogram(
                observations=obs,
                forecasts=sim,
                dim=StandardDim.time,  # Preserve stations, forecast_period
                member_dim=StandardDim.realization,
            )

            # Reset the station coordinates, because they get removed in the calculation
            def _reassign_station_auxiliary_coords(
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

            result.name = f"{ScoreKind.rank_histogram}_{variable_pair.sim_obs_string}"

            result = set_data_array_attributes(
                result,
                long_name="rank_histogram",
                units="1",
                config=self.config,
            )
            result = _reassign_station_auxiliary_coords(result, sim)
        results.append(result)
        return xr.merge(results)
