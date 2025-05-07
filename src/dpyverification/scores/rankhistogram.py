"""Compute the rank histogram (Talagrand diagram) over the specified dimensions.

For external documentation, see below:
https://xskillscore.readthedocs.io/en/stable/api/xskillscore.rank_histogram.html?highlight=rank%20histogram#xskillscore.rank_histogram
"""

import numpy as np
import xarray as xr
from xskillscore import rank_histogram as _rank_histogram  # type: ignore[import-untyped]

from dpyverification.configuration import RankHistogramConfig
from dpyverification.constants import (
    DataModelDims,
)
from dpyverification.datamodel import DataModel
from dpyverification.scores.base import BaseScore


class RankHistogram(BaseScore):
    """Rank histogram implementation."""

    kind = "rankhistogram"
    config_class = RankHistogramConfig

    def __init__(self, config: RankHistogramConfig) -> None:
        self.config: RankHistogramConfig = config

    def compute(self, data: DataModel) -> xr.Dataset:
        """Compute the histogram of ranks over the specified dimensions."""
        # Select sim and obs.
        obs = data.intermediate[self.config.variablepairs[0].obs]
        sim = data.intermediate[self.config.variablepairs[0].sim]

        rankhistograms_per_leadtime = []
        for leadtime in sim[DataModelDims.leadtime]:  # type: ignore[misc]
            # Get a subset of the simulations dataset
            sim_subset = sim.sel(leadtime=leadtime)  # type: ignore[misc]

            # Compute the rank for
            rank_for_leadtime: xr.DataArray | xr.Dataset = _rank_histogram(
                observations=obs,
                forecasts=sim_subset,
                dim=self.config.reduce_dims,
                member_dim=DataModelDims.ensemble,
            )

            # Check a DataArray is returned
            if not isinstance(rank_for_leadtime, xr.DataArray):  # type: ignore[misc]
                msg = f"Expected xr.DataArray, got {type(rank_for_leadtime)}"
                raise TypeError(msg)

            # Set the variable name with specific lead time
            leadtime_seconds = int(leadtime.to_numpy() / np.timedelta64(1, "s"))  # type: ignore[misc]
            name = f"{self.kind}_leadtime_{leadtime_seconds}s"
            rank_for_leadtime.name = name

            # Set the long_name attribute on the variable
            # Set units to 1 (CF-compliant indication for dimensionless variable)
            rank_for_leadtime.attrs = {"long_name": name, "units": 1} | {  # type: ignore[misc]
                str(k): str(v)  # type: ignore[misc]
                for k, v in self.config.__dict__.items()  # type: ignore[misc]
            }

            # Append to the list
            rankhistograms_per_leadtime.append(rank_for_leadtime)

        # Compute data variables
        data_vars = {k.name: k for k in rankhistograms_per_leadtime}

        return xr.Dataset(data_vars=data_vars)
