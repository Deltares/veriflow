"""Compute the rank histogram (Talagrand diagram) over the specified dimensions.

For external documentation, see below:
https://xskillscore.readthedocs.io/en/stable/api/xskillscore.rank_histogram.html?highlight=rank%20histogram#xskillscore.rank_histogram
"""

import numpy as np
import xarray as xr
from xskillscore import rank_histogram as _rank_histogram  # type: ignore[import-untyped]

from dpyverification.configuration import Calculation, RankHistogram
from dpyverification.constants import (
    DataModelDims,
)
from dpyverification.datamodel import DataModel


def rankhistogram(
    calcconfig: Calculation,
    data: DataModel,
) -> xr.Dataset:
    """Compute the histogram of ranks over the specified dimensions."""
    if not isinstance(calcconfig, RankHistogram):
        msg = "Input calcconfig does not have calculationtype RankHistogram"
        raise TypeError(msg)

    # Select sim and obs.
    obs = data.intermediate[calcconfig.variablepair.obs]
    sim = data.intermediate[calcconfig.variablepair.sim]

    rankhistograms_per_leadtime = []
    for leadtime in sim[DataModelDims.leadtime]:  # type: ignore[misc]
        # Get a subset of the simulations dataset
        sim_subset = sim.sel(leadtime=leadtime)  # type: ignore[misc]

        # Compute the rank for
        rank_for_leadtime: xr.DataArray | xr.Dataset = _rank_histogram(
            observations=obs,
            forecasts=sim_subset,
            dim=calcconfig.reduce_dims,
            member_dim=DataModelDims.ensemble,
        )

        # Check a DataArray is returned
        if not isinstance(rank_for_leadtime, xr.DataArray):  # type: ignore[misc]
            msg = f"Expected xr.DataArray, got {type(rank_for_leadtime)}"
            raise TypeError(msg)

        # Set the variable name with specific lead time
        leadtime_seconds = int(leadtime.to_numpy() / np.timedelta64(1, "s"))  # type: ignore[misc]
        name = f"rank_histogram_leadtime_{leadtime_seconds}s"
        rank_for_leadtime.name = name

        # Set the long_name attribute on the variable
        # Set units to 1 (CF-compliant indication for dimensionless variable)
        rank_for_leadtime.attrs = {"long_name": name, "units": 1} | {  # type: ignore[misc]
            str(k): str(v)  # type: ignore[misc]
            for k, v in calcconfig.__dict__.items()  # type: ignore[misc]
        }

        # Append to the list
        rankhistograms_per_leadtime.append(rank_for_leadtime)

    # Compute data variables
    data_vars = {k.name: k for k in rankhistograms_per_leadtime}

    return xr.Dataset(data_vars=data_vars)
