"""Compute the histogram of ranks over the specified dimensions."""

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

    # For now, temporary hardcoded solution to
    # select obs and sim from simobs dataset. It
    # is assumed here that the simobspairs calculation
    # is run before rankhistogram. Once the intermediate
    # datamodel is implemented, this should be updated.
    obs = data.output["Q.m_simobspair_Q.m"]
    sim = data.output["Q.m_simobspair_Q.fs"]
    _result: xr.DataArray | xr.Dataset = _rank_histogram(
        observations=obs,
        forecasts=sim,
        dim=calcconfig.reduce_dims,
        member_dim=DataModelDims.ensemble,
    )

    # Covert to xr.Dataset if type xr.DataArray.
    # since it is required by the add_to_output
    # method in the pipiline.
    if hasattr(_result, "to_dataset"):
        result: xr.Dataset = _result.to_dataset()
    return result
