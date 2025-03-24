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

    # Select sim and obs.
    obs = data.intermediate[calcconfig.variablepair.obs]
    sim = data.intermediate[calcconfig.variablepair.sim]

    # https://github.com/Deltares-research/DPyVerification/issues/63
    if len(sim[DataModelDims.leadtime]) != 1:
        msg = "Computation of rank histogram for multiple lead times is not yet supported."
        raise NotImplementedError(msg)

    # Compute
    _result: xr.DataArray | xr.Dataset = _rank_histogram(
        observations=obs,
        forecasts=sim,
        dim=calcconfig.reduce_dims,
        member_dim=DataModelDims.ensemble,
    )

    # Set attrs on xr.DataArray
    # For now, store config as dict
    # General config could be stored as xr.Dataset attrs
    _result.attrs = {str(k): str(v) for k, v in calcconfig.__dict__.items()}  # type: ignore[misc]

    # Covert to xr.Dataset if type xr.DataArray.
    # since it is required by the add_to_output
    # method in the pipiline.
    if hasattr(_result, "to_dataset"):
        result: xr.Dataset = _result.to_dataset()
    return result
