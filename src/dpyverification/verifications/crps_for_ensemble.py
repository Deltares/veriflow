"""Compute the histogram of ranks over the specified dimensions."""

import xarray as xr
from scores.probability import crps_for_ensemble as _crps_for_ensemble
from xskillscore import rank_histogram  # type: ignore[import-untyped]

from dpyverification.configuration import Calculation, CRPSForEnsemble
from dpyverification.constants import (
    DataModelDims,
)
from dpyverification.datamodel import DataModel


def crps_for_ensemble(
    calcconfig: Calculation,
    data: DataModel,
) -> xr.Dataset:
    """Compute the CRPS for an ensemble of forecasts and observations."""
    if not isinstance(calcconfig, CRPSForEnsemble):
        msg = "Input calcconfig does not have calculationtype CRPSForEnsemble"
        raise TypeError(msg)

    # For now, temporary hardcoded solution to
    # select obs and sim from simobs dataset. It
    # is assumed here that the simobspairs calculation
    # is run before rankhistogram. Once the intermediate
    # datamodel is implemented, this should be updated.
    obs = data.output["Q.m_simobspair_Q.m"]
    sim = data.output["Q.m_simobspair_Q.fs"]
    _result: xr.DataArray | xr.Dataset = _crps_for_ensemble(
        fcst=sim,
        obs=obs,
        ensemble_member_dim=DataModelDims.ensemble,
        reduce_dims=calcconfig.reduce_dims,
    )

    # Covert to xr.Dataset if type xr.DataArray.
    # since it is required by the add_to_output
    # method in the pipiline.
    if hasattr(_result, "to_dataset"):
        result: xr.Dataset = _result.to_dataset()
    return result
