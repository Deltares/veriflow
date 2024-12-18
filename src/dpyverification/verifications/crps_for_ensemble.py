"""Compute the CRPS for an ensemble of forecasts over specified dimensions."""

import xarray as xr
from scores.probability import (  # type: ignore[import-untyped]
    crps_for_ensemble as _crps_for_ensemble,
)

from dpyverification.configuration import Calculation, CRPSForEnsemble
from dpyverification.constants import (
    CalculationType,
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

    # Select sim and obs.
    obs = data.intermediate[calcconfig.simobsvariables.obs]
    sim = data.intermediate[calcconfig.simobsvariables.sim]

    # Compute
    _result: xr.DataArray | xr.Dataset = _crps_for_ensemble(
        fcst=sim,
        obs=obs,
        ensemble_member_dim=DataModelDims.ensemble,
        reduce_dims=calcconfig.reduce_dims,
    )

    # Set variable name
    _result.name = CalculationType.CRPSForEnsemble

    # Covert to xr.Dataset if type xr.DataArray.
    # since it is required by the add_to_output
    # method in the pipiline.
    if hasattr(_result, "to_dataset"):
        result: xr.Dataset = _result.to_dataset()
    return result
