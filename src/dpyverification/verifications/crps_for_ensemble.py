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
) -> xr.DataArray:
    """Compute the CRPS for an ensemble of forecasts and observations."""
    if not isinstance(calcconfig, CRPSForEnsemble):
        msg = "Input calcconfig does not have calculationtype CRPSForEnsemble"
        raise TypeError(msg)

    # Select sim and obs.
    obs: xr.DataArray = data.intermediate[calcconfig.variablepair.obs]
    sim: xr.DataArray = data.intermediate[calcconfig.variablepair.sim]

    # Compute
    _result: xr.Dataset | xr.DataArray = _crps_for_ensemble(
        fcst=sim,
        obs=obs,
        ensemble_member_dim=DataModelDims.ensemble,
        preserve_dims=calcconfig.preserve_dims,
    )

    if not isinstance(_result, xr.DataArray):  # type: ignore[misc]
        msg = f"Expected xr.DataArray, got {type(_result)}"
        raise NotImplementedError(msg)

    # Set variable name on xr.DataArray
    _result.name = CalculationType.CRPSForEnsemble

    # Set attrs on xr.DataArray
    # For now, store config as dict
    # General config could be stored as xr.Dataset attrs
    _result.attrs = {str(k): str(v) for k, v in calcconfig.__dict__.items()}  # type: ignore[misc]

    return _result
