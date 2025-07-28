"""Compute the Continuous Ranked Propability Score (CRPS) for an ensemble of forecasts.

For documentation, see below:
https://scores.readthedocs.io/en/1.0.0/tutorials/CRPS_for_Ensembles.html
"""

import xarray as xr
from scores.probability import (  # type: ignore[import-untyped]
    crps_for_ensemble as _crps_for_ensemble,
)

from dpyverification.configuration import CrpsForEnsembleConfig
from dpyverification.constants import (
    DataModelDims,
    ScoreKind,
)
from dpyverification.datamodel import DataModel
from dpyverification.scores.base import BaseScore


class CrpsForEnsemble(BaseScore):
    """Implementation for CRPS for an ensemble."""

    kind = "crps_for_ensemble"
    config_class = CrpsForEnsembleConfig

    def __init__(self, config: CrpsForEnsembleConfig) -> None:
        self.config: CrpsForEnsembleConfig = config

    def compute(
        self,
        data: DataModel,
    ) -> xr.DataArray:
        """Compute the CRPS for an ensemble of forecasts and observations."""
        # Select sim and obs.
        obs: xr.DataArray = data.intermediate[self.config.variablepairs[0].obs]
        sim: xr.DataArray = data.intermediate[self.config.variablepairs[0].sim]

        # Compute
        result: xr.Dataset | xr.DataArray = _crps_for_ensemble(
            fcst=sim,
            obs=obs,
            ensemble_member_dim=DataModelDims.ensemble,
            preserve_dims=self.config.preserve_dims,
        )

        if not isinstance(result, xr.DataArray):  # type: ignore[misc]
            msg = f"Expected xr.DataArray, got {type(result)}"
            raise NotImplementedError(msg)

        # Set variable name on xr.DataArray
        result.name = ScoreKind.CRPSFORENSEMBLE

        # Set attrs on xr.DataArray
        # For now, store config as dict
        # General config could be stored as xr.Dataset attrs
        result.attrs = {str(k): str(v) for k, v in self.config.__dict__.items()}  # type: ignore[misc]

        return result
