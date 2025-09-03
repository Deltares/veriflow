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
    ScoreKind,
    StandardDim,
)
from dpyverification.datamodel import SimObsDataset
from dpyverification.scores.base import BaseScore
from dpyverification.scores.utils import set_data_array_attributes


class CrpsForEnsemble(BaseScore):
    """Implementation for CRPS for an ensemble."""

    kind = "crps_for_ensemble"
    config_class = CrpsForEnsembleConfig

    def __init__(self, config: CrpsForEnsembleConfig) -> None:
        self.config: CrpsForEnsembleConfig = config

    def compute(
        self,
        data: SimObsDataset,
    ) -> xr.Dataset:
        """Compute the CRPS for an ensemble of forecasts and observations."""
        # Select sim and obs.

        results = []
        for variable_pair in self.config.variable_pairs:
            obs: xr.DataArray = data.dataset[variable_pair.obs_variable_name]
            sim: xr.DataArray = data.dataset[variable_pair.sim_variable_name]

            # Compute
            result: xr.Dataset | xr.DataArray = _crps_for_ensemble(
                fcst=sim,
                obs=obs,
                ensemble_member_dim=StandardDim.realization,
                preserve_dims=self.config.preserve_dims,
            )

            if not isinstance(result, xr.DataArray):  # type: ignore[misc]
                msg = f"Expected xr.DataArray, got {type(result)}"
                raise NotImplementedError(msg)

            # Set variable name on xr.DataArray
            result.name = f"{ScoreKind.crps_for_ensemble}_{variable_pair.sim_obs_string}"

            # Get unit, default to 1 (CF-compliant definition for dimensionless)
            units: str = sim.attrs.get("units", "1")  # type:ignore[misc]
            result = set_data_array_attributes(
                result,
                long_name="continuous ranked probability score",
                units=units,
                config=self.config,
            )
            results.append(result)
        return xr.merge(results)
