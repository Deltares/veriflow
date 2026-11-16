# mypy: disable-error-code="misc, no-any-return, explicit-any"
# This module implements the SAL score from Pysteps, which is Any typed.
"""
Spatial verification scores for gridded (lat/lon) data.

Currently implements the SAL (Structure-Amplitude-Location) score from ``pysteps``,
computed on 2D ``(lat, lon)`` fields.

The SAL score requires the optional ``pysteps`` dependency, which is not installed by
default. Install it with::

    pip install veriflow[spatial]

For reference, see:
https://pysteps.readthedocs.io/en/stable/generated/pysteps.verification.salscores.sal.html
"""

from typing import ClassVar

import numpy as np
import xarray as xr

from veriflow.configuration.default.scores import SALScoreConfig
from veriflow.constants import DataType, SpatialType, StandardDim
from veriflow.scores.base import BaseScore
from veriflow.types import DataSpec

__all__ = [
    "SALScore",
    "SALScoreConfig",
]

_PYSTEPS_IMPORT_HINT = (
    "The SAL score requires the optional 'pysteps' dependency, which is not installed. "
    "Install it with: pip install veriflow[spatial]"
)

_SAL_COMPONENTS = ("structure", "amplitude", "location")


class SALScore(BaseScore):
    """Structure-Amplitude-Location (SAL) spatial verification score.

    Computes the SAL score of ``pysteps`` on 2D ``(y, x)`` fields for deterministic
    or ensemble gridded forecasts against gridded observations. For each forecast slice
    (along any non-spatial dimensions such as ``forecast_reference_time``, ``lead_time``
    and ``realization``) the structure, amplitude and location components are returned
    as separate variables. Ensemble members are evaluated independently; the
    ``realization`` dimension is preserved without reduction.

    For reference, see:
    https://pysteps.readthedocs.io/en/stable/generated/pysteps.verification.salscores.sal.html
    """

    kind = "sal"
    config_class = SALScoreConfig
    supported_data_specs: ClassVar[set[DataSpec]] = {
        (DataType.simulated_forecast_single, SpatialType.gridded),
        (DataType.simulated_forecast_ensemble, SpatialType.gridded),
    }

    def __init__(self, config: SALScoreConfig) -> None:
        self.config: SALScoreConfig = config

    def compute(self, obs: xr.DataArray, sim: xr.DataArray) -> xr.Dataset:
        """Compute the SAL score over the ``(y, x)`` plane for each forecast slice."""
        try:
            from pysteps.verification.salscores import sal  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(_PYSTEPS_IMPORT_HINT) from exc

        thr_factor = self.config.thr_factor
        thr_quantile = self.config.thr_quantile

        def _sal(prediction: np.ndarray, observation: np.ndarray) -> np.ndarray:
            structure, amplitude, location = sal(
                prediction,
                observation,
                thr_factor=thr_factor,
                thr_quantile=thr_quantile,
            )
            return np.array([structure, amplitude, location], dtype="float64")

        result = xr.apply_ufunc(
            _sal,
            sim,
            obs,
            input_core_dims=[
                [StandardDim.y, StandardDim.x],
                [StandardDim.y, StandardDim.x],
            ],
            output_core_dims=[["sal_component"]],
            vectorize=True,
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"output_sizes": {"sal_component": len(_SAL_COMPONENTS)}},
            dask="parallelized",
        )
        result = result.assign_coords(sal_component=list(_SAL_COMPONENTS))
        return result.to_dataset(dim="sal_component")
