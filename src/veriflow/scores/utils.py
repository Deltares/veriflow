"""Utility functions shared across the scores module."""

from collections.abc import Hashable, Sequence
from typing import Protocol

import xarray as xr

from veriflow.configuration.base import BaseConfig
from veriflow.constants import StandardDim


def set_data_array_attributes(
    da: xr.DataArray,
    long_name: str,
    units: str,
    standard_name: str | None = None,
    config: BaseConfig | None = None,
) -> xr.DataArray:
    """Set configuration attributes on xr.DataArray."""
    cf_attrs: dict[str, str] = {
        "long_name": long_name,
        "units": units,
    }

    if standard_name is not None:
        cf_attrs.update({"standard_name": standard_name})

    union: dict[str, str]

    if config is not None:
        config_attrs: dict[str, str] = config.model_dump()
        union = cf_attrs | config_attrs
    else:
        union = cf_attrs

    return da.assign_attrs(union)


def assign_station_auxiliary_coords(
    result: xr.DataArray,
    sim: xr.DataArray,
) -> xr.DataArray:
    """Reassign auxiliary coordinates on dimension station.

    These typically include, station_id, station_name, lat, lon, x, y, z.
    """
    for coord in sim.coords:  # type:ignore[misc]
        # Reassign only coords with dim station
        if sim[coord].dims == (StandardDim.station,):
            result = result.assign_coords({coord: sim[coord]})  # type:ignore[misc]
    return result


def compute_reduce_and_preserve_dims(
    config_reduce_dims: Sequence[StandardDim],
    data_dims: Sequence[Hashable],
    *,
    allow_realization_dim: bool = False,
) -> tuple[list[StandardDim], list[StandardDim]]:
    """Compute reduce_dims and preserve_dims filtered by actual data dimensions.

    Args:
        config_reduce_dims: Dimensions to reduce from config (may not all exist in data)
        data_dims: Actual dimensions present in the data
        allow_realization_dim: Whether to include the realization dimension in the computation.

    Returns
    -------
        Tuple of (reduce_dims, preserve_dims), each filtered to only include
        dimensions that exist in the data.
    """
    all_possible_dims = [
        StandardDim.station,
        StandardDim.forecast_reference_time,
        StandardDim.lead_time,
        StandardDim.time,
        StandardDim.x,
        StandardDim.y,
    ]

    # Optionally include the realization dimension if allowed
    if allow_realization_dim:
        all_possible_dims.append(StandardDim.realization)

    # Filter config reduce_dims to only those present in data
    reduce_dims = [d for d in config_reduce_dims if d in data_dims]
    # Compute preserve_dims as all possible dims not in reduce_dims, filtered to data dims
    preserve_dims = [d for d in all_possible_dims if d not in reduce_dims and d in data_dims]
    return reduce_dims, preserve_dims


class ScoreFunc(Protocol):
    """Callable score taking two DataArrays and returning a DataArray."""

    def __call__(  # noqa: D102
        self,
        fcst: xr.DataArray,
        obs: xr.DataArray,
        reduce_dims: Sequence[StandardDim],
        **kwargs: object,
    ) -> xr.DataArray | xr.Dataset: ...
