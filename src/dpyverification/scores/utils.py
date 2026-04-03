"""Utility functions shared across the scores module."""

from collections.abc import Sequence
from typing import Protocol

import xarray as xr

from dpyverification.configuration.base import BaseConfig
from dpyverification.constants import StandardDim


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


class ScoreFunc(Protocol):
    """Callable score taking two DataArrays and returning a DataArray."""

    def __call__(  # noqa: D102
        self,
        fcst: xr.DataArray,
        obs: xr.DataArray,
        reduce_dims: Sequence[StandardDim],
        **kwargs: object,
    ) -> xr.DataArray | xr.Dataset: ...
