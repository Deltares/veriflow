"""Utility functions shared across the scores module."""

import xarray as xr

from dpyverification.configuration.base import BaseConfig


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
