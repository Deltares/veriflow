"""General utility functions for Veriflow."""

from collections.abc import Hashable, Sequence

import numpy as np
import xarray as xr


def _decode_utf8(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def convert_byte_string_coord_to_utf8(coord: xr.DataArray) -> xr.DataArray:
    """Convert byte strings in a coordinate to regular strings."""
    if coord.dtype.kind != "S":  # type:ignore[misc]
        return coord

    values = coord.to_numpy()  # type:ignore[misc]
    if values.ndim == 0:  # type:ignore[misc]
        decoded = _decode_utf8(values.item())  # type:ignore[misc]
    else:
        decoded = np.array(  # type:ignore[type-var, assignment]
            [_decode_utf8(value) for value in values.flat],  # type:ignore[misc]
            dtype=str,
        ).reshape(values.shape)  # type:ignore[misc]

    return xr.DataArray(decoded, dims=coord.dims, attrs=coord.attrs, name=coord.name)  # type:ignore[misc]


def convert_byte_string_coords_to_utf8(
    ds: xr.Dataset,
    coords: Sequence[Hashable] | None = None,
) -> xr.Dataset:
    """Convert byte-string coordinates on a dataset to regular strings."""
    coords_to_convert = ds.coords if coords is None else coords
    for coord in coords_to_convert:
        ds = ds.assign_coords({coord: convert_byte_string_coord_to_utf8(ds[coord])})  # type:ignore[misc]
    return ds
