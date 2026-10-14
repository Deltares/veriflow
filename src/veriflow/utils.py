"""General utility functions for Veriflow."""

from collections.abc import Hashable, Sequence

import numpy as np
import xarray as xr

from veriflow.constants import StandardDim


def get_chunk_settings_along_forecast_reference_time(
    obj: xr.Dataset | xr.DataArray | xr.DataTree,
) -> dict[Hashable, int]:
    """Build a dask chunking dict safe for writing ``obj`` to Zarr.

    Chunks ``forecast_reference_time`` to size 1, matching how the cache appends new forecast
    cycles one at a time so future appends add chunks instead of rewriting existing ones, and
    collapses every other dimension to a single chunk. Variables combined from multiple source
    files/forecast cycles (e.g. scores like crps) can otherwise end up with irregular dask
    chunking on those dims even though every slice is the same length; Zarr requires all chunks
    but the last to be equal size, so this also normalizes that away.

    For a ``DataTree``, dims are collected from every node in the subtree rather than from
    ``obj.dims`` alone: a DataTree's own ``.dims`` only reflects dims defined on its *own* local
    dataset (e.g. the root), not a union across descendant groups - relying on it alone would
    silently skip rechunking any dims that only appear in a nested group's data.
    """
    if isinstance(obj, xr.DataTree):
        dims: set[Hashable] = set()
        for node in obj.subtree:
            dims.update(node.dims)
    else:
        dims = set(obj.dims)
    return {dim: 1 if dim == StandardDim.forecast_reference_time else -1 for dim in dims}


def drop_chunk_encoding(obj: xr.Dataset | xr.DataTree) -> None:
    """Strip 'chunks'/'preferred_chunks' from every variable's ``encoding``, in place.

    These are commonly inherited from wherever a variable was originally read from (e.g. a
    coordinate like 'time' carrying the chunking of the source FEWS files, or of a previous
    write of this same store) and, left in place, take priority over the dask chunking applied
    via :func:`zarr_write_chunks` - causing ``to_zarr`` to reject the mismatch as "would overlap
    multiple Dask chunks" instead of simply deriving the on-disk chunk shape from the current
    (correct) dask chunks. Call this before ``.chunk(zarr_write_chunks(obj))``/``to_zarr(...)``.
    """
    datasets = (node.dataset for node in obj.subtree) if isinstance(obj, xr.DataTree) else (obj,)
    for dataset in datasets:
        for variable in dataset.variables.values():
            variable.encoding.pop("chunks", None)  # type:ignore[misc]
            variable.encoding.pop("preferred_chunks", None)  # type:ignore[misc]


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
