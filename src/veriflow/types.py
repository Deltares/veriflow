"""Type aliases for the veriflow package."""

from typing import TypeAlias, TypeVar

import xarray as xr

from veriflow.constants import DataType, SpatialType

DataSpec: TypeAlias = tuple[DataType, SpatialType]

DataSetTree: TypeAlias = xr.Dataset | xr.DataTree
DataArraySetTree: TypeAlias = xr.Dataset | xr.DataTree | xr.DataArray

TDataSetTree = TypeVar("TDataSetTree", xr.Dataset, xr.DataTree)
