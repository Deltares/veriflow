"""Read and write NetCDF files in a fews compatible format."""

from typing import ClassVar, Self

import xarray as xr

from veriflow.configuration.default.datasources import NetCDFConfig
from veriflow.constants import (
    DataType,
)
from veriflow.datasources.base import BaseDatasource

__all__ = [
    "NetCDF",
    "NetCDFConfig",
]


class NetCDF(BaseDatasource):
    """A datasource for reading NetCDF files compatible with the internal datamodel.

    You can validate that the NetCDF file satisfies the internal datamodel using the following
    example:

    .. code-block:: python

        import xarray as xr
        from veriflow.datasources import validate_input_data

        data_array = xr.open_dataarray("path/to/netcdf/file/example.nc")
        validated_data = validate_input_data(data_array)

    .. note::
        The NetCDF file must contain exactly one data variable. The data variable
        must also have a ``data_type`` attribute that matches one of the supported data types.
    """

    kind = "netcdf"
    config_class = NetCDFConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.observed_historical,
        DataType.simulated_forecast_ensemble,
        DataType.simulated_forecast_single,
        DataType.simulated_forecast_probabilistic,
        DataType.threshold,
    }

    def __init__(self, config: NetCDFConfig) -> None:
        self.config: NetCDFConfig = config

    def fetch_data(self) -> Self:
        """Retrieve NetCDF file content as an xarray DataArray."""
        dataset = xr.open_mfdataset(self.config.paths)  # type:ignore[arg-type] # Generator is accepted by open_mfdataset, but not correctly typed in xarray

        if len(dataset.data_vars) != 1:
            msg = (
                f"Expected exactly one data variable in the NetCDF file, but found "
                f"{len(dataset.data_vars)}.",
            )
            raise ValueError(msg)
        self.data_array = dataset[next(iter(dataset.data_vars))]
        self.data_array.attrs["data_type"] = self.config.data_type  # type: ignore[misc]
        return self
