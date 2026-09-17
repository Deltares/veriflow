"""The various datasinks that can be used for writing data."""

from .base import BaseDatasink
from .cf_compliant_netcdf import CFCompliantNetCDF, CFCompliantNetCDFConfig
from .cf_compliant_zarr import CFCompliantZarr, CFCompliantZarrConfig

DEFAULT_DATASINKS: list[type[BaseDatasink]] = [CFCompliantNetCDF, CFCompliantZarr]
