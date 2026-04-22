"""The various datasinks that can be used for writing data."""

from .base import BaseDatasink
from .cf_compliant_netcdf import CFCompliantNetCDF, CFCompliantNetCDFConfig

DEFAULT_DATASINKS: list[type[BaseDatasink]] = [CFCompliantNetCDF]
