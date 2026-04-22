"""Read and write netcdf files in a fews compatible format."""

from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

from veriflow.configuration.default.datasinks import CFCompliantNetCDFConfig
from veriflow.constants import NAME, VERSION
from veriflow.datasinks.base import BaseDatasink

__all__ = [
    "CFCompliantNetCDF",
    "CFCompliantNetCDFConfig",
]


class CFCompliantNetCDF(BaseDatasink):
    """For writing data to a fews netcdf file."""

    kind = "cf_compliant_netcdf"
    config_class = CFCompliantNetCDFConfig

    def __init__(self, config: CFCompliantNetCDFConfig) -> None:
        self.config: CFCompliantNetCDFConfig = config

    def write_data(self, dataset: xr.Dataset) -> None:
        """Write the data in the xarray Dataset to the file as specified in the output config."""
        filepath = Path(self.config.directory) / self.config.filename
        if filepath.exists() and self.config.force_overwrite is False:
            msg = "File already exists: " + str(filepath)
            raise FileExistsError(msg)

        # Metadata attrs according to CF-compliancy
        dataset.attrs = {
            "title": self.config.title,
            "institution": self.config.institution,
            "source": f"{NAME}: version: {VERSION}",
            "history": "",
            "references": "",
            "comment": self.config.comment,
            "time_coverage_start": self.config.verification_period.start.isoformat(),
            "time_coverage_end": self.config.verification_period.end.isoformat(),
            "production_time": datetime.now(tz=timezone.utc).isoformat(),
            "Conventions": "CF-1.11",
        }
        dataset.to_netcdf(filepath)
