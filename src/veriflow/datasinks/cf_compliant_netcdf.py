"""Read and write netcdf files in a fews compatible format."""

from datetime import datetime, timezone
from pathlib import Path

from veriflow.configuration.default.datasinks import CFCompliantNetCDFConfig
from veriflow.constants import NAME, VERSION
from veriflow.datasinks.base import BaseDatasink
from veriflow.datatree.datatree import VeriflowDataTree

__all__ = [
    "CFCompliantNetCDF",
    "CFCompliantNetCDFConfig",
]


class CFCompliantNetCDF(BaseDatasink):
    """For writing veriflow output to NetCDF.

    This datasink will write one NetCDF file for each verification pair. Input data are included and
    results are written in NetCDF groups. Each file is named ``"<stem>_<verification_pair_id>
    <suffix>"``, where ``<stem>``/``<suffix>`` are derived from the configured ``filename``.

    .. note::
        CF-compliancy is not yet fully implemented.
    """

    kind = "cf_compliant_netcdf"
    config_class = CFCompliantNetCDFConfig

    def __init__(self, config: CFCompliantNetCDFConfig) -> None:
        self.config: CFCompliantNetCDFConfig = config

    def write_data(self, dt: VeriflowDataTree) -> None:
        """Write the data in the xarray DataTree to the file as specified in the output config."""
        filepath = Path(self.config.directory) / self.config.filename

        # Metadata attrs according to CF-compliancy
        attrs = {
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

        filtered_dt = dt.veriflow.filter_nodes(
            include_input_data=self.config.include_input_data,
            include_aligned_input_data=self.config.include_aligned_input_data,
            include_output=self.config.include_output,
        )

        # Persist the `xr.DataTree` attributes to the filtered DataTree
        filtered_dt.attrs = attrs

        def validate_write_allowed(filepath: Path, *, force_overwrite: bool) -> None:
            if not force_overwrite:
                msg = "File already exists: " + str(filepath)
                raise FileExistsError(msg)

        validate_write_allowed(filepath, force_overwrite=self.config.force_overwrite)
        filtered_dt.to_netcdf(filepath)
