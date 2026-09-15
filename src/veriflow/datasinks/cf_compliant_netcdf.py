"""Read and write netcdf files in a fews compatible format."""

from datetime import datetime, timezone
from pathlib import Path

from veriflow.configuration.default.datasinks import CFCompliantNetCDFConfig
from veriflow.constants import NAME, VERSION
from veriflow.datasinks.base import BaseDatasink
from veriflow.datatree.datatree import DataTreeNode, VeriflowDataTree

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
        directory = Path(self.config.directory)
        filename = Path(self.config.filename)

        if self.config.include_input_data:
            # Write the raw input data to a separate NetCDF file.
            dt[DataTreeNode.INPUT_DATA].to_netcdf(
                directory / f"{filename.stem}_input_data{filename.suffix}",
            )
        for pair in dt.veriflow.verification_pairs:
            if not (self.config.include_aligned_input_data or self.config.include_output):
                continue

            filepath = directory / f"{filename.stem}_{pair}{filename.suffix}"
            if filepath.exists() and self.config.force_overwrite is False:
                msg = "File already exists: " + str(filepath)
                raise FileExistsError(msg)

            if self.config.include_aligned_input_data and self.config.include_output:
                # Write each verification pair to its own NetCDF file.
                # Naming convention: "<stem>_<pair_id><suffix>".
                dt[pair].to_netcdf(filepath)
            if self.config.include_aligned_input_data and not self.config.include_output:
                dt.veriflow.get_aligned_input_data(pair).to_netcdf(filepath)
            if not self.config.include_aligned_input_data and self.config.include_output:
                dt.veriflow.get_outputs(pair).to_netcdf(filepath)

            subset = dt[pair]

            # Metadata attrs according to CF-compliancy
            subset.attrs = {
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
            subset.to_netcdf(filepath)
