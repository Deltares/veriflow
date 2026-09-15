"""Read and write netcdf files in a fews compatible format."""

from datetime import datetime, timezone
from pathlib import Path

from veriflow.configuration.default.datasinks import CFCompliantZarrConfig
from veriflow.constants import NAME, VERSION
from veriflow.datasinks.base import BaseDatasink
from veriflow.datatree.datatree import VeriflowDataTree

__all__ = [
    "CFCompliantZarr",
    "CFCompliantZarrConfig",
]


class CFCompliantZarr(BaseDatasink):
    """For writing data to a CF-compliant zarr store.

    This datasinks is compatible with both local filesystems and remote object stores.

    This datasink writes the full pipeline output to a single zarr store. Each verification pair
    will be stored as a separate group within the zarr store. Each subgroup corresponds to a
    specific verification pair and contains both input data and output data.

    .. note::
        CF-compliancy is not yet fully implemented.
    """

    kind = "cf_compliant_zarr"
    config_class = CFCompliantZarrConfig

    def __init__(self, config: CFCompliantZarrConfig) -> None:
        self.config: CFCompliantZarrConfig = config

    def write_data(self, dt: VeriflowDataTree) -> None:
        """Write the data in the xarray DataTree to zarr as specified in the output config."""
        filepath = Path(self.config.path)
        if filepath.exists() and self.config.force_overwrite is False:
            msg = "Zarr store already exists: " + str(filepath)
            raise FileExistsError(msg)

        # Metadata attrs according to CF-compliancy
        dt.attrs = {
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

        filtered_dt.to_zarr(
            self.config.path,
            storage_options=self.config.storage_options,
            # "None" means "let xarray auto-detect" only for reading; to_zarr requires a bool.
            consolidated=self.config.consolidated if self.config.consolidated is not None else True,
            mode="w" if self.config.force_overwrite else "w-",
        )
