"""Read and write netcdf files in a fews compatible format."""

from datetime import datetime, timezone
from pathlib import Path

from veriflow.configuration.default.datasinks import CFCompliantZarrConfig
from veriflow.constants import NAME, VERSION
from veriflow.datasinks.base import BaseDatasink
from veriflow.datatree.datatree import VeriflowDataTree
from veriflow.utils import drop_chunk_encoding, get_chunk_settings_along_forecast_reference_time

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

        # Persist the `xr.DataTree` attributes to the filtered DataTree
        filtered_dt.attrs = dt.attrs.copy()  # type: ignore[misc]

        # Score/output variables are frequently built by combining per-forecast_reference_time
        # slices (e.g. crps), which can leave irregular dask chunking along shared dims even
        # though every slice has the same length; Zarr requires all chunks but the last to be
        # equal size, so normalize the chunking before writing. Stale 'chunks'/'preferred_chunks'
        # encoding (e.g. on the auxiliary 'time' coordinate, inherited from the source files)
        # must be cleared first, otherwise it overrides the new dask chunking and to_zarr
        # rejects the mismatch.
        drop_chunk_encoding(filtered_dt)
        filtered_dt = filtered_dt.chunk(
            get_chunk_settings_along_forecast_reference_time(filtered_dt),
        )

        filtered_dt.to_zarr(
            self.config.path,
            storage_options=self.config.resolved_storage_options,
            # "None" means "let xarray auto-detect" only for reading; to_zarr requires a bool.
            consolidated=self.config.consolidated if self.config.consolidated is not None else True,
            mode="w" if self.config.force_overwrite else "w-",
        )
