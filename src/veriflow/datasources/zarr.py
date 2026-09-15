"""Read Zarr stores from local disk or S3."""

from typing import ClassVar, Self

import xarray as xr

from veriflow.configuration.default.datasources import ZarrConfig
from veriflow.constants import (
    DataType,
    SpatialType,
    StandardAttribute,
    StandardDim,
)
from veriflow.datasources.base import BaseDatasource
from veriflow.types import DataSpec
from veriflow.utils import convert_byte_string_coords_to_utf8

__all__ = [
    "Zarr",
    "ZarrConfig",
]


class Zarr(BaseDatasource):
    """A datasource for reading Zarr stores compatible with the internal datatree.

    Wraps :func:`xarray.open_zarr` and supports both local filesystem paths and remote
    URLs (currently ``s3://`` is exercised). For S3 stores, credentials are taken from a
    :class:`~veriflow.configuration.utils.S3AuthConfig` instance loaded from environment
    variables prefixed with ``S3_``. Additional ``storage_options`` configured on the
    :class:`ZarrConfig` are merged on top and forwarded to ``xr.open_zarr``.

    .. note::
        The dataset must carry a ``data_type`` attribute that matches one of the supported
        data types; if the attribute is missing it will be set from the configuration.
    """

    kind = "zarr"
    config_class = ZarrConfig
    supported_data_specs: ClassVar[set[DataSpec]] = {
        (DataType.observed_historical, SpatialType.point),
        (DataType.simulated_forecast_ensemble, SpatialType.point),
        (DataType.simulated_forecast_single, SpatialType.point),
        (DataType.simulated_forecast_probabilistic, SpatialType.point),
        (DataType.threshold, SpatialType.point),
        (DataType.observed_historical, SpatialType.gridded),
        (DataType.simulated_forecast_single, SpatialType.gridded),
        (DataType.simulated_forecast_ensemble, SpatialType.gridded),
    }

    def __init__(self, config: ZarrConfig) -> None:
        self.config: ZarrConfig = config

    @property
    def configured_stations(self) -> set[str] | None:
        """Return the internal station identifiers configured for this datasource.

        This is needed for standardization of station identifiers across sources.
        """
        return set(self.config.stations) if self.config.stations is not None else None

    @configured_stations.setter
    def configured_stations(self, stations: set[str] | None) -> None:
        """Set the internal station identifiers configured for this datasource.

        This is needed for standardization of station identifiers across sources.
        """
        self.config.stations = list(stations) if stations is not None else None

    @property
    def configured_variables(self) -> set[str] | None:
        """Return the internal variable identifiers configured for this datasource.

        This is needed for standardization of variable identifiers across sources.
        """
        return set(self.config.variables) if self.config.variables is not None else None

    @configured_variables.setter
    def configured_variables(self, variables: set[str] | None) -> None:
        """Set the internal variable identifiers configured for this datasource.

        This is needed for standardization of variable identifiers across sources.
        """
        self.config.variables = list(variables) if variables is not None else None

    def _build_storage_options(self) -> dict[str, object] | None:
        """Build storage_options for xr.open_zarr based on path and config.

        Returns ``None`` for non-remote (local) paths so that xarray opens the store
        directly from the local filesystem.
        """
        if not self._is_remote_path(self.config.path):
            return None

        options: dict[str, object] = {}
        if self.config.auth_config is not None:
            options.update(self.config.auth_config.to_storage_options())
        if self.config.storage_options is not None:
            options.update(self.config.storage_options)
        return options

    @staticmethod
    def _is_remote_path(path: str) -> bool:
        """Return True if ``path`` looks like a remote/fsspec URL (e.g. ``s3://``)."""
        return "://" in path

    def fetch_data(self) -> Self:
        """Retrieve the configured Zarr store as an xarray Dataset."""
        storage_options = self._build_storage_options()
        dataset = xr.open_zarr(  # type:ignore[misc] # xarray's stubs are loose here
            self.config.path,
            storage_options=storage_options,
            consolidated=self.config.consolidated,
        )

        dataset = convert_byte_string_coords_to_utf8(dataset, [StandardDim.station])  # type:ignore[misc]

        # Filter the dataset by configured stations and variables, if any. Set the
        # data_type attribute to the configured value, if any.
        if self.config.stations is not None:
            dataset = dataset.sel(station=self.config.stations)
        if self.config.variables is not None:
            dataset = dataset[self.config.variables]
        dataset.attrs["data_type"] = self.config.data_type  # type: ignore[misc]

        # Resolve the CRS: a configured 'crs' takes precedence, otherwise the one already
        # present on the dataset attributes (if any) is used. For gridded data the CRS is the
        # canonical spatial reference; lat/lon are only derived on demand (for a score or
        # output) rather than eagerly at ingestion.
        if self.config.crs is not None:
            dataset.attrs[StandardAttribute.crs] = self.config.crs  # type: ignore[misc]

        # Assign the dataset to the instance and return self for chaining.
        self.dataset = dataset
        return self
