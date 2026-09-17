"""Datasources to fetch thresholds."""

from pathlib import Path
from typing import ClassVar, Self

import pandas as pd
import xarray as xr

from veriflow.configuration.default.datasources import CsvConfig
from veriflow.constants import DataSourceKind, DataType, SpatialType, StandardDim
from veriflow.datasources.base import BaseDatasource
from veriflow.types import DataSpec

__all__ = [
    "Csv",
    "CsvConfig",
]


class Csv(BaseDatasource):
    """Datasource for reading CSV files."""

    kind: str = DataSourceKind.CSV
    config_class = CsvConfig
    supported_data_specs: ClassVar[set[DataSpec]] = {
        (DataType.threshold, SpatialType.point),
    }

    def __init__(self, config: CsvConfig) -> None:
        self.config: CsvConfig = config
        self.dataset: xr.Dataset = xr.Dataset()

    @property
    def configured_stations(self) -> set[str]:
        """Return the internal station identifiers configured for this datasource.

        This is needed for standardization of station identifiers across sources.
        """
        return set(self.config.stations)

    @configured_stations.setter
    def configured_stations(self, stations: set[str]) -> None:
        """Set the internal station identifiers configured for this datasource.

        This is needed for standardization of station identifiers across sources.
        """
        self.config.stations = list(stations)

    @property
    def configured_variables(self) -> set[str]:
        """Return the internal variable identifiers configured for this datasource.

        This is needed for standardization of variable identifiers across sources.
        """
        return set(self.config.variables)

    @configured_variables.setter
    def configured_variables(self, variables: set[str]) -> None:
        """Set the internal variable identifiers configured for this datasource.

        This is needed for standardization of variable identifiers across sources.
        """
        self.config.variables = list(variables)

    def fetch_data(self) -> Self:
        """Parse thresholds from csv file."""
        file_path = Path(self.config.directory) / self.config.filename
        threshold_df = pd.read_csv(file_path)

        # Check that the df has the correct structure
        expected_columns = [
            StandardDim.station,
            "variable",
            StandardDim.threshold,
            "value",
        ]
        if not all(k in expected_columns for k in threshold_df.columns):
            msg = f"Expected columns: {expected_columns}. Got: {threshold_df.columns}"
            raise ValueError(msg)

        # Pivot the long-form table into a Dataset where each unique variable becomes a
        # data variable with dims (station, threshold).
        pivoted = threshold_df.set_index(
            [StandardDim.station, "variable", StandardDim.threshold],
        ).to_xarray()["value"]

        # Filter the array based on the configured station, variable and threshold ids
        try:
            pivoted = pivoted.sel(
                station=self.config.stations,
                variable=self.config.variables,
                threshold=self.config.thresholds,
            )
        except KeyError as e:
            msg = "One of the configured station, variable or threshold ids was not found in the . "
            f"data. Details: {e}"
            raise ValueError(msg) from e

        # Convert the variable dim into separate data variables, one per variable.
        dataset = pivoted.to_dataset(dim="variable")

        # Set the data type and source as attributes for later use in the verification process
        dataset.attrs["data_type"] = "threshold"  # type:ignore[misc]
        dataset.attrs["source_id"] = self.config.source_id  # type:ignore[misc]
        self.dataset = dataset
        return self
