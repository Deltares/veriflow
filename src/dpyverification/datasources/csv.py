"""Datasources to fetch thresholds."""

from pathlib import Path
from typing import ClassVar, Self

import pandas as pd
import xarray as xr

from dpyverification.configuration.default.datasources import CsvConfig
from dpyverification.constants import DataSourceKind, DataType, StandardDim
from dpyverification.datasources.base import BaseDatasource


class Csv(BaseDatasource):
    """Parse thresholds from a csv file."""

    kind: str = DataSourceKind.CSV
    config_class = CsvConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.threshold,
    }

    def __init__(self, config: CsvConfig) -> None:
        self.config: CsvConfig = config
        self.data_array = xr.DataArray()

    def fetch_data(self) -> Self:
        """Parse thresholds from csv file."""
        file_path = Path(self.config.directory) / self.config.filename
        threshold_df = pd.read_csv(file_path)

        # Check that the df has the correct structure
        expected_columns = [
            StandardDim.station,
            StandardDim.variable,
            StandardDim.threshold,
            "value",
        ]
        if not all(k in expected_columns for k in threshold_df.columns):
            msg = f"Expected columns: {expected_columns}. Got: {threshold_df.columns}"
            raise ValueError(msg)

        # Convert it to the internal datamodel
        self.data_array = threshold_df.set_index(
            [StandardDim.station, StandardDim.variable, StandardDim.threshold],
        ).to_xarray()["value"]

        # Filter the data array based on the configured station, variable and threshold ids
        try:
            self.data_array = self.data_array.sel(
                station=self.config.stations,
                variable=self.config.variables,
                threshold=self.config.thresholds,
            )
        except KeyError as e:
            msg = "One of the configured station, variable or threshold ids was not found in the . "
            f"data. Details: {e}"
            raise ValueError(msg) from e

        # Set the data type as an attribute for later use in the verification process
        self.data_array.attrs["data_type"] = "threshold"  # type:ignore[misc]
        return self
