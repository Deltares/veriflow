"""Test the threshold datasource."""

from pathlib import Path

import pandas as pd

from dpyverification.configuration.config import GeneralInfoConfig
from dpyverification.configuration.default.datasources import ThresholdCsvConfig
from dpyverification.constants import DataSourceKind, DataType
from dpyverification.datasources.thresholds import ThresholdCsv


def test_fetch_thresholds_from_csv(
    dummy_threshold_df: pd.DataFrame,
    general_info_config_single: GeneralInfoConfig,
    tmp_path: Path,
) -> None:
    """Test we can fetch thresholds from csv file."""
    file_path = tmp_path / "thresholds.csv"
    dummy_threshold_df.to_csv(file_path, index=False)
    config = ThresholdCsvConfig(
        import_adapter=DataSourceKind.THRESHOLD_CSV,
        general=general_info_config_single,
        data_type=DataType.threshold,
        source="threshold_source",
        directory=file_path.parent,
        filename=file_path.name,
        stations=["station_2"],
        variables=["variable_1"],
        thresholds=["warn_1"],
    )
    instance = ThresholdCsv(config)
    instance.fetch_data()
