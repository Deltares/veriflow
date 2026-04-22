"""Test the threshold datasource."""

import numpy as np
import pandas as pd
import pytest

from veriflow.datasources.csv import Csv

# mypy: disable-error-code=misc
# we are using xarray DataArrays in the tests, which have attributes that mypy cannot verify


def test_fetch_thresholds(
    xarray_thresholds: Csv,
    dummy_threshold_df: pd.DataFrame,
) -> None:
    """Test we can fetch thresholds from csv file."""
    xarray_thresholds.fetch_data()

    # Test the threshold ids are as expected (coordinate values)
    np.testing.assert_array_equal(
        xarray_thresholds.data_array.threshold.to_numpy(),  # type:ignore[misc]
        np.array(["warn_1", "warn_2"]),  # type:ignore[misc]
    )

    # Test one threshold value matches the source csv content.
    expected_value = dummy_threshold_df[
        (dummy_threshold_df["station"] == "station_2")
        & (dummy_threshold_df["variable"] == "var_1")
        & (dummy_threshold_df["threshold"].isin(["warn_1"]))
    ]["value"].iloc[0]

    np.testing.assert_approx_equal(
        xarray_thresholds.data_array.isel(station=0, variable=0, threshold=0).to_numpy(),  # type:ignore[misc]
        expected_value,  # type:ignore[misc]
    )


def test_fetch_thresholds_raises_error_on_invalid_location_config(
    xarray_thresholds: Csv,
) -> None:
    """Test we can fetch thresholds from csv file."""
    config = xarray_thresholds.config
    config.stations = ["non_existent_station"]
    instance = Csv(config)

    expected_error_msg = "One of the configured station, variable or threshold ids was not found *"

    with pytest.raises(ValueError, match=expected_error_msg):
        instance.fetch_data()
