"""Test the threshold datasource."""

import pytest

from dpyverification.datasources.csv import Csv
from dpyverification.datasources.inputschemas import validate_input_data


def test_validate_thresholds(xarray_thresholds: Csv) -> None:
    """Test we can validate thresholds."""
    validate_input_data(xarray_thresholds.data_array)


def test_validate_thresholds_fails_on_missing_data_type_attr(
    xarray_thresholds: Csv,
) -> None:
    """Test we can validate thresholds."""
    xarray_thresholds.data_array.attrs.pop("data_type", None)  # type: ignore[misc]
    expected_error_msg = "Input data array is missing required 'data_type' attribute."
    with pytest.raises(
        ValueError,
        match=expected_error_msg,
    ):
        validate_input_data(xarray_thresholds.data_array)
