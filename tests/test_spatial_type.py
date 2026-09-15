"""Tests for the orthogonal ``spatial_type`` axis (point/gridded) and SAL score."""

# mypy: ignore-errors

import pytest
import xarray as xr

from veriflow.constants import DataType, SpatialType
from veriflow.datasources.inputschemas import INPUT_SCHEMAS, validate_input_data
from veriflow.datatree.datatree import VeriflowAccessor
from veriflow.scores.spatial import SALScore


def test_gridded_observed_validates(
    xarray_observed_historical_gridded: xr.Dataset,
) -> None:
    """A gridded observed dataset validates and keeps its spatial_type attr."""
    validate_input_data(xarray_observed_historical_gridded)
    assert xarray_observed_historical_gridded.attrs["spatial_type"] == SpatialType.gridded


def test_gridded_forecast_single_validates(
    xarray_simulated_forecast_single_gridded: xr.Dataset,
) -> None:
    """A gridded single forecast validates against the gridded schema."""
    validate_input_data(xarray_simulated_forecast_single_gridded)
    assert xarray_simulated_forecast_single_gridded.attrs["spatial_type"] == SpatialType.gridded


def test_spatial_type_defaults_to_point_and_is_backfilled(
    xarray_observed_historical: xr.Dataset,
) -> None:
    """A dataset without a spatial_type attr validates as point and gets backfilled."""
    xarray_observed_historical.attrs.pop("spatial_type", None)
    assert "spatial_type" not in xarray_observed_historical.attrs
    validate_input_data(xarray_observed_historical)
    assert xarray_observed_historical.attrs["spatial_type"] == SpatialType.point


def test_verification_accessor_spatial_type(
    xarray_observed_historical: xr.Dataset,
    xarray_observed_historical_gridded: xr.Dataset,
) -> None:
    """The verification accessor exposes spatial_type (defaulting to point)."""
    assert xarray_observed_historical.verification.spatial_type == SpatialType.point
    assert xarray_observed_historical_gridded.verification.spatial_type == SpatialType.gridded


def test_unsupported_combination_raises(
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    """An unsupported (data_type, spatial_type) combination raises a clear error."""
    ds = xarray_simulated_forecast_ensemble
    ds.attrs["data_type"] = DataType.simulated_forecast_probabilistic
    ds.attrs["spatial_type"] = SpatialType.gridded
    assert (DataType.simulated_forecast_probabilistic, SpatialType.gridded) not in INPUT_SCHEMAS
    with pytest.raises(ValueError, match="No input schema defined"):
        validate_input_data(ds)


def test_point_score_rejects_gridded() -> None:
    """SAL only supports gridded; a point spec is not part of its supported specs."""
    assert (
        DataType.simulated_forecast_single,
        SpatialType.point,
    ) not in SALScore.supported_data_specs
    assert (
        DataType.simulated_forecast_single,
        SpatialType.gridded,
    ) in SALScore.supported_data_specs


def test_map_historical_gridded_into_forecast_space(
    xarray_observed_historical_gridded: xr.Dataset,
    xarray_simulated_forecast_single_gridded: xr.Dataset,
) -> None:
    """Historical gridded obs can be mapped into a gridded forecast structure."""
    variable = "var_0"
    obs = xarray_observed_historical_gridded[variable]
    sim = xarray_simulated_forecast_single_gridded[variable]
    mapped = VeriflowAccessor.map_historical_into_forecast_space(obs, sim)
    assert "y" in mapped.dims
    assert "x" in mapped.dims
