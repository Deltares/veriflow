"""Test input data is valid according to schema."""

# mypy: ignore-errors
# ruff: noqa: D103

import pytest
import xarray as xr
from pydantic import ValidationError

from veriflow.constants import DataType, SpatialType
from veriflow.datasources.inputschemas import (
    INPUT_SCHEMAS,
    BaseAttrs,
    HistoricalTimeCoord,
    ObservedHistorical,
)


def test_time_coord_good() -> None:
    good = {
        "dtype": "datetime64[ns]",
        "dims": ("time",),
    }
    HistoricalTimeCoord(**good)


def test_time_coord_bad() -> None:
    bad = {
        "dtype": "datetime64[ns]",
        "dims": ("a_bad_time",),
    }
    with pytest.raises(ValidationError):
        HistoricalTimeCoord(**bad)


class TestDatasetBaseAttrs:
    """Test BaseAttrs schema validation."""

    @property
    def valid_dict(self) -> dict[str, str]:
        """Default valid dict."""
        return {
            "data_type": "observed_historical",
            "spatial_type": "point",
            "crs": "EPSG:4326",
        }

    def test_missing_crs_defaults_to_epsg4326(self) -> None:
        """Test that missing 'crs' field defaults to 'EPSG:4326'."""
        valid_dict = self.valid_dict.copy()
        valid_dict.pop("crs", None)

        base_attrs = BaseAttrs.model_validate(valid_dict)
        assert base_attrs.crs == "EPSG:4326"

    def test_missing_spatial_type_defaults_to_point(self) -> None:
        """Test that missing 'spatial_type' field defaults to 'point'."""
        valid_dict = self.valid_dict.copy()
        valid_dict.pop("spatial_type", None)

        base_attrs = BaseAttrs.model_validate(valid_dict)
        assert base_attrs.spatial_type == SpatialType.point


def test_xarray_observations(xarray_observed_historical: xr.Dataset) -> None:
    ObservedHistorical.model_validate(xarray_observed_historical.to_dict(data=False))


def test_xarray_observations_invalid_dims(xarray_observed_historical: xr.Dataset) -> None:
    ds = xarray_observed_historical.copy()
    ds = ds.expand_dims("invalid_dimension")
    with pytest.raises(ValidationError):
        ObservedHistorical.model_validate(ds.to_dict(data=False))


def test_xarray_simulation_ensemble(
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    ds = xarray_simulated_forecast_ensemble
    schema = INPUT_SCHEMAS[(DataType(ds.attrs["data_type"]), SpatialType.point)]
    schema.model_validate(ds.to_dict(data=False))


def test_xarray_simulation_no_ensemble(
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    ds = xarray_simulated_forecast_ensemble.drop_vars("realization")
    schema = INPUT_SCHEMAS[(DataType(ds.attrs["data_type"]), SpatialType.point)]

    with pytest.raises(ValidationError):
        schema.model_validate(ds.to_dict(data=False))
