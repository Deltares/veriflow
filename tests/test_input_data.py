"""Test input data is valid according to schema."""

# mypy: ignore-errors
# ruff: noqa: D100, D101, D102, D103, D104, D105, D106, D107

import pytest
import xarray as xr
from dpyverification.datasources.inputschemas import (
    ObservedHistorical,
    TimeCoord,
    input_schemas,
)
from pydantic import ValidationError


def test_time_coord_good() -> None:
    good = {
        "dtype": "datetime64[ns]",
        "dims": ("time",),
    }
    TimeCoord(**good)


def test_time_coord_bad() -> None:
    bad = {
        "dtype": "datetime64[ns]",
        "dims": ("a_bad_time",),
    }
    with pytest.raises(ValidationError):
        TimeCoord(**bad)


def test_xarray_observations(xarray_data_array_observations: xr.DataArray) -> None:
    ObservedHistorical.model_validate(xarray_data_array_observations.to_dict(data=False))


def test_xarray_observations_invalid_dims(xarray_data_array_observations: xr.DataArray) -> None:
    ds = xarray_data_array_observations.copy()
    ds = ds.expand_dims("invalid_dimension")
    with pytest.raises(ValidationError):
        ObservedHistorical.model_validate(ds.to_dict(data=False))


def test_xarray_simulation_ensemble(
    xarray_data_array_simulation: xr.DataArray,
) -> None:
    da = xarray_data_array_simulation
    schema = input_schemas[da.attrs["timeseries_kind"]]
    schema.model_validate(da.to_dict(data=False))


def test_xarray_simulation_no_ensemble(
    xarray_data_array_simulation: xr.DataArray,
) -> None:
    da = xarray_data_array_simulation.drop_vars("realization")
    schema = input_schemas[da.attrs["timeseries_kind"]]

    with pytest.raises(ValidationError):
        schema.model_validate(da.to_dict(data=False))
