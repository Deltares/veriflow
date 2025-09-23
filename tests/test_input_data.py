"""Test input data is valid according to schema."""

# mypy: ignore-errors
# ruff: noqa: D100, D101, D102, D103, D104, D105, D106, D107

import pytest
import xarray as xr
from dpyverification.datasources.inputschemas import (
    Observation,
    Simulation,
    TimeCoord,
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


def test_xarray_observations(xarray_data_array_observation: xr.DataArray) -> None:
    Observation.model_validate(xarray_data_array_observation.to_dict(data=False))


def test_xarray_observations_invalid_dims(xarray_data_array_observation: xr.DataArray) -> None:
    ds = xarray_data_array_observation.copy()
    ds = ds.expand_dims("invalid_dimension")
    with pytest.raises(ValidationError):
        Observation.model_validate(ds.to_dict(data=False))


def test_xarray_simulation_ensemble(
    xarray_data_array_simulation: xr.DataArray,
) -> None:
    ds = xarray_data_array_simulation
    Simulation.model_validate(ds.to_dict(data=False))


def test_xarray_simulation_no_ensemble(
    xarray_data_array_simulation: xr.DataArray,
) -> None:
    ds = xarray_data_array_simulation.drop_vars("realization")
    with pytest.raises(ValidationError):
        Simulation.model_validate(ds.to_dict(data=False))
