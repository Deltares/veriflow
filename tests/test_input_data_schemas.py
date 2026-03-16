"""Test input data is valid according to schema."""

# mypy: ignore-errors
# ruff: noqa: D103

import pytest
import xarray as xr
from pydantic import ValidationError

from dpyverification.datasources.inputschemas import (
    INPUT_SCHEMAS,
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


def test_xarray_observations(xarray_observed_historical: xr.DataArray) -> None:
    ObservedHistorical.model_validate(xarray_observed_historical.to_dict(data=False))


def test_xarray_observations_invalid_dims(xarray_observed_historical: xr.DataArray) -> None:
    ds = xarray_observed_historical.copy()
    ds = ds.expand_dims("invalid_dimension")
    with pytest.raises(ValidationError):
        ObservedHistorical.model_validate(ds.to_dict(data=False))


def test_xarray_simulation_ensemble(
    xarray_simulated_forecast_ensemble: xr.DataArray,
) -> None:
    da = xarray_simulated_forecast_ensemble
    schema = INPUT_SCHEMAS[da.attrs["data_type"]]
    schema.model_validate(da.to_dict(data=False))


def test_xarray_simulation_no_ensemble(
    xarray_simulated_forecast_ensemble: xr.DataArray,
) -> None:
    da = xarray_simulated_forecast_ensemble.drop_vars("realization")
    schema = INPUT_SCHEMAS[da.attrs["data_type"]]

    with pytest.raises(ValidationError):
        schema.model_validate(da.to_dict(data=False))
