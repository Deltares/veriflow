"""Test input data is valid according to schema."""

# mypy: ignore-errors
# ruff: noqa: D100, D101, D102, D103, D104, D105, D106, D107

import pytest
import xarray as xr
from dpyverification.datasources.inputschemas import (
    TimeCoord,
    XarrayDatasetObservations,
    XarrayDatasetSimulationsByForecastPeriod,
    XarrayDatasetSimulationsByForecastReferenceTime,
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


def test_xarray_observations(xarray_dataset_observations: xr.Dataset) -> None:
    XarrayDatasetObservations.model_validate(xarray_dataset_observations.to_dict(data=False))


def test_xarray_observations_invalid_dims(xarray_dataset_observations: xr.Dataset) -> None:
    ds = xarray_dataset_observations.copy()
    ds = ds.expand_dims("invalid_dimension")
    with pytest.raises(ValidationError):
        XarrayDatasetObservations.model_validate(ds.to_dict(data=False))


def test_xarray_simulations_by_forecast_ref_time(
    xarray_dataset_simulations_forecast_reference_time: xr.Dataset,
) -> None:
    ds = xarray_dataset_simulations_forecast_reference_time
    XarrayDatasetSimulationsByForecastReferenceTime.model_validate(ds.to_dict(data=False))


def test_xarray_simulations_by_forecast_period_ensemble(
    xarray_dataset_simulations_forecast_period: xr.Dataset,
) -> None:
    ds = xarray_dataset_simulations_forecast_period
    XarrayDatasetSimulationsByForecastPeriod.model_validate(ds.to_dict(data=False))


def test_xarray_simulations_by_forecast_period_no_ensemble(
    xarray_dataset_simulations_forecast_period: xr.Dataset,
) -> None:
    ds = xarray_dataset_simulations_forecast_period.drop_vars("realization")
    XarrayDatasetSimulationsByForecastPeriod.model_validate(ds.to_dict(data=False))
