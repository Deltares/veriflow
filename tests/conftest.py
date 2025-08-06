"""Shared resources across the test suite."""

# mypy: ignore-errors

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dpyverification.configuration import GeneralInfoConfig
from dpyverification.configuration.utils import LeadTimes, TimePeriod, TimeUnits
from dpyverification.datamodel.main import InputDataset

rng = np.random.default_rng(seed=42)

# Dims
n_time = 60
n_frt = 10  # One forecast every 6 hours
n_forecast_period = 4  # Hours
n_realization = 10
n_stations = 3

# Coords
start_date = "2025-01-01T00:00"
time = pd.date_range(start_date, periods=n_time, freq="h")
station_ids = [f"station{n}" for n in range(n_stations)]
x = rng.uniform(0, 100, size=n_stations)
y = rng.uniform(0, 100, size=n_stations)
z = rng.uniform(0, 10, size=n_stations)
realization = np.arange(1, n_realization + 1)

# One forecast every 6 hours
forecast_reference_time = pd.date_range(
    start_date,
    periods=n_frt,
    freq="6h",
)
forecast_period = np.array([np.timedelta64(i, "h") for i in range(1, n_forecast_period + 1)])


@pytest.fixture()
def xarray_dataset_observations() -> xr.Dataset:
    """Return example observations."""
    # Create observation data
    obs_data = rng.random((len(time), len(station_ids)))

    # Create dataset
    return xr.Dataset(
        data_vars={
            "obs": (("time", "stations"), obs_data),
        },
        coords={
            "time": time,
            "station_id": ("stations", station_ids),
            "x": ("stations", x),
            "y": ("stations", y),
            "z": ("stations", z),
        },
    )


@pytest.fixture()
def xarray_dataset_simulations_forecast_reference_time() -> xr.Dataset:
    """Return example simulations compatible with the internal datamodel."""
    # Generate random forecast data using Generator
    data = rng.random((n_time, n_frt, n_realization, n_stations))

    # Create Dataset
    ds = xr.Dataset(
        {
            "forecast": (("time", "forecast_reference_time", "realization", "stations"), data),
        },
        coords={
            "time": time,
            "forecast_reference_time": forecast_reference_time,
            "realization": realization,
            "station_id": ("stations", station_ids),
            "x": ("stations", x),
            "y": ("stations", y),
            "z": ("stations", z),
        },
    )

    # Mask some forecast values to be more realistic
    mask = (ds["time"] >= ds["forecast_reference_time"]) & (
        ds["time"] <= ds["forecast_reference_time"] + max(forecast_period)
    )
    return ds.where(mask)


@pytest.fixture()
def xarray_dataset_simulations_forecast_period() -> xr.Dataset:
    """Return example simulations compatible with the internal datamodel.

    Uses forecast_period as dimension and coordinates.
    """
    data = rng.random((n_time, n_forecast_period, n_realization, n_stations))

    return xr.Dataset(
        {
            "forecast": (("time", "forecast_period", "realization", "stations"), data),
        },
        coords={
            "time": time,
            "forecast_period": forecast_period,
            "realization": realization,
            "station_id": ("stations", station_ids),
            "x": ("stations", x),
            "y": ("stations", y),
            "z": ("stations", z),
        },
    )


@pytest.fixture()
def datamodel_forecast_reference_time(
    xarray_dataset_observations: xr.Dataset,
    xarray_dataset_simulations_forecast_reference_time: xr.Dataset,
) -> InputDataset:
    """Initialize datamodel with observations and forecasts (based on frt)."""
    general_config = GeneralInfoConfig(
        verificationperiod=TimePeriod(
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 3, tzinfo=timezone.utc),
        ),
        leadtimes=LeadTimes(unit=TimeUnits.HOUR, values=[1, 2, 3, 4]),
    )

    return InputDataset(
        data=[xarray_dataset_observations, xarray_dataset_simulations_forecast_reference_time],
        general_config=general_config,
    )
