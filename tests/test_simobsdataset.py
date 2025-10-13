"""Test the dpyverification.datamodel package."""

from datetime import datetime, timezone

import xarray as xr
from dpyverification.configuration import GeneralInfoConfig
from dpyverification.configuration.utils import (
    ForecastPeriods,
    TimePeriod,
    TimeUnits,
    VerificationPair,
)
from dpyverification.datamodel.main import InputDataset
from dpyverification.datasources.fewsnetcdf import FewsNetCDF

# mypy: disable-error-code="misc"


def test_init_input_dataset(
    xarray_data_array_observations: xr.DataArray,
    xarray_data_array_simulation: xr.DataArray,
) -> None:
    """Test the input_dataset initializes successfully with forecast period (fp) input."""
    general_config = GeneralInfoConfig(
        verification_period=TimePeriod(
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 3, tzinfo=timezone.utc),
        ),
        forecast_periods=ForecastPeriods(unit=TimeUnits.HOUR, values=[1, 2, 3, 4]),
        verification_pairs=[
            VerificationPair(
                id="pair1",
                obs="source_1",
                sim="source_1",
            ),
        ],
    )

    _ = InputDataset(
        data=[xarray_data_array_observations, xarray_data_array_simulation],
        general_config=general_config,
    )


def test_init_input_dataset_fewsnetcdf(
    datasource_fewsnetcdf_obs: FewsNetCDF,
    datasource_fewsnetcdf_sim_per_forecast_reference_time: FewsNetCDF,
    general_info_config_fewsnetcdf: GeneralInfoConfig,
) -> None:
    """Test the fewsnetcdf is accepted by the input_dataset."""
    InputDataset(
        data=[
            datasource_fewsnetcdf_obs.get_data().data_array,
            datasource_fewsnetcdf_sim_per_forecast_reference_time.get_data().data_array,
        ],
        general_config=general_info_config_fewsnetcdf,
    )
