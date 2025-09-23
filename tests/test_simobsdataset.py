"""Test the dpyverification.datamodel package."""

from datetime import datetime, timezone

import xarray as xr
from dpyverification.configuration import GeneralInfoConfig
from dpyverification.configuration.utils import (
    ForecastPeriods,
    Pair,
    TimePeriod,
    TimeUnits,
    VerificationPair,
)
from dpyverification.datamodel.main import SimObsDataset
from dpyverification.datasources.fewsnetcdf import FewsNetCDFFile

# mypy: disable-error-code="misc"


def test_init_simobsdataset(
    xarray_data_array_observation: xr.DataArray,
    xarray_data_array_simulation: xr.DataArray,
) -> None:
    """Test the simobsdataset initializes successfully with forecast period (fp) input."""
    general_config = GeneralInfoConfig(
        verification_period=TimePeriod(
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 3, tzinfo=timezone.utc),
        ),
        forecast_periods=ForecastPeriods(unit=TimeUnits.HOUR, values=[1, 2, 3, 4]),
        verification_pairs=[
            VerificationPair(
                id="pair1",
                source=Pair(obs="source_1", sim="source_1"),
            ),
        ],
    )

    _ = SimObsDataset(
        data=[xarray_data_array_observation, xarray_data_array_simulation],
        general_config=general_config,
    )


def test_init_simobsdataset_fewsnetcdf(
    datasource_fewsnetcdf_obs: FewsNetCDFFile,
    datasource_fewsnetcdf_sim: FewsNetCDFFile,
    general_info_config_fewsnetcdf: GeneralInfoConfig,
) -> None:
    """Test the fewsnetcdf is accepted by the simobsdataset."""
    SimObsDataset(
        data=[
            datasource_fewsnetcdf_obs.get_data().data_array,
            datasource_fewsnetcdf_sim.get_data().data_array,
        ],
        general_config=general_info_config_fewsnetcdf,
    )
