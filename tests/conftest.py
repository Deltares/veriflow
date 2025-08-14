"""Shared resources across the test suite."""

# mypy: ignore-errors

from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dpyverification.configuration import GeneralInfoConfig
from dpyverification.configuration.default.datasinks import CFCompliantNetCDFConfig
from dpyverification.configuration.default.datasources import (
    FewsNetcdfKind,
    FewsWebserviceAuthConfig,
    FewsWebserviceInputConfig,
)
from dpyverification.configuration.default.scores import CrpsForEnsembleConfig, RankHistogramConfig
from dpyverification.configuration.utils import (
    ForecastPeriods,
    SimObsVariables,
    TimePeriod,
    TimeUnits,
)
from dpyverification.constants import DataSinkKind, ScoreKind, StandardCoord, StandardDim
from dpyverification.datamodel.main import SimObsDataset
from dpyverification.datasinks.cf_compliant_netdf import CFCompliantNetCDF
from dpyverification.datasources.fewsnetcdf import FewsNetcdfFile
from dpyverification.datasources.fewswebservice import FewsWebservice, SimulationRetrievalMethod

from tests import TESTS_FEWS_COMPLIANT_FILE

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
lat = y
lon = x
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
            "obs": ((StandardDim.time, StandardDim.station), obs_data),
        },
        coords={
            StandardCoord.time.name: time,
            StandardCoord.station_id.name: (StandardDim.station, station_ids),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
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
            "forecast": (
                (
                    StandardDim.time,
                    StandardDim.forecast_reference_time,
                    StandardDim.realization,
                    StandardDim.station,
                ),
                data,
            ),
        },
        coords={
            StandardCoord.time.name: time,
            StandardCoord.forecast_reference_time.name: forecast_reference_time,
            StandardCoord.realization.name: realization,
            StandardCoord.station_id.name: (StandardDim.station, station_ids),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
    )

    # Mask some forecast values to be more realistic
    mask = (ds[StandardDim.time] >= ds[StandardDim.forecast_reference_time]) & (
        ds[StandardDim.time] <= ds[StandardDim.forecast_reference_time] + max(forecast_period)
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
            "forecast": (
                (
                    StandardDim.time,
                    StandardDim.forecast_period,
                    StandardDim.realization,
                    StandardDim.station,
                ),
                data,
            ),
        },
        coords={
            StandardCoord.time.name: time,
            StandardCoord.forecast_period.name: forecast_period,
            StandardCoord.realization.name: realization,
            StandardCoord.station_id.name: (StandardDim.station, station_ids),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
    )


@pytest.fixture()
def testconfig_general_info_simobsdataset_from_dummy_data() -> GeneralInfoConfig:
    """General info config to be used across tests."""
    return GeneralInfoConfig(
        verification_period=TimePeriod(
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 3, tzinfo=timezone.utc),
        ),
        forecast_periods=ForecastPeriods(unit=TimeUnits.HOUR, values=[1, 2, 3, 4]),
    )


@pytest.fixture()
def general_info_config_fewsnetcdf() -> GeneralInfoConfig:
    """Get general info config matching the test data."""
    return GeneralInfoConfig(
        verification_period=TimePeriod(
            start=datetime(2024, 11, 10, tzinfo=timezone.utc),
            end=datetime(2024, 12, 1, tzinfo=timezone.utc),
        ),
        forecast_periods=ForecastPeriods(unit=TimeUnits.DAY, values=[1, 2, 3, 4]),
    )


@pytest.fixture()
def datasource_fewsnetcdf_obs(general_info_config_fewsnetcdf: GeneralInfoConfig) -> FewsNetcdfFile:
    """Fewsnetcdf datasource obs config."""
    return FewsNetcdfFile.from_config(
        {
            "kind": "fewsnetcdf",
            "simobskind": "obs",
            "netcdf_kind": "observation",
            "directory": "tests/data/webservice_responses_netcdf/obs",
            "filename_pattern": "*.nc",
            "station_ids": ["H-RN-0001", "H-RN-0689"],
            "general": general_info_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture()
def datasource_fewsnetcdf_sim(general_info_config_fewsnetcdf: GeneralInfoConfig) -> FewsNetcdfFile:
    """Fewsnetcdf datasource sim config."""
    return FewsNetcdfFile.from_config(
        {
            "kind": "fewsnetcdf",
            "simobskind": "sim",
            "netcdf_kind": FewsNetcdfKind.one_full_simulation,
            "directory": "tests/data/webservice_responses_netcdf/sim",
            "filename_pattern": "*.nc",
            "station_ids": ["H-RN-0001", "H-RN-0689"],
            "general": general_info_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture()
def fews_webservice_auth_config(
    monkeypatch: Generator[pytest.MonkeyPatch, None, None],
) -> FewsWebserviceAuthConfig:
    """Create a mock environment for testing secret env vars."""
    # The dummy url, username and password
    url = "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
    monkeypatch.setenv("FEWSWEBSERVICE_URL", url)  # type: ignore  # noqa: PGH003
    monkeypatch.setenv("FEWSWEBSERVICE_USERNAME", "")  # type: ignore  # noqa: PGH003
    monkeypatch.setenv("FEWSWEBSERVICE_PASSWORD", "")  # type: ignore  # noqa: PGH003
    return FewsWebserviceAuthConfig()


@pytest.fixture()
def datasource_fewswebservice_sim(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceInputConfig(
        kind="fewswebservice",
        simobskind="sim",
        location_ids=["H-RN-0001", "H-RN-0689"],
        parameter_ids=["Q_fs"],
        module_instance_ids=["SBK3_MaxRTK_ECMWF_ENS"],
        ensemble_id="ECMWF_ENS",
        simulation_retrieval_method=SimulationRetrievalMethod.retrieve_all_forecast_data,
        general=general_info_config_fewsnetcdf,
        auth_config=fews_webservice_auth_config,
    )
    return FewsWebservice(config)


@pytest.fixture()
def datasource_fewswebservice_obs(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceInputConfig(
        kind="fewswebservice",
        simobskind="obs",
        location_ids=["H-RN-0001", "H-RN-0689"],
        parameter_ids=["Q_m"],
        module_instance_ids=["Hydro_Prep"],
        general=general_info_config_fewsnetcdf,
        auth_config=fews_webservice_auth_config,
    )
    return FewsWebservice(config)


@pytest.fixture()
def datasource_fewsnetcdf_compliant(
    datasource_fewsnetcdf_obs: dict[
        str,
        str | list[str] | dict[str, dict[str, str | list[str]]],
    ],
) -> FewsNetcdfFile:
    """Get a fews netcdf datasource."""
    config = datasource_fewsnetcdf_obs
    config.station_ids = None
    config.directory = TESTS_FEWS_COMPLIANT_FILE.parent
    return FewsNetcdfFile(datasource_fewsnetcdf_obs)


@pytest.fixture()
def simobsdataset_dummy_data_forecast_reference_time(
    xarray_dataset_observations: xr.Dataset,
    xarray_dataset_simulations_forecast_reference_time: xr.Dataset,
    testconfig_general_info_simobsdataset_from_dummy_data: GeneralInfoConfig,
) -> SimObsDataset:
    """Initialize datamodel with observations and forecasts (based on frt)."""
    return SimObsDataset(
        data=[xarray_dataset_observations, xarray_dataset_simulations_forecast_reference_time],
        general_config=testconfig_general_info_simobsdataset_from_dummy_data,
    )


@pytest.fixture()
def simobsdataset_fews_netcdf_data(
    datasource_fewsnetcdf_obs: FewsNetcdfFile,
    datasource_fewsnetcdf_sim: FewsNetcdfFile,
    general_info_config_fewsnetcdf: GeneralInfoConfig,
) -> SimObsDataset:
    """Initialize datamodel with observations and forecasts (based on frt)."""
    return SimObsDataset(
        data=[
            datasource_fewsnetcdf_obs.get_data().dataset,
            datasource_fewsnetcdf_sim.get_data().dataset,
        ],
        general_config=general_info_config_fewsnetcdf,
    )


@pytest.fixture()
def score_config_crps(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
) -> CrpsForEnsembleConfig:
    """Flexible fixture for scores config, sharing general config."""
    return CrpsForEnsembleConfig(
        kind=ScoreKind.crps_for_ensemble,
        general=general_info_config_fewsnetcdf,
        variable_pairs=[(SimObsVariables(sim="Q_fs", obs="Q_m"))],
    )


@pytest.fixture()
def score_config_rank_histogram(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
) -> CrpsForEnsembleConfig:
    """Flexible fixture for scores config, sharing general config."""
    return RankHistogramConfig(
        kind=ScoreKind.rank_histogram,
        general=general_info_config_fewsnetcdf,
        variable_pairs=[(SimObsVariables(sim="Q_fs", obs="Q_m"))],
    )


@pytest.fixture()
def datasink_cf_compliant_netcdf(tmp_path: Path) -> CFCompliantNetCDF:
    """CF Compliant NetCDF datasink."""
    return CFCompliantNetCDF(
        config=CFCompliantNetCDFConfig(
            kind=DataSinkKind.cf_compliant_netcdf,
            directory=tmp_path,
            filename="test.nc",
        ),
    )
