"""Shared resources across the test suite."""

# mypy: ignore-errors

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dpyverification.configuration import GeneralInfoConfig
from dpyverification.configuration.base import IdMappingConfig
from dpyverification.configuration.default.datasinks import CFCompliantNetCDFConfig
from dpyverification.configuration.default.datasources import (
    FewsNetCDFKind,
    FewsWebserviceAuthConfig,
    FewsWebserviceInputConfig,
)
from dpyverification.configuration.default.scores import CrpsForEnsembleConfig, RankHistogramConfig
from dpyverification.configuration.utils import (
    ForecastPeriods,
    Pair,
    TimePeriod,
    TimeUnits,
    VerificationPair,
)
from dpyverification.constants import DataSinkKind, ScoreKind, StandardCoord, StandardDim
from dpyverification.datamodel.main import SimObsDataset
from dpyverification.datasinks.cf_compliant_netdf import CFCompliantNetCDF
from dpyverification.datasources.fewsnetcdf import FewsNetCDFFile
from dpyverification.datasources.fewswebservice import FewsWebservice, SimulationRetrievalMethod

from tests import TESTS_FEWS_COMPLIANT_FILE

cache_dir = Path(".verification_cache")


# Before each test - remove the cache directory
@pytest.fixture(autouse=True)
def _rm_cache_dir_before_each_test() -> None:
    """Remove the cache directory before each test."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


rng = np.random.default_rng(seed=42)

# Dims
n_time = 60
n_frt = 10  # One forecast every 6 hours
n_forecast_period = 4  # Hours
n_realization = 10
n_stations = 3
n_variables = 1
n_sources = 1

# Coords
start_date = "2025-01-01T00:00"
time = pd.date_range(start_date, periods=n_time, freq="h")
stations = [f"station{n}" for n in range(n_stations)]
x = rng.uniform(0, 100, size=n_stations)
y = rng.uniform(0, 100, size=n_stations)
z = rng.uniform(0, 10, size=n_stations)
lat = y
lon = x
realization = np.arange(1, n_realization + 1)
variables = [f"var_{x}" for x in range(n_variables)]
obs_sources = [f"obs_source_{x}" for x in range(n_sources)]
sim_sources = [f"sim_source_{x}" for x in range(n_sources)]
units = [f"unit_{x}" for x in range(n_sources)]

# One forecast every 6 hours
forecast_reference_time = pd.date_range(
    start_date,
    periods=n_frt,
    freq="6h",
)
forecast_period = np.array([np.timedelta64(i, "h") for i in range(1, n_forecast_period + 1)])


@pytest.fixture()
def xarray_data_array_observation() -> xr.DataArray:
    """Return example observations."""
    # Create observation data
    obs_data = rng.random((len(time), len(stations), 1, len(obs_sources)))

    # Create dataset
    return xr.DataArray(
        data=obs_data,
        name="observations",
        dims=[StandardDim.time, StandardDim.station, StandardDim.variable, StandardDim.source],
        coords={
            StandardCoord.time.name: time,
            StandardCoord.source.name: obs_sources,
            StandardCoord.variable.name: variables,
            StandardCoord.units.name: (StandardDim.variable, units),
            StandardDim.station: (StandardDim.station, stations),
            StandardCoord.station.name: (StandardDim.station, stations),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
    )


@pytest.fixture()
def xarray_data_array_simulation() -> xr.DataArray:
    """Return example simulations compatible with the internal datamodel.

    Uses forecast_period as dimension and coordinates.
    """
    data = rng.random(
        (n_time, n_forecast_period, n_realization, n_stations, 1, len(obs_sources)),
    )

    return xr.DataArray(
        data=data,
        name="simulations",
        dims=[
            StandardDim.time,
            StandardDim.forecast_period,
            StandardDim.realization,
            StandardDim.station,
            StandardDim.variable,
            StandardDim.source,
        ],
        coords={
            StandardCoord.time.name: time,
            StandardCoord.forecast_period.name: forecast_period,
            StandardCoord.realization.name: realization,
            StandardCoord.source.name: sim_sources,
            StandardCoord.variable.name: variables,
            StandardCoord.units.name: (StandardDim.variable, units),
            StandardCoord.station.name: (StandardDim.station, stations),
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
        verification_pairs=[
            VerificationPair(
                id="dummy_var",
                source={"obs": "source_1", "sim": "source_1"},
            ),
        ],
    )


@pytest.fixture()
def id_mapping_config_fewsnetcdf() -> IdMappingConfig:
    """Id mapping config to be used across tests."""
    return IdMappingConfig(
        variable={"discharge": {"observed": "Q_m", "Sobek3": "Q_fs"}},
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
        verification_pairs=[
            VerificationPair(
                id="pair1",
                source=Pair(obs="observed", sim="Sobek3"),
            ),
        ],
    )


@pytest.fixture()
def datasource_fewsnetcdf_obs(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
    id_mapping_config_fewsnetcdf: IdMappingConfig,
) -> FewsNetCDFFile:
    """Fewsnetcdf datasource obs config."""
    return FewsNetCDFFile.from_config(
        {
            "kind": "fewsnetcdf",
            "simobskind": "obs",
            "netcdf_kind": "observation",
            "directory": "tests/data/webservice_responses_netcdf/obs",
            "filename_glob": "*.nc",
            "station_ids": ["H-RN-0001", "H-RN-0689"],
            "source": "observed",
            "general": general_info_config_fewsnetcdf.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture()
def datasource_fewsnetcdf_sim(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
    id_mapping_config_fewsnetcdf: IdMappingConfig,
) -> FewsNetCDFFile:
    """Fewsnetcdf datasource sim config."""
    return FewsNetCDFFile.from_config(
        {
            "kind": "fewsnetcdf",
            "simobskind": "sim",
            "netcdf_kind": FewsNetCDFKind.simulation_per_forecast_reference_time,
            "directory": "tests/data/webservice_responses_netcdf/sim",
            "filename_glob": "*.nc",
            "station_ids": ["H-RN-0001", "H-RN-0689"],
            "source": "Sobek3",
            "general": general_info_config_fewsnetcdf.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture()
def fews_webservice_auth_config() -> FewsWebserviceAuthConfig:
    """Create a mock environment for testing secret env vars."""
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
) -> FewsNetCDFFile:
    """Get a fews netcdf datasource."""
    config = datasource_fewsnetcdf_obs
    config.station_ids = None
    config.directory = TESTS_FEWS_COMPLIANT_FILE.parent
    return FewsNetCDFFile(datasource_fewsnetcdf_obs)


@pytest.fixture()
def simobsdataset_dummy_data_forecast_reference_time(
    xarray_data_array_observation: xr.DataArray,
    xarray_dataset_simulations_forecast_reference_time: xr.DataArray,
    testconfig_general_info_simobsdataset_from_dummy_data: GeneralInfoConfig,
) -> SimObsDataset:
    """Initialize datamodel with observations and forecasts (based on frt)."""
    return SimObsDataset(
        data=[xarray_data_array_observation, xarray_dataset_simulations_forecast_reference_time],
        general_config=testconfig_general_info_simobsdataset_from_dummy_data,
    )


@pytest.fixture()
def simobsdataset_fews_netcdf_data(
    datasource_fewsnetcdf_obs: FewsNetCDFFile,
    datasource_fewsnetcdf_sim: FewsNetCDFFile,
    general_info_config_fewsnetcdf: GeneralInfoConfig,
) -> SimObsDataset:
    """Initialize datamodel with observations and forecasts (based on frt)."""
    return SimObsDataset(
        data=[
            datasource_fewsnetcdf_obs.get_data().data_array,
            datasource_fewsnetcdf_sim.get_data().data_array,
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
    )


@pytest.fixture()
def score_config_rank_histogram(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
) -> RankHistogramConfig:
    """Flexible fixture for scores config, sharing general config."""
    return RankHistogramConfig(
        kind=ScoreKind.rank_histogram,
        general=general_info_config_fewsnetcdf,
    )


@pytest.fixture()
def datasink_cf_compliant_netcdf(
    tmp_path: Path,
    general_info_config_fewsnetcdf: GeneralInfoConfig,
) -> CFCompliantNetCDF:
    """CF Compliant NetCDF datasink."""
    return CFCompliantNetCDF(
        config=CFCompliantNetCDFConfig(
            kind=DataSinkKind.cf_compliant_netcdf,
            directory=str(tmp_path),
            filename="test.nc",
            general=general_info_config_fewsnetcdf,
            institution="Deltares",
        ),
    )
