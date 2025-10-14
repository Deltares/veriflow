"""Shared resources across the test suite."""

# mypy: ignore-errors

import shutil
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dpyverification.configuration import GeneralInfoConfig
from dpyverification.configuration.base import IdMappingConfig
from dpyverification.configuration.default.datasinks import CFCompliantNetCDFConfig
from dpyverification.configuration.default.datasources import (
    ArchiveKind,
    FewsWebserviceAuthConfig,
    FewsWebserviceConfig,
)
from dpyverification.configuration.default.scores import CrpsForEnsembleConfig, RankHistogramConfig
from dpyverification.configuration.utils import (
    ForecastPeriods,
    TimePeriod,
    TimeUnits,
    VerificationPair,
)
from dpyverification.constants import (
    DataSinkKind,
    ScoreKind,
    StandardCoord,
    StandardDim,
    TimeseriesKind,
)
from dpyverification.datamodel.main import InputDataset
from dpyverification.datasinks.cf_compliant_netdf import CFCompliantNetCDF
from dpyverification.datasources.fewsnetcdf import FewsNetCDF, FewsNetCDFKind
from dpyverification.datasources.fewswebservice import FewsWebservice, SimulationRetrievalMethod

TESTS_DATA_DIR = Path(__file__).parent / "data"

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


class DummySource(StrEnum):
    """Dummy sources."""

    observation_source = "observation_source"
    simulation_ensemble_source = "simulation_ensemble_source"
    simulation_single_source = "simulation_single_source"


# One forecast every 6 hours
forecast_reference_time = pd.date_range(
    start_date,
    periods=n_frt,
    freq="6h",
)
forecast_period = np.array([np.timedelta64(i, "h") for i in range(1, n_forecast_period + 1)])


@pytest.fixture()
def xarray_dataset_fews_compliant() -> xr.Dataset:
    """Return xarray dataset for FEWS Compliant file."""
    return xr.open_dataset(TESTS_DATA_DIR / "fews_compliant_test_file.nc")


@pytest.fixture()
def xarray_observed_historical() -> xr.Dataset:
    """Return example observations."""
    # Create observation data
    obs_data = rng.random((len(time), len(stations), 1))

    # Create dataset
    return xr.DataArray(
        data=obs_data,
        name=DummySource.observation_source,
        dims=[StandardDim.time, StandardDim.station, StandardDim.variable],
        coords={
            StandardCoord.time.name: time,
            StandardCoord.variable.name: variables,
            StandardCoord.units.name: (StandardDim.variable, ["dummy_unit"]),
            StandardDim.station: (StandardDim.station, stations),
            StandardCoord.station.name: (StandardDim.station, stations),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
        attrs={"timeseries_kind": TimeseriesKind.observed_historical},
    )


@pytest.fixture()
def xarray_simulated_forecast_ensemble() -> xr.DataArray:
    """Return example simulations compatible with the internal datamodel.

    Uses forecast_period as dimension and coordinates.
    """
    data = rng.random(
        (n_time, n_forecast_period, n_realization, n_stations, 1),
    )

    return xr.DataArray(
        data=data,
        name=DummySource.simulation_ensemble_source,
        dims=[
            StandardDim.time,
            StandardDim.forecast_period,
            StandardDim.realization,
            StandardDim.station,
            StandardDim.variable,
        ],
        coords={
            StandardCoord.time.name: time,
            StandardCoord.forecast_period.name: forecast_period,
            StandardCoord.realization.name: realization,
            StandardCoord.variable.name: variables,
            StandardCoord.units.name: (StandardDim.variable, ["dummy_unit"]),
            StandardCoord.station.name: (StandardDim.station, stations),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
        attrs={"timeseries_kind": TimeseriesKind.simulated_forecast_ensemble},
    )


@pytest.fixture()
def xarray_simulated_forecast_single() -> xr.DataArray:
    """Return example simulations compatible with the internal datamodel.

    Uses forecast_period as dimension and coordinates.
    """
    data = rng.random(
        (n_time, n_forecast_period, n_stations, 1),
    )

    return xr.DataArray(
        data=data,
        name=DummySource.simulation_single_source,
        dims=[
            StandardDim.time,
            StandardDim.forecast_period,
            StandardDim.station,
            StandardDim.variable,
        ],
        coords={
            StandardCoord.time.name: time,
            StandardCoord.forecast_period.name: forecast_period,
            StandardCoord.variable.name: variables,
            StandardCoord.units.name: (StandardDim.variable, ["dummy_unit"]),
            StandardCoord.station.name: (StandardDim.station, stations),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
        attrs={"timeseries_kind": TimeseriesKind.simulated_forecast_single},
    )


@pytest.fixture()
def testconfig_general_info_input_dataset_from_dummy_data() -> GeneralInfoConfig:
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
                obs=DummySource.observation_source,
                sim=DummySource.simulation_ensemble_source,
            ),
        ],
    )


@pytest.fixture()
def id_mapping_config_fewsnetcdf() -> IdMappingConfig:
    """Id mapping config to be used across tests."""
    return IdMappingConfig(
        variable={
            "discharge": {
                "observed": "Q_m",
                "Hydro_Prep": "Q_m",
                "SBK3_MaxRTK_ECMWF_ENS": "Q_fs",
                "Sobek3": "Q_fs",
            },
        },
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
                obs="observed",
                sim="Sobek3",
            ),
        ],
    )


@pytest.fixture()
def fews_netcdf_observed_historical(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
    id_mapping_config_fewsnetcdf: IdMappingConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource obs config."""
    return FewsNetCDF.from_config(
        {
            "kind": "fewsnetcdf",
            "timeseries_kind": "observed_historical",
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
def fews_netcdf_simulated_forecast_ensemble_frt(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
    id_mapping_config_fewsnetcdf: IdMappingConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    return FewsNetCDF.from_config(
        {
            "kind": "fewsnetcdf",
            "timeseries_kind": "simulated_forecast_ensemble",
            "netcdf_kind": FewsNetCDFKind.simulated_forecast_per_forecast_reference_time,
            "directory": "tests/data/webservice_responses_netcdf/sim_per_forecast_reference_time",
            "filename_glob": "*.nc",
            "station_ids": ["H-RN-0001", "H-RN-0689"],
            "source": "Sobek3",
            "general": general_info_config_fewsnetcdf.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture()
def fews_webservice_auth_config() -> FewsWebserviceAuthConfig:
    """Read authorization config from environment."""
    return FewsWebserviceAuthConfig()


@pytest.fixture()
def fews_webservice_observed_historical(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
    id_mapping_config_fewsnetcdf: IdMappingConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        kind="fewswebservice",
        timeseries_kind="observed_historical",
        location_ids=["H-RN-0001", "H-RN-0689"],
        parameter_ids=["Q_m"],
        module_instance_id="Hydro_Prep",
        general=general_info_config_fewsnetcdf,
        auth_config=fews_webservice_auth_config,
        id_mapping=id_mapping_config_fewsnetcdf,
    )
    return FewsWebservice(config)


@pytest.fixture()
def fews_webservice_simulated_forecast_ensemble_by_forecast_reference_time(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
    id_mapping_config_fewsnetcdf: IdMappingConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        kind="fewswebservice",
        timeseries_kind="simulated_forecast_ensemble",
        location_ids=["H-RN-0001", "H-RN-0689"],
        parameter_ids=["Q_fs"],
        module_instance_id="SBK3_MaxRTK_ECMWF_ENS",
        ensemble_id="ECMWF_ENS",
        archive_kind=ArchiveKind.external_storage_archive,
        forecast_retrieval_method=SimulationRetrievalMethod.retrieve_all_forecast_data,
        general=general_info_config_fewsnetcdf,
        auth_config=fews_webservice_auth_config,
        id_mapping=id_mapping_config_fewsnetcdf,
    )
    return FewsWebservice(config)


@pytest.fixture()
def fews_webservice_simulated_forecast_ensemble_by_forecast_period(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
    id_mapping_config_fewsnetcdf: IdMappingConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        kind="fewswebservice",
        timeseries_kind="simulated_forecast_ensemble",
        location_ids=["H-RN-0001", "H-RN-0689"],
        parameter_ids=["Q_fs"],
        module_instance_id="SBK3_MaxRTK_ECMWF_ENS",
        ensemble_id="ECMWF_ENS",
        archive_kind=ArchiveKind.external_storage_archive,
        forecast_retrieval_method=SimulationRetrievalMethod.retrieve_forecast_data_per_lead_time,
        general=general_info_config_fewsnetcdf,
        auth_config=fews_webservice_auth_config,
        id_mapping=id_mapping_config_fewsnetcdf,
    )
    return FewsWebservice(config)


# Test data from Meuse
#   - simulated_forecast_single
#   - simulated_forecast_probabilistic

test_data_meuse_locations = ["H-MS-EIJS"]
test_data_meuse_parameters = ["waterlevel", "discharge"]
test_data_meuse_module_instance_ids = {
    TimeseriesKind.simulated_forecast_single: "fews_riv_ecmwf_hres_sobek3_choozkeiz_bias",
}
test_data_meuse_general_info_config = GeneralInfoConfig(
    verification_period=TimePeriod(
        start=datetime(2025, 9, 1, tzinfo=timezone.utc),
        end=datetime(2025, 9, 4, tzinfo=timezone.utc),
    ),
    forecast_periods=ForecastPeriods(unit=TimeUnits.DAY, values=[1, 2, 3, 4]),
    verification_pairs=[
        VerificationPair(
            id="fews_riv_ecmwf_hres_sobek3_choozkeiz_bias",
            obs="observed",
            sim="fews_riv_ecmwf_hres_sobek3_choozkeiz_bias",
        ),
    ],
)


@pytest.fixture()
def fews_webservice_simulated_forecast_single_by_forecast_reference_time(
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        kind="fewswebservice",
        timeseries_kind=TimeseriesKind.simulated_forecast_single,
        source="fews_riv_ecmwf_hres_sobek3_choozkeiz_bias",
        location_ids=test_data_meuse_locations,
        parameter_ids=test_data_meuse_parameters,
        module_instance_id=test_data_meuse_module_instance_ids[
            TimeseriesKind.simulated_forecast_single
        ],
        archive_kind=ArchiveKind.external_storage_archive,
        forecast_retrieval_method=SimulationRetrievalMethod.retrieve_all_forecast_data,
        general=test_data_meuse_general_info_config,
        auth_config=fews_webservice_auth_config,
    )
    return FewsWebservice(config)


@pytest.fixture()
def fews_netcdf_compliant_file(
    fews_netcdf_observed_historical: dict[
        str,
        str | list[str] | dict[str, dict[str, str | list[str]]],
    ],
) -> FewsNetCDF:
    """Get a fews netcdf datasource."""
    config = fews_netcdf_observed_historical
    config.station_ids = None
    config.directory = TESTS_DATA_DIR
    return FewsNetCDF(fews_netcdf_observed_historical)


@pytest.fixture()
def input_dataset_dummy_data_forecast_reference_time(
    xarray_data_array_observation: xr.DataArray,
    xarray_dataset_simulations_forecast_reference_time: xr.DataArray,
    testconfig_general_info_input_dataset_from_dummy_data: GeneralInfoConfig,
) -> InputDataset:
    """Initialize datamodel with observations and forecasts (based on frt)."""
    return InputDataset(
        data=[xarray_data_array_observation, xarray_dataset_simulations_forecast_reference_time],
        general_config=testconfig_general_info_input_dataset_from_dummy_data,
    )


@pytest.fixture()
def input_dataset_fews_netcdf_data(
    fews_netcdf_observed_historical: FewsNetCDF,
    fews_netcdf_simulated_forecast_ensemble_frt: FewsNetCDF,
    general_info_config_fewsnetcdf: GeneralInfoConfig,
) -> InputDataset:
    """Initialize datamodel with observations and forecasts (based on frt)."""
    return InputDataset(
        data=[
            fews_netcdf_observed_historical.get_data().data_array,
            fews_netcdf_simulated_forecast_ensemble_frt.get_data().data_array,
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
