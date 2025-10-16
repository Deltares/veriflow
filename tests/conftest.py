"""Shared resources across the test suite."""

# mypy: ignore-errors

import json
import shutil
from copy import deepcopy
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
from dpyverification.configuration.default.scores import (
    ContinuousScoresConfig,
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    RankHistogramConfig,
)
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
def fews_webservice_auth_config() -> FewsWebserviceAuthConfig:
    """Read authorization config from environment."""
    return FewsWebserviceAuthConfig()


# Test data from Meuse
#   - simulated_forecast_single
#   - simulated_forecast_probabilistic


test_data_meuse_locations = ["H-MS-SINT"]
test_data_meuse_parameters = ["waterlevel", "discharge"]
test_data_meuse_module_instance_ids = {
    TimeseriesKind.simulated_forecast_single: "fews_riv_ecmwf_hres_sobek3_choozkeiz_bias",
    TimeseriesKind.simulated_forecast_probabilistic: "fews_riv_ecmwf_ens_sobek3_choozkeiz_ens_dres",
}
test_data_general_info_config_single = GeneralInfoConfig(
    verification_period=TimePeriod(
        start=datetime(2025, 9, 1, tzinfo=timezone.utc),
        end=datetime(2025, 9, 4, tzinfo=timezone.utc),
    ),
    forecast_periods=ForecastPeriods(unit=TimeUnits.DAY, values=[1, 2, 3, 4]),
    verification_pairs=[
        VerificationPair(
            id="pair1",
            obs="observed",
            sim="source_single",
        ),
    ],
)
test_data_general_info_config_probabilistic = GeneralInfoConfig(
    verification_period=TimePeriod(
        start=datetime(2025, 7, 1, tzinfo=timezone.utc),
        end=datetime(2025, 7, 4, tzinfo=timezone.utc),
    ),
    forecast_periods=ForecastPeriods(unit=TimeUnits.DAY, values=[1, 2, 3, 4]),
    verification_pairs=[
        VerificationPair(
            id="pair1",
            obs="observed",
            sim="source_probabilistic",
        ),
    ],
)
test_data_general_info_config_ensemble = GeneralInfoConfig(
    verification_period=TimePeriod(
        start=datetime(2024, 11, 10, tzinfo=timezone.utc),
        end=datetime(2024, 12, 1, tzinfo=timezone.utc),
    ),
    forecast_periods=ForecastPeriods(unit=TimeUnits.DAY, values=[1, 2, 3, 4]),
    verification_pairs=[
        VerificationPair(
            id="pair1",
            obs="observed",
            sim="source_ensemble",
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
                "source_single": "Q_fs",
                "source_ensemble": "Q_fs",
                "source_probabilistic": "Q_fs",
            },
        },
    )


@pytest.fixture()
def fews_webservice_observed_historical(
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
        general=test_data_general_info_config_ensemble,
        auth_config=fews_webservice_auth_config,
        id_mapping=id_mapping_config_fewsnetcdf,
    )
    return FewsWebservice(config)


@pytest.fixture()
def fews_webservice_simulated_forecast_ensemble_frt(
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
        general=test_data_general_info_config_ensemble,
        auth_config=fews_webservice_auth_config,
        id_mapping=id_mapping_config_fewsnetcdf,
    )
    return FewsWebservice(config)


@pytest.fixture()
def fews_webservice_simulated_forecast_ensemble_fp(
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
        general=test_data_general_info_config_ensemble,
        auth_config=fews_webservice_auth_config,
        id_mapping=id_mapping_config_fewsnetcdf,
    )
    return FewsWebservice(config)


@pytest.fixture()
def fews_webservice_simulated_forecast_single_frt(
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        kind="fewswebservice",
        timeseries_kind=TimeseriesKind.simulated_forecast_single,
        location_ids=test_data_meuse_locations,
        parameter_ids=test_data_meuse_parameters,
        module_instance_id=test_data_meuse_module_instance_ids[
            TimeseriesKind.simulated_forecast_single
        ],
        archive_kind=ArchiveKind.external_storage_archive,
        forecast_retrieval_method=SimulationRetrievalMethod.retrieve_all_forecast_data,
        general=test_data_general_info_config_single.model_dump(),
        auth_config=fews_webservice_auth_config,
    )
    return FewsWebservice(config)


@pytest.fixture()
def fews_webservice_simulated_forecast_single_fp(
    fews_webservice_simulated_forecast_single_frt: FewsWebservice,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    instance = deepcopy(
        fews_webservice_simulated_forecast_single_frt,
    )
    instance.config.forecast_retrieval_method = (
        SimulationRetrievalMethod.retrieve_forecast_data_per_lead_time
    )
    return instance


@pytest.fixture()
def fews_webservice_simulated_forecast_probabilistic_frt(
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        kind="fewswebservice",
        timeseries_kind=TimeseriesKind.simulated_forecast_probabilistic,
        location_ids=test_data_meuse_locations,
        parameter_ids=["discharge"],
        module_instance_id=test_data_meuse_module_instance_ids[
            TimeseriesKind.simulated_forecast_probabilistic
        ],
        ensemble_id="ensembleQR",
        archive_kind=ArchiveKind.external_storage_archive,
        forecast_retrieval_method=SimulationRetrievalMethod.retrieve_all_forecast_data,
        general=test_data_general_info_config_probabilistic.model_dump(),
        auth_config=fews_webservice_auth_config,
    )
    return FewsWebservice(config)


@pytest.fixture()
def fews_webservice_simulated_forecast_probabilistic_fp(
    fews_webservice_simulated_forecast_probabilistic_frt: FewsWebservice,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    instance = deepcopy(
        fews_webservice_simulated_forecast_probabilistic_frt,
    )
    instance.config.forecast_retrieval_method = (
        SimulationRetrievalMethod.retrieve_forecast_data_per_lead_time
    )
    return instance


@pytest.fixture()
def fews_webservice_timeseries_headers_only() -> xr.Dataset:
    """Return xarray dataset for FEWS Compliant file."""
    file_path = Path("tests/data/webservice_responses_netcdf/timeseries_headers.json")
    with file_path.open(mode="r", encoding="utf8") as f:
        return json.load(f)


# Fews NetCDF fixtures - for local testing on test data


## Observed
@pytest.fixture()
def fews_netcdf_observed_historical(
    id_mapping_config_fewsnetcdf: IdMappingConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource obs config."""
    return FewsNetCDF.from_config(
        {
            "kind": "fewsnetcdf",
            "timeseries_kind": "observed_historical",
            "netcdf_kind": "observation",
            "directory": "tests/data/webservice_responses_netcdf/observations",
            "filename_glob": "*.nc",
            "station_ids": ["H-RN-0001", "H-RN-0689"],
            "source": "observed",
            "general": test_data_general_info_config_ensemble.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


## Simulated Forecast Ensemble
@pytest.fixture()
def fews_netcdf_simulated_forecast_ensemble_frt(
    id_mapping_config_fewsnetcdf: IdMappingConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    return FewsNetCDF.from_config(
        {
            "kind": "fewsnetcdf",
            "timeseries_kind": "simulated_forecast_ensemble",
            "netcdf_kind": FewsNetCDFKind.simulated_forecast_per_forecast_reference_time,
            "directory": "tests/data/webservice_responses_netcdf/simulations_per_forecast_reference_time/ensemble",  # noqa: E501
            "filename_glob": "*.nc",
            "source": "source_ensemble",
            "general": test_data_general_info_config_ensemble.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture()
def fews_netcdf_simulated_forecast_ensemble_fp(
    fews_netcdf_simulated_forecast_ensemble_frt: FewsNetCDF,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    instance = deepcopy(fews_netcdf_simulated_forecast_ensemble_frt)
    instance.config.netcdf_kind = FewsNetCDFKind.simulated_forecast_per_forecast_period
    instance.config.directory = (
        "tests/data/webservice_responses_netcdf/simulations_per_forecast_period/ensemble"
    )
    return instance


## Simulated Forecast Single
@pytest.fixture()
def fews_netcdf_simulated_forecast_single_frt(
    id_mapping_config_fewsnetcdf: IdMappingConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    return FewsNetCDF.from_config(
        {
            "kind": "fewsnetcdf",
            "timeseries_kind": TimeseriesKind.simulated_forecast_single,
            "netcdf_kind": FewsNetCDFKind.simulated_forecast_per_forecast_reference_time,
            "directory": "tests/data/webservice_responses_netcdf/simulations_per_forecast_reference_time/single",  # noqa: E501
            "filename_glob": "*.nc",
            "source": "source_single",
            "general": test_data_general_info_config_single.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture()
def fews_netcdf_simulated_forecast_single_fp(
    fews_netcdf_simulated_forecast_single_frt: FewsNetCDF,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    instance = deepcopy(fews_netcdf_simulated_forecast_single_frt)
    instance.config.netcdf_kind = FewsNetCDFKind.simulated_forecast_per_forecast_period
    instance.config.directory = (
        "tests/data/webservice_responses_netcdf/simulations_per_forecast_period/single"
    )
    return instance


## Simulated Forecast Probabilistic
@pytest.fixture()
def fews_netcdf_simulated_forecast_probabilistic_frt() -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    return FewsNetCDF.from_config(
        {
            "kind": "fewsnetcdf",
            "timeseries_kind": TimeseriesKind.simulated_forecast_probabilistic,
            "netcdf_kind": FewsNetCDFKind.simulated_forecast_per_forecast_reference_time,
            "directory": "tests/data/webservice_responses_netcdf/simulations_per_forecast_reference_time/probabilistic",  # noqa: E501
            "filename_glob": "*.nc",
            "source": "source_probabilistic",
            "general": test_data_general_info_config_probabilistic.model_dump(),
        },
    )


@pytest.fixture()
def fews_netcdf_simulated_forecast_probabilistic_fp(
    fews_netcdf_simulated_forecast_probabilistic_frt: FewsNetCDF,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    instance = deepcopy(fews_netcdf_simulated_forecast_probabilistic_frt)
    instance.config.netcdf_kind = FewsNetCDFKind.simulated_forecast_per_forecast_period
    instance.config.directory = (
        "tests/data/webservice_responses_netcdf/simulations_per_forecast_period/probabilistic"
    )
    return instance


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


# Input dataset


@pytest.fixture()
def input_dataset_dummy_data_forecast_reference_time(
    xarray_data_array_observation: xr.DataArray,
    xarray_dataset_simulations_forecast_reference_time: xr.DataArray,
) -> InputDataset:
    """Initialize datamodel with observations and forecasts (based on frt)."""
    return InputDataset(
        data=[xarray_data_array_observation, xarray_dataset_simulations_forecast_reference_time],
    )


@pytest.fixture()
def input_dataset_fews_netcdf_simulated_forecast_ensemble(
    fews_netcdf_observed_historical: FewsNetCDF,
    fews_netcdf_simulated_forecast_ensemble_frt: FewsNetCDF,
) -> InputDataset:
    """Initialize datamodel with observations and forecasts (based on frt)."""
    return InputDataset(
        data=[
            fews_netcdf_observed_historical.get_data().data_array,
            fews_netcdf_simulated_forecast_ensemble_frt.get_data().data_array,
        ],
    )


# Scores


@pytest.fixture()
def score_config_crps() -> CrpsForEnsembleConfig:
    """Flexible fixture for scores config, sharing general config."""
    return CrpsForEnsembleConfig(
        kind=ScoreKind.crps_for_ensemble,
        general=test_data_general_info_config_ensemble.model_dump(),
    )


@pytest.fixture()
def score_config_rank_histogram() -> RankHistogramConfig:
    """Flexible fixture for scores config, sharing general config."""
    return RankHistogramConfig(
        kind=ScoreKind.rank_histogram,
        general=test_data_general_info_config_ensemble.model_dump(),
    )


@pytest.fixture()
def score_config_crps_cdf() -> CrpsCDFConfig:
    """Flexible fixture for scores config, sharing general config."""
    return CrpsCDFConfig(
        kind=ScoreKind.crps_cdf,
        general=test_data_general_info_config_probabilistic.model_dump(),
    )


@pytest.fixture()
def score_config_continuous() -> ContinuousScoresConfig:
    """Flexible fixture for scores config, sharing general config."""
    return ContinuousScoresConfig(
        kind=ScoreKind.continuous_scores,
        general=test_data_general_info_config_single.model_dump(),
        scores=["mae", "rmse"],
    )


# Sinks


@pytest.fixture()
def datasink_cf_compliant_netcdf(
    tmp_path: Path,
) -> CFCompliantNetCDF:
    """CF Compliant NetCDF datasink."""
    return CFCompliantNetCDF(
        config=CFCompliantNetCDFConfig(
            kind=DataSinkKind.cf_compliant_netcdf,
            directory=str(tmp_path),
            filename="test.nc",
            general=test_data_general_info_config_ensemble.model_dump(),
            institution="Deltares",
        ),
    )
