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
from dpyverification.configuration.config import IdMappingConfig
from dpyverification.configuration.default.datasinks import CFCompliantNetCDFConfig
from dpyverification.configuration.default.datasources import (
    ArchiveKind,
    FewsWebserviceAuthConfig,
    FewsWebserviceConfig,
    ThresholdCsvConfig,
)
from dpyverification.configuration.default.scores import (
    CategoricalScoresConfig,
    ContinuousScoresConfig,
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    EventOperator,
    RankHistogramConfig,
    ThresholdOperator,
)
from dpyverification.configuration.utils import (
    ForecastPeriods,
    TimeUnits,
    VerificationPair,
    VerificationPeriod,
)
from dpyverification.constants import (
    DataSinkKind,
    DataSourceKind,
    DataType,
    ScoreKind,
    StandardCoord,
    StandardDim,
)
from dpyverification.datamodel.main import InputDataset
from dpyverification.datasinks.cf_compliant_netcdf import CFCompliantNetCDF
from dpyverification.datasources.fewsnetcdf import FewsNetCDF, FewsNetCDFKind
from dpyverification.datasources.fewswebservice import FewsWebservice, ForecastRetrievalMethod
from dpyverification.datasources.thresholds import ThresholdCsv

TESTS_DATA_DIR = Path(__file__).parent / "data"


rng = np.random.default_rng(seed=42)

# Settings for dummy xarray data.
#   fp  = forecast period
#   frt = forecast reference time
day_multiplier = 365 * 1  # Easily scale the verification period
dtype = "float32"

time_start = "2025-01-01T00:00"
time_step = fp_step = "h"
time_n = day_multiplier * 24

frt_start = "2025-01-01T00:00"
frt_step = "d"
frt_n = day_multiplier

fp_n = 96
realization_n = 10
station_n = 3
variable_n = 2
threshold_n = 4

# Coords
times = pd.date_range(time_start, periods=time_n, freq=time_step)
forecast_reference_times = pd.date_range(
    frt_start,
    periods=frt_n,
    freq=frt_step,
)
forecast_periods = pd.timedelta_range(0, periods=fp_n, freq=fp_step)
forecast_times = forecast_reference_times.to_numpy()[:, None] + forecast_periods.to_numpy()[None, :]
stations = [f"station_{n}" for n in range(station_n)]
x = rng.uniform(0, 100, size=station_n)
y = rng.uniform(0, 100, size=station_n)
z = rng.uniform(0, 10, size=station_n)
lat = y
lon = x
realization = np.arange(1, realization_n + 1)
variables = [f"var_{x}" for x in range(variable_n)]
thresholds = [f"warn_{x}" for x in range(threshold_n)]


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Pytest cache directory."""
    return tmp_path / "sub"


# Before each test - remove the cache directory
@pytest.fixture(autouse=True)
def _ensure_empty_cache_dir_before_each_test(cache_dir: Path) -> None:
    """Remove the cache directory before each test."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)


class DummySource(StrEnum):
    """Dummy sources."""

    observation_source = "observation_source"
    simulation_ensemble_source = "simulation_ensemble_source"
    simulation_single_source = "simulation_single_source"


@pytest.fixture
def xarray_dataset_fews_compliant() -> xr.Dataset:
    """Return xarray dataset for FEWS Compliant file."""
    return xr.open_dataset(TESTS_DATA_DIR / "fews_compliant_test_file.nc")


@pytest.fixture
def xarray_observed_historical() -> xr.DataArray:
    """Return example observations."""
    # Create observation data
    obs_data = rng.random((len(times), len(stations), variable_n), dtype=dtype)

    # Create dataset
    return xr.DataArray(
        data=obs_data,
        name=DummySource.observation_source,
        dims=[StandardDim.time, StandardDim.station, StandardDim.variable],
        coords={
            StandardCoord.time.name: times,
            StandardCoord.variable.name: variables,
            StandardCoord.units.name: (
                StandardDim.variable,
                [f"dummy_unit_{x}" for x in range(variable_n)],
            ),
            StandardDim.station: (StandardDim.station, stations),
            StandardCoord.station.name: (StandardDim.station, stations),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
        attrs={"data_type": DataType.observed_historical},
    )


@pytest.fixture
def xarray_simulated_forecast_ensemble() -> xr.DataArray:
    """Return example simulations compatible with the internal datamodel.

    Uses forecast_period as dimension and coordinates.
    """
    data = rng.random(
        (variable_n, station_n, frt_n, fp_n, realization_n),
        dtype=dtype,
    )

    return xr.DataArray(
        data=data,
        name=DummySource.simulation_ensemble_source,
        dims=[
            StandardDim.variable,
            StandardDim.station,
            StandardDim.forecast_reference_time,
            StandardDim.forecast_period,
            StandardDim.realization,
        ],
        coords={
            StandardCoord.forecast_reference_time.name: forecast_reference_times,
            StandardCoord.forecast_period.name: forecast_periods,
            StandardCoord.realization.name: realization,
            StandardCoord.variable.name: variables,
            StandardCoord.time.name: (
                (StandardDim.forecast_reference_time, StandardDim.forecast_period),
                forecast_times,
            ),
            StandardCoord.units.name: (
                StandardDim.variable,
                [f"dummy_unit_{x}" for x in range(variable_n)],
            ),
            StandardCoord.station.name: (StandardDim.station, stations),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
        attrs={"data_type": DataType.simulated_forecast_ensemble},
    )


@pytest.fixture
def xarray_simulated_forecast_single() -> xr.DataArray:
    """Return example simulations compatible with the internal datamodel.

    Uses forecast_period as dimension and coordinates.
    """
    data = rng.random(
        (variable_n, station_n, frt_n, fp_n),
        dtype=dtype,
    )

    return xr.DataArray(
        data=data,
        name=DummySource.simulation_single_source,
        dims=[
            StandardDim.variable,
            StandardDim.station,
            StandardDim.forecast_reference_time,
            StandardDim.forecast_period,
        ],
        coords={
            StandardCoord.forecast_reference_time.name: forecast_reference_times,
            StandardCoord.forecast_period.name: forecast_periods,
            StandardCoord.variable.name: variables,
            StandardCoord.time.name: (
                (StandardDim.forecast_reference_time, StandardDim.forecast_period),
                forecast_times,
            ),
            StandardCoord.units.name: (
                StandardDim.variable,
                [f"dummy_unit_{x}" for x in range(variable_n)],
            ),
            StandardCoord.station.name: (StandardDim.station, stations),
            StandardCoord.lat.name: (StandardDim.station, lat),
            StandardCoord.lon.name: (StandardDim.station, lon),
            StandardCoord.x.name: (StandardDim.station, x),
            StandardCoord.y.name: (StandardDim.station, y),
            StandardCoord.z.name: (StandardDim.station, z),
        },
        attrs={"data_type": DataType.simulated_forecast_single},
    )


@pytest.fixture
def fews_webservice_auth_config() -> FewsWebserviceAuthConfig:
    """Read authorization config from environment."""
    return FewsWebserviceAuthConfig()


# Test data from Meuse
#   - simulated_forecast_single
#   - simulated_forecast_probabilistic


test_data_meuse_locations = ["H-MS-SINT"]
test_data_meuse_parameters = ["waterlevel", "discharge"]
test_data_meuse_module_instance_ids = {
    DataType.simulated_forecast_single: "fews_riv_ecmwf_hres_sobek3_choozkeiz_bias",
    DataType.simulated_forecast_probabilistic: "fews_riv_ecmwf_ens_sobek3_choozkeiz_ens_dres",
}


@pytest.fixture
def general_info_config_single(cache_dir: Path) -> GeneralInfoConfig:
    """GeneralInfoConfig for a single forecast."""
    return GeneralInfoConfig(
        verification_period=VerificationPeriod(
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
        cache_dir=cache_dir,
    )


@pytest.fixture
def general_info_config_ensemble(cache_dir: Path) -> GeneralInfoConfig:
    """GeneralInfoConfig for an ensemble."""
    return GeneralInfoConfig(
        verification_period=VerificationPeriod(
            start=datetime(2024, 11, 10, tzinfo=timezone.utc),
            end=datetime(2024, 11, 12, tzinfo=timezone.utc),
        ),
        forecast_periods=ForecastPeriods(unit=TimeUnits.DAY, values=[1, 2, 3, 4]),
        verification_pairs=[
            VerificationPair(
                id="pair1",
                obs="observed",
                sim="source_ensemble",
            ),
        ],
        cache_dir=cache_dir,
    )


@pytest.fixture
def general_info_config_probabilistic(cache_dir: Path) -> GeneralInfoConfig:
    """GeneralInfoConfig for probabilistic forecast."""
    return GeneralInfoConfig(
        verification_period=VerificationPeriod(
            start=datetime(2025, 6, 27, tzinfo=timezone.utc),
            end=datetime(2025, 6, 28, tzinfo=timezone.utc),
        ),
        forecast_periods=ForecastPeriods(unit=TimeUnits.DAY, values=[1, 2, 3, 4]),
        verification_pairs=[
            VerificationPair(
                id="pair1",
                obs="observed",
                sim="source_probabilistic",
            ),
        ],
        cache_dir=cache_dir,
    )


@pytest.fixture
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


# Datasource fixtures - FewsWebservice


@pytest.fixture
def fews_webservice_observed_historical(
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
    id_mapping_config_fewsnetcdf: IdMappingConfig,
    general_info_config_ensemble: GeneralInfoConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        import_adapter="fewswebservice",
        source="observed",
        data_type="observed_historical",
        location_ids=["H-RN-0001", "H-RN-0689"],
        parameter_ids=["Q_m"],
        module_instance_id="Hydro_Prep",
        general=general_info_config_ensemble,
        auth_config=fews_webservice_auth_config,
        id_mapping=id_mapping_config_fewsnetcdf,
        webservice_version="2025.01",
    )
    return FewsWebservice(config)


@pytest.fixture
def fews_webservice_simulated_forecast_ensemble_frt(
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
    id_mapping_config_fewsnetcdf: IdMappingConfig,
    general_info_config_ensemble: GeneralInfoConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        import_adapter="fewswebservice",
        source="source_ensemble",
        data_type="simulated_forecast_ensemble",
        location_ids=["H-RN-0001", "H-RN-0689"],
        parameter_ids=["Q_fs"],
        module_instance_id="SBK3_MaxRTK_ECMWF_ENS",
        ensemble_id="ECMWF_ENS",
        archive_kind=ArchiveKind.external_storage_archive,
        forecast_retrieval_method=ForecastRetrievalMethod.retrieve_all_forecast_data,
        general=general_info_config_ensemble,
        auth_config=fews_webservice_auth_config,
        id_mapping=id_mapping_config_fewsnetcdf,
        webservice_version="2025.01",
    )
    return FewsWebservice(config)


@pytest.fixture
def fews_webservice_simulated_forecast_ensemble_fp(
    fews_webservice_simulated_forecast_ensemble_frt: FewsWebservice,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    instance = deepcopy(
        fews_webservice_simulated_forecast_ensemble_frt,
    )
    instance.config.forecast_retrieval_method = (
        ForecastRetrievalMethod.retrieve_forecast_data_per_lead_time
    )
    return instance


@pytest.fixture
def fews_webservice_simulated_forecast_single_frt(
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
    general_info_config_single: GeneralInfoConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        import_adapter="fewswebservice",
        source="source_single",
        data_type=DataType.simulated_forecast_single,
        location_ids=test_data_meuse_locations,
        parameter_ids=test_data_meuse_parameters,
        module_instance_id=test_data_meuse_module_instance_ids[DataType.simulated_forecast_single],
        archive_kind=ArchiveKind.external_storage_archive,
        forecast_retrieval_method=ForecastRetrievalMethod.retrieve_all_forecast_data,
        general=general_info_config_single.model_dump(),
        auth_config=fews_webservice_auth_config,
        webservice_version="2025.01",
    )
    return FewsWebservice(config)


@pytest.fixture
def fews_webservice_simulated_forecast_single_fp(
    fews_webservice_simulated_forecast_single_frt: FewsWebservice,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    instance = deepcopy(
        fews_webservice_simulated_forecast_single_frt,
    )
    instance.config.forecast_retrieval_method = (
        ForecastRetrievalMethod.retrieve_forecast_data_per_lead_time
    )
    return instance


@pytest.fixture
def fews_webservice_simulated_forecast_probabilistic_frt(
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
    general_info_config_probabilistic: GeneralInfoConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        import_adapter="fewswebservice",
        source="source_probabilistic",
        data_type=DataType.simulated_forecast_probabilistic,
        location_ids=test_data_meuse_locations,
        parameter_ids=["discharge"],
        module_instance_id=test_data_meuse_module_instance_ids[
            DataType.simulated_forecast_probabilistic
        ],
        ensemble_id="ensembleQR",
        archive_kind=ArchiveKind.external_storage_archive,
        forecast_retrieval_method=ForecastRetrievalMethod.retrieve_all_forecast_data,
        general=general_info_config_probabilistic.model_dump(),
        auth_config=fews_webservice_auth_config,
        webservice_version="2025.01",
    )
    return FewsWebservice(config)


@pytest.fixture
def fews_webservice_simulated_forecast_probabilistic_fp(
    fews_webservice_simulated_forecast_probabilistic_frt: FewsWebservice,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    instance = deepcopy(
        fews_webservice_simulated_forecast_probabilistic_frt,
    )
    instance.config.forecast_retrieval_method = (
        ForecastRetrievalMethod.retrieve_forecast_data_per_lead_time
    )
    return instance


@pytest.fixture
def fews_webservice_timeseries_headers_only() -> xr.Dataset:
    """Return xarray dataset for FEWS Compliant file."""
    file_path = Path("tests/data/webservice_responses_netcdf/timeseries_headers.json")
    with file_path.open(mode="r", encoding="utf8") as f:
        return json.load(f)


# Datasource fixtures - FewsNetCDF


@pytest.fixture
def fews_netcdf_observed_historical(
    id_mapping_config_fewsnetcdf: IdMappingConfig,
    general_info_config_ensemble: GeneralInfoConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource obs config."""
    return FewsNetCDF.from_config(
        {
            "import_adapter": "fewsnetcdf",
            "data_type": "observed_historical",
            "netcdf_kind": "observation",
            "directory": "tests/data/webservice_responses_netcdf/observations",
            "filename_glob": "*.nc",
            "station_ids": ["H-RN-0001", "H-RN-0689"],
            "source": "observed",
            "general": general_info_config_ensemble.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture
def fews_netcdf_simulated_forecast_ensemble_frt(
    id_mapping_config_fewsnetcdf: IdMappingConfig,
    general_info_config_ensemble: GeneralInfoConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    return FewsNetCDF.from_config(
        {
            "import_adapter": "fewsnetcdf",
            "data_type": "simulated_forecast_ensemble",
            "netcdf_kind": FewsNetCDFKind.simulated_forecast_per_forecast_reference_time,
            "directory": "tests/data/webservice_responses_netcdf/simulations_per_forecast_reference_time/ensemble",  # noqa: E501
            "filename_glob": "*.nc",
            "source": "source_ensemble",
            "general": general_info_config_ensemble.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture
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


@pytest.fixture
def fews_netcdf_simulated_forecast_single_frt(
    id_mapping_config_fewsnetcdf: IdMappingConfig,
    general_info_config_single: GeneralInfoConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    return FewsNetCDF.from_config(
        {
            "import_adapter": "fewsnetcdf",
            "data_type": DataType.simulated_forecast_single,
            "netcdf_kind": FewsNetCDFKind.simulated_forecast_per_forecast_reference_time,
            "directory": "tests/data/webservice_responses_netcdf/simulations_per_forecast_reference_time/single",  # noqa: E501
            "filename_glob": "*.nc",
            "source": "source_single",
            "general": general_info_config_single.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture
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


@pytest.fixture
def fews_netcdf_simulated_forecast_probabilistic_frt(
    general_info_config_probabilistic: GeneralInfoConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    return FewsNetCDF.from_config(
        {
            "import_adapter": "fewsnetcdf",
            "data_type": DataType.simulated_forecast_probabilistic,
            "netcdf_kind": FewsNetCDFKind.simulated_forecast_per_forecast_reference_time,
            "directory": "tests/data/webservice_responses_netcdf/simulations_per_forecast_reference_time/probabilistic",  # noqa: E501
            "filename_glob": "*.nc",
            "source": "source_probabilistic",
            "general": general_info_config_probabilistic.model_dump(),
        },
    )


@pytest.fixture
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


@pytest.fixture
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


# Input dataset fixtures


@pytest.fixture
def input_dataset_dummy_data_forecast_reference_time(
    xarray_data_array_observation: xr.DataArray,
    xarray_dataset_simulations_forecast_reference_time: xr.DataArray,
) -> InputDataset:
    """Initialize datamodel with observations and forecasts (based on frt)."""
    return InputDataset(
        data=[xarray_data_array_observation, xarray_dataset_simulations_forecast_reference_time],
    )


@pytest.fixture
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


# Score fixtures


@pytest.fixture
def score_config_crps(
    general_info_config_ensemble: GeneralInfoConfig,
) -> CrpsForEnsembleConfig:
    """Flexible fixture for scores config, sharing general config."""
    return CrpsForEnsembleConfig(
        score_adapter=ScoreKind.crps_for_ensemble,
        general=general_info_config_ensemble.model_dump(),
    )


@pytest.fixture
def score_config_rank_histogram(
    general_info_config_ensemble: GeneralInfoConfig,
) -> RankHistogramConfig:
    """Flexible fixture for scores config, sharing general config."""
    return RankHistogramConfig(
        score_adapter=ScoreKind.rank_histogram,
        general=general_info_config_ensemble.model_dump(),
    )


@pytest.fixture
def score_config_crps_cdf(
    general_info_config_probabilistic: GeneralInfoConfig,
) -> CrpsCDFConfig:
    """Flexible fixture for scores config, sharing general config."""
    return CrpsCDFConfig(
        score_adapter=ScoreKind.crps_cdf,
        general=general_info_config_probabilistic.model_dump(),
    )


@pytest.fixture
def score_config_continuous(
    general_info_config_single: GeneralInfoConfig,
) -> ContinuousScoresConfig:
    """Flexible fixture for scores config, sharing general config."""
    return ContinuousScoresConfig(
        score_adapter=ScoreKind.continuous_scores,
        general=general_info_config_single.model_dump(),
        scores=["mae", "rmse"],
    )


@pytest.fixture
def score_config_categorical(
    general_info_config_single: GeneralInfoConfig,
) -> ContinuousScoresConfig:
    """Flexible fixture for scores config, sharing general config."""
    return CategoricalScoresConfig(
        score_adapter=ScoreKind.categorical_scores,
        general=general_info_config_single.model_dump(),
        scores=["accuracy", "false_alarm_rate"],
        events=[ThresholdOperator(threshold="warn_2", operator=EventOperator.GREATER_THAN)],
        verification_pair_ids=["pair1"],
        reduce_dims=[StandardDim.forecast_reference_time],
    )


# Datasink fixtures


@pytest.fixture
def datasink_cf_compliant_netcdf(
    tmp_path: Path,
    general_info_config_ensemble: GeneralInfoConfig,
) -> CFCompliantNetCDF:
    """CF Compliant NetCDF datasink."""
    return CFCompliantNetCDF(
        config=CFCompliantNetCDFConfig(
            export_adapter=DataSinkKind.cf_compliant_netcdf,
            directory=str(tmp_path),
            filename="test.nc",
            general=general_info_config_ensemble.model_dump(),
            institution="Deltares",
        ),
    )


@pytest.fixture
def dummy_threshold_df() -> pd.DataFrame:
    """Get dummy thresholds."""
    station_ids = np.array(stations)
    threshold_ids = np.array(thresholds)
    variable_ids = np.array(variables)

    station_idx, threshold_idx, variable_idx = np.meshgrid(
        station_ids,
        threshold_ids,
        variable_ids,
        indexing="ij",
    )
    data = rng.random(size=(station_n, threshold_n, variable_n))

    return pd.DataFrame(
        {
            "station": station_idx.ravel(),
            "variable": variable_idx.ravel(),
            "threshold": threshold_idx.ravel(),
            "value": data.ravel(),
        },
    )


@pytest.fixture
def xarray_thresholds(
    dummy_threshold_df: pd.DataFrame,
    general_info_config_single: GeneralInfoConfig,
    tmp_path: Path,
) -> ThresholdCsv:
    """Get threshold datasource from csv file."""
    file_path = tmp_path / "thresholds.csv"
    dummy_threshold_df.to_csv(file_path, index=False)
    config = ThresholdCsvConfig(
        import_adapter=DataSourceKind.THRESHOLD_CSV,
        data_type=DataType.threshold,
        source="threshold_source",
        general=general_info_config_single,
        directory=file_path.parent,
        filename=file_path.name,
        stations=["station_2"],
        variables=["variable_1"],
        thresholds=["warn_1"],
    )
    return ThresholdCsv(config).fetch_data().data_array
