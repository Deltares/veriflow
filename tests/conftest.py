"""Shared resources across the test suite."""

# mypy: ignore-errors

import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from veriflow.configuration import Config
from veriflow.configuration.base import GeneralInfoConfig, IdMappingConfig
from veriflow.configuration.config import SupportedSchemaVersion
from veriflow.configuration.default.datasinks import CFCompliantNetCDFConfig
from veriflow.configuration.default.datasources import (
    ArchiveKind,
    CsvConfig,
    FewsWebserviceAuthConfig,
    FewsWebserviceConfig,
    NetCDFConfig,
)
from veriflow.configuration.default.scores import (
    CategoricalScoresConfig,
    ContinuousScoresConfig,
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    EventOperator,
    RankHistogramConfig,
    ThresholdEvent,
)
from veriflow.configuration.utils import (
    LeadTimes,
    TimeUnits,
    VerificationPair,
    VerificationPeriod,
)
from veriflow.constants import (
    DataSinkKind,
    DataSourceKind,
    DataType,
    ScoreKind,
    SpatialType,
    StandardCoord,
    StandardDim,
)
from veriflow.datasinks.cf_compliant_netcdf import CFCompliantNetCDF
from veriflow.datasources.csv import Csv
from veriflow.datasources.fewsnetcdf import FewsNetCDF, FewsNetCDFKind
from veriflow.datasources.fewswebservice import FewsWebservice, ForecastRetrievalMethod
from veriflow.datasources.netcdf import NetCDF
from veriflow.datatree.datatree import VeriflowDataTree
from veriflow.types import DataSpec

TESTS_DATA_DIR = Path(__file__).parent / "data"


rng = np.random.default_rng(seed=42)

# Settings for dummy xarray data.
#   fp  = lead time
#   frt = forecast reference time
day_multiplier = 10  # Easily scale the verification period
dtype = "float32"

time_start = frt_start = "2025-01-01T00:00"
time_step = fp_step = "h"
time_n = day_multiplier * 24

frt_step = "d"
frt_n = day_multiplier

fp_n = 24 * 5
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

# Start and end of the verification period, based on the forecast reference times and forecast
# periods.
vp_start = pd.to_datetime(frt_start)
vp_end = pd.to_datetime(frt_start) + pd.to_timedelta(time_n, unit=time_step)

lead_times = pd.timedelta_range(0, periods=fp_n, freq=fp_step)
forecast_times = forecast_reference_times.to_numpy()[:, None] + lead_times.to_numpy()[None, :]
stations = [f"station_{n}" for n in range(station_n)]
x = rng.uniform(0, 100, size=station_n)
y = rng.uniform(0, 100, size=station_n)
z = rng.uniform(0, 10, size=station_n)
lat = y
lon = x
realization = np.arange(1, realization_n + 1)
variables = [f"var_{x}" for x in range(variable_n)]
thresholds = [f"warn_{x}" for x in range(threshold_n)]

# Gridded (x/y dimension) coordinates for spatial_type == gridded fixtures.
grid_x = np.linspace(4.0, 7.0, 6)
grid_y = np.linspace(50.0, 52.0, 5)
grid_crs = "EPSG:4326"


@pytest.fixture
def cache_dir_local(tmp_path: Path) -> Path:
    """Pytest cache directory."""
    return str(tmp_path / "sub")


@pytest.fixture
def cache_dir_remote() -> str:
    """Remote cache directory (s3)."""
    return "https://s3.dummy.com/veriflow-cache"


# Before each test - remove the cache directory
@pytest.fixture(autouse=True)
def _ensure_empty_cache_dir_before_each_test(cache_dir_local: str) -> None:
    """Remove the cache directory before each test."""
    cache_dir_local = Path(cache_dir_local)
    if cache_dir_local.exists():
        shutil.rmtree(cache_dir_local)
    cache_dir_local.mkdir(parents=True)


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
def xarray_observed_historical() -> xr.Dataset:
    """Return example observations as a Dataset (one data_var per variable)."""
    coords = {
        StandardCoord.time.name: times,
        StandardDim.station: (StandardDim.station, stations),
        StandardCoord.station.name: (StandardDim.station, stations),
        StandardCoord.lat.name: (StandardDim.station, lat),
        StandardCoord.lon.name: (StandardDim.station, lon),
        StandardCoord.x.name: (StandardDim.station, x),
        StandardCoord.y.name: (StandardDim.station, y),
        StandardCoord.z.name: (StandardDim.station, z),
    }
    data_vars = {}
    for i, v in enumerate(variables):
        arr = rng.random((len(times), len(stations)), dtype=dtype)
        data_vars[v] = xr.DataArray(
            data=arr,
            dims=[StandardDim.time, StandardDim.station],
            attrs={"units": f"dummy_unit_{i}"},
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "data_type": DataType.observed_historical,
            "source_id": DummySource.observation_source,
            "crs": "dummy_crs",
            "spatial_type": SpatialType.point,
        },
    )


@pytest.fixture
def xarray_observed_historical_datasource(
    tmp_path: Path,
    xarray_general_info_config: GeneralInfoConfig,
    xarray_observed_historical: xr.Dataset,
) -> NetCDF:
    """Return example observations."""
    # Write the xarray to a NetCDF file in the temporary directory
    file_path = tmp_path / "observations.nc"
    xarray_observed_historical.to_netcdf(file_path)

    # Initialize the datasource with the path to the NetCDF file
    datasource = NetCDF(
        config=NetCDFConfig(
            general=xarray_general_info_config,
            source_id=xarray_general_info_config.verification_pairs[0].observations_source_id,
            data_type=DataType.observed_historical,
            directory=str(tmp_path),
            filename_glob="observations.nc",
            import_adapter=DataSourceKind.NETCDF,
        ),
    )
    datasource.fetch_data()
    return datasource


@pytest.fixture
def xarray_simulated_forecast_ensemble() -> xr.Dataset:
    """Return example simulations compatible with the internal datatree.

    Uses lead_time as dimension and coordinates.
    """
    coords = {
        StandardCoord.forecast_reference_time.name: forecast_reference_times,
        StandardCoord.lead_time.name: lead_times,
        StandardCoord.realization.name: realization,
        StandardCoord.time.name: (
            (StandardDim.forecast_reference_time, StandardDim.lead_time),
            forecast_times,
        ),
        StandardCoord.station.name: (StandardDim.station, stations),
        StandardCoord.lat.name: (StandardDim.station, lat),
        StandardCoord.lon.name: (StandardDim.station, lon),
        StandardCoord.x.name: (StandardDim.station, x),
        StandardCoord.y.name: (StandardDim.station, y),
        StandardCoord.z.name: (StandardDim.station, z),
    }
    data_vars = {}
    for i, v in enumerate(variables):
        arr = rng.random((station_n, frt_n, fp_n, realization_n), dtype=dtype)
        data_vars[v] = xr.DataArray(
            data=arr,
            dims=[
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
                StandardDim.realization,
            ],
            attrs={"units": f"dummy_unit_{i}"},
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "data_type": DataType.simulated_forecast_ensemble,
            "source_id": DummySource.simulation_ensemble_source,
            "crs": "dummy_crs",
            "spatial_type": SpatialType.point,
        },
    )


@pytest.fixture
def xarray_simulated_forecast_single() -> xr.Dataset:
    """Return example simulations compatible with the internal datatree.

    Uses lead_time as dimension and coordinates.
    """
    coords = {
        StandardCoord.forecast_reference_time.name: forecast_reference_times,
        StandardCoord.lead_time.name: lead_times,
        StandardCoord.time.name: (
            (StandardDim.forecast_reference_time, StandardDim.lead_time),
            forecast_times,
        ),
        StandardCoord.station.name: (StandardDim.station, stations),
        StandardCoord.lat.name: (StandardDim.station, lat),
        StandardCoord.lon.name: (StandardDim.station, lon),
        StandardCoord.x.name: (StandardDim.station, x),
        StandardCoord.y.name: (StandardDim.station, y),
        StandardCoord.z.name: (StandardDim.station, z),
    }
    data_vars = {}
    for i, v in enumerate(variables):
        arr = rng.random((station_n, frt_n, fp_n), dtype=dtype)
        data_vars[v] = xr.DataArray(
            data=arr,
            dims=[
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
            ],
            attrs={"units": f"dummy_unit_{i}"},
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "data_type": DataType.simulated_forecast_single,
            "source_id": DummySource.simulation_single_source,
            "crs": "dummy_crs",
            "spatial_type": SpatialType.point,
        },
    )


@pytest.fixture
def xarray_observed_historical_gridded() -> xr.Dataset:
    """Return example gridded observations (dims: time, y, x)."""
    coords = {
        StandardCoord.time.name: times,
        StandardCoord.x.name: (StandardDim.x, grid_x),
        StandardCoord.y.name: (StandardDim.y, grid_y),
    }
    data_vars = {}
    for i, v in enumerate(variables):
        arr = rng.random((len(times), len(grid_y), len(grid_x)), dtype=dtype)
        data_vars[v] = xr.DataArray(
            data=arr,
            dims=[StandardDim.time, StandardDim.y, StandardDim.x],
            attrs={"units": f"dummy_unit_{i}"},
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "data_type": DataType.observed_historical,
            "spatial_type": SpatialType.gridded,
            "crs": grid_crs,
            "source_id": DummySource.observation_source,
        },
    )


@pytest.fixture
def xarray_simulated_forecast_single_gridded() -> xr.Dataset:
    """Return example gridded single forecast (dims: frt, lead_time, y, x)."""
    coords = {
        StandardCoord.forecast_reference_time.name: forecast_reference_times,
        StandardCoord.lead_time.name: lead_times,
        StandardCoord.time.name: (
            (StandardDim.forecast_reference_time, StandardDim.lead_time),
            forecast_times,
        ),
        StandardCoord.x.name: (StandardDim.x, grid_x),
        StandardCoord.y.name: (StandardDim.y, grid_y),
    }
    data_vars = {}
    for i, v in enumerate(variables):
        arr = rng.random((frt_n, fp_n, len(grid_y), len(grid_x)), dtype=dtype)
        data_vars[v] = xr.DataArray(
            data=arr,
            dims=[
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
                StandardDim.y,
                StandardDim.x,
            ],
            attrs={"units": f"dummy_unit_{i}"},
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "data_type": DataType.simulated_forecast_single,
            "spatial_type": SpatialType.gridded,
            "crs": grid_crs,
            "source_id": DummySource.simulation_single_source,
        },
    )


@pytest.fixture
def xarray_simulated_forecast_ensemble_gridded() -> xr.Dataset:
    """Return example gridded ensemble forecast (dims: frt, lead_time, realization, y, x)."""
    coords = {
        StandardCoord.forecast_reference_time.name: forecast_reference_times,
        StandardCoord.lead_time.name: lead_times,
        StandardCoord.realization.name: realization,
        StandardCoord.time.name: (
            (StandardDim.forecast_reference_time, StandardDim.lead_time),
            forecast_times,
        ),
        StandardCoord.x.name: (StandardDim.x, grid_x),
        StandardCoord.y.name: (StandardDim.y, grid_y),
    }
    data_vars = {}
    for i, v in enumerate(variables):
        arr = rng.random((frt_n, fp_n, realization_n, len(grid_y), len(grid_x)), dtype=dtype)
        data_vars[v] = xr.DataArray(
            data=arr,
            dims=[
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
                StandardDim.realization,
                StandardDim.y,
                StandardDim.x,
            ],
            attrs={"units": f"dummy_unit_{i}"},
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "data_type": DataType.simulated_forecast_ensemble,
            "spatial_type": SpatialType.gridded,
            "crs": grid_crs,
            "source_id": DummySource.simulation_ensemble_source,
        },
    )


@pytest.fixture
def xarray_observed_forecast_single_datasource(
    tmp_path: Path,
    xarray_general_info_config: GeneralInfoConfig,
    xarray_simulated_forecast_single: xr.Dataset,
) -> NetCDF:
    """Return example forecast single datasource."""
    # Write the xarray to a NetCDF file in the temporary directory
    file_path = tmp_path / "forecast_single.nc"
    xarray_simulated_forecast_single.to_netcdf(file_path)

    # Initialize the datasource with the path to the NetCDF file
    datasource = NetCDF(
        config=NetCDFConfig(
            general=xarray_general_info_config,
            source_id=xarray_general_info_config.verification_pairs[0].simulations_source_id,
            data_type=DataType.simulated_forecast_single,
            directory=str(tmp_path),
            filename_glob="forecast_single.nc",
            import_adapter=DataSourceKind.NETCDF,
        ),
    )
    datasource.fetch_data()
    return datasource


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
def fews_general_info_config_single() -> GeneralInfoConfig:
    """GeneralInfoConfig for a single forecast."""
    return GeneralInfoConfig(
        verification_period=VerificationPeriod(
            start=datetime(2025, 9, 1, tzinfo=timezone.utc),
            end=datetime(2025, 9, 4, tzinfo=timezone.utc),
        ),
        lead_times=LeadTimes(unit=TimeUnits.day, values=[1, 2, 3, 4]),
        verification_pairs=[
            VerificationPair(
                id="pair1",
                observations_source_id="observed",
                simulations_source_id="source_single",
                variable="discharge",
            ),
        ],
    )


@pytest.fixture
def xarray_general_info_config() -> GeneralInfoConfig:
    """GeneralInfoConfig for a single forecast."""
    return GeneralInfoConfig(
        verification_period=VerificationPeriod(
            start=vp_start,
            end=vp_end,
        ),
        lead_times=LeadTimes(unit=TimeUnits.day, values=[1, 2, 3, 4]),
        verification_pairs=[
            VerificationPair(
                id="pair1",
                observations_source_id="observation_source",
                simulations_source_id="simulation_ensemble_source",
                variable="var_1",
            ),
        ],
    )


@pytest.fixture
def xarray_general_info_config_historical() -> GeneralInfoConfig:
    """GeneralInfoConfig for a single forecast."""
    return GeneralInfoConfig(
        verification_period=VerificationPeriod(
            start=vp_start,
            end=vp_end,
            dimension=StandardDim.time,
        ),
        verification_pairs=[
            VerificationPair(
                id="pair1",
                observations_source_id="observation_source",
                simulations_source_id="simulation_ensemble_source",
                variable="var_1",
            ),
        ],
    )


@pytest.fixture
def fews_general_info_config_ensemble() -> GeneralInfoConfig:
    """GeneralInfoConfig for an ensemble."""
    return GeneralInfoConfig(
        verification_period=VerificationPeriod(
            start=datetime(2024, 11, 10, tzinfo=timezone.utc),
            end=datetime(2024, 11, 12, tzinfo=timezone.utc),
        ),
        lead_times=LeadTimes(unit=TimeUnits.day, values=[1, 2, 3, 4]),
        verification_pairs=[
            VerificationPair(
                id="pair1",
                observations_source_id="observed",
                simulations_source_id="source_ensemble",
                variable="discharge",
            ),
        ],
    )


@pytest.fixture
def fews_general_info_config_probabilistic() -> GeneralInfoConfig:
    """GeneralInfoConfig for probabilistic forecast."""
    return GeneralInfoConfig(
        verification_period=VerificationPeriod(
            start=datetime(2025, 6, 27, tzinfo=timezone.utc),
            end=datetime(2025, 6, 28, tzinfo=timezone.utc),
        ),
        lead_times=LeadTimes(unit=TimeUnits.day, values=[1, 2, 3, 4]),
        verification_pairs=[
            VerificationPair(
                id="pair1",
                observations_source_id="observed",
                simulations_source_id="source_probabilistic",
                variable="discharge",
            ),
        ],
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
    fews_general_info_config_ensemble: GeneralInfoConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        import_adapter="fewswebservice",
        source_id="observed",
        data_type="observed_historical",
        location_ids=["H-RN-0001", "H-RN-0689"],
        parameter_ids=["Q_m"],
        module_instance_id="Hydro_Prep",
        general=fews_general_info_config_ensemble,
        auth_config=fews_webservice_auth_config,
        id_mapping=id_mapping_config_fewsnetcdf,
        webservice_version="2025.01",
    )
    return FewsWebservice(config)


@pytest.fixture
def fews_webservice_simulated_forecast_ensemble_frt(
    fews_webservice_auth_config: FewsWebserviceAuthConfig,
    id_mapping_config_fewsnetcdf: IdMappingConfig,
    fews_general_info_config_ensemble: GeneralInfoConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        import_adapter="fewswebservice",
        source_id="source_ensemble",
        data_type="simulated_forecast_ensemble",
        location_ids=["H-RN-0001", "H-RN-0689"],
        parameter_ids=["Q_fs"],
        module_instance_id="SBK3_MaxRTK_ECMWF_ENS",
        ensemble_id="ECMWF_ENS",
        archive_kind=ArchiveKind.external_storage_archive,
        forecast_retrieval_method=ForecastRetrievalMethod.retrieve_all_forecast_data,
        general=fews_general_info_config_ensemble,
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
    fews_general_info_config_single: GeneralInfoConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        import_adapter="fewswebservice",
        source_id="source_single",
        data_type=DataType.simulated_forecast_single,
        location_ids=test_data_meuse_locations,
        parameter_ids=test_data_meuse_parameters,
        module_instance_id=test_data_meuse_module_instance_ids[DataType.simulated_forecast_single],
        archive_kind=ArchiveKind.external_storage_archive,
        forecast_retrieval_method=ForecastRetrievalMethod.retrieve_all_forecast_data,
        general=fews_general_info_config_single.model_dump(),
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
    fews_general_info_config_probabilistic: GeneralInfoConfig,
) -> FewsWebservice:
    """Fewsnetcdf datasource sim config."""
    config = FewsWebserviceConfig(
        import_adapter="fewswebservice",
        source_id="source_probabilistic",
        data_type=DataType.simulated_forecast_probabilistic,
        location_ids=test_data_meuse_locations,
        parameter_ids=["discharge"],
        module_instance_id=test_data_meuse_module_instance_ids[
            DataType.simulated_forecast_probabilistic
        ],
        ensemble_id="ensembleQR",
        archive_kind=ArchiveKind.external_storage_archive,
        forecast_retrieval_method=ForecastRetrievalMethod.retrieve_all_forecast_data,
        general=fews_general_info_config_probabilistic.model_dump(),
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
    fews_general_info_config_ensemble: GeneralInfoConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource obs config."""
    return FewsNetCDF.from_config(
        {
            "import_adapter": "fewsnetcdf",
            "data_type": "observed_historical",
            "netcdf_kind": "external_historical",
            "directory": "tests/data/webservice_responses_netcdf/observations",
            "filename_glob": "*.nc",
            "station_ids": ["H-RN-0001", "H-RN-0689"],
            "source_id": "observed",
            "general": fews_general_info_config_ensemble.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture
def fews_netcdf_simulated_historical() -> FewsNetCDF:
    """Fewsnetcdf datasource for simulated_historical."""
    general = GeneralInfoConfig(
        verification_period=VerificationPeriod(
            start=datetime(2024, 10, 28, tzinfo=timezone.utc),
            end=datetime(2024, 10, 30, tzinfo=timezone.utc),
            dimension=StandardDim.time,
        ),
        verification_pairs=[
            VerificationPair(
                id="pair1",
                observations_source_id="observed",
                simulations_source_id="source_ensemble",
                variable="discharge",
            ),
        ],
    )
    return FewsNetCDF.from_config(
        {
            "import_adapter": "fewsnetcdf",
            "data_type": "simulated_historical",
            "netcdf_kind": "simulated_historical",
            "directory": "tests/data/webservice_responses_netcdf/simulated_historical",
            "filename_glob": "*.nc",
            "station_ids": ["T508HMS", "T509HMS"],
            "source_id": "some_simulated_historical_source",
            "general": general.model_dump(),
        },
    )


@pytest.fixture
def fews_netcdf_simulated_forecast_ensemble_frt(
    id_mapping_config_fewsnetcdf: IdMappingConfig,
    fews_general_info_config_ensemble: GeneralInfoConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    return FewsNetCDF.from_config(
        {
            "import_adapter": "fewsnetcdf",
            "data_type": "simulated_forecast_ensemble",
            "netcdf_kind": FewsNetCDFKind.simulated_forecast_per_forecast_reference_time,
            "directory": "tests/data/webservice_responses_netcdf/simulations_per_forecast_reference_time/ensemble",  # noqa: E501
            "filename_glob": "*.nc",
            "source_id": "source_ensemble",
            "general": fews_general_info_config_ensemble.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture
def fews_netcdf_simulated_forecast_ensemble_fp(
    fews_netcdf_simulated_forecast_ensemble_frt: FewsNetCDF,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    instance = deepcopy(fews_netcdf_simulated_forecast_ensemble_frt)
    instance.config.netcdf_kind = FewsNetCDFKind.simulated_forecast_per_lead_time
    instance.config.directory = (
        "tests/data/webservice_responses_netcdf/simulations_per_lead_time/ensemble"
    )
    return instance


@pytest.fixture
def fews_netcdf_simulated_forecast_single_frt(
    id_mapping_config_fewsnetcdf: IdMappingConfig,
    fews_general_info_config_single: GeneralInfoConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    return FewsNetCDF.from_config(
        {
            "import_adapter": "fewsnetcdf",
            "data_type": DataType.simulated_forecast_single,
            "netcdf_kind": FewsNetCDFKind.simulated_forecast_per_forecast_reference_time,
            "directory": "tests/data/webservice_responses_netcdf/simulations_per_forecast_reference_time/single",  # noqa: E501
            "filename_glob": "*.nc",
            "source_id": "source_single",
            "general": fews_general_info_config_single.model_dump(),
            "id_mapping": id_mapping_config_fewsnetcdf.model_dump(),
        },
    )


@pytest.fixture
def fews_netcdf_simulated_forecast_single_fp(
    fews_netcdf_simulated_forecast_single_frt: FewsNetCDF,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    instance = deepcopy(fews_netcdf_simulated_forecast_single_frt)
    instance.config.netcdf_kind = FewsNetCDFKind.simulated_forecast_per_lead_time
    instance.config.directory = (
        "tests/data/webservice_responses_netcdf/simulations_per_lead_time/single"
    )
    return instance


@pytest.fixture
def fews_netcdf_simulated_forecast_probabilistic_frt(
    fews_general_info_config_probabilistic: GeneralInfoConfig,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    return FewsNetCDF.from_config(
        {
            "import_adapter": "fewsnetcdf",
            "data_type": DataType.simulated_forecast_probabilistic,
            "netcdf_kind": FewsNetCDFKind.simulated_forecast_per_forecast_reference_time,
            "directory": "tests/data/webservice_responses_netcdf/simulations_per_forecast_reference_time/probabilistic",  # noqa: E501
            "filename_glob": "*.nc",
            "source_id": "source_probabilistic",
            "general": fews_general_info_config_probabilistic.model_dump(),
        },
    )


@pytest.fixture
def fews_netcdf_simulated_forecast_probabilistic_fp(
    fews_netcdf_simulated_forecast_probabilistic_frt: FewsNetCDF,
) -> FewsNetCDF:
    """Fewsnetcdf datasource sim config."""
    instance = deepcopy(fews_netcdf_simulated_forecast_probabilistic_frt)
    instance.config.netcdf_kind = FewsNetCDFKind.simulated_forecast_per_lead_time
    instance.config.directory = (
        "tests/data/webservice_responses_netcdf/simulations_per_lead_time/probabilistic"
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
def input_data_datatree(
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> VeriflowDataTree:
    """Initialize a datatree with observations and forecasts (based on frt) as input data."""
    dt = cast("VeriflowDataTree", xr.DataTree(name="veriflow_output"))
    dt.veriflow.add_input_data([xarray_observed_historical, xarray_simulated_forecast_ensemble])
    return dt


# Score fixtures


@pytest.fixture
def score_config_crps(
    fews_general_info_config_ensemble: GeneralInfoConfig,
) -> CrpsForEnsembleConfig:
    """Flexible fixture for scores config, sharing general config."""
    return CrpsForEnsembleConfig(
        score_adapter=ScoreKind.crps_for_ensemble,
        general=fews_general_info_config_ensemble.model_dump(),
    )


@pytest.fixture
def score_config_rank_histogram(
    fews_general_info_config_ensemble: GeneralInfoConfig,
) -> RankHistogramConfig:
    """Flexible fixture for scores config, sharing general config."""
    return RankHistogramConfig(
        score_adapter=ScoreKind.rank_histogram,
        general=fews_general_info_config_ensemble.model_dump(),
        reduce_dims=[StandardDim.forecast_reference_time],
    )


@pytest.fixture
def score_config_crps_cdf(
    fews_general_info_config_probabilistic: GeneralInfoConfig,
) -> CrpsCDFConfig:
    """Flexible fixture for scores config, sharing general config."""
    return CrpsCDFConfig(
        score_adapter=ScoreKind.crps_cdf,
        general=fews_general_info_config_probabilistic.model_dump(),
    )


@pytest.fixture
def score_config_continuous(
    fews_general_info_config_single: GeneralInfoConfig,
) -> ContinuousScoresConfig:
    """Flexible fixture for scores config, sharing general config."""
    return ContinuousScoresConfig(
        score_adapter=ScoreKind.continuous_scores,
        general=fews_general_info_config_single.model_dump(),
        scores=["mae", "rmse", "nse", "kge"],
        reduce_dims=[StandardDim.forecast_reference_time],
    )


@pytest.fixture
def score_config_categorical(
    fews_general_info_config_single: GeneralInfoConfig,
) -> ContinuousScoresConfig:
    """Flexible fixture for scores config, sharing general config."""
    return CategoricalScoresConfig(
        score_adapter=ScoreKind.categorical_scores,
        general=fews_general_info_config_single.model_dump(),
        scores=["accuracy", "false_alarm_rate"],
        events=[ThresholdEvent(threshold="warn_2", operator=EventOperator.GREATER_THAN)],
        verification_pair_ids=["pair1"],
        reduce_dims=[StandardDim.forecast_reference_time],
    )


# ----------------------------------------------------------------------------
# Fake datasource & cache fixtures (for cache tests)
# ----------------------------------------------------------------------------

from collections.abc import Callable, Iterator  # noqa: E402
from copy import deepcopy as _deepcopy  # noqa: E402
from typing import ClassVar  # noqa: E402

from veriflow.cache.config import ReadWriteMode, ZarrCacheConfig  # noqa: E402
from veriflow.configuration.base import BaseDatasourceConfig  # noqa: E402
from veriflow.datasources.base import BaseDatasource  # noqa: E402

# Module-level registry mapping (source, data_type) → seed dataset.
# Allows from_config(model_dump()) to find the right seed without
# embedding xarray data in pydantic config.
_FAKE_SEEDS: dict[tuple[str, str], xr.Dataset] = {}


class FakeDatasourceConfig(BaseDatasourceConfig):
    """Pydantic config for the in-test FakeDatasource."""

    import_adapter: str = "fake"
    stations: list[str]
    variables: list[str]


class FakeDatasource(BaseDatasource):
    """In-test datasource that returns slices of a pre-registered seed dataset.

    The seed dataset is looked up via the module-level ``_FAKE_SEEDS`` registry
    using ``(source, data_type)``. ``fetch_data`` slices on
    variables/stations/verification-period/lead_times based on ``self.config``.
    """

    kind: str = "fake"
    config_class = FakeDatasourceConfig
    supported_data_specs: ClassVar[set[DataSpec]] = {
        (DataType.observed_historical, SpatialType.point),
        (DataType.simulated_forecast_single, SpatialType.point),
        (DataType.simulated_forecast_ensemble, SpatialType.point),
    }

    @property
    def configured_stations(self) -> set[str] | None:
        """Return the internal station identifiers configured for this datasource."""
        return set(self.config.stations)

    @configured_stations.setter
    def configured_stations(self, stations: set[str]) -> None:
        """Set the internal station identifiers configured for this datasource."""
        self.config.stations = list(stations)

    @property
    def configured_variables(self) -> set[str] | None:
        """Return the internal variable identifiers configured for this datasource."""
        return set(self.config.variables)

    @configured_variables.setter
    def configured_variables(self, variables: set[str]) -> None:
        """Set the internal variable identifiers configured for this datasource."""
        self.config.variables = list(variables)

    def fetch_data(self) -> "FakeDatasource":
        """Slice the registered seed dataset and store the result on ``self.dataset``."""
        seed = _FAKE_SEEDS[(str(self.config.source_id), str(self.config.data_type))]
        ds = seed
        # variables
        ds = ds[list(self.config.variables)]
        # stations
        if StandardDim.station in ds.dims:
            ds = ds.sel({StandardDim.station: list(self.config.stations)})
        # frt period (forecast)
        if StandardDim.forecast_reference_time in ds.dims:
            vp = self.config.verification_period_on_frt
            ds = ds.sel(
                {
                    StandardDim.forecast_reference_time: slice(
                        np.datetime64(vp.start),
                        np.datetime64(vp.end),
                    ),
                },
            )
            if self.config.lead_times is not None:
                # Filter to requested lead times present in the seed
                requested = list(self.config.lead_times.timedelta64)
                seed_lt = ds[StandardDim.lead_time].to_numpy()
                keep = [lt for lt in requested if lt in seed_lt]
                ds = ds.sel({StandardDim.lead_time: keep})
        # historical time slice
        if StandardDim.time in ds.dims and StandardDim.forecast_reference_time not in ds.dims:
            vp = self.config.verification_period_on_time
            ds = ds.sel(
                {StandardDim.time: slice(np.datetime64(vp.start), np.datetime64(vp.end))},
            )

        ds = ds.copy()
        ds.attrs["data_type"] = self.config.data_type
        ds.attrs["source_id"] = self.config.source_id
        self.dataset = ds
        return self


def register_fake_seed(source: str, data_type: DataType, dataset: xr.Dataset) -> None:
    """Register a seed dataset for FakeDatasource lookup."""
    _FAKE_SEEDS[(str(source), str(data_type))] = dataset


@pytest.fixture
def fake_seed_registry() -> Iterator[Callable[[str, DataType, xr.Dataset], None]]:
    """Yield a registration callable; clear ``_FAKE_SEEDS`` before and after the test."""
    _FAKE_SEEDS.clear()
    yield register_fake_seed
    _FAKE_SEEDS.clear()


@pytest.fixture
def fake_fetch_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Record every ``fetch_data`` call's config (deep-copied).

    Returns a list of dicts: ``[{"variables": [...], "stations": [...], ...}]``.
    """
    calls: list[dict[str, object]] = []
    original = FakeDatasource.fetch_data

    def spy(self: FakeDatasource) -> "FakeDatasource":
        calls.append(
            {
                "variables": list(self.config.variables),
                "stations": list(self.config.stations),
                "vp_start": self.config.verification_period.start,
                "vp_end": self.config.verification_period.end,
                "lead_times": (
                    list(self.config.lead_times.values) if self.config.lead_times else None
                ),
                "config": _deepcopy(self.config),
            },
        )
        return original(self)

    monkeypatch.setattr(FakeDatasource, "fetch_data", spy)
    return calls


@pytest.fixture
def cache_zarr_config(cache_dir_local: str) -> ZarrCacheConfig:
    """Return a writable local-disk ZarrCacheConfig under ``cache_dir``."""
    return ZarrCacheConfig(
        path=str(Path(cache_dir_local) / "veriflow-cache.zarr"),
        read_write_mode=ReadWriteMode.read_write,
    )


@pytest.fixture
def cache_zarr_config_readonly(cache_dir_local: str) -> ZarrCacheConfig:
    """Return a read-only ZarrCacheConfig under ``cache_dir``."""
    return ZarrCacheConfig(
        path=str(Path(cache_dir_local) / "veriflow-cache.zarr"),
        read_write_mode=ReadWriteMode.read,
    )


@pytest.fixture
def general_info_config_with_cache(
    xarray_general_info_config: GeneralInfoConfig,
    cache_zarr_config: ZarrCacheConfig,
) -> GeneralInfoConfig:
    """Forecast general config with a writable cache.

    Uses a tighter verification period than ``xarray_general_info_config`` so that
    ``verification_period_on_frt`` is strictly inside the seed dataset's frt range
    [2025-01-01..2025-01-10]. This is required for cache hit/partial-hit tests:
    requested frt range must be ⊆ cached frt range.
    """
    cfg = xarray_general_info_config.model_copy(deep=True)
    # Default vp.dimension="forecast_reference_time": vp_on_frt = vp directly.
    # Seed has frts at midnight in [2025-01-01..2025-01-10]. Use vp = (01-05..01-10)
    # so cached frts after first fetch = vp = requested vp_on_frt on second call.
    cfg.verification_period = VerificationPeriod(
        start=pd.Timestamp("2025-01-05"),
        end=pd.Timestamp("2025-01-10"),
    )
    cfg.cache = cache_zarr_config
    return cfg


@pytest.fixture
def general_info_config_historical_with_cache(
    xarray_general_info_config_historical: GeneralInfoConfig,
    cache_zarr_config: ZarrCacheConfig,
) -> GeneralInfoConfig:
    """Historical general config with a writable cache."""
    cfg = xarray_general_info_config_historical.model_copy(deep=True)
    cfg.cache = cache_zarr_config
    return cfg


# Datasink fixtures


@pytest.fixture
def datasink_cf_compliant_netcdf(
    tmp_path: Path,
    fews_general_info_config_ensemble: GeneralInfoConfig,
) -> CFCompliantNetCDF:
    """CF Compliant NetCDF datasink."""
    return CFCompliantNetCDF(
        config=CFCompliantNetCDFConfig(
            export_adapter=DataSinkKind.cf_compliant_netcdf,
            directory=str(tmp_path),
            filename="test.nc",
            general=fews_general_info_config_ensemble.model_dump(),
            institution="Deltares",
        ),
    )


@pytest.fixture
def dummy_threshold_df() -> pd.DataFrame:
    """Get dummy thresholds."""
    # Use a local RNG so threshold test data is independent of global RNG state.
    local_rng = np.random.default_rng(seed=42)
    station_ids = np.array(stations)
    threshold_ids = np.array(thresholds)
    variable_ids = np.array(variables)

    station_idx, threshold_idx, variable_idx = np.meshgrid(
        station_ids,
        threshold_ids,
        variable_ids,
        indexing="ij",
    )
    data = local_rng.random(size=(station_n, threshold_n, variable_n))

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
    fews_general_info_config_single: GeneralInfoConfig,
    tmp_path: Path,
) -> Csv:
    """Get threshold datasource from csv file."""
    file_path = tmp_path / "thresholds.csv"
    dummy_threshold_df.to_csv(file_path, index=False)
    config = CsvConfig(
        import_adapter=DataSourceKind.CSV,
        data_type=DataType.threshold,
        source_id="threshold_source",
        general=fews_general_info_config_single,
        directory=file_path.parent,
        filename=file_path.name,
        stations=["station_2"],
        variables=["var_1"],
        thresholds=["warn_1", "warn_2"],
    )
    instance = Csv(config)
    instance.fetch_data()
    return instance


@pytest.fixture
def cli_dummy_pipeline_config_yaml(tmp_path: Path) -> Path:
    """dummy_pipeline_config_yaml based on an in-memory dummy Config object."""
    general = GeneralInfoConfig(
        verification_period=VerificationPeriod(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        lead_times=LeadTimes(unit=TimeUnits.day, values=[1, 2]),
        verification_pairs=[
            VerificationPair(
                id="pair1",
                observations_source_id="observed",
                simulations_source_id="simulated",
                variable="variable_1",
            ),
        ],
    )

    datasource_config = CsvConfig(
        import_adapter=DataSourceKind.CSV,
        source_id="threshold_source",
        data_type=DataType.threshold,
        general=general,
        directory=tmp_path,
        filename="thresholds.csv",
        stations=["station_1"],
        variables=["variable_1"],
        thresholds=["warn_1"],
    )

    score_config = CrpsForEnsembleConfig(
        score_adapter=ScoreKind.crps_for_ensemble,
        general=general,
        method="ecdf",
    )

    datasink_config = CFCompliantNetCDFConfig(
        export_adapter=DataSinkKind.cf_compliant_netcdf,
        directory=tmp_path,
        filename="results.nc",
        institution="Dummy Institution",
        general=general,
    )

    config_obj = Config(
        version=SupportedSchemaVersion.V0,
        general=general,
        datasources=[datasource_config],
        scores=[score_config],
        datasinks=[datasink_config],
    )

    data = config_obj.model_dump(mode="json", serialize_as_any=True)
    destination = tmp_path / "config.yaml"
    destination.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return destination


# ----------------------------------------------------------------------------
# Fake output dataset fixtures (for output dataset and datasinks tests)
# ----------------------------------------------------------------------------


@pytest.fixture
def xarray_fake_score_result() -> xr.DataArray:
    """Fixture for a fake score result."""
    return xr.DataArray(
        data=[1, 2, 3],
        dims=["station"],
        coords={"station": ["station_1", "station_2", "station_3"]},
        name="fake_score",
    )


@pytest.fixture
def output_datatree_without_scores(
    input_data_datatree: VeriflowDataTree,
) -> VeriflowDataTree:
    """Fixture for an OutputDataset instance."""
    # Initialize the output dataset
    output_dataset = cast("VeriflowDataTree", xr.DataTree(name="veriflow_output"))
    verification_pair = VerificationPair(
        observations_source_id="observation_source",
        simulations_source_id="simulation_ensemble_source",
        id="test_pair",
        variable="var_0",
    )
    obs, sim = input_data_datatree.veriflow.get_pair(verification_pair)
    output_dataset.veriflow.add_staged_input_data(
        verification_pair=verification_pair,
        obs=obs,
        sim=sim,
    )
    return output_dataset


@pytest.fixture
def fake_verification_pair() -> VerificationPair:
    """Fixture for a fake verification pair."""
    return VerificationPair(
        observations_source_id="observation_source",
        simulations_source_id="simulation_ensemble_source",
        id="test_pair",
        variable="var_0",
    )


@pytest.fixture
def output_datatree_with_scores(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
) -> VeriflowDataTree:
    """Fixture for an OutputDataset instance."""
    output_datatree_without_scores.veriflow.add_score(
        verification_pair=fake_verification_pair,
        result=xr.DataArray(
            data=[1, 2, 3],
            dims=["station"],
            coords={"station": ["station_1", "station_2", "station_3"]},
            name="fake_score",
        ),
        name="fake_score",
    )
    return output_datatree_without_scores


@pytest.fixture
def output_datatree_with_multiple_pairs(
    input_data_datatree: VeriflowDataTree,
) -> VeriflowDataTree:
    """Fixture for an OutputDataset instance with two verification pairs."""
    output_dataset = cast("VeriflowDataTree", xr.DataTree(name="veriflow_output"))
    for pair_id in ("test_pair_1", "test_pair_2"):
        verification_pair = VerificationPair(
            observations_source_id="observation_source",
            simulations_source_id="simulation_ensemble_source",
            id=pair_id,
            variable="var_0",
        )
        obs, sim = input_data_datatree.veriflow.get_pair(verification_pair)
        output_dataset.veriflow.add_staged_input_data(
            verification_pair=verification_pair,
            obs=obs,
            sim=sim,
        )
    return output_dataset
