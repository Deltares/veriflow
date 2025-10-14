"""Test the main module of the dpyverification.configuration package."""

from collections.abc import Generator
from copy import deepcopy
from pathlib import Path

import pytest
import xarray as xr
from dpyverification.configuration import Config
from dpyverification.configuration.base import IdMap, IdMappingConfig
from dpyverification.configuration.default.scores import CrpsForEnsembleConfig
from dpyverification.configuration.utils import (
    FewsWebserviceAuthConfig,
    ForecastPeriods,
    Range,
    TimeUnits,
)


@pytest.fixture()
def _mock_env(monkeypatch: Generator[pytest.MonkeyPatch, None, None]) -> None:
    """Create a mock environment for testing secret env vars."""
    monkeypatch.setenv("FEWSWEBSERVICE_URL", "https://fixture_url.test")  # type: ignore  # noqa: PGH003
    monkeypatch.setenv("FEWSWEBSERVICE_USERNAME", "fixture_user")  # type: ignore  # noqa: PGH003
    monkeypatch.setenv("FEWSWEBSERVICE_PASSWORD", "fixture_pass")  # type: ignore  # noqa: PGH003


@pytest.mark.usefixtures("_mock_env")
def test_auth_config_from_fixture() -> None:
    """Test authorization configuration."""
    config = FewsWebserviceAuthConfig()
    assert str(config.url) == "https://fixture_url.test/"
    assert config.username.get_secret_value() == "fixture_user"
    assert config.password.get_secret_value() == "fixture_pass"


def test_schema_jsonable(tmp_path: Path) -> None:
    """Check that the schema for our config is jsonable.

    This so we can be sure it will generate correctly for the documentation of our configuration.
    """
    tmpfile = tmp_path / "config.json"

    Config.write_yaml_schema(tmpfile)

    assert tmpfile.exists()

    # TODO(AU): Additional tests on the configuration schema # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/37
    #   When adding documentation, can add the json schema in the doc. Then, also compare the
    #   version in the documentation with the current version as per this test.


def test_forecast_period_config() -> None:
    """Check forecast periods config identical when using list or range."""
    list_instance = ForecastPeriods(unit=TimeUnits.HOUR, values=[1, 2, 3])
    range_instance = ForecastPeriods(
        unit=TimeUnits.HOUR,
        values=Range(start=1, end=3, step=1).to_list(),
    )
    assert list_instance == range_instance
    assert list_instance.timedelta64 == range_instance.timedelta64
    assert list_instance.stdlib_timedelta == range_instance.stdlib_timedelta
    assert list_instance.max == range_instance.max
    assert list_instance.min == range_instance.min


def test_single_id_map_get_mapping() -> None:
    """Test id mapping get_mapping method."""
    config = IdMap({"intId1": {"sourceA": "extId1"}})
    assert config.get_external_to_internal_mapping("sourceA") == {"extId1": "intId1"}


def test_id_mapping_rename_dataset(xarray_observed_historical: xr.DataArray) -> None:
    """Test partial renaming of stations on dummy dataset."""
    config = IdMappingConfig(
        station=IdMap({"newstation1": {"observation_source": "station0"}}),
    )
    new_da = config.rename_data_array(xarray_observed_historical)
    assert next(iter(new_da.station.to_numpy())) == "newstation1"  # type:ignore[misc]


def test_id_mapping_rename_dataset_fails_on_invalid_source(
    xarray_observed_historical: xr.DataArray,
) -> None:
    """Test partial renaming of stations on dummy dataset."""
    config = IdMappingConfig(
        station=IdMap({"newstation1": {"invalid_observation_source": "station0"}}),
    )
    with pytest.raises(ValueError, match="No IdMapping found for source"):
        config.rename_data_array(xarray_observed_historical)


def test_score_config_with_invalid_pair_reference(
    score_config_crps: CrpsForEnsembleConfig,
) -> None:
    """Test CRPS."""
    modified_config = deepcopy(score_config_crps.model_dump())  # type:ignore[misc]
    modified_config["filter_verification_pairs"] = ["invalid_id"]  # type:ignore[misc]
    with pytest.raises(ValueError, match="Pair id"):
        _ = CrpsForEnsembleConfig(**modified_config)  # type:ignore[misc]
