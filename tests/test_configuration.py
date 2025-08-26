"""Test the main module of the dpyverification.configuration package."""

from collections.abc import Generator
from pathlib import Path

import pytest
import yaml
from dpyverification.configuration import Config
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
    assert not tmpfile.exists()

    with tmpfile.open(mode="w") as myfile:
        yaml.dump(Config.model_json_schema(), myfile)  # type: ignore[misc] # model_json_schema output has Any

    assert tmpfile.exists()

    # TODO(AU): Additional tests on the configuration schema # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/37
    #   When adding documentation, can add the json schema in the doc. Then, also compare the
    #   version in the documentation with the current version as per this test.


def test_forecast_period_config() -> None:
    """Check forecast periods config identical when using list or range."""
    list_instance = ForecastPeriods(unit=TimeUnits.HOUR, values=[1, 2, 3])
    range_instance = ForecastPeriods(unit=TimeUnits.HOUR, values=Range(start=1, end=3, step=1))
    assert list_instance == range_instance
    assert list_instance.timedelta64 == range_instance.timedelta64
    assert list_instance.py_timedelta == range_instance.py_timedelta
    assert list_instance.max == range_instance.max
    assert list_instance.min == range_instance.min
