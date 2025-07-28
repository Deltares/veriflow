"""Test the main module of the dpyverification.configuration package."""

from collections.abc import Generator

import pytest
from dpyverification.configuration.utils import FewsWebserviceAuthConfig


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
