"""Test the main module of the dpyverification.configuration package."""

from dpyverification.configuration.main import Config, ConfigTypes

from tests import TESTS_CONFIGURATION_FILE


def test_main_yaml_happy() -> None:
    """Check the returned config object has the expected content."""
    config = Config(TESTS_CONFIGURATION_FILE, configtype=ConfigTypes.YAML)

    assert config.filename == TESTS_CONFIGURATION_FILE
    assert config.configtype == ConfigTypes.YAML
    assert config.datasources[0].model_dump() == {  # type: ignore[misc]
        "datasourcetype": "pixml",
        "simobstype": "obs",
        "directory": "iets",
        "filename": "anders",
    }
