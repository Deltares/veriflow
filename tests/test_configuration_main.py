"""Test the main module of the dpyverification.configuration package."""

from pathlib import Path

import yaml
from dpyverification.configuration import Config, ConfigFile, ConfigTypes

from tests import TESTS_CONFIGURATION_FILE


def test_main_yaml_happy() -> None:
    """Check the returned config object has the expected content."""
    config = ConfigFile(TESTS_CONFIGURATION_FILE, configtype=ConfigTypes.YAML)

    assert config.filename == TESTS_CONFIGURATION_FILE
    assert config.configtype == ConfigTypes.YAML
    assert list(config.content.datasources[0].model_dump().keys()) == [  # type: ignore[misc] # model_dump can have Any
        "kind",
        "simobstype",
        "general",
        "directory",
        "filename",
    ]
    assert (
        config.content.general.verificationperiod.model_dump_json()
        == '{"start":"2000-01-01T00:00:00Z","end":"2001-01-01T00:00:00Z"}'
    )

    assert list(config.content.scores[0].model_dump().keys()) == [  # type: ignore[misc] # model_dump can have Any
        "kind",
        "general",
        "variablepairs",
    ]
    assert config.content.datasinks[0].model_dump() == {  # type: ignore[misc] # model_dump can have Any
        "kind": "fewsnetcdf",
        "directory": "somewhere",
        "filename": "something",
    }


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


# TODO(AU): Additional tests on the configuration schema # noqa: FIX002
#   https://github.com/Deltares-research/DPyVerification/issues/37
#   Do we want to test that all schema fields (recursive, and even private ones?) have a
#   description?
