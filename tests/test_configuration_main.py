"""Test the main module of the dpyverification.configuration package."""

from pathlib import Path

import yaml
from dpyverification.configuration import Config, ConfigSchema, ConfigTypes

from tests import TESTS_CONFIGURATION_FILE


def test_main_yaml_happy() -> None:
    """Check the returned config object has the expected content."""
    config = Config(TESTS_CONFIGURATION_FILE, configtype=ConfigTypes.YAML)

    assert config.filename == TESTS_CONFIGURATION_FILE
    assert config.configtype == ConfigTypes.YAML
    assert config.datasources[0].model_dump() == {  # type: ignore[misc] # model_dump can have Any
        "datasourcetype": "pixml",
        "simobstype": "obs",
        "directory": "iets",
        "filename": "anders",
    }
    assert config.general.verificationperiod.model_dump() == {  # type: ignore[misc] # model_dump can have Any
        "start": {"format": "%Y-%m-%dT%H:%M:%S%z", "value": "2000-01-01T00:00:00Z"},
        "end": {"format": "%Y-%m-%dT%H:%M:%S%z", "value": "2001-01-01T00:00:00Z"},
    }
    assert config.calculations[0].model_dump() == {  # type: ignore[misc] # model_dump can have Any
        "calculationtype": "simobspair",
        "leadtimes": None,
        "variablepairs": [{"obs": "Q.m", "sim": "Q.fs"}],
    }
    assert config.output[0].model_dump() == {  # type: ignore[misc] # model_dump can have Any
        "datasourcetype": "fewsnetcdf",
        "directory": "somewhere",
        "filename": "something",
        "institution": "Deltares",
        "title": None,
    }


def test_schema_jsonable(tmp_path: Path) -> None:
    """Check that the schema for our config is jsonable.

    This so we can be sure it will generate correctly for the documentation of our configuration.
    """
    tmpfile = tmp_path / "configschema.json"
    assert not tmpfile.exists()

    with tmpfile.open(mode="w") as myfile:
        yaml.dump(ConfigSchema.model_json_schema(), myfile)  # type: ignore[misc] # model_json_schema output has Any

    assert tmpfile.exists()

    # When adding documentation, can add the json schema in the doc. Then, also compare the version
    #  in the documentation with the current version as per this test.


# Do we want to test that all schema fields (recursive, and even private ones?) have a description?
