"""Test the functions in the pipeline module."""

import copy
from pathlib import Path

import pytest
import yaml
from dpyverification import pipeline
from pydantic import ValidationError

from tests import (
    TESTS_CONFIGURATION_FILE,
    TESTS_FORECASTS_2_FILE,
    TESTS_FORECASTS_FILE,
    TESTS_OBSERVATIONS_FILE,
)


def test_execute_pipeline_happy_yaml(tmp_path: Path) -> None:
    """Test at least one valid conf file for each conf type."""
    tmpfile = tmp_path / "tempconf.yaml"
    tmpout = tmp_path / "out.netcdf"

    assert not tmpout.exists()

    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)
        testconf["datasources"][0]["directory"] = str(TESTS_OBSERVATIONS_FILE.parent)
        testconf["datasources"][0]["filename"] = TESTS_OBSERVATIONS_FILE.name
        testconf["datasources"][1]["directory"] = str(TESTS_FORECASTS_FILE.parent)
        testconf["datasources"][1]["filename"] = TESTS_FORECASTS_FILE.name
        testconf["datasources"].append(copy.deepcopy(testconf["datasources"][1]))
        testconf["datasources"][2]["filename"] = TESTS_FORECASTS_2_FILE.name
        testconf["output"][0]["directory"] = str(tmpout.parent)
        testconf["output"][0]["filename"] = tmpout.name
    with tmpfile.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    pipeline.execute_pipeline(tmpfile, conf_type="yaml")

    assert tmpout.exists()


def test_execute_pipeline_happy_runinfo() -> None:
    """Test that runinfo is not implemented yet."""
    with pytest.raises(ValidationError, match="Field required"):
        pipeline.execute_pipeline(Path(), conf_type="runinfo")


def test_execute_pipeline_bad_conf_type() -> None:
    """Check that error on invalid conf type."""
    with pytest.raises(ValueError, match="'1234567890' is not a valid ConfigTypes"):
        pipeline.execute_pipeline(Path(), conf_type="1234567890")
