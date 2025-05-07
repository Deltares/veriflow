"""Test the functions in the pipeline module."""

import copy
from pathlib import Path

import pytest
import xarray as xr
import yaml
from dpyverification import pipeline
from dpyverification.constants import ScoreKind

from tests import (
    TEST_DIR_FEWS_NETCDF_OBS,
    TEST_DIR_FEWS_NETCDF_SIM,
    TESTS_CONFIGURATION_FILE,
    TESTS_FORECASTS_2_FILE,
    TESTS_FORECASTS_FILE,
    TESTS_OBSERVATIONS_FILE,
)


def get_specific_score_config(scorekind: str) -> dict:
    """Get specific config added during test."""
    variablepairs = [{"sim": "Q_fs", "obs": "Q_m"}]

    if scorekind == ScoreKind.CRPSFORENSEMBLE:
        return {"variablepairs": variablepairs, "preserve_dims": ["time", "leadtime"]}  # type: ignore[misc]
    if scorekind == ScoreKind.RANKHISTOGRAM:
        return {"variablepairs": variablepairs}  # type: ignore[misc]
    if scorekind == ScoreKind.SIMOBSPAIRS:
        return {"variablepairs": variablepairs}  # type: ignore[misc]
    return {}  # type: ignore[misc]


def test_execute_pipeline_happy_yaml(tmp_path: Path) -> None:
    """Test at least one valid conf file for each conf type."""
    tmpout = tmp_path / "out.netcdf"
    assert not tmpout.exists()

    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)

    # Create an adapted testconfig, based on default testconfig, and write to temporary file
    testconf["datasources"][0]["directory"] = str(TESTS_OBSERVATIONS_FILE.parent)
    testconf["datasources"][0]["filename"] = TESTS_OBSERVATIONS_FILE.name
    testconf["datasources"][1]["directory"] = str(TESTS_FORECASTS_FILE.parent)
    testconf["datasources"][1]["filename"] = TESTS_FORECASTS_FILE.name
    testconf["datasources"].append(copy.deepcopy(testconf["datasources"][1]))
    testconf["datasources"][2]["filename"] = TESTS_FORECASTS_2_FILE.name
    testconf["datasinks"][0]["directory"] = str(tmpout.parent)
    testconf["datasinks"][0]["filename"] = tmpout.name
    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    pipeline.execute_pipeline(tmp_conf_file, configtype="yaml")

    assert tmpout.exists()


def test_execute_pipeline_bad_conf_type() -> None:
    """Check that error on invalid conf type."""
    with pytest.raises(ValueError, match="'1234567890' is not a valid ConfigTypes"):
        pipeline.execute_pipeline(Path(), configtype="1234567890")


@pytest.mark.parametrize(
    "score_kind",
    [ScoreKind.SIMOBSPAIRS, ScoreKind.CRPSFORENSEMBLE, ScoreKind.RANKHISTOGRAM],
)
def test_execute_pipeline_ext_storage(
    tmp_path: Path,
    score_kind: str,
) -> None:
    """Test at least one valid conf file for each conf type."""
    tmpout = tmp_path / "out.netcdf"
    assert not tmpout.exists()

    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)

    obs_file = next(iter(TEST_DIR_FEWS_NETCDF_OBS.rglob("*.nc")))
    # Create an adapted testconfig, based on default testconfig, and write to temporary file
    testconf["datasources"][0]["directory"] = str(obs_file.parent)
    testconf["datasources"][0]["filename"] = obs_file.name
    testconf["datasources"][0]["kind"] = "fewsnetcdf"

    # Delete the second pre-configured datasource
    del testconf["datasources"][1]

    # Create datasources for all forecasts files
    for file in TEST_DIR_FEWS_NETCDF_SIM.rglob("*.nc"):
        testconf["datasources"].append(
            {
                "directory": str(file.parent),
                "filename": file.name,
                "kind": "fewsnetcdf",
                "simobstype": "sim",
            },
        )
    # Scores
    testconf["scores"][0].pop("variablepairs")
    testconf["scores"][0]["kind"] = str(score_kind)
    testconf["scores"][0].update(get_specific_score_config(scorekind=score_kind))  # type: ignore[misc]

    # Sinks
    testconf["datasinks"][0]["directory"] = str(tmpout.parent)
    testconf["datasinks"][0]["filename"] = tmpout.name
    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)

    pipeline.execute_pipeline(tmp_conf_file, configtype="yaml")
    assert tmpout.exists()

    ds = xr.open_dataset(tmpout)
    # Check each variable name contains the score kind
    assert all(str(score_kind) in var for var in ds.data_vars)  # type: ignore[operator]
