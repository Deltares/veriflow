"""Test the pixml module of the dpyverification.datasources package."""

from pathlib import Path

import numpy as np
import yaml
from dpyverification.configuration import Config
from dpyverification.datasources.pixml import PiXmlFile

from tests import TESTS_CONFIGURATION_FILE, TESTS_FORECASTS_FILE, TESTS_OBSERVATIONS_FILE


def test_sim_happy(tmp_path: Path) -> None:
    """Check that the imported pixml gives an xarray with the expected content."""
    # Create an adapted testconfig, based on default testconfig
    # - load default config
    # - adapt config
    # - create config object from adapted config
    # Load:
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)
    # Adapt:
    testconf["datasources"][0]["directory"] = str(TESTS_FORECASTS_FILE.parent)
    testconf["datasources"][0]["filename"] = TESTS_FORECASTS_FILE.name
    testconf["datasources"][0]["simobstype"] = "sim"
    # Create:
    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = Config(tmp_conf_file, "yaml")

    data = PiXmlFile.get_data(conf.content.datasources[0])

    assert (
        data[0]
        .xarray["Q.fs"]
        .loc[  # type: ignore[misc]
            np.datetime64("2024-07-03T05:00"),
            "LOC2",
            20:21,
            np.datetime64("2024-07-02T20:00"),
        ]
        .data.tolist()
        == [468.22, 468.27]
    )


def test_obs_happy(tmp_path: Path) -> None:
    """Check that the imported pixml gives an xarray with the expected content."""
    # Create an adapted testconfig, based on default testconfig
    # - load default config
    # - adapt config
    # - create config object from adapted config
    # Load:
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)
    # Adapt:
    testconf["datasources"][0]["directory"] = str(TESTS_OBSERVATIONS_FILE.parent)
    testconf["datasources"][0]["filename"] = TESTS_OBSERVATIONS_FILE.name
    testconf["datasources"][0]["simobstype"] = "obs"
    # Create:
    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = Config(tmp_conf_file, "yaml")

    data = PiXmlFile.get_data(conf.content.datasources[0])

    assert data[0].xarray["Q.m"].loc[np.datetime64("2024-07-03T05:00"), :].data.tolist() == [  # type: ignore[misc]
        3128.42,
        483.28,
    ]
