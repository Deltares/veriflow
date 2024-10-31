"""Test the fewswebservice module of the dpyverification.datasources package."""

import os
from pathlib import Path

import pytest
import requests
import yaml
from dpyverification.configuration import Config
from dpyverification.datasources.fewswebservice import FewsWebService

from tests import TESTS_CONFIGURATION_FILE

SIM_TIME_DIM_LENGTH = 49
OBS_TIME_DIM_LENGTH = 49
VALID_RESPONSE_CODE = 200

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_webservice_live() -> None:
    """Test that a webservice is live and can find filters."""
    url = "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
    endpoint = "filters"
    test_endpoint_url = url + "/" + endpoint
    response = requests.get(test_endpoint_url, timeout=10)
    assert response.status_code == VALID_RESPONSE_CODE


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_get_timeseries_sim_happy(tmp_path: Path) -> None:
    """Check that the imported pixml gives an xarray with the expected content."""
    leadtimes = {"unit": "h", "values": [3, 6]}
    verificationperiod = {
        "start": {"value": "2024-06-01T00:00:00Z"},
        "end": {"value": "2024-06-03T00:00:00Z"},
    }

    # Create an adapted testconfig, based on default testconfig
    # - load default config
    # - adapt config
    # - create config object from adapted config
    # Load:
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)
    # Adapt:
    testconf["general"]["verificationperiod"] = verificationperiod  # type: ignore[call-overload] # Indeed this assignment does not match with our faked type def of testconf
    testconf["datasources"][0]["simobstype"] = "sim"
    testconf["datasources"][0]["datasourcetype"] = "fewswebservice"
    testconf["datasources"][0]["url"] = (
        "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
    )
    testconf["datasources"][0]["location_ids"] = ["H-RN-0001"]  # type: ignore[assignment] # Indeed this assignment does not match with our faked type def of testconf
    testconf["datasources"][0]["parameter_ids"] = ["Q.fs"]  # type: ignore[assignment] # Indeed this assignment does not match with our faked type def of testconf
    testconf["datasources"][0]["module_instance_ids"] = ["SBK3_MaxRTK_ECMWF_ENS"]  # type: ignore[assignment] # Indeed this assignment does not match with our faked type def of testconf
    testconf["datasources"][0]["qualifier_ids"] = []  # type: ignore[assignment] # Indeed this assignment does not match with our faked type def of testconf
    testconf["datasources"][0]["document_format"] = "PI_XML"
    testconf["datasources"][0]["document_version"] = "1.32"
    testconf["datasources"][0]["leadtimes"] = leadtimes  # type: ignore[assignment] # Indeed this assignment does not match with our faked type def of testconf
    # Create:
    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = Config(tmp_conf_file, "yaml")

    data = FewsWebService.get_data(conf.content.datasources[0])

    assert len(data[0].xarray.time) == SIM_TIME_DIM_LENGTH  # type: ignore[misc]


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_get_timeseries_obs_happy(tmp_path: Path) -> None:
    """Check that the imported pixml gives an xarray with the expected content."""
    verificationperiod = {
        "start": {"value": "2024-06-01T00:00:00Z"},
        "end": {"value": "2024-06-03T00:00:00Z"},
    }

    # Create an adapted testconfig, based on default testconfig
    # - load default config
    # - adapt config
    # - create config object from adapted config
    # Load:
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)
    # Adapt:
    testconf["general"]["verificationperiod"] = verificationperiod  # type: ignore[call-overload] # Indeed this assignment does not match with our faked type def of testconf
    testconf["datasources"][0]["simobstype"] = "obs"
    testconf["datasources"][0]["datasourcetype"] = "fewswebservice"
    testconf["datasources"][0]["url"] = (
        "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
    )
    testconf["datasources"][0]["location_ids"] = ["H-RN-0001"]  # type: ignore[assignment] # Indeed this assignment does not match with our faked type def of testconf
    testconf["datasources"][0]["parameter_ids"] = ["Q.m"]  # type: ignore[assignment] # Indeed this assignment does not match with our faked type def of testconf
    testconf["datasources"][0]["module_instance_ids"] = ["Import_LMW"]  # type: ignore[assignment] # Indeed this assignment does not match with our faked type def of testconf
    testconf["datasources"][0]["qualifier_ids"] = []  # type: ignore[assignment] # Indeed this assignment does not match with our faked type def of testconf
    testconf["datasources"][0]["document_format"] = "PI_XML"
    testconf["datasources"][0]["document_version"] = "1.32"
    # Create:
    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = Config(tmp_conf_file, "yaml")

    data = FewsWebService.get_data(conf.content.datasources[0])

    assert len(data[0].xarray.time) == OBS_TIME_DIM_LENGTH  # type: ignore[misc]
