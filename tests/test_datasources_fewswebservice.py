"""Test the fewswebservice module of the dpyverification.datasources package."""

import os

import pytest
import requests
import yaml
from dpyverification.configuration import ConfigSchema
from dpyverification.configuration.schema import LeadTimes
from dpyverification.constants import TimeUnits
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
def test_get_timeseries_sim_happy() -> None:
    """Check that the imported pixml gives an xarray with the expected content."""
    with TESTS_CONFIGURATION_FILE.open() as cf:
        leadtimes = LeadTimes(unit=TimeUnits.hour, values=[3, 6])
        verificationperiod = {
            "start": {"value": "2024-06-01T00:00:00Z"},
            "end": {"value": "2024-06-03T00:00:00Z"},
        }
        testconf = yaml.safe_load(cf)  # type: ignore[misc]
        testconf["general"]["verificationperiod"] = verificationperiod  # type: ignore[misc]
        testconf["datasources"][0]["simobstype"] = "sim"  # type: ignore[misc]
        testconf["datasources"][0]["datasourcetype"] = "fewswebservice"  # type: ignore[misc]
        testconf["datasources"][0]["url"] = (  # type: ignore[misc]
            "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
        )
        testconf["datasources"][0]["location_ids"] = ["H-RN-0001"]  # type: ignore[misc]
        testconf["datasources"][0]["parameter_ids"] = ["Q.fs"]  # type: ignore[misc]
        testconf["datasources"][0]["module_instance_ids"] = ["SBK3_MaxRTK_ECMWF_ENS"]  # type: ignore[misc]
        testconf["datasources"][0]["qualifier_ids"] = []  # type: ignore[misc]
        testconf["datasources"][0]["document_format"] = "PI_XML"  # type: ignore[misc]
        testconf["datasources"][0]["document_version"] = "1.32"  # type: ignore[misc]
        testconf["datasources"][0]["leadtimes"] = leadtimes  # type: ignore[misc]
        testconf["datasources"][0]["verificationperiod"] = verificationperiod  # type: ignore[misc]

    parsed_content = ConfigSchema(**testconf)  # type: ignore[misc]

    data = FewsWebService.get_data(parsed_content.datasources[0])

    assert len(data[0].xarray.time) == SIM_TIME_DIM_LENGTH  # type: ignore[misc]


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_get_timeseries_obs_happy() -> None:
    """Check that the imported pixml gives an xarray with the expected content."""
    with TESTS_CONFIGURATION_FILE.open() as cf:
        verificationperiod = {
            "start": {"value": "2024-06-01T00:00:00Z"},
            "end": {"value": "2024-06-03T00:00:00Z"},
        }
        testconf = yaml.safe_load(cf)  # type: ignore[misc]
        testconf["general"]["verificationperiod"] = verificationperiod  # type: ignore[misc]
        testconf["datasources"][0]["simobstype"] = "obs"  # type: ignore[misc]
        testconf["datasources"][0]["datasourcetype"] = "fewswebservice"  # type: ignore[misc]
        testconf["datasources"][0]["url"] = (  # type: ignore[misc]
            "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
        )
        testconf["datasources"][0]["location_ids"] = ["H-RN-0001"]  # type: ignore[misc]
        testconf["datasources"][0]["parameter_ids"] = ["Q.m"]  # type: ignore[misc]
        testconf["datasources"][0]["module_instance_ids"] = ["Import_LMW"]  # type: ignore[misc]
        testconf["datasources"][0]["qualifier_ids"] = []  # type: ignore[misc]
        testconf["datasources"][0]["document_format"] = "PI_XML"  # type: ignore[misc]
        testconf["datasources"][0]["document_version"] = "1.32"  # type: ignore[misc]
        testconf["datasources"][0]["verificationperiod"] = verificationperiod  # type: ignore[misc]

    parsed_content = ConfigSchema(**testconf)  # type: ignore[misc]

    data = FewsWebService.get_data(parsed_content.datasources[0])

    assert len(data[0].xarray.time) == OBS_TIME_DIM_LENGTH  # type: ignore[misc]
