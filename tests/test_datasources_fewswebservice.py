"""Test the fewswebservice module of the dpyverification.datasources package."""

# mypy: ignore-errors

import os
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import requests
import yaml
from dpyverification.configuration import ConfigFile
from dpyverification.datasources.fewswebservice import FewsWebservice

from tests import TESTS_CONFIGURATION_FILE

SIM_TIME_DIM_LENGTH = 373
OBS_TIME_DIM_LENGTH = 721
VALID_RESPONSE_CODE = 200
TASK_START_SUCCESS_TEXT = '{"started":true,"message":"Task started"}'

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(scope="module", autouse=True)
def _initialize_archive() -> None:
    @dataclass
    class _ArchiveTask:
        name: str
        task_id: str

    clear_catalogue = _ArchiveTask("Clear internal catalogue", "clear internal catalogue")
    internal_harvester = _ArchiveTask("Internal harvester", "harvester internal catalogue")

    def get_archive_task_status(archive_task: _ArchiveTask) -> dict[str, bool | str]:
        archive_status_url = "http://localhost:8080/deltares-archive-server/api/v1/archive/status"

        archive_status_response = requests.get(archive_status_url, timeout=10)
        assert archive_status_response.status_code == VALID_RESPONSE_CODE
        archive_status: dict[str, list[dict[str, bool | str]]] = yaml.safe_load(
            archive_status_response.text,
        )

        for task in archive_status["list"]:
            if task["name"] == archive_task.name:
                return task
        msg = (
            f"Task with name {archive_task.name} not found in archive status"
            f" information: {archive_status_response.text}"
        )
        raise ValueError(msg)

    def start_and_wait_for_task(archive_task: _ArchiveTask) -> None:
        archive_task_post_url = "http://localhost:8080/deltares-archive-server/api/v1/runtask"

        # The taskId should match the predefinedArchiveTask entry in the ArchiveTasksSchedule.xml
        body = {"taskId": archive_task.task_id}
        # Use argument 'data' (that will pass the data as application/x-www-form-urlencoded), and
        #  NOT 'json', as that is not properly processed by the other side
        archive_task_post_response = requests.post(archive_task_post_url, data=body, timeout=10)
        assert archive_task_post_response.status_code == VALID_RESPONSE_CODE
        assert archive_task_post_response.text == TASK_START_SUCCESS_TEXT

        task_status = get_archive_task_status(archive_task)
        max_wait = 15.0
        waited_time = 0.0
        sleep_time = 0.5
        while task_status["running"] and waited_time < max_wait:
            # running, wait for finish
            time.sleep(sleep_time)
            waited_time = waited_time + sleep_time
            task_status = get_archive_task_status(archive_task)

        assert "finished" in task_status["status"]  # type: ignore[operator] # Indeed the use of in does not fully match with our faked type def of task_status

    # Check archive is up by requesting status
    _ = get_archive_task_status(clear_catalogue)
    # Always run these two tasks, before any of the tests on the webservice
    #   Do not check lastruntime or running status beforehand, unnecessary complication
    start_and_wait_for_task(clear_catalogue)
    start_and_wait_for_task(internal_harvester)


@pytest.fixture()
def _fews_webservice_mock_env(
    monkeypatch: Generator[pytest.MonkeyPatch, None, None],
) -> None:
    """Create a mock environment for testing secret env vars."""
    # The dummy url, username and password
    url = "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
    monkeypatch.setenv("FEWSWEBSERVICE_URL", url)  # type: ignore  # noqa: PGH003
    monkeypatch.setenv("FEWSWEBSERVICE_USERNAME", "")  # type: ignore  # noqa: PGH003
    monkeypatch.setenv("FEWSWEBSERVICE_PASSWORD", "")  # type: ignore  # noqa: PGH003


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_webservice_live() -> None:
    """Test that a webservice is live and can find filters."""
    url = "http://localhost:8080/FewsWebServices/rest/fewspiservice/v1"
    endpoint = "archive/locations"
    test_endpoint_url = url + "/" + endpoint
    response = requests.get(test_endpoint_url, timeout=10)
    assert response.status_code == VALID_RESPONSE_CODE


@pytest.mark.usefixtures("_fews_webservice_mock_env")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_get_obs_netcdf(
    tmp_path: Path,
) -> None:
    """Check that the imported pixml gives an xarray with the expected content."""
    verificationperiod = {
        "start": "2024-11-01T00:00:00Z",
        "end": "2024-12-01T00:00:00Z",
    }

    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)

    testconf["general"]["verificationperiod"] = verificationperiod  # type: ignore[call-overload] # Indeed this assignment does not match with our faked type def of testconf

    testconf["datasources"][0] = {
        "simobstype": "obs",
        "kind": "fewswebservice",
        "location_ids": ["H-RN-0001"],
        "parameter_ids": ["Q_m"],
        "module_instance_ids": ["Hydro_Prep"],
        "auth_config": {
            "url": os.environ.get("FEWSWEBSERVICE_URL"),
            "username": os.environ.get("FEWSWEBSERVICE_USERNAME"),
            "password": os.environ.get("FEWSWEBSERVICE_PASSWORD"),
        },
    }

    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = ConfigFile(tmp_conf_file, "yaml")
    instance = FewsWebservice.from_config(conf.content.datasources[0].model_dump()).get_data()  # type: ignore[misc] # Yes, allow any
    assert "Q_m" in instance.xarray
    np.testing.assert_array_equal(instance.xarray["lat"].values, np.float64(51.85059))


@pytest.mark.usefixtures("_fews_webservice_mock_env")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_get_sim_netcdf(
    tmp_path: Path,
) -> None:
    """Check that the imported pixml gives an xarray with the expected content."""
    leadtimes = {"unit": "h", "values": [24, 48, 72, 96]}
    verificationperiod = {
        "start": "2024-11-01T00:00:00Z",
        "end": "2024-12-01T00:00:00Z",
    }

    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)

    testconf["general"]["verificationperiod"] = verificationperiod  # type: ignore[call-overload] # Indeed this assignment does not match with our faked type def of testconf
    testconf["general"]["leadtimes"] = leadtimes  # type: ignore[call-overload] # Indeed this assignment does not match with our faked type def of testconf

    testconf["datasources"][0] = {
        "simobstype": "sim",
        "kind": "fewswebservice",
        "location_ids": ["H-RN-0001"],
        "parameter_ids": ["Q_fs"],
        "module_instance_ids": ["SBK3_MaxRTK_ECMWF_ENS"],
        "ensemble_id": ["ECMWF_ENS"],
        "auth_config": {
            "url": os.environ.get("FEWSWEBSERVICE_URL"),
            "username": os.environ.get("FEWSWEBSERVICE_USERNAME"),
            "password": os.environ.get("FEWSWEBSERVICE_PASSWORD"),
        },
    }

    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = ConfigFile(tmp_conf_file, "yaml")
    with pytest.raises(NotImplementedError, match="Simulations are not yet supported"):
        FewsWebservice.from_config(conf.content.datasources[0].model_dump()).get_data()  # type: ignore[misc] # Yes, allow any
