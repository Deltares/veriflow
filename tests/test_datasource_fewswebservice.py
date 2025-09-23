"""Test the fewswebservice module of the dpyverification.datasources package."""

# mypy: ignore-errors

import os
import time
from dataclasses import dataclass

import pytest
import requests
import yaml
from dpyverification.datasources.fewswebservice import FewsWebservice

SIM_TIME_DIM_LENGTH = 373
OBS_TIME_DIM_LENGTH = 721
VALID_RESPONSE_CODE = 200
TASK_START_SUCCESS_TEXT = '{"started":true,"message":"Task started"}'

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(scope="module", autouse=False)
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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_webservice_live() -> None:
    """Test that a webservice is live and can find filters."""
    url = "https://rwsos-dataservices.avi.deltares.nl/testarchive/FewsWebServices/rest/fewspiservice/v1"
    endpoint = "archive/locations"
    test_endpoint_url = url + "/" + endpoint
    response = requests.get(test_endpoint_url, timeout=10)
    assert response.status_code == VALID_RESPONSE_CODE


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_get_obs_netcdf(datasource_fewswebservice_obs: FewsWebservice) -> None:
    """Check that the webservice gives expected outcome for obs."""
    _ = datasource_fewswebservice_obs.get_data()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Cannot yet test webservice in GitHub CI")
def test_get_sim_netcdf(
    datasource_fewswebservice_sim: FewsWebservice,
) -> None:
    """Check that the webservice gives expected outcome for sim."""
    _ = datasource_fewswebservice_sim.get_data()
