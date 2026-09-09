"""Test the fewswebservice module of the veriflow.datasources package."""

# mypy: ignore-errors

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
import requests
import xarray as xr
import yaml

from veriflow.configuration.base import GeneralInfoConfig
from veriflow.configuration.config import SupportedSchemaVersion
from veriflow.configuration.default.datasources import FewsWebserviceConfig
from veriflow.configuration.utils import VerificationPair, VerificationPeriod
from veriflow.constants import DataType, SpatialType, StandardDim
from veriflow.datasources.fewswebservice import FewsWebservice
from veriflow.datasources.inputschemas import INPUT_SCHEMAS

SIM_TIME_DIM_LENGTH = 373
OBS_TIME_DIM_LENGTH = 721

TASK_START_SUCCESS_TEXT = '{"started":true,"message":"Task started"}'

SKIP_LIVE_WEBSERVICE_TEST = True


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
        assert archive_status_response.ok
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
        assert archive_task_post_response.ok
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

    if not SKIP_LIVE_WEBSERVICE_TEST:
        # Check archive is up by requesting status
        _ = get_archive_task_status(clear_catalogue)
        # Always run these two tasks, before any of the tests on the webservice
        #   Do not check lastruntime or running status beforehand, unnecessary complication
        start_and_wait_for_task(clear_catalogue)
        start_and_wait_for_task(internal_harvester)


@pytest.mark.skipif(SKIP_LIVE_WEBSERVICE_TEST, reason="Skipping live webservice tests")
def test_webservice_live() -> None:
    """Test that a webservice is live and can find filters."""
    url = os.environ["FEWSWEBSERVICE_URL"]
    endpoint = "archive/locations"
    test_endpoint_url = url + "/" + endpoint
    response = requests.get(test_endpoint_url, timeout=10)
    assert response.ok


@pytest.mark.skipif(SKIP_LIVE_WEBSERVICE_TEST, reason="Skipping live webservice tests")
def test_get_data_fews_webservice_observed_historical(
    fews_webservice_observed_historical: FewsWebservice,
) -> None:
    """Check that the webservice gives expected outcome for obs."""
    _ = fews_webservice_observed_historical.get_data()


@pytest.mark.skipif(SKIP_LIVE_WEBSERVICE_TEST, reason="Skipping live webservice tests")
@pytest.mark.parametrize(
    ("frt", "fp"),
    [
        (
            "fews_webservice_simulated_forecast_single_frt",
            "fews_webservice_simulated_forecast_single_fp",
        ),
        (
            "fews_webservice_simulated_forecast_ensemble_frt",
            "fews_webservice_simulated_forecast_ensemble_fp",
        ),
        (
            "fews_webservice_simulated_forecast_probabilistic_frt",
            "fews_webservice_simulated_forecast_probabilistic_fp",
        ),
    ],
    ids=[
        "simulated_forecast_single",
        "simulated_forecast_ensemble",
        "simulated_forecast_probabilistic",
    ],
)
def test_get_data_retrieval_methods_return_equal_data_arrays(
    request: pytest.FixtureRequest,
    frt: FewsWebservice,
    fp: FewsWebservice,
) -> None:
    """Check that 'frt' and 'fp' data sources produce identical datasets."""
    ds_a: FewsWebservice = request.getfixturevalue(frt)
    ds_b: FewsWebservice = request.getfixturevalue(fp)

    a = ds_a.get_data().dataset
    b = ds_b.get_data().dataset

    xr.testing.assert_equal(a, b)


@pytest.mark.skipif(SKIP_LIVE_WEBSERVICE_TEST, reason="Skipping live webservice tests")
@pytest.mark.parametrize(
    "fews_webservice",
    [
        "fews_webservice_simulated_forecast_single_frt",
        "fews_webservice_simulated_forecast_single_fp",
        "fews_webservice_simulated_forecast_ensemble_frt",
        "fews_webservice_simulated_forecast_ensemble_fp",
        "fews_webservice_simulated_forecast_probabilistic_frt",
        "fews_webservice_simulated_forecast_probabilistic_fp",
    ],
)
def test_get_data_returns_valid_data_array(
    request: pytest.FixtureRequest,
    fews_webservice: FewsWebservice,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected lead times."""
    fews_netcdf: FewsWebservice = request.getfixturevalue(fews_webservice)
    datasource = fews_netcdf.get_data()

    schema = INPUT_SCHEMAS[(fews_netcdf.config.data_type, fews_netcdf.config.spatial_type)]
    schema.model_validate(fews_netcdf.dataset.to_dict(data=False))

    assert all(
        datasource.dataset[StandardDim.lead_time] == datasource.config.lead_times.timedelta64,
    )


@pytest.mark.skipif(SKIP_LIVE_WEBSERVICE_TEST, reason="Skipping live webservice tests")
def test_get_data_for_simulated_historical() -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected time dimension."""
    general = GeneralInfoConfig(
        version=SupportedSchemaVersion.V0,
        verification_period=VerificationPeriod(
            start=datetime(2024, 10, 28, tzinfo=timezone.utc),
            end=datetime(2024, 10, 30, tzinfo=timezone.utc),
            dimension=StandardDim.time,
        ),
        verification_pairs=[
            VerificationPair(
                id="idtest",
                reference_source_id="observed_historical",
                simulations_source_id="simulated_historical",
                variable="Qsim",
            ),
        ],
    )
    config = FewsWebserviceConfig(
        general=general,
        import_adapter="fewswebservice",
        source_id="source_single",
        data_type="simulated_historical",
        location_ids=["T508HMS", "T509HMS"],
        parameter_ids=["Q.sim"],
        module_instance_id="HMS_TM05_Update",
        webservice_version="2025.02",
    )
    fews_webservice = FewsWebservice(config)
    fews_webservice.get_data()

    schema = INPUT_SCHEMAS[(DataType.simulated_historical, SpatialType.point)]
    schema.model_validate(fews_webservice.dataset.to_dict(data=False))
