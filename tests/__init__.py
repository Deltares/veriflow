"""Collection of tests for the dpyverification package."""

import pathlib

TESTS_DATA_DIR = pathlib.Path(__file__).parent / "data"
TESTS_CONFIGURATION_FILE = TESTS_DATA_DIR / "testconfig.yaml"
TESTS_OBSERVATIONS_FILE = TESTS_DATA_DIR / "webservice_responses_pi_xml/example_observations.xml"
TESTS_FORECASTS_FILE = TESTS_DATA_DIR / "webservice_responses_pi_xml/example_forecasts.xml"
