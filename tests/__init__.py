"""Collection of tests for the dpyverification package."""

import pathlib

TESTS_DATA_DIR = pathlib.Path(__file__).parent / "data"
TESTS_CONFIGURATION_FILE = TESTS_DATA_DIR / "testconfig.yaml"
TESTS_FEWS_COMPLIANT_FILE = TESTS_DATA_DIR / "fews_compliant_test_file.nc"

# Files from the webservice
TEST_DIR_FEWS_NETCDF_OBS = TESTS_DATA_DIR / "webservice_responses_netcdf/obs"
TEST_DIR_FEWS_NETCDF_SIM = TESTS_DATA_DIR / "webservice_responses_netcdf/sim"
