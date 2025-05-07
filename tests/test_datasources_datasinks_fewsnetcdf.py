"""Test the fewsnetcdf module of the dpyverification.datasources package."""

from pathlib import Path

import xarray as xr
import yaml
from dpyverification.configuration import ConfigFile
from dpyverification.constants import DataModelAttributes, DataModelCoords, DataModelDims
from dpyverification.datamodel.main import DataModel
from dpyverification.datasinks.fewsnetcdf import FewsNetcdfFileSink, FewsNetcdfOutputSchema
from dpyverification.datasources.fewsnetcdf import FewsNetcdfFile

from tests import (
    TEST_DIR_FEWS_NETCDF_OBS,
    TEST_DIR_FEWS_NETCDF_SIM,
    TESTS_CONFIGURATION_FILE,
    TESTS_FEWS_COMPLIANT_FILE,
)


def test_get_data_compliant_file_happy(tmp_path: Path) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    # Create an adapted testconfig, based on default testconfig
    # - load default config
    # - adapt config
    # - create config object from adapted config
    # Load:
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)
    # Adapt:
    testconf["datasources"][0]["kind"] = "fewsnetcdf"
    testconf["datasources"][0]["directory"] = str(TESTS_FEWS_COMPLIANT_FILE.parent)
    testconf["datasources"][0]["filename"] = TESTS_FEWS_COMPLIANT_FILE.name
    testconf["datasources"][0]["simobstype"] = "sim"
    # Create:
    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = ConfigFile(tmp_conf_file, "yaml")

    instance = FewsNetcdfFile.from_config(conf.content.datasources[0].model_dump()).get_data()  # type: ignore[misc] # Yes, allow any

    assert instance.xarray.attrs["date_created"] == "2014-03-10 07:57:01 GMT"  # type: ignore[misc] # Yes, allow any


def test_schema_compliant_file() -> None:
    """Test FEWS-compliant file is compliant with schema."""
    ds = xr.open_dataset(TESTS_FEWS_COMPLIANT_FILE)
    schema_like = ds.to_dict()  # type: ignore[misc] # Yes, the dict could have any content, it will be checked against the FewsNetcdfSchema
    # This will throw an error when not compliant
    _ = FewsNetcdfOutputSchema(**schema_like)  # type: ignore[misc] # See above


def test_get_data_obs_and_check_dims_coords(tmp_path: Path) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)

    # Get the path to the obs file
    test_file = next(iter(TEST_DIR_FEWS_NETCDF_OBS.rglob("*.nc")))
    testconf["datasources"][0]["kind"] = "fewsnetcdf"
    testconf["datasources"][0]["directory"] = str(test_file.parent)
    testconf["datasources"][0]["filename"] = test_file.name
    testconf["datasources"][0]["simobstype"] = "obs"
    # Create:
    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = ConfigFile(tmp_conf_file, "yaml")

    instance = FewsNetcdfFile.from_config(conf.content.datasources[0].model_dump()).get_data()  # type: ignore[misc] # Yes, allow any
    DataModel._check_source_dims_and_coords(instance)


def test_get_data_sim_and_check_dims_coords(tmp_path: Path) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)

    # Get the path to the obs file
    test_file = next(iter(TEST_DIR_FEWS_NETCDF_SIM.rglob("*.nc")))
    testconf["datasources"][0]["kind"] = "fewsnetcdf"
    testconf["datasources"][0]["directory"] = str(test_file.parent)
    testconf["datasources"][0]["filename"] = test_file.name
    testconf["datasources"][0]["simobstype"] = "sim"
    # Create:
    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = ConfigFile(tmp_conf_file, "yaml")

    instance = FewsNetcdfFile.from_config(conf.content.datasources[0].model_dump()).get_data()  # type: ignore[misc] # Yes, allow any
    DataModel._check_source_dims_and_coords(instance)


def test_write_happy(tmp_path: Path) -> None:
    """Test writing a netcdf succeeds."""
    ds = xr.open_dataset(TESTS_FEWS_COMPLIANT_FILE)

    # A fewscompliant nc file uses different names than our internal datamodel
    # Adapt the ds to look like our internal datamodel
    # When get_data has been implemented for fewsnetcdf, use that instead
    ds_datamodel = ds.rename_dims({"analysis_time": DataModelDims.simstart})  # type: ignore[misc] # attrs is a dict[Any,Any]
    ds_datamodel = ds_datamodel.rename_vars({"analysis_time": DataModelCoords.simstart.name})  # type: ignore[misc] # attrs is a dict[Any,Any]
    ds_datamodel.attrs[DataModelAttributes.timestep] = 1  # type: ignore[misc] # attrs is a dict[Any,Any]

    tmpfile = tmp_path / "test.nc"
    assert not tmpfile.exists()

    # Create an adapted testconfig, based on default testconfig
    # - load default config
    # - adapt config
    # - create config object from adapted config
    # Load:
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)
    # Adapt:
    testconf["datasinks"][0]["directory"] = str(tmpfile.parent)
    testconf["datasinks"][0]["filename"] = tmpfile.name
    # Create:
    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = ConfigFile(tmp_conf_file, "yaml")

    # rename_dims
    # rename_vars

    FewsNetcdfFileSink.from_config(conf.content.datasinks[0].model_dump()).write_data(ds_datamodel)  # type: ignore[misc] # Yes, allow any

    assert tmpfile.exists()


def test_read_write_equal(tmp_path: Path) -> None:
    """Test written content is equal to input dataset."""
    ds = xr.open_dataset(TESTS_FEWS_COMPLIANT_FILE)

    tmpfile = tmp_path / "test.nc"
    assert not tmpfile.exists()

    # Create an adapted testconfig, based on default testconfig
    # - load default config
    # - adapt config
    # - create config object from adapted config
    # Load:
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf: dict[str, list[dict[str, str]]] = yaml.safe_load(cf)
    # Adapt:
    testconf["datasinks"][0]["directory"] = str(tmpfile.parent)
    testconf["datasinks"][0]["filename"] = tmpfile.name
    # Create:
    tmp_conf_file = tmp_path / "tempconf.yaml"
    with tmp_conf_file.open(mode="w") as tf:
        yaml.dump(testconf, tf)
    conf = ConfigFile(tmp_conf_file, "yaml")

    # A fewscompliant nc file uses different names than our internal datamodel
    # Adapt the ds to look like our internal datamodel
    # When get_data has been implemented for fewsnetcdf, use that instead
    ds_datamodel = ds.rename_dims({"analysis_time": DataModelDims.simstart})  # type: ignore[misc] # attrs is a dict[Any,Any]
    ds_datamodel = ds_datamodel.rename_vars({"analysis_time": DataModelCoords.simstart.name})  # type: ignore[misc] # attrs is a dict[Any,Any]
    ds_datamodel.attrs[DataModelAttributes.timestep] = 1  # type: ignore[misc] # attrs is a dict[Any,Any]

    FewsNetcdfFileSink.from_config(conf.content.datasinks[0].model_dump()).write_data(ds_datamodel)  # type: ignore[misc] # Yes, allow any

    assert tmpfile.exists()

    ds2 = xr.open_dataset(tmpfile)

    # "Two Datasets are equal if they have matching variables and coordinates, all of which
    #  are equal." Thus, attributes are probably not checked by the following.
    assert ds.equals(ds2)


# Add a test for cf compliance of output, with the tool cfchecker?
#  However, cfchecker needs cfunits, which needs the udunits2 library. And udunits2 is not on
#  pypi. It is on conda-forge, so could use it in CI, and/or for developers. And as an optional for
#  the package? Definitely not as a required. If optional, then also use it in code to generate
#  warnings, or only in the tests?
#  Note that, even though the cf checker doc says it does not work on windows, it actually does
#  work, at least for me in my conda environment on windows.
