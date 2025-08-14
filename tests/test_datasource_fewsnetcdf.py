"""Test the fewsnetcdf module of the dpyverification.datasources package."""

from pathlib import Path

import xarray as xr
import yaml
from dpyverification.configuration import ConfigFile
from dpyverification.constants import StandardAttribute, StandardCoord, StandardDim
from dpyverification.datasinks.fewsnetcdf import FewsNetcdfFileSink, FewsNetcdfOutputSchema
from dpyverification.datasources.fewsnetcdf import FewsNetcdfFile

from tests import (
    TESTS_CONFIGURATION_FILE,
    TESTS_FEWS_COMPLIANT_FILE,
)


def test_get_data_compliant_file_happy(
    datasource_fewsnetcdf_compliant: FewsNetcdfFile,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    _ = datasource_fewsnetcdf_compliant


def test_fewsnetcdf_output_schema_compliant_file() -> None:
    """Test FEWS-compliant file is compliant with schema."""
    ds = xr.open_dataset(TESTS_FEWS_COMPLIANT_FILE)
    dataset_dict = ds.to_dict()  # type: ignore[misc] # Yes, the dict could have any content, it will be checked against the FewsNetcdfSchema
    # This will throw an error when not compliant
    FewsNetcdfOutputSchema.model_validate(dataset_dict)  # type: ignore[misc] # See above


def test_get_data_obs(
    datasource_fewsnetcdf_obs: FewsNetcdfFile,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    _ = datasource_fewsnetcdf_obs.get_data()


def test_get_data_sim(
    datasource_fewsnetcdf_sim: FewsNetcdfFile,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    datasource = datasource_fewsnetcdf_sim.get_data()

    # Assert resulting forecast periods in dataset match
    #   configured forecast periods
    assert all(
        datasource.dataset[StandardDim.forecast_period]
        == datasource.config.forecast_periods.timedelta64,
    )


def test_write_happy(tmp_path: Path) -> None:
    """Test writing a netcdf succeeds."""
    ds = xr.open_dataset(TESTS_FEWS_COMPLIANT_FILE)

    # A fewscompliant nc file uses different names than our internal datamodel
    # Adapt the ds to look like our internal datamodel
    # When get_data has been implemented for fewsnetcdf, use that instead
    ds_datamodel = ds.rename_dims({"analysis_time": StandardDim.forecast_reference_time})  # type: ignore[misc] # attrs is a dict[Any,Any]
    ds_datamodel = ds_datamodel.rename_vars(
        {"analysis_time": StandardCoord.forecast_reference_time.name},  # type: ignore[misc] # attrs is a dict[Any,Any]
    )
    ds_datamodel.attrs[StandardAttribute.timestep] = 1  # type: ignore[misc] # attrs is a dict[Any,Any]

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
    ds_datamodel = ds.rename_dims({"analysis_time": StandardDim.forecast_reference_time})  # type: ignore[misc] # attrs is a dict[Any,Any]
    ds_datamodel = ds_datamodel.rename_vars(
        {"analysis_time": StandardCoord.forecast_reference_time.name},  # type: ignore[misc] # attrs is a dict[Any,Any]
    )
    ds_datamodel.attrs[StandardAttribute.timestep] = 1  # type: ignore[misc] # attrs is a dict[Any,Any]

    FewsNetcdfFileSink.from_config(conf.content.datasinks[0].model_dump()).write_data(ds_datamodel)  # type: ignore[misc] # Yes, allow any

    assert tmpfile.exists()

    ds2 = xr.open_dataset(tmpfile)

    # "Two Datasets are equal if they have matching variables and coordinates, all of which
    #  are equal." Thus, attributes are probably not checked by the following.
    assert ds.equals(ds2)
