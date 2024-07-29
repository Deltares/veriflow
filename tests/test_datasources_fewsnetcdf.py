"""Test the fewsnetcdf module of the dpyverification.datasources package."""

from pathlib import Path

import pytest
import xarray as xr
import yaml
from dpyverification.configuration import YamlSchema
from dpyverification.datasources.fewsnetcdf import (
    FewsNetcdfFile,
    FewsNetcdfSchema,
)

from tests import TESTS_CONFIGURATION_FILE, TESTS_FEWS_COMPLIANT_FILE


# Not so very happy yet
def test_get_data_happy() -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    with TESTS_CONFIGURATION_FILE.open() as cf:
        testconf = yaml.safe_load(cf)  # type: ignore[misc]
        testconf["datasources"][0]["datasourcetype"] = "fewsnetcdf"  # type: ignore[misc]
        testconf["datasources"][0]["directory"] = str(TESTS_FEWS_COMPLIANT_FILE.parent)  # type: ignore[misc]
        testconf["datasources"][0]["filename"] = TESTS_FEWS_COMPLIANT_FILE.name  # type: ignore[misc]
        testconf["datasources"][0]["simobstype"] = "sim"  # type: ignore[misc]
    parsed_content = YamlSchema(**testconf)  # type: ignore[misc]

    with pytest.raises(NotImplementedError):
        _ = FewsNetcdfFile.get_data(parsed_content.datasources[0])

    # When no longer not-implemented, assert the contents
    # Also, when implemented, add to TESTS_CONFIGURATION_FILE?
    assert (
        True
    )  # ncdata[0].xarray["rainfall"].loc["a", "b:c"].data.tolist() == [somevalue, anothervalue]


def test_schema_testfile() -> None:
    """Test FEWS-compliant file is compliant with schema."""
    ds = xr.open_dataset(TESTS_FEWS_COMPLIANT_FILE)
    schema_like = ds.to_dict()  # type: ignore[misc] # Yes, the dict could have any content, it will be checked against the FewsNetcdfSchema
    # This will throw an error when not compliant
    _ = FewsNetcdfSchema(**schema_like)  # type: ignore[misc] # See above


def test_write_happy(tmp_path: Path) -> None:
    """Test writing a netcdf succeeds."""
    ds = xr.open_dataset(TESTS_FEWS_COMPLIANT_FILE)

    tmpfile = tmp_path / "test.nc"
    assert not tmpfile.exists()

    FewsNetcdfFile.write_to_file(tmpfile, ds)

    assert tmpfile.exists()


def test_read_write_equal(tmp_path: Path) -> None:
    """Test written content is equal to input dataset."""
    ds = xr.open_dataset(TESTS_FEWS_COMPLIANT_FILE)

    tmpfile = tmp_path / "test.nc"
    assert not tmpfile.exists()

    FewsNetcdfFile.write_to_file(tmpfile, ds)

    assert tmpfile.exists()

    ds2 = xr.open_dataset(tmpfile)

    # "Two Datasets are equal if they have matching variables and coordinates, all of which
    #  are equal." Thus, attributes are probably not checked by the following.
    assert ds.equals(ds2)
