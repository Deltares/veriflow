"""Test the fewsnetcdf module of the dpyverification.datasources package."""

import xarray as xr
from dpyverification.constants import StandardDim
from dpyverification.datasinks.fewsnetcdf import FewsNetcdfOutputSchema
from dpyverification.datasources.fewsnetcdf import FewsNetCDFFile

from tests import (
    TESTS_FEWS_COMPLIANT_FILE,
)


def test_get_data_compliant_file_happy(
    datasource_fewsnetcdf_compliant: FewsNetCDFFile,
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
    datasource_fewsnetcdf_obs: FewsNetCDFFile,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    _ = datasource_fewsnetcdf_obs.get_data()


def test_get_data_sim(
    datasource_fewsnetcdf_sim: FewsNetCDFFile,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    datasource = datasource_fewsnetcdf_sim.get_data()

    # Assert resulting forecast periods in dataset match
    #   configured forecast periods
    assert all(
        datasource.data_array[StandardDim.forecast_period]
        == datasource.config.forecast_periods.timedelta64,
    )
