"""Test the fewsnetcdf module of the dpyverification.datasources package."""

import xarray as xr
from dpyverification.constants import StandardDim
from dpyverification.datasinks.fewsnetcdf import FewsNetcdfOutputSchema
from dpyverification.datasources.fewsnetcdf import FewsNetCDF


def test_get_data_compliant_file_happy(
    fews_netcdf_compliant_file: FewsNetCDF,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    _ = fews_netcdf_compliant_file


def test_fewsnetcdf_output_schema_compliant_file(xarray_dataset_fews_compliant: xr.Dataset) -> None:
    """Test FEWS-compliant file is compliant with schema."""
    dataset_dict = xarray_dataset_fews_compliant.to_dict()  # type: ignore[misc] # Yes, the dict could have any content, it will be checked against the FewsNetcdfSchema
    # This will throw an error when not compliant
    FewsNetcdfOutputSchema.model_validate(dataset_dict)  # type: ignore[misc] # See above


def test_get_data_obs(
    fews_netcdf_observed_historical: FewsNetCDF,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    _ = fews_netcdf_observed_historical.get_data()


def test_get_data_sim_for_full_simulations(
    fews_netcdf_simulated_forecast_ensemble_frt: FewsNetCDF,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    datasource = fews_netcdf_simulated_forecast_ensemble_frt.get_data()

    # Assert resulting forecast periods in dataset match
    #   configured forecast periods
    assert all(
        datasource.data_array[StandardDim.forecast_period]
        == datasource.config.forecast_periods.timedelta64,
    )
