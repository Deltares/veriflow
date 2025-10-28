"""Test the dpyverification.datamodel package."""

import xarray as xr
from dpyverification.datamodel.main import InputDataset
from dpyverification.datasources.fewsnetcdf import FewsNetCDF

# mypy: disable-error-code="misc"


def test_init_input_dataset_simulated_forecast_ensemble(
    xarray_observed_historical: xr.DataArray,
    xarray_simulated_forecast_ensemble: xr.DataArray,
) -> None:
    """Test the input_dataset initializes successfully with forecast period (fp) input."""
    _ = InputDataset(
        data=[xarray_observed_historical, xarray_simulated_forecast_ensemble],
    )


def test_init_input_dataset_simulated_forecast_single(
    xarray_observed_historical: xr.DataArray,
    xarray_simulated_forecast_single: xr.DataArray,
) -> None:
    """Test the input_dataset initializes successfully with forecast period (fp) input."""
    _ = InputDataset(
        data=[xarray_observed_historical, xarray_simulated_forecast_single],
    )


def test_init_input_dataset_fewsnetcdf(
    fews_netcdf_observed_historical: FewsNetCDF,
    fews_netcdf_simulated_forecast_ensemble_frt: FewsNetCDF,
) -> None:
    """Test the fewsnetcdf is accepted by the input_dataset."""
    InputDataset(
        data=[
            fews_netcdf_observed_historical.get_data().data_array,
            fews_netcdf_simulated_forecast_ensemble_frt.get_data().data_array,
        ],
    )
