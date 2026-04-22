"""Test the fewsnetcdf module of the veriflow.datasources package."""

import pytest
import xarray as xr

from veriflow.constants import StandardDim
from veriflow.datasinks.fewsnetcdf import FewsNetcdfOutputSchema
from veriflow.datasources.fewsnetcdf import FewsNetCDF
from veriflow.datasources.inputschemas import INPUT_SCHEMAS


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


# Observed Historical
def test_get_data_observed_historical(
    fews_netcdf_observed_historical: FewsNetCDF,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    _ = fews_netcdf_observed_historical.get_data()


@pytest.mark.parametrize(
    "fews_netcdf_fixture",
    [
        "fews_netcdf_simulated_forecast_single_frt",
        "fews_netcdf_simulated_forecast_single_fp",
        "fews_netcdf_simulated_forecast_ensemble_frt",
        "fews_netcdf_simulated_forecast_ensemble_fp",
        "fews_netcdf_simulated_forecast_probabilistic_frt",
        "fews_netcdf_simulated_forecast_probabilistic_fp",
    ],
)
def test_get_data_returns_valid_data_array(
    request: pytest.FixtureRequest,
    fews_netcdf_fixture: str,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected forecast periods."""
    fews_netcdf: FewsNetCDF = request.getfixturevalue(fews_netcdf_fixture)
    datasource = fews_netcdf.get_data()

    schema = INPUT_SCHEMAS[fews_netcdf.config.data_type]
    schema.model_validate(fews_netcdf.data_array.to_dict(data=False))  # type:ignore[misc]

    assert all(
        datasource.data_array[StandardDim.forecast_period]
        == datasource.config.forecast_periods.timedelta64,
    )


@pytest.mark.parametrize(
    ("frt", "fp"),
    [
        ("fews_netcdf_simulated_forecast_single_frt", "fews_netcdf_simulated_forecast_single_fp"),
        (
            "fews_netcdf_simulated_forecast_ensemble_frt",
            "fews_netcdf_simulated_forecast_ensemble_fp",
        ),
        (
            "fews_netcdf_simulated_forecast_probabilistic_frt",
            "fews_netcdf_simulated_forecast_probabilistic_fp",
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
    frt: str,
    fp: str,
) -> None:
    """Check that 'frt' and 'fp' data sources produce identical datasets."""
    ds_a: FewsNetCDF = request.getfixturevalue(frt)
    ds_b: FewsNetCDF = request.getfixturevalue(fp)

    a = ds_a.get_data().data_array
    b = ds_b.get_data().data_array

    xr.testing.assert_equal(a, b)
