"""Test the fewsnetcdf module of the veriflow.datasources package."""

import pytest
import xarray as xr

from veriflow.constants import DataType, StandardDim
from veriflow.datasources.fewsnetcdf import FewsNetCDF
from veriflow.datasources.inputschemas import INPUT_SCHEMAS, validate_input_data


def test_get_data_compliant_file_happy(
    fews_netcdf_compliant_file: FewsNetCDF,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    _ = fews_netcdf_compliant_file


# Observed Historical
def test_get_data_observed_historical(
    fews_netcdf_observed_historical: FewsNetCDF,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    obj = fews_netcdf_observed_historical.get_data()
    validate_input_data(obj.dataset)


def test_get_data_simulated_historical(
    fews_netcdf_simulated_historical: FewsNetCDF,
) -> None:
    """Check that the imported fewsnetcdf gives an xarray with the expected content."""
    obj = fews_netcdf_simulated_historical.get_data()
    assert obj.dataset.attrs["data_type"] == DataType.simulated_historical  # type:ignore[misc]
    validate_input_data(obj.dataset)


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
    """Check that the imported fewsnetcdf gives an xarray with the expected lead times."""
    fews_netcdf: FewsNetCDF = request.getfixturevalue(fews_netcdf_fixture)
    datasource = fews_netcdf.get_data()

    schema = INPUT_SCHEMAS[(fews_netcdf.config.data_type, fews_netcdf.config.spatial_type)]
    schema.model_validate(fews_netcdf.dataset.to_dict(data=False))  # type:ignore[misc]
    assert datasource.config.lead_times is not None
    assert all(
        datasource.dataset[StandardDim.lead_time] == datasource.config.lead_times.timedelta64,
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

    a = ds_a.get_data().dataset
    b = ds_b.get_data().dataset

    xr.testing.assert_equal(a, b)
