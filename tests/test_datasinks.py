"""Module for tests of datasinks."""

from pathlib import Path
from typing import cast

import pytest
import xarray as xr

from veriflow.configuration.base import GeneralInfoConfig
from veriflow.constants import DataSinkKind
from veriflow.datasinks.cf_compliant_netcdf import CFCompliantNetCDF, CFCompliantNetCDFConfig
from veriflow.datasinks.cf_compliant_zarr import CFCompliantZarr, CFCompliantZarrConfig
from veriflow.datatree.datatree import VeriflowDataTree


@pytest.mark.parametrize(
    "dt_fixture",
    [
        "output_datatree_without_scores",
        "output_datatree_with_scores",
    ],
)
def test_cf_compliant_netcdf_write(
    request: pytest.FixtureRequest,
    dt_fixture: str,
    tmpdir: Path,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Test writing data to a cf-compliant NetCDF file."""
    output_datatree: VeriflowDataTree = request.getfixturevalue(dt_fixture)
    datasink_cf_compliant_netcdf = CFCompliantNetCDF(
        CFCompliantNetCDFConfig(
            institution="Test Institution",
            comment="Test Comment",
            export_adapter=DataSinkKind.cf_compliant_netcdf,
            crs="EPSG:4326",
            directory=Path(tmpdir),
            filename="test_output.nc",
            general=xarray_general_info_config,
        ),
    )
    assert output_datatree is not None
    assert isinstance(output_datatree, xr.DataTree)
    datasink_cf_compliant_netcdf.write_data(
        output_datatree,
    )
    # File name is derived from the configured filename plus the verification pair id.
    assert (Path(tmpdir) / f"{datasink_cf_compliant_netcdf.config.filename}").exists()


def test_cf_compliant_netcdf_write_multiple_pairs(
    output_datatree_with_multiple_pairs: VeriflowDataTree,
    tmpdir: Path,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Test that one distinct NetCDF file is written per verification pair."""
    datasink_cf_compliant_netcdf = CFCompliantNetCDF(
        CFCompliantNetCDFConfig(
            institution="Test Institution",
            comment="Test Comment",
            export_adapter=DataSinkKind.cf_compliant_netcdf,
            crs="EPSG:4326",
            directory=Path(tmpdir),
            filename="test_output.nc",
            general=xarray_general_info_config,
        ),
    )
    datasink_cf_compliant_netcdf.write_data(output_datatree_with_multiple_pairs)

    filepath = Path(tmpdir) / f"{datasink_cf_compliant_netcdf.config.filename}"
    assert filepath.exists()

    with xr.open_dataset(filepath) as written:
        assert cast("str", written.attrs["institution"]) == "Test Institution"  # type: ignore[misc]


def test_cf_compliant_netcdf_write_force_overwrite_false_raises(
    output_datatree_without_scores: VeriflowDataTree,
    tmpdir: Path,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Test that an existing per-pair file raises FileExistsError when force_overwrite=False."""
    # Pre-create the file that would be written for verification pair "test_pair".
    (Path(tmpdir) / "test_output_test_pair.nc").touch()
    datasink_cf_compliant_netcdf = CFCompliantNetCDF(
        CFCompliantNetCDFConfig(
            institution="Test Institution",
            comment="Test Comment",
            export_adapter=DataSinkKind.cf_compliant_netcdf,
            crs="EPSG:4326",
            directory=Path(tmpdir),
            filename="test_output.nc",
            force_overwrite=False,
            general=xarray_general_info_config,
        ),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        datasink_cf_compliant_netcdf.write_data(output_datatree_without_scores)


@pytest.mark.parametrize(
    "dt_fixture",
    [
        "output_datatree_without_scores",
        "output_datatree_with_scores",
    ],
)
def test_cf_compliant_zarr_write_local(
    request: pytest.FixtureRequest,
    dt_fixture: str,
    tmpdir: Path,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Test writing data to a cf-compliant Zarr store."""
    output_datatree: VeriflowDataTree = request.getfixturevalue(dt_fixture)
    datasink_cf_compliant_zarr = CFCompliantZarr(
        CFCompliantZarrConfig(
            institution="Test Institution",
            comment="Test Comment",
            export_adapter=DataSinkKind.cf_compliant_zarr,
            crs="EPSG:4326",
            path=f"{tmpdir!s}/test_output.zarr",
            general=xarray_general_info_config,
        ),
    )
    assert output_datatree is not None
    assert isinstance(output_datatree, xr.DataTree)
    datasink_cf_compliant_zarr.write_data(
        output_datatree,
    )
    assert (Path(datasink_cf_compliant_zarr.config.path)).exists()


def test_cf_compliant_zarr_write_explicit_consolidated(
    output_datatree_without_scores: VeriflowDataTree,
    tmpdir: Path,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Test that an explicit (non-None) consolidated value is passed through as-is."""
    datasink_cf_compliant_zarr = CFCompliantZarr(
        CFCompliantZarrConfig(
            institution="Test Institution",
            comment="Test Comment",
            export_adapter=DataSinkKind.cf_compliant_zarr,
            crs="EPSG:4326",
            path=f"{tmpdir!s}/test_output.zarr",
            consolidated=False,
            general=xarray_general_info_config,
        ),
    )
    datasink_cf_compliant_zarr.write_data(output_datatree_without_scores)
    assert Path(datasink_cf_compliant_zarr.config.path).exists()


def test_cf_compliant_zarr_write_force_overwrite_false_raises(
    output_datatree_without_scores: VeriflowDataTree,
    tmpdir: Path,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Test that an existing store raises FileExistsError when force_overwrite=False."""
    store_path = Path(tmpdir) / "test_output.zarr"
    store_path.mkdir()
    datasink_cf_compliant_zarr = CFCompliantZarr(
        CFCompliantZarrConfig(
            institution="Test Institution",
            comment="Test Comment",
            export_adapter=DataSinkKind.cf_compliant_zarr,
            crs="EPSG:4326",
            path=str(store_path),
            force_overwrite=False,
            general=xarray_general_info_config,
        ),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        datasink_cf_compliant_zarr.write_data(output_datatree_without_scores)
