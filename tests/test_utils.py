"""Tests for general utility functions."""

import numpy as np
import pytest
import xarray as xr

from veriflow.utils import (
    convert_byte_string_coord_to_utf8,
    convert_byte_string_coords_to_utf8,
    convert_timedelta64_list_to_largest_common_unit,
    drop_chunk_encoding,
    get_chunk_settings_along_forecast_reference_time,
    stringify_timedelta64_list,
)

# mypy: disable-error-code=misc


@pytest.fixture(scope="module")
def dt() -> xr.DataTree:
    """Create DataTree for this test module."""
    forecast_reference_time = np.datetime64("2024-01-01T00:00:00")
    da1 = xr.DataArray(
        np.ones((4, 3)),
        dims=("forecast_reference_time", "station"),
        coords={
            "forecast_reference_time": [
                forecast_reference_time + np.timedelta64(i, "h") for i in range(4)
            ],
            "station": ["A", "B", "C"],
        },
    )

    da2 = xr.DataArray(
        np.ones((4, 3)),
        dims=("dummy", "station"),
        coords={
            "dummy": [forecast_reference_time + np.timedelta64(i, "h") for i in range(4)],
            "station": ["A", "B", "C"],
        },
    )

    dt = xr.DataTree()
    dt["da1"] = da1
    dt["da2"] = da2
    return dt


def test_convert_byte_string_coord_to_utf8_decodes_non_ascii_utf8() -> None:
    """Byte-string coordinates are decoded with UTF-8, not ASCII."""
    expected = "chäråcter"
    coord = xr.DataArray(np.array([expected.encode()], dtype="S"), dims="station")

    result = convert_byte_string_coord_to_utf8(coord)

    assert result.to_numpy().tolist() == [expected]


def test_convert_byte_string_coords_to_utf8_decodes_specified_coords() -> None:
    """Only specified byte-string coordinates are decoded on a dataset."""
    expected = "chäråcter"
    dataset = xr.Dataset(
        coords={
            "station": ("station", np.array([expected.encode("utf-8")], dtype="S")),
            "scenario": ("scenario", np.array([b"dry"], dtype="S")),
        },
    )

    result = convert_byte_string_coords_to_utf8(dataset, ["station"])

    assert result["station"].to_numpy().tolist() == [expected]
    assert result["scenario"].dtype.kind == "S"


def test_convert_byte_string_coords_to_utf8_decodes_all_coords_by_default() -> None:
    """All byte-string coordinates are decoded when no coord list is provided."""
    dataset = xr.Dataset(
        coords={
            "station": ("station", np.array(["chäråcter".encode()], dtype="S")),
            "scenario": ("scenario", np.array([b"dry"], dtype="S")),
        },
    )
    result = convert_byte_string_coords_to_utf8(dataset)
    assert result["station"].to_numpy().tolist() == ["chäråcter"]
    assert result["scenario"].to_numpy().tolist() == ["dry"]


def test_format_lead_times_to_string() -> None:
    """Lead times are formatted as strings with the appropriate unit."""
    lead_times = [np.timedelta64(1, "h"), np.timedelta64(2, "h"), np.timedelta64(3, "h")]
    expected = ["1h", "2h", "3h"]
    result = convert_timedelta64_list_to_largest_common_unit(lead_times)
    assert result == expected


def test_stringify_timedelta64_list() -> None:
    """Lead times are converted to strings with the largest common unit."""
    lead_times = [np.timedelta64(1, "h"), np.timedelta64(2, "h"), np.timedelta64(3, "h")]
    expected = ["1h", "2h", "3h"]
    result = stringify_timedelta64_list(lead_times)
    assert result == expected


def test_get_chunk_settings_along_forecast_reference_time(dt: xr.DataTree) -> None:
    """Chunk settings along the forecast reference time are correctly determined."""
    chunked_dt = dt.chunk(get_chunk_settings_along_forecast_reference_time(dt))
    assert chunked_dt["da1"].chunksizes == {
        "forecast_reference_time": (1, 1, 1, 1),
        "station": (3,),
    }
    assert chunked_dt["da2"].chunksizes == {"dummy": (4,), "station": (3,)}


def test_drop_chunk_encoding(dt: xr.DataTree) -> None:
    """Chunk encoding is correctly dropped."""
    dt["da1"].encoding["chunks"] = (1, 3)
    assert "chunks" in dt["da1"].encoding
    assert dt["da1"].encoding["chunks"] == (1, 3)
    drop_chunk_encoding(dt)
    assert "chunks" not in dt["da1"].encoding
