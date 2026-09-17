"""Tests for general utility functions."""

import numpy as np
import xarray as xr

from veriflow.utils import convert_byte_string_coord_to_utf8, convert_byte_string_coords_to_utf8

# mypy: disable-error-code=misc


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
