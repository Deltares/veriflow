"""Collection of constant values definitions.

Both single values, and enum classes.
These values are can be used throughout DPyVerification.
"""

import pathlib
import subprocess
from dataclasses import dataclass
from enum import StrEnum

import importlib_metadata


class DataSourceTypeEnum(StrEnum):
    """Enumeration of the supported input and / or output datasource types."""

    pixml = "pixml"
    fewsnetcdf = "fewsnetcdf"
    fewswebservice = "fewswebservice"


class SimObsType(StrEnum):
    """Enumeration of the supported types of input data."""

    sim = "sim"
    obs = "obs"
    combined = "combined"


class CalculationTypeEnum(StrEnum):
    """Enumeration of the implemented verification calculations."""

    simobspairs = "simobspair"
    pinscore = "pinscore"


class TimeUnits(StrEnum):
    """Time unit strings, compatible with numpy datetime64 and timedelta64."""

    year = "Y"
    month = "M"
    week = "W"
    day = "D"
    hour = "h"
    minute = "m"
    second = "s"


class DataModelDims:
    """List of dimension names.

    To avoid hardcoded strings in multiple places,
    have a single list with the names of known dimensions.
    """

    time = "time"
    location = "location_id"
    ensemble = "ensemble_member"
    simstart = "simulation_starttime"


class DataModelCoords:
    """List of coordinate names and attributes.

    To avoid hardcoded strings in multiple places, have a single list with the names of known
    coordinates, and their known attributes. At the very least the attributes needed for the
    coordinates to be CF compliant are included in the attributes info.

    Coordinates with matching dimension will have the same name as the dimension
    """

    @dataclass
    class CoordinateProperties:
        """Collect name and known attributes of coordinate in one object."""

        name: str
        # The attributes given here are meant to be
        # - used directly in a attrs.update() call
        # - immutable
        # so use tuples of tuples (inner tuples length two).
        attributes: tuple[tuple[str, str], ...]

    # Define unitsattribute here, or when putting in the values? How to make the values match the
    #  units, for the time-like coordinates mainly?
    time = CoordinateProperties(
        DataModelDims.time,
        (("standard_name", "time"), ("long_name", "time"), ("axis", "T")),
    )
    location = CoordinateProperties(
        DataModelDims.location,
        # Having cf_role: timeseries_id on this coordinate, and featureType: timeSeries on the full
        #  dataset, was copied from example fews netcdf files. However, it appears to not be fully
        #  in line with how these are supposed to be used, according to CF 1.6?
        (("long_name", "station identification code"), ("cf_role", "timeseries_id")),
    )
    lat = CoordinateProperties(
        "lat",
        (
            ("standard_name", "latitude"),
            ("long_name", "Station coordinates, latitude"),
            ("units", "degrees_north"),
            ("axis", "Y"),
        ),
    )
    lon = CoordinateProperties(
        "lon",
        (
            ("standard_name", "longitude"),
            ("long_name", "Station coordinates, longitude"),
            ("units", "degrees_east"),
            ("axis", "X"),
        ),
    )
    ensemble = CoordinateProperties(
        DataModelDims.ensemble,
        (
            ("standard_name", "realization"),
            ("long_name", "Index of an ensemble member within an ensemble"),
            ("units", "1"),
        ),
    )
    simstart = CoordinateProperties(
        DataModelDims.simstart,
        (
            ("standard_name", "forecast_reference_time"),
            ("long_name", "forecast_reference_time"),
        ),
    )


class DataModelAttributes:
    """List of attribute names on the main xarray.

    To avoid hardcoded strings in multiple places,
    have a single list with the names of known attributes.
    """

    # Rework to be somewhat similar to DataModelCoords, with both names and (default) values?
    source = "source"
    timestep = "timestep"
    featuretype = "featureType"


def _set_version_info() -> tuple[str, str]:
    version = importlib_metadata.version("dpyverification")
    version_extra = ""

    # It would be preferable to use available tools for our versioning, e.g.
    # https://github.com/mtkennerly/dunamai and / or https://github.com/mtkennerly/poetry-dynamic-versioning
    # so we also do not rely on (updating the) hardcoded version in the pyproject.toml
    this_dir = pathlib.Path(__file__).parent
    command = ["git", "rev-parse", "HEAD"]
    completed = subprocess.run(command, cwd=this_dir, capture_output=True, check=False, text=True)  # noqa: S603 # No S603 since we are pretty certain the input is not dangerous
    if completed.returncode == 0:
        # Git is installed, and this file is in a directory that is part of a git repository
        # Add git commit hash as version_extra
        stdout = completed.stdout
        version_extra = "+" + stdout.strip()
        # Also check whether there are uncommitted changes on top of the commit
        command = ["git", "diff", "HEAD"]
        completed = subprocess.run(  # noqa: S603 # No S603 since we are pretty certain the input is not dangerous
            command,
            cwd=this_dir,
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.stdout:
            version_extra += ".dirty"
    else:
        origin = importlib_metadata.distribution("dpyverification").origin  # type: ignore[misc] # Since not sure what will be returned, there is the long if statement
        if (
            origin  # type: ignore[misc] # Since not sure what will be returned, there is the long if statement
            and hasattr(origin, "dir_info")  # type: ignore[misc] # Since not sure what will be returned, there is the long if statement
            and hasattr(origin.dir_info, "editable")  # type: ignore[misc] # Since not sure what will be returned, there is the long if statement
            and origin.dir_info.editable  # type: ignore[misc] # Since not sure what will be returned, there is the long if statement
        ):
            # Package installed in editable mode, but no way to check whether it has been edited
            version_extra = "+editable"

    version_full = version + version_extra
    return version, version_full


NAME = "DPyVerification"
VERSION, VERSION_FULL = _set_version_info()
