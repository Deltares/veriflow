"""Collection of constant values definitions.

Both single values, and enum classes.
These values are can be used throughout DPyVerification.
"""

import pathlib
import subprocess
from dataclasses import dataclass
from enum import StrEnum, unique

import importlib_metadata


@unique
class DataSourceKind(StrEnum):
    """Enumeration of the supported datasource types."""

    FEWSNETCDF = "fewsnetcdf"
    FEWSWEBSERVICE = "fewswebservice"


class DataSinkKind(StrEnum):
    """Enumeration of the supported datasink types."""

    fews_netcdf = "fewsnetcdf"
    cf_compliant_netcdf = "cf_compliant_netcdf"


@unique
class TimeseriesKind(StrEnum):
    """Enumeration of the supported types of input data."""

    observed_historical = "observed_historical"
    simulated_historical = "simulated_historical"
    simulated_forecast_single = "simulated_forecast_single"
    simulated_forecast_ensemble = "simulated_forecast_ensemble"
    simulated_forecast_probabilistic = "simulated_forecast_probabilistic"


FORECAST_TIMESERIES_KIND = (
    TimeseriesKind.simulated_forecast_single,
    TimeseriesKind.simulated_forecast_probabilistic,
    TimeseriesKind.simulated_forecast_ensemble,
)


class ForecastTimeseriesKind(StrEnum):
    """Enumeration of forecast timeseries kinds."""

    simulated_forecast_single = "simulated_forecast_single"
    simulated_forecast_ensemble = "simulated_forecast_ensemble"
    simulated_forecast_probabilistic = "simulated_forecast_probabilistic"


@unique
class ScoreKind(StrEnum):
    """Enumeration of the implemented verification scores."""

    rank_histogram = "rank_histogram"
    crps_for_ensemble = "crps_for_ensemble"
    crps_cdf = "crps_cdf"
    continuous_scores = "continuous_scores"


@unique
class SupportedContinuousScore(StrEnum):
    """Supported continuous scores."""

    additive_bias = "additive_bias"
    mean_error = "mean_error"
    mae = "mae"
    mse = "mse"
    rmse = "rmse"
    nse = "nse"
    kge = "kge"


@unique
class TimeUnits(StrEnum):
    """Time unit strings, compatible with numpy datetime64 and timedelta64."""

    YEAR = "Y"
    MONTH = "M"
    WEEK = "W"
    DAY = "D"
    HOUR = "h"
    MINUTE = "m"
    SECOND = "s"


class StandardDim(StrEnum):
    """List of dimension names.

    To avoid hardcoded strings in multiple places,
    have a single list with the names of known dimensions.
    """

    time = "time"
    station = "station"
    realization = "realization"
    forecast_reference_time = "forecast_reference_time"
    forecast_period = "forecast_period"
    source = "source"
    variable = "variable"
    threshold = "threshold"


class StandardCoord:
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
        # - used directly in an attrs.update() call
        # - immutable
        # so use tuples of tuples (inner tuples length two).
        attributes: tuple[tuple[str, str], ...]

    # TODO(AU): Add units attribute on time-like variables # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/34
    #   Define units attribute here, or when putting in the values? How to make the values match the
    #   units, for the time-like coordinates mainly?
    time = CoordinateProperties(
        StandardDim.time,
        (("standard_name", "time"), ("long_name", "time"), ("axis", "T")),
    )
    station = CoordinateProperties(
        "station",
        # TODO(AU): Check location coordinate attributes and CF compliance # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/35
        #   Having cf_role: timeseries_id on this coordinate, and featureType: timeSeries on the
        #   full dataset, was copied from example fews netcdf files. However, it appears to not be
        #   fully in line with how these are supposed to be used, according to CF 1.6?
        (
            ("long_name", "station identification code"),
            ("cf_role", "timeseries_id"),
        ),
    )
    station_name = CoordinateProperties(
        "station_name",
        (("long_name", "station name"),),
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
    x = CoordinateProperties(
        "x",
        (
            ("standard_name", "projection_x_coordinate"),
            ("long_name", "x coordinate according to WGS 1984"),
            ("units", "degrees"),
            ("axis", "X"),
        ),
    )
    y = CoordinateProperties(
        "x",
        (
            ("standard_name", "projection_y_coordinate"),
            ("long_name", "y coordinate according to WGS 1984"),
            ("units", "degrees"),
            ("axis", "Y"),
        ),
    )
    z = CoordinateProperties(
        "z",
        (
            ("standard_name", "projection_z_coordinate"),
            ("long_name", "z coordinate according to WGS 1984"),
            ("units", "degrees"),
            ("axis", "Z"),
        ),
    )
    realization = CoordinateProperties(
        StandardDim.realization,
        (
            ("standard_name", "realization"),
            ("long_name", "Index of an ensemble member within an ensemble"),
            ("units", "1"),
        ),
    )
    forecast_reference_time = CoordinateProperties(
        StandardDim.forecast_reference_time,
        (
            ("standard_name", "forecast_reference_time"),
            ("long_name", "forecast_reference_time"),
        ),
    )
    forecast_period = CoordinateProperties(
        StandardDim.forecast_period,
        (
            ("standard_name", "forecast_period"),
            ("long_name", "forecast_period"),
        ),
    )
    source = CoordinateProperties(
        StandardDim.source,
        (
            ("standard_name", "source"),
            ("long_name", "source"),
        ),
    )
    variable = CoordinateProperties(
        StandardDim.variable,
        (("long_name", "simulation_or_observation_kind"),),
    )
    units = CoordinateProperties(
        "units",
        (("long_name", "units"),),
    )


class StandardAttribute:
    """List of attribute names on the main xarray.

    To avoid hardcoded strings in multiple places,
    have a single list with the names of known attributes.
    """

    # TODO(AU): Make similar to DataModelCoords, with both name and (default) value # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/17
    source = "source"
    timestep = "timestep"
    featuretype = "featureType"


def _set_version_info() -> tuple[str, str]:
    version = importlib_metadata.version("dpyverification")
    version_extra = ""

    # TODO(AU): Version numbering of our package # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/36
    #   See issue link for details
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
