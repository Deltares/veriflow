"""Collection of constant values definitions.

Both single values, and enum classes.
These values are can be used throughout veriflow.
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
    CSV = "csv"
    NETCDF = "netcdf"


class DataSinkKind(StrEnum):
    """Enumeration of the supported datasink types."""

    fews_netcdf = "fewsnetcdf"
    cf_compliant_netcdf = "cf_compliant_netcdf"


@unique
class DataType(StrEnum):
    """Input timeseries data kind."""

    observed_historical = "observed_historical"
    simulated_historical = "simulated_historical"
    simulated_forecast_single = "simulated_forecast_single"
    simulated_forecast_ensemble = "simulated_forecast_ensemble"
    simulated_forecast_probabilistic = "simulated_forecast_probabilistic"
    threshold = "threshold"


FORECAST_DATA_TYPES = (
    DataType.simulated_forecast_single,
    DataType.simulated_forecast_ensemble,
    DataType.simulated_forecast_probabilistic,
)
HISTORICAL_DATA_TYPES = (
    DataType.observed_historical,
    DataType.simulated_historical,
)


@unique
class ScoreKind(StrEnum):
    """Enumeration of the implemented verification scores."""

    rank_histogram = "rank_histogram"
    crps_for_ensemble = "crps_for_ensemble"
    crps_cdf = "crps_cdf"
    continuous_scores = "continuous_scores"
    categorical_scores = "categorical_scores"


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


class SupportedCategoricalScores(StrEnum):
    """The supported categorical scores.

    Explicitly excluded scores
    - gilberts_skill_score              (identical to equitable_threat_score)
    - threat_score                      (identical to critical_success_index)
    - heidke_skill_score                (identical to cohens_kappa)
    - frequency_bias                    (identical to bias score)
    - fraction_correct                  (identical to accuracy)
    - hanssen_and_kuipers_discriminant  (identical to peirce_skill_score)
    - true_skill_statistic              (identical to peirce_skill_score)
    - yules_q                           (identical to odds_ratio_skill_score)
    - positive_predictive_value         (identical to precision)
    - success_ratio                     (identical to precision)
    - probability_of_detection          (identical to hit_rate)
    - true_positive_rate                (identical to hit rate)
    - sensitivity                       (identical to hit rate)
    - recall                            (identical to hit rate)
    - true_negative_rate                (identical to specificity)
    - probability of false detection    (identical to false alarm rate)
    """

    accuracy = "accuracy"
    base_rate = "base_rate"
    bias_score = "bias_score"
    cohens_kappa = "cohens_kappa"
    critical_success_index = "critical_success_index"
    equitable_threat_score = "equitable_threat_score"
    f1_score = "f1_score"
    false_alarm_rate = "false_alarm_rate"
    false_alarm_ratio = "false_alarm_ratio"
    forecast_rate = "forecast_rate"
    peirce_skill_score = "peirce_skill_score"
    hit_rate = "hit_rate"
    negative_predictive_value = "negative_predictive_value"
    odds_ratio = "odds_ratio"
    odds_ratio_skill_score = "odds_ratio_skill_score"
    precision = "precision"
    specificity = "specificity"
    symmetric_extremal_dependence_index = "symmetric_extremal_dependence_index"


@unique
class TimeUnits(StrEnum):
    """Time unit strings, compatible with numpy datetime64 and timedelta64."""

    year = "Y"
    month = "M"
    week = "W"
    day = "D"
    hour = "h"
    minute = "m"
    second = "s"


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
    #   https://github.com/Deltares/veriflow/issues/34
    #   Define units attribute here, or when putting in the values? How to make the values match the
    #   units, for the time-like coordinates mainly?
    time = CoordinateProperties(
        StandardDim.time,
        (("standard_name", "time"), ("long_name", "time"), ("axis", "T")),
    )
    station = CoordinateProperties(
        "station",
        # TODO(AU): Check location coordinate attributes and CF compliance # noqa: FIX002
        #   https://github.com/Deltares/veriflow/issues/35
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
    #   https://github.com/Deltares/veriflow/issues/17
    source = "source"
    timestep = "timestep"
    featuretype = "featureType"


def _set_version_info() -> tuple[str, str]:
    version = importlib_metadata.version("veriflow")
    version_extra = ""

    # TODO(AU): Version numbering of our package # noqa: FIX002
    #   https://github.com/Deltares/veriflow/issues/36
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
        origin = importlib_metadata.distribution("veriflow").origin  # type: ignore[misc] # Since not sure what will be returned, there is the long if statement
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


NAME = "veriflow"
VERSION, VERSION_FULL = _set_version_info()
