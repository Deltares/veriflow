"""A collection of schemas for input data.

The pipeline in the package ingests a number of predefined data inputs. To validate
that the input data has the correct structure, we use Pydantic models in this module.
For now, input data are always an instance of xarray.Dataset.

Using the xarray.Dataset.to_dict(data=False) method on the provided Dataset instances
returns a dictonary which can be used as input into a Pydantic model. Each of the
accepted input datasets has it's own dedicated schema, built up of smaller sub-models.
This allows us to re-use much of the code and structure in this module, which keeps
this module readable and understandable.

xarray.Dataset.to_dict(data=False), returns a dictonary with the structure of the datset.
Because we use the option data=False, we only receive the dtype of the underlying data
array, which we can use for testing.

For now, we validate
- Dimensions: name and data type
- Coordinates: name, datatype, dimensions

On the following input datasets:
- Observations
- Simulations (with dimensions: time, forecast_reference_time, stations, optional[realization])
- Simulations (with dimensions: time, forecast_period, stations, optional[realization])

We do not yet validate:
- Attributes
"""

# mypy: ignore-errors
# ruff: noqa: D100, D101, D102, D103, D104, D105, D106, D107

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field

AllowedDTypeInt = Literal["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
AllowedDTypeFloat = Literal["float16", "float32", "float64"]
AllowedDTypeDateTime = Literal["datetime64[ns]"]
AllowedDTypeTimeDelta = Literal["timedelta64[ns]"]


def check_tuple(
    required: set[str],
    optional: set[str] | None = None,
) -> callable:
    """Check a if a tuple has correct elements in Pydantic AfterValidator."""
    allowed = required | optional if optional else required

    def validator(value: tuple[str, ...]) -> tuple[str, ...]:
        value_set = set(value)

        # Check for missing required
        missing = required - value_set
        if missing:
            msg = f"Missing required dims: {missing}"
            raise ValueError(msg)

        # Check for disallowed dims
        disallowed = value_set - allowed
        if disallowed:
            msg = f"Invalid dims: {disallowed}. Allowed: {allowed}"
            raise ValueError(msg)

        return value

    return validator


class SharedDims(BaseModel):
    time: int
    stations: int


class ObsDims(SharedDims):
    pass


class TimeCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_tuple({"time"}))]
    dtype: AllowedDTypeDateTime


class ForecastReferenceTimeCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_tuple({"forecast_reference_time"}))]
    dtype: AllowedDTypeDateTime


class StationsCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_tuple({"stations"}))]


class XYZCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_tuple({"stations"}))]
    dtype: AllowedDTypeFloat


class SharedCoords(BaseModel):
    time: TimeCoord
    stations: StationsCoord
    x: XYZCoord | None
    y: XYZCoord | None
    z: XYZCoord | None


class ObsCoords(SharedCoords):
    pass


class XarrayObservationsDataArray(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_tuple({"time", "stations"}))]
    dtype: AllowedDTypeFloat


ValidVarName = Annotated[str, Field(pattern=r"[a-zA-Z_]*")]


class XarrayDatasetObservations(BaseModel):
    dims: ObsDims
    coords: ObsCoords
    data_vars: dict[
        ValidVarName,
        XarrayObservationsDataArray,
    ]


class RealizationCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_tuple({"realization"}))]
    dtype: AllowedDTypeInt


class XarrayDatasetSimByForecastReferenceTimeDims(SharedDims):
    forecast_reference_time: int
    realization: int | None


class XarrayDatasetSimByForecastReferenceTimeCoords(SharedCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    realization: RealizationCoord | None  # Optional to handle ensemble and deterministic forecasts


class XarrayDataArraySimulationsByForecastReferenceTime(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_tuple(
                required={"time", "stations", "forecast_reference_time"},
                optional={"realization"},
            ),
        ),
    ]


class XarrayDatasetSimulationsByForecastReferenceTime(BaseModel):
    dims: XarrayDatasetSimByForecastReferenceTimeDims
    coords: XarrayDatasetSimByForecastReferenceTimeCoords
    data_vars: dict[
        ValidVarName,
        XarrayDataArraySimulationsByForecastReferenceTime,
    ]


class XarrayDatasetSimByForecastPeriodDimsDims(SharedDims):
    forecast_period: int
    realization: int | None


class ForecastPeriodCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_tuple({"forecast_period"}))]
    dtype: AllowedDTypeTimeDelta


class XarrayDatasetSimByForecastPeriodDimsCoords(SharedCoords):
    forecast_period: ForecastPeriodCoord
    realization: RealizationCoord | None = (
        None  # Optional to handle ensemble and deterministic forecasts
    )


class XarrayDataArraySimulationsByForecastPeriod(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_tuple(
                required={"time", "stations", "forecast_period"},
                optional={"realization"},
            ),
        ),
    ]


class XarrayDatasetSimulationsByForecastPeriod(BaseModel):
    dims: XarrayDatasetSimByForecastPeriodDimsDims
    coords: XarrayDatasetSimByForecastPeriodDimsCoords
    data_vars: dict[
        ValidVarName,
        XarrayDataArraySimulationsByForecastPeriod,
    ]
