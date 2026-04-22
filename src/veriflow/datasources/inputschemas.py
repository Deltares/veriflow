"""A collection of schemas for input data.

The pipeline in the package ingests a number of predefined data inputs. To validate
that the input data has the correct structure, we use Pydantic models in this module.

Using the xarray.Dataset.to_dict(data=False) method on the provided Dataset instances
returns a dictionary which can be used as input into a Pydantic model. Each of the
accepted input datasets has its own dedicated schema, built up of smaller sub-models.
This allows us to re-use much of the code and structure in this module, which keeps
this module readable and understandable.

For now, we validate
- Dimensions: name and data type
- Coordinates: name, datatype, dimensions
- Variables: names, coords, dims
- Attributes

On the following input datasets:
- Observations
- Simulations
"""


# mypy: ignore-errors
# ruff: noqa: D101

from typing import Annotated, Literal

import xarray as xr
from pydantic import AfterValidator, BaseModel, Field

from veriflow.constants import DataType, StandardDim

AllowedDTypeInt = Literal["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
AllowedDTypeFloat = Literal["float16", "float32", "float64"]
AllowedDTypeDateTime = Literal["datetime64[ns]"]
AllowedDTypeTimeDelta = Literal["timedelta64[ns]"]


def check_dims(
    required: set[str],
    optional: set[str] | None = None,
) -> callable:
    """Check a if a tuple contains the expected dimensions. Used in a Pydantic AfterValidator."""
    allowed = required | optional if optional else required

    def validator(value: tuple[str, ...]) -> tuple[str, ...]:
        value_set = set(value)

        # Check for missing required
        missing = required - value_set
        if len(missing) > 0:
            msg = f"Missing required dims: {', '.join(missing)}"
            raise ValueError(msg)

        # Check for disallowed dims
        disallowed = value_set - allowed
        if len(disallowed) > 0:
            msg = f"Invalid dims: {disallowed}. Allowed: {', '.join(allowed)}"
            raise ValueError(msg)

        return value

    return validator


class HistoricalTimeCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.time}))]
    dtype: AllowedDTypeDateTime


class ForecastTimeCoord(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims({StandardDim.forecast_reference_time, StandardDim.forecast_period}),
        ),
    ]
    dtype: AllowedDTypeDateTime


class ForecastReferenceTimeCoord(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(check_dims({StandardDim.forecast_reference_time})),
    ]
    dtype: AllowedDTypeDateTime


class StationCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.station}))]


class XYZCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.station}))]
    dtype: AllowedDTypeFloat


class VariableCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.variable}))]


class UnitsCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.variable}))]


class ForecastPeriodCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.forecast_period}))]
    dtype: AllowedDTypeTimeDelta


class RealizationCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.realization}))]
    dtype: AllowedDTypeInt


class ThresholdCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.threshold}))]


class BaseCoords(BaseModel):
    station: StationCoord
    station_name: StationCoord | None = None  # Optional station name coordinate
    variable: VariableCoord
    units: UnitsCoord
    lat: XYZCoord  # Always required lat, lon
    lon: XYZCoord
    x: XYZCoord | None = None  # Optional x, y, z
    y: XYZCoord | None = None
    z: XYZCoord | None = None


class BaseHistoricalCoords(BaseCoords):
    time: HistoricalTimeCoord


class SimulatedForecastSingleCoords(BaseCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    forecast_period: ForecastPeriodCoord
    time: ForecastTimeCoord


class SimulatedForecastEnsembleCoords(BaseCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    forecast_period: ForecastPeriodCoord
    realization: RealizationCoord
    time: ForecastTimeCoord


class SimulatedForecastProbabilisticCoords(BaseCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    forecast_period: ForecastPeriodCoord
    threshold: ThresholdCoord
    time: ForecastTimeCoord


class ThresholdCoords(BaseModel):
    """The structure of a threshold array."""

    station: StationCoord
    station_name: StationCoord | None = None  # Optional station name coordinate
    variable: VariableCoord
    threshold: ThresholdCoord


CFCompliantName = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="It is required that variable, dimension, attribute and group names"
        "begin with a letter and be composed of letters, digits, and underscores."
        "(https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_naming_conventions)",
    ),
]


class BaseAttrs(BaseModel):
    data_type: str


class Base(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.variable,
                    StandardDim.time,
                    StandardDim.station,
                },
            ),
        ),
    ]
    coords: BaseHistoricalCoords
    attrs: BaseAttrs


# Below, the final allowed input data structures
class ObservedHistorical(Base):
    pass


class SimulatedHistorical(Base):
    pass


class SimulatedForecastSingle(Base):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.variable,
                    StandardDim.forecast_reference_time,
                    StandardDim.forecast_period,
                    StandardDim.station,
                },
            ),
        ),
        Field(
            description="Tuple of dimensions. Must contain: variable, forecast_reference_time, "
            "forecast_period and station.",
        ),
    ]
    coords: SimulatedForecastSingleCoords


class SimulatedForecastEnsemble(Base):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.variable,
                    StandardDim.forecast_reference_time,
                    StandardDim.forecast_period,
                    StandardDim.station,
                    StandardDim.realization,
                },
            ),
        ),
    ]
    coords: SimulatedForecastEnsembleCoords


class SimulatedForecastProbabilistic(Base):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.variable,
                    StandardDim.forecast_reference_time,
                    StandardDim.forecast_period,
                    StandardDim.station,
                    StandardDim.threshold,
                },
            ),
        ),
    ]
    coords: SimulatedForecastProbabilisticCoords


class Thresholds(Base):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.variable,
                    StandardDim.station,
                    StandardDim.threshold,
                },
            ),
        ),
    ]
    coords: ThresholdCoords


# All input schemas, keyed by the corresponding data type
INPUT_SCHEMAS: dict[DataType, BaseModel] = {
    DataType.observed_historical: ObservedHistorical,
    DataType.simulated_historical: SimulatedHistorical,
    DataType.simulated_forecast_single: SimulatedForecastSingle,
    DataType.simulated_forecast_ensemble: SimulatedForecastEnsemble,
    DataType.simulated_forecast_probabilistic: SimulatedForecastProbabilistic,
    DataType.threshold: Thresholds,
}


def validate_input_data(data_array: xr.DataArray) -> BaseModel:
    """Validate input data against the expected schema for the given data type.

    The data type is determined from the 'data_type' attribute of the provided xarray DataArray.
    """
    if not isinstance(data_array, xr.DataArray):
        msg = f"Expected an xarray DataArray. Got: {type(data_array)}"
        raise TypeError(msg)

    if "data_type" not in data_array.attrs:
        msg = "Input data array is missing required 'data_type' attribute."
        raise ValueError(msg)

    data_type = data_array.attrs["data_type"]
    schema_class = INPUT_SCHEMAS.get(data_type)
    if not schema_class:
        msg = f"No input schema defined for data type: {data_type}"
        raise ValueError(msg)

    # Convert the xarray DataArray to a dictionary that can be used as input for the Pydantic model
    data_dict = data_array.to_dict(data=False)

    # Validate the data against the schema
    schema_class.model_validate(data_dict)
