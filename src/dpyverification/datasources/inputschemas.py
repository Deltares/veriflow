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

# TODO(jb): remove mypy ignore errors  # noqa: FIX002, TD003

# mypy: ignore-errors
# ruff: noqa: D100, D101, D102, D103, D104, D105, D106, D107

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field

from dpyverification.constants import StandardDim, TimeseriesKind

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


class TimeCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.time}))]
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


class BaseCoords(BaseModel):
    time: TimeCoord
    station: StationCoord
    station_name: StationCoord | None = None  # Optional station name coordinate
    variable: VariableCoord
    units: UnitsCoord
    lat: XYZCoord  # Always required lat, lon
    lon: XYZCoord
    x: XYZCoord | None = None  # Optional x, y, z
    y: XYZCoord | None = None
    z: XYZCoord | None = None


CFCompliantName = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description=(
            "It is required that variable, dimension, attribute and group names",
            "begin with a letter and be composed of letters, digits, and underscores.",
            "(https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_naming_conventions)",
        ),
    ),
]


class RealizationCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.realization}))]
    dtype: AllowedDTypeInt


class ForecastPeriodCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.forecast_period}))]
    dtype: AllowedDTypeTimeDelta


class BaseSimulationCoords(BaseCoords):
    forecast_period: ForecastPeriodCoord
    realization: RealizationCoord


class BaseAttrs(BaseModel):
    timeseries_kind: str


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
    coords: BaseCoords
    attrs: BaseAttrs


# Below, the final allowed input data structures
class ObservedHistorical(Base):
    pass


class SimulatedHistorical(Base):
    pass


class SimulatedForecastSingleCoords(BaseCoords):
    forecast_period: ForecastPeriodCoord


class SimulatedForecastSingle(Base):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.variable,
                    StandardDim.time,
                    StandardDim.station,
                    StandardDim.forecast_period,
                },
            ),
        ),
    ]
    coords: SimulatedForecastSingleCoords


class SimulatedForecastEnsembleCoords(BaseCoords):
    realization: RealizationCoord
    forecast_period: ForecastPeriodCoord


class SimulatedForecastEnsemble(Base):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.variable,
                    StandardDim.time,
                    StandardDim.station,
                    StandardDim.forecast_period,
                    StandardDim.realization,
                },
            ),
        ),
    ]
    coords: SimulatedForecastEnsembleCoords


class SimulatedForecastProbabilisticCoords(BaseCoords):
    forecast_period: ForecastPeriodCoord


class SimulatedForecastProbabilistic(Base):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.variable,
                    StandardDim.time,
                    StandardDim.station,
                    StandardDim.forecast_period,
                    StandardDim.realization,
                },
            ),
        ),
    ]
    coords: SimulatedForecastProbabilisticCoords


# All input schemas, keyed by the corresponding timeseries kind
input_schemas: dict[TimeseriesKind, BaseModel] = {
    TimeseriesKind.observed_historical: ObservedHistorical,
    TimeseriesKind.simulated_historical: SimulatedHistorical,
    TimeseriesKind.simulated_forecast_single: SimulatedForecastSingle,
    TimeseriesKind.simulated_forecast_ensemble: SimulatedForecastEnsemble,
    TimeseriesKind.simulated_forecast_probabilistic: SimulatedForecastProbabilistic,
}
