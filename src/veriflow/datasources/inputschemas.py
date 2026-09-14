"""A collection of schemas for input data.

The pipeline ingests a number of predefined input data types as ``xr.Dataset`` instances. To
validate that the input data has the correct structure, we use Pydantic models in this module.

Using ``xr.Dataset.to_dict(data=False)`` returns a dictionary that can be used as input to a
Pydantic model. Each accepted data type has its own schema, built up of smaller sub-models.

For now, we validate
- Coordinates: name, dtype, dimensions
- Data variables: required dims, required attributes (notably ``units``), CF-compliant naming
- Dataset attributes (notably ``data_type``)
"""


# mypy: ignore-errors
# ruff: noqa: D101

from typing import Annotated, Literal, Self

import xarray as xr
from pydantic import AfterValidator, BaseModel, Field, RootModel, model_validator

from veriflow.constants import DataType, SpatialType, StandardDim
from veriflow.types import DataSpec

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


# ---------------------------------------------------------------------------
# Coordinate schemas
# ---------------------------------------------------------------------------


class HistoricalTimeCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.time}))]
    dtype: AllowedDTypeDateTime


class ForecastTimeCoord(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims({StandardDim.forecast_reference_time, StandardDim.lead_time}),
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


class LeadTimeCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.lead_time}))]
    dtype: AllowedDTypeTimeDelta


class RealizationCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.realization}))]
    dtype: AllowedDTypeInt


class ThresholdCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.threshold}))]


class BaseCoords(BaseModel):
    station: StationCoord
    station_name: StationCoord | None = None  # Optional station name coordinate
    # Point data is valid with either ``lat``/``lon`` (assumed EPSG:4326) or ``x``/``y`` plus a
    # ``crs`` dataset attribute. At least one of the two pairs must be present.
    lat: XYZCoord | None = None
    lon: XYZCoord | None = None
    x: XYZCoord | None = None
    y: XYZCoord | None = None
    z: XYZCoord | None = None

    @model_validator(mode="after")
    def validate_spatial_coords(self) -> Self:
        """Require either ``lat`` and ``lon``, or ``x`` and ``y`` to be present."""
        has_latlon = self.lat is not None and self.lon is not None
        has_xy = self.x is not None and self.y is not None
        if not has_latlon and not has_xy:
            msg = "Point data requires either 'lat' and 'lon', or 'x' and 'y' coordinates."
            raise ValueError(msg)
        return self


class BaseHistoricalCoords(BaseCoords):
    time: HistoricalTimeCoord


class SimulatedForecastSingleCoords(BaseCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    lead_time: LeadTimeCoord
    time: ForecastTimeCoord


class SimulatedForecastEnsembleCoords(BaseCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    lead_time: LeadTimeCoord
    realization: RealizationCoord
    time: ForecastTimeCoord


class SimulatedForecastProbabilisticCoords(BaseCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    lead_time: LeadTimeCoord
    threshold: ThresholdCoord
    time: ForecastTimeCoord


class ThresholdCoords(BaseModel):
    """The structure of a threshold dataset's coords."""

    station: StationCoord
    station_name: StationCoord | None = None  # Optional station name coordinate
    threshold: ThresholdCoord


# ---------------------------------------------------------------------------
# Gridded (spatial_type=gridded) coordinate schemas
#
# For gridded data, ``x`` and ``y`` are dimension coordinates (each with its own
# dimension) expressed in the dataset's CRS (see the required ``crs`` attribute).
# ``x``/``y`` plus the CRS are the canonical spatial datamodel; ``lat``/``lon`` are
# not stored and are only derived on demand (for a score or output).
# ---------------------------------------------------------------------------


class XDimCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.x}))]
    dtype: AllowedDTypeFloat


class YDimCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.y}))]
    dtype: AllowedDTypeFloat


class BaseGriddedCoords(BaseModel):
    x: XDimCoord
    y: YDimCoord


class GriddedHistoricalCoords(BaseGriddedCoords):
    time: HistoricalTimeCoord


class GriddedForecastSingleCoords(BaseGriddedCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    lead_time: LeadTimeCoord
    time: ForecastTimeCoord


class GriddedForecastEnsembleCoords(BaseGriddedCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    lead_time: LeadTimeCoord
    realization: RealizationCoord
    time: ForecastTimeCoord


# ---------------------------------------------------------------------------
# Data variable schemas
# ---------------------------------------------------------------------------


CFCompliantName = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="It is required that variable, dimension, attribute and group names "
        "begin with a letter and be composed of letters, digits, and underscores. "
        "(https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_naming_conventions)",
    ),
]


class DataVarAttrs(BaseModel):
    """Required attributes on a data variable."""

    units: Annotated[
        str,
        Field(
            min_length=1,
            description="Units of the data variable. Required for downstream interpretation and "
            "CF-compliant output.",
        ),
    ]

    model_config = {"extra": "allow"}


class HistoricalDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims({StandardDim.station, StandardDim.time}),
        ),
    ]
    attrs: DataVarAttrs


class SimulatedForecastSingleDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.station,
                    StandardDim.forecast_reference_time,
                    StandardDim.lead_time,
                },
            ),
        ),
    ]
    attrs: DataVarAttrs


class SimulatedForecastEnsembleDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.station,
                    StandardDim.forecast_reference_time,
                    StandardDim.lead_time,
                    StandardDim.realization,
                },
            ),
        ),
    ]
    attrs: DataVarAttrs


class SimulatedForecastProbabilisticDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.station,
                    StandardDim.forecast_reference_time,
                    StandardDim.lead_time,
                    StandardDim.threshold,
                },
            ),
        ),
    ]
    attrs: DataVarAttrs


class ThresholdDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims({StandardDim.station, StandardDim.threshold}),
        ),
    ]
    # Threshold variables don't strictly need units; allow any attrs
    attrs: dict | None = None


class GriddedHistoricalDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims({StandardDim.y, StandardDim.x, StandardDim.time}),
        ),
    ]
    attrs: DataVarAttrs


class GriddedForecastSingleDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.y,
                    StandardDim.x,
                    StandardDim.forecast_reference_time,
                    StandardDim.lead_time,
                },
            ),
        ),
    ]
    attrs: DataVarAttrs


class GriddedForecastEnsembleDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.y,
                    StandardDim.x,
                    StandardDim.forecast_reference_time,
                    StandardDim.lead_time,
                    StandardDim.realization,
                },
            ),
        ),
    ]
    attrs: DataVarAttrs


# ---------------------------------------------------------------------------
# Data variable collections (dict of CF-compliant name -> DataVar schema)
#
# Each ``RootModel`` validates that data variable names (the dict keys) are
# CF-compliant via the ``CFCompliantName`` constraint, and that each value
# matches the corresponding per-data-type ``*DataVar`` schema.
# ---------------------------------------------------------------------------


HistoricalDataVars = RootModel[dict[CFCompliantName, HistoricalDataVar]]
SimulatedForecastSingleDataVars = RootModel[dict[CFCompliantName, SimulatedForecastSingleDataVar]]
SimulatedForecastEnsembleDataVars = RootModel[
    dict[CFCompliantName, SimulatedForecastEnsembleDataVar]
]
SimulatedForecastProbabilisticDataVars = RootModel[
    dict[CFCompliantName, SimulatedForecastProbabilisticDataVar]
]
ThresholdDataVars = RootModel[dict[CFCompliantName, ThresholdDataVar]]
GriddedHistoricalDataVars = RootModel[dict[CFCompliantName, GriddedHistoricalDataVar]]
GriddedForecastSingleDataVars = RootModel[dict[CFCompliantName, GriddedForecastSingleDataVar]]
GriddedForecastEnsembleDataVars = RootModel[dict[CFCompliantName, GriddedForecastEnsembleDataVar]]


# ---------------------------------------------------------------------------
# Dataset-level attribute schemas
# ---------------------------------------------------------------------------


class BaseAttrs(BaseModel):
    source: str
    data_type: str
    spatial_type: SpatialType = SpatialType.point
    crs: str = "EPSG:4326"

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Top-level dataset schemas
# ---------------------------------------------------------------------------


class ObservedHistorical(BaseModel):
    coords: BaseHistoricalCoords
    data_vars: HistoricalDataVars
    attrs: BaseAttrs

    @model_validator(mode="after")
    def validate_dataset(self) -> "ObservedHistorical":
        """Validate that the dataset has at least one data variable."""
        if not self.data_vars.root or len(self.data_vars.root) == 0:
            msg = "Observed historical dataset must have at least one data variable."
            raise ValueError(msg)
        return self


class SimulatedHistorical(BaseModel):
    coords: BaseHistoricalCoords
    data_vars: HistoricalDataVars
    attrs: BaseAttrs


class SimulatedForecastSingle(BaseModel):
    coords: SimulatedForecastSingleCoords
    data_vars: SimulatedForecastSingleDataVars
    attrs: BaseAttrs


class SimulatedForecastEnsemble(BaseModel):
    coords: SimulatedForecastEnsembleCoords
    data_vars: SimulatedForecastEnsembleDataVars
    attrs: BaseAttrs


class SimulatedForecastProbabilistic(BaseModel):
    coords: SimulatedForecastProbabilisticCoords
    data_vars: SimulatedForecastProbabilisticDataVars
    attrs: BaseAttrs


class Thresholds(BaseModel):
    coords: ThresholdCoords
    data_vars: ThresholdDataVars
    attrs: BaseAttrs


class ObservedHistoricalGridded(BaseModel):
    coords: GriddedHistoricalCoords
    data_vars: GriddedHistoricalDataVars
    attrs: BaseAttrs


class SimulatedForecastSingleGridded(BaseModel):
    coords: GriddedForecastSingleCoords
    data_vars: GriddedForecastSingleDataVars
    attrs: BaseAttrs


class SimulatedForecastEnsembleGridded(BaseModel):
    coords: GriddedForecastEnsembleCoords
    data_vars: GriddedForecastEnsembleDataVars
    attrs: BaseAttrs


# All input schemas, keyed by the (data_type, spatial_type) pair that fully describes
# a dataset. ``spatial_type`` defaults to ``point`` so existing station-based data and
# configuration keep working unchanged.
INPUT_SCHEMAS: dict[DataSpec, BaseModel] = {
    (DataType.observed_historical, SpatialType.point): ObservedHistorical,
    (DataType.simulated_historical, SpatialType.point): SimulatedHistorical,
    (DataType.simulated_forecast_single, SpatialType.point): SimulatedForecastSingle,
    (DataType.simulated_forecast_ensemble, SpatialType.point): SimulatedForecastEnsemble,
    (DataType.simulated_forecast_probabilistic, SpatialType.point): SimulatedForecastProbabilistic,
    (DataType.threshold, SpatialType.point): Thresholds,
    (DataType.observed_historical, SpatialType.gridded): ObservedHistoricalGridded,
    (DataType.simulated_forecast_single, SpatialType.gridded): SimulatedForecastSingleGridded,
    (DataType.simulated_forecast_ensemble, SpatialType.gridded): SimulatedForecastEnsembleGridded,
}


def validate_input_data(dataset: xr.Dataset) -> None:
    """Validate an input ``xr.Dataset`` against its schema.

    The schema is determined from the ``data_type`` (temporal kind) and ``spatial_type``
    (spatial structure) attributes on the dataset. ``spatial_type`` defaults to ``point``
    when absent, and is backfilled onto ``dataset.attrs`` so the full data type is always
    retrievable downstream.
    """
    if not isinstance(dataset, xr.Dataset):
        msg = f"Expected an xarray Dataset. Got: {type(dataset)}"
        raise TypeError(msg)

    if "data_type" not in dataset.attrs:
        msg = "Input dataset is missing required 'data_type' attribute."
        raise ValueError(msg)

    data_type = DataType(dataset.attrs["data_type"])
    spatial_type = SpatialType(dataset.attrs.get("spatial_type", SpatialType.point))
    # Always persist spatial_type on the dataset so the full (temporal + spatial) data
    # type can be determined at any later stage.
    dataset.attrs["spatial_type"] = spatial_type
    # Guarantee a CRS on the dataset (defaulting to EPSG:4326) so downstream reprojection
    # can always rely on its presence without runtime checks.
    dataset.attrs.setdefault("crs", "EPSG:4326")

    schema_class = INPUT_SCHEMAS.get((data_type, spatial_type))
    if not schema_class:
        supported = sorted(f"({dt}, {st})" for dt, st in INPUT_SCHEMAS)
        msg = (
            f"No input schema defined for (data_type, spatial_type) = "
            f"({data_type}, {spatial_type}). Supported combinations: {', '.join(supported)}."
        )
        raise ValueError(msg)

    data_dict = dataset.to_dict(data=False)
    schema_class.model_validate(data_dict)
