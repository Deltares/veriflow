"""A module for default implementation of preprocessor configurations."""

from typing import Annotated, Literal

from pydantic import Field

from veriflow.configuration.base import BasePreprocessorConfig
from veriflow.configuration.utils import TimeDelta
from veriflow.constants import PreprocessorKind, TimeAggregationMethod


class TimeAggregationConfig(BasePreprocessorConfig):
    """Configuration for aggregating a historical dataset to a coarser time step.

    Wraps ``xarray`` resampling (``Dataset.resample(time=...)``). For example, to compute hourly
    values as the mean over the past 60 minutes from 10-minute data, use ``frequency`` of 1 hour,
    ``method`` mean and (the defaults) ``closed`` and ``label`` right.
    """

    preprocessor_adapter: Literal[PreprocessorKind.time_aggregation]
    frequency: Annotated[
        TimeDelta,
        Field(description="The output time step, as a value and a unit (e.g. 1 hour)."),
    ]
    method: Annotated[
        TimeAggregationMethod,
        Field(description="The aggregation method applied to the values in each output interval."),
    ]
    closed: Annotated[
        Literal["left", "right"],
        Field(
            description="Which side of the output interval is closed. With 'right' an interval "
            "covers (t - frequency, t], so a timestamp aggregates the values over the past "
            "'frequency'. With 'left' an interval covers [t, t + frequency).",
        ),
    ] = "right"
    label: Annotated[
        Literal["left", "right"],
        Field(
            description="Which edge of the output interval is used as the timestamp of the "
            "aggregated value.",
        ),
    ] = "right"
    min_count: Annotated[
        int | None,
        Field(
            ge=1,
            description="Optional minimum number of valid (non-missing) input values required in "
            "an output interval. Intervals with fewer values (e.g. incomplete intervals at the "
            "start or end of the data) become NaN. When omitted, all intervals with at least one "
            "value are aggregated.",
        ),
    ] = None
