"""A module for default implementation of preprocessor configurations."""

from datetime import datetime, timedelta
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
    origin: Annotated[
        Literal["epoch", "start", "start_day", "end", "end_day"] | datetime,
        Field(
            description="Anchor for the resampling intervals, passed to xarray.resample. Use "
            "'epoch', 'start', 'start_day', 'end', 'end_day', or an ISO datetime matching the "
            "time coordinate's timezone. Defaults to midnight on the first day ('start_day').",
        ),
    ] = "start_day"
    offset: Annotated[
        str | timedelta | None,
        Field(
            description="Duration added to the resampling origin, passed to xarray.resample "
            "(e.g. '7h' or '-1h'). Changes interval boundaries, not just output timestamps. "
            "Omit for no offset.",
        ),
    ] = None
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


class RollingWindowConfig(BasePreprocessorConfig):
    """Configuration for aggregating using a centered or aligned rolling window.

    This preprocessor computes rolling window statistics over the time dimension. For example,
    to compute the maximum value over a centered 6-hour window (3 hours before and after each
    timestamp), use ``window_size`` of 6 hours, ``method`` max and ``center`` true.

    When ``center`` is false, the window is right-aligned (aligned to the past), similar to
    moving averages or exponential smoothing in forecasting.

    Examples:
        - Centered 6h max: window_size=6h, method=max, center=true
        - Right-aligned 12h mean: window_size=12h, method=mean, center=false
    """

    preprocessor_adapter: Literal[PreprocessorKind.rolling_window]
    window_size: Annotated[
        TimeDelta,
        Field(description="The size of the rolling window (e.g. 6 hours)."),
    ]
    method: Annotated[
        TimeAggregationMethod,
        Field(description="The aggregation method applied to values in each window."),
    ]
    center: Annotated[
        bool,
        Field(
            description="If True, center the window (equal time before and after each timestamp). "
            "If False, right-align the window (past values only). Default is True.",
        ),
    ] = True
    min_count: Annotated[
        int | None,
        Field(
            ge=1,
            description="Optional minimum number of valid (non-missing) input values required in "
            "a window. Windows with fewer values become NaN. When omitted, all windows with at "
            "least one value are aggregated.",
        ),
    ] = None
