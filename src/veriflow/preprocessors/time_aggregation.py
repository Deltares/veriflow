"""Time aggregation preprocessor: aggregate a historical dataset to a coarser time step."""

from typing import cast

import pandas as pd
import xarray as xr

from veriflow.configuration.default.preprocessors import TimeAggregationConfig
from veriflow.constants import PreprocessorKind, StandardDim
from veriflow.datasources.inputschemas import validate_input_data
from veriflow.preprocessors.base import BasePreprocessor

__all__ = ["TimeAggregation", "TimeAggregationConfig"]


class TimeAggregation(BasePreprocessor):
    """Aggregate a historical dataset along ``time`` using xarray resampling.

    Examples are 10-minute observations to hourly means over the past 60 minutes, or hourly
    observations to daily maxima. Only data variables with a ``time`` dimension are aggregated;
    all other variables are passed on unchanged.
    """

    kind = PreprocessorKind.time_aggregation
    config_class = TimeAggregationConfig

    config: TimeAggregationConfig

    def __init__(self, config: TimeAggregationConfig) -> None:
        super().__init__(config)

    def preprocess(self, dataset: xr.Dataset) -> xr.Dataset:
        """Resample the dataset along ``time`` to the configured time step."""
        if not dataset.verification.is_historical:  # type:ignore[misc]
            msg = (
                f"Time aggregation of source '{self.config.source_id}' is only supported for "
                f"historical data types, but the data type is "
                f"'{dataset.verification.data_type}'."  # type:ignore[misc]
            )
            raise ValueError(msg)

        if not dataset.data_vars:
            msg = f"Time aggregation of source '{self.config.source_id}' found no data variables."
            raise ValueError(msg)

        resampler = dataset.resample(
            time=pd.Timedelta(self.config.frequency.timedelta64),
            closed=self.config.closed,
            label=self.config.label,
        )
        aggregated = cast(
            "xr.Dataset",
            getattr(resampler, self.config.method.value)(keep_attrs=True),
        )

        # Empty intervals must not yield a value. For example, an empty sum would otherwise be 0.
        min_count = self.config.min_count if self.config.min_count is not None else 1
        aggregated = aggregated.where(resampler.count() >= min_count)

        # Every historical data variable has a time dimension (guaranteed by the input schema)
        for name in dataset.data_vars:
            aggregated[name] = aggregated[name].transpose(*dataset[name].dims)
            aggregated[name].attrs = {
                **dataset[name].attrs,  # type:ignore[misc]
                "cell_methods": f"{StandardDim.time}: {self.config.method.value}",
            }

        aggregated[StandardDim.time].attrs = dataset[StandardDim.time].attrs
        aggregated.attrs = dict(dataset.attrs)  # type:ignore[misc]
        validate_input_data(aggregated)
        return aggregated
