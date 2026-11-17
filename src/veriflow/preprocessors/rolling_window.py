"""Rolling window preprocessor: aggregate using a centered or aligned rolling window."""

from typing import cast

import pandas as pd
import xarray as xr

from veriflow.configuration.default.preprocessors import RollingWindowConfig
from veriflow.constants import FORECAST_DATA_TYPES, PreprocessorKind, StandardDim
from veriflow.datasources.inputschemas import validate_input_data
from veriflow.preprocessors.base import BasePreprocessor

__all__ = ["RollingWindow", "RollingWindowConfig"]


class RollingWindow(BasePreprocessor):
    """Aggregate a historical dataset using a centered or aligned rolling window.

    For example, compute the maximum value over a centered 6-hour window (3h before, 3h after)
    or an aligned window (e.g., past 6 hours). Only data variables with a ``time`` dimension are
    aggregated; all other variables are passed on unchanged.

    For forecast data types, the rolling window is applied along the ``lead_time`` dimension
    instead of ``time``.

    This preprocessor uses xarray's rolling window functionality with optional centering.
    """

    kind = PreprocessorKind.rolling_window
    config_class = RollingWindowConfig

    config: RollingWindowConfig

    def __init__(self, config: RollingWindowConfig) -> None:
        super().__init__(config)

    def preprocess(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply rolling window aggregation along the time or lead_time dimension."""
        if not dataset.data_vars:
            msg = f"Rolling window of source '{self.config.source_id}' found no data variables."
            raise ValueError(msg)

        # Determine which dimension to apply the rolling window to
        data_type = dataset.verification.data_type  # type:ignore[misc]
        is_forecast = data_type in FORECAST_DATA_TYPES
        window_dim = StandardDim.lead_time if is_forecast else StandardDim.time

        # Calculate the window size in terms of time steps
        window_td = pd.Timedelta(self.config.window_size.timedelta64)
        dim_coord = dataset[window_dim]

        # Estimate time step from data (assume uniform spacing)
        if len(dim_coord) < 2:
            msg = (
                f"Rolling window of source '{self.config.source_id}' requires at least 2 "
                f"{window_dim} steps to determine the resolution."
            )
            raise ValueError(msg)

        # For lead_time, convert to timedelta64 if needed
        if is_forecast and dim_coord.dtype == object:
            # lead_time might be stored as timedeltas
            time_step = pd.Timedelta(dim_coord[1].values - dim_coord[0].values)
        else:
            time_step = pd.Timedelta(dim_coord[1].values - dim_coord[0].values)

        window_size = int(window_td / time_step)

        if window_size < 1:
            msg = (
                f"Rolling window of source '{self.config.source_id}' is smaller than the "
                f"{window_dim} step ({window_td} < {time_step}). Window size must be at least one "
                f"{window_dim} step."
            )
            raise ValueError(msg)

        # Apply rolling window aggregation
        min_periods = self.config.min_count if self.config.min_count is not None else 1
        

        if hasattr(dataset, 'chunks') and dataset.chunks:
            # Build chunking strategy that works with rolling
            rechunk_dict = {}

            # Consolidate realization/ensemble to single chunk
            if StandardDim.realization in dataset.dims:
                rechunk_dict[StandardDim.realization] = -1
                dataset = dataset.chunk(rechunk_dict)
        
        aggregated = dataset.copy()

        for var_name in dataset.data_vars:
            if window_dim not in dataset[var_name].dims:
                # Skip variables without the target dimension
                continue

            # Apply rolling window with dynamic dimension name
            rolling_kwargs = {
                window_dim: window_size,
                'center': self.config.center,
                'min_periods': min_periods,
            }
            rolling_result = getattr(
                dataset[var_name].rolling(**rolling_kwargs),
                self.config.method.value,
            )()

            # Rename variable by appending the method name
            new_var_name = f'{var_name}_{self.config.method.value}'
            aggregated[new_var_name] = rolling_result
            aggregated[new_var_name].attrs = {
                **dataset[var_name].attrs,  # type:ignore[misc]
                "cell_methods": (
                    f"{window_dim}: {self.config.method.value} "
                    f"({window_td}, center={self.config.center})"
                ),
            }
            
            # Remove the original variable if it was modified
            if new_var_name != var_name and var_name in aggregated.data_vars:
                aggregated = aggregated.drop_vars(var_name)

        aggregated[window_dim].attrs = dataset[window_dim].attrs
        aggregated.attrs = dict(dataset.attrs)  # type:ignore[misc]
        aggregated = aggregated.chunk({StandardDim.realization: -1})
        validate_input_data(aggregated)
        return aggregated
