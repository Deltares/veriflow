"""Module for preprocessors in the veriflow framework."""

from .base import BasePreprocessor
from .rolling_window import RollingWindow
from .time_aggregation import TimeAggregation

DEFAULT_PREPROCESSORS: list[type[BasePreprocessor]] = [TimeAggregation, RollingWindow]
