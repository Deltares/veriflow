"""Categorical scores, based on a 2x2 contingency table."""

import operator
from collections.abc import Callable
from enum import StrEnum
from typing import ClassVar

import xarray as xr
from scores.categorical import (  # type:ignore[import-untyped]
    BasicContingencyManager,
    BinaryContingencyManager,
)

from dpyverification.configuration.default.scores import (
    BaseEvent,
    CategoricalScoresConfig,
    EventOperator,
    ThresholdEvent,
)
from dpyverification.constants import DataType, SupportedCategoricalScores
from dpyverification.scores.base import BaseCategoricalScore


def get_categorical_score(score_name: SupportedCategoricalScores) -> type:
    """Get a categorical score from the scores package."""
    return getattr(BasicContingencyManager, score_name.value)  # type:ignore[no-any-return, misc]


def get_event_operator(
    operator_name: EventOperator,
) -> Callable[[xr.DataArray, xr.DataArray], xr.DataArray]:
    """Get an event operator function based on the operator name."""
    if operator_name == EventOperator.GREATER_THAN:
        return operator.gt  # type:ignore[misc]
    if operator_name == EventOperator.LESS_THAN:
        return operator.lt  # type:ignore[misc]
    if operator_name == EventOperator.GREATER_THAN_OR_EQUAL_TO:
        return operator.ge  # type:ignore[misc]
    if operator_name == EventOperator.LESS_THAN_OR_EQUAL_TO:
        return operator.le  # type:ignore[misc]
    msg = f"Unsupported operator: {operator_name}"  # type:ignore[unreachable] # runtime check
    raise ValueError(msg)


class CategoricalScoreDim(StrEnum):
    """Names of dimensions added when computing a categorical score."""

    EVENT_THRESHOLD = "event_threshold"
    EVENT_OPERATOR = "event_operator"


def create_binary_array(
    data: xr.DataArray,
    thresholds: xr.DataArray,
    operator: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
) -> xr.DataArray:
    """Given data and thresholds, compute the binary events."""
    # Align along dimension station
    data_aligned, thresholds_aligned = xr.align(data, thresholds, join="inner")
    result = operator(data_aligned, thresholds_aligned)
    if isinstance(result, xr.DataArray):  # type:ignore[misc]
        return result
    msg = "Failed to create a binary xr.DataArray based on data and thresholds."  # type:ignore[unreachable] # runtime check
    raise ValueError(msg)


def set_event_coordinates_on_result(
    data_array: xr.Dataset,
    threshold: str,
    operator: EventOperator,
) -> xr.Dataset:
    """Set coordinates on data array to represent the event for which a score was computed."""
    data_array = data_array.expand_dims(
        {CategoricalScoreDim.EVENT_THRESHOLD: 1, CategoricalScoreDim.EVENT_OPERATOR: 1},
    )
    return data_array.assign_coords(
        {  # type:ignore[misc]
            CategoricalScoreDim.EVENT_OPERATOR: [operator.name],  # type:ignore[misc]
            CategoricalScoreDim.EVENT_THRESHOLD: [threshold],  # type:ignore[misc]
        },
    )


class CategoricalScores(BaseCategoricalScore):
    """Implementation for CRPS for probabilistic forecasts, expressed as cdf."""

    kind = "categorical_scores"
    config_class = CategoricalScoresConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.simulated_forecast_single,
    }

    def __init__(self, config: CategoricalScoresConfig) -> None:
        self.config: CategoricalScoresConfig = config

    def compute_score_for_single_event(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
        thresholds: xr.DataArray,
        event: BaseEvent,
    ) -> xr.Dataset | xr.DataArray:
        """Compute any number of categorical scores for a single event."""
        if not isinstance(event, ThresholdEvent):
            msg = f"Unsupported event type: {type(event)}. Expected a ThresholdEvent."
            raise TypeError(msg)

        operator_func = get_event_operator(event.operator)
        obs_events = create_binary_array(
            obs,
            thresholds=thresholds,
            operator=operator_func,
        )
        sim_events = create_binary_array(
            sim,
            thresholds=thresholds,
            operator=operator_func,
        )
        binary_contingency_manager = BinaryContingencyManager(  # type:ignore[misc]
            fcst_events=sim_events,
            obs_events=obs_events,
        )
        basic_contingency_manager = binary_contingency_manager.transform(  # type:ignore[misc]
            preserve_dims=self.config.preserve_dims,
        )
        scores = []
        for score in self.config.scores:
            score_func = get_categorical_score(score)
            score_array = score_func(basic_contingency_manager)  # type:ignore[misc]
            score_array.name = str(score.value)  # type:ignore[misc]
            scores.append(score_array)  # type:ignore[misc]

        if self.config.return_contingency_table is True:
            table: xr.DataArray = basic_contingency_manager.get_table()  # type:ignore[misc]
            table.name = "contingency_table"
            scores.append(table)  # type:ignore[misc]

        merged_scores: xr.Dataset = xr.merge(scores)  # type:ignore[misc, assignment]
        return set_event_coordinates_on_result(
            merged_scores,
            threshold=event.threshold,
            operator=event.operator,
        )
