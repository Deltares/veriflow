"""A module for default implementation of scores."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, RootModel

from dpyverification.configuration.base import (
    BaseCategoricalScoreConfig,
    BaseEvent,
    BaseScoreConfig,
)
from dpyverification.constants import (
    ScoreKind,
    StandardDim,
    SupportedCategoricalScores,
    SupportedContinuousScore,
)


class EventOperator(Enum):
    """Event operators."""

    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL_TO = "greater_than_or_equal_to"
    LESS_THAN_OR_EQUAL_TO = "less_than_or_equal_to"


class ReduceDimsForecast(BaseModel):
    """The dimensions over which a forecast can be reduced."""

    reduce_dims: Annotated[
        list[
            Literal[
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.forecast_period,
            ]
        ],
        Field(default_factory=list),
    ]

    @property
    def preserve_dims(self) -> list[StandardDim]:
        """The dimensions to preserve."""
        return [
            k
            for k in [
                StandardDim.variable,
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.forecast_period,
            ]
            if k not in self.reduce_dims
        ]


class IdMap(RootModel[dict[str, dict[str, str]]]):
    """Mapping from internal IDs to external IDs per data source."""

    def get_external_to_internal_mapping(self, data_source: str) -> dict[str, str]:
        """Return external → internal mapping for this data source."""
        return {v[data_source]: k for k, v in self.root.items()}


class RankHistogramConfig(BaseScoreConfig, ReduceDimsForecast):
    """A rank histogram config element."""

    score_adapter: Literal[ScoreKind.rank_histogram]


class CrpsForEnsembleConfig(BaseScoreConfig, ReduceDimsForecast):
    """Configuration for CRPS for ensemble.

    For reference, see: See: https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_for_ensemble
    """

    score_adapter: Literal[ScoreKind.crps_for_ensemble]
    method: Annotated[
        Literal["ecdf", "fair"],
        Field(
            description=(
                "Method to compute the cumulative distribution function from an ensemble."
            ),
            default="ecdf",
        ),
    ]


class CrpsCDFConfig(BaseScoreConfig, ReduceDimsForecast):
    """Configuration for CRPS for CDF.

    For reference, see: https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_cdf
    """

    score_adapter: Literal[ScoreKind.crps_cdf]
    integration_method: Annotated[
        Literal["exact", "trapz"],
        Field(
            description="The method of integration. 'exact' computes the exact integral, "
            "'trapz' uses a trapezoidal rule and is an approximation of the CRPS.",
        ),
    ] = "exact"


class ContinuousScoresConfig(BaseScoreConfig, ReduceDimsForecast):
    """Configure multiple continuous scores."""

    score_adapter: Literal[ScoreKind.continuous_scores]
    scores: list[SupportedContinuousScore]


class ThresholdEvent(BaseEvent):
    """An event definition for a threshold."""

    threshold: Annotated[
        str,
        Field(description="Threshold id to use in event definition."),
    ]
    operator: Annotated[
        EventOperator,
        Field(description="The operator to use for creating the events."),
    ]


class CategoricalScoresConfig(BaseCategoricalScoreConfig, ReduceDimsForecast):
    """Config to compute categorical scores, based on an event definition."""

    score_adapter: Literal[ScoreKind.categorical_scores]
    scores: Annotated[
        list[SupportedCategoricalScores],
        Field(
            description="For reference, see: https://scores.readthedocs.io/en/stable/api.html#module-scores.categorical.",
        ),
    ]
    events: Annotated[
        list[ThresholdEvent],
        Field(
            description="A list of threshold event definitions. For each event, a categorical "
            "score will be computed. A threshold event is defined by a threshold and an operator. "
            "The threshold is a string that corresponds to a threshold id defined in the "
            "configuration. The operator defines how the threshold is applied to the data to "
            "create the event. For example, if the threshold is '10' and the operator is "
            "'greater_than', the event will be created by applying the operator to the data "
            "and the threshold, i.e. data > 10. ",
        ),
    ]
    return_contingency_table: Annotated[
        bool,
        Field(description="Whether to return the contingency table in the output."),
    ] = True
