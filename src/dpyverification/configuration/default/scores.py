"""A module for default implementation of scores."""

import operator
from enum import Enum
from typing import Annotated, Literal

import xarray as xr
from pydantic import BaseModel, Field, RootModel
from scores.categorical import (  # type:ignore[import-untyped]
    BasicContingencyManager,
)

from dpyverification.configuration.config import BaseScoreConfig
from dpyverification.constants import ScoreKind, StandardDim, SupportedContinuousScore


class EventOperator(Enum):
    """Enumerator for even operators."""

    GREATER_THAN = operator.gt  # type:ignore[misc]
    LESS_THAN = operator.lt  # type:ignore[misc]
    GREATER_THAN_OR_EQUAL_TO = operator.ge  # type:ignore[misc]
    LESS_THAN_OR_EQUAL_TO = operator.le  # type:ignore[misc]


class SupportedCategoricalScores(Enum):
    """The supported categorical scores.

    Explicitly excluded scores
    - gilberts_skill_score              (identical to equitable_threat_score)
    - threat_score                      (identical to critical_success_index)
    - heidke_skill_score                (identical to cohens_kappa)
    - frequency_bias                    (identical to bias score)
    - fraction_correct                  (identical to accuracy)
    - hanssen_and_kuipers_discriminant  (identical to peirce_skill_score)
    - true_skill_statistic              (identical to peirce_skill_score)
    - yules_q                           (identical to odds_ratio_skill_score)
    - positive_predictive_value         (identical to precision)
    - success_ratio                     (identical to precision)
    - probability_of_detection          (identical to hit_rate)
    - true_positive_rate                (identical to hit rate)
    - sensitivity                       (identical to hit rate)
    - recall                            (identical to hit rate)
    - true_negative_rate                (identical to specificity)
    - probability of false detection    (identical to false alarm rate)
    """

    ACCURACY = "accuracy"
    BASE_RATE = "base_rate"
    BIAS_SCORE = "bias_score"
    COHENS_KAPPA = "cohens_kappa"
    CRITICAL_SUCCESS_INDEX = "critical_success_index"
    EQUITABLE_THREAT_SCORE = "equitable_threat_score"
    F1_SCORE = "f1_score"
    FALSE_ALARM_RATE = "false_alarm_rate"
    FALSE_ALARM_RATIO = "false_alarm_ratio"
    FORECAST_RATE = "forecast_rate"
    PEIRCE_SKILL_SCORE = "peirce_skill_score"
    HIT_RATE = "hit_rate"
    NEGATIVE_PREDICTIVE_VALUE = "negative_predictive_value"
    ODDS_RATIO = "odds_ratio"
    ODDS_RATIO_SKILL_SCORE = "odds_ratio_skill_score"
    PRECISION = "precision"
    SPECIFICITY = "specificity"
    SYMMETRIC_EXTREMAL_DEPENDENCE_INDEX = "symmetric_extremal_dependence_index"

    def __call__(self, contingency_manager: BasicContingencyManager) -> xr.DataArray:
        """Compute the score on the contingency manager."""
        method = getattr(contingency_manager, self.value)  # type:ignore[misc]
        return method()  # type:ignore[misc, no-any-return]


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
            description="Method to compute the cumulative distribution function from an ensemble.",
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


class ThresholdOperator(BaseModel):
    """An event definition."""

    threshold: Annotated[
        str,
        Field(description="Threshold id to use in event definition."),
    ]
    operator: Annotated[
        EventOperator,
        Field(description="The operator to use for creating the events."),
    ]


class CategoricalScoresConfig(BaseScoreConfig, ReduceDimsForecast):
    """Config to compute categorical scores, based on an event definition."""

    scores: Annotated[
        list[SupportedCategoricalScores],
        Field(
            description="For reference, see: https://scores.readthedocs.io/en/stable/api.html#module-scores.continuous.",
        ),
    ]
    events: Annotated[
        list[ThresholdOperator],
        Field(
            description="List of events. An event is a combination of a threshold and an operator. "
            "Events are used to compute the 2x2 contingency tables can categorical scores.",
        ),
    ]
    return_contingency_table: Annotated[
        bool,
        Field(description="Wether to return the contingency table in the output."),
    ] = True
