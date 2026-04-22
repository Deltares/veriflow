"""The various verification scores that can be requested to be applied to the datamodel."""

from .base import BaseCategoricalScore, BaseScore, BaseScoreConfig
from .categorical import CategoricalScores, CategoricalScoresConfig
from .continuous import ContinuousScores, ContinuousScoresConfig
from .probabilistic import (
    CrpsCDF,
    CrpsCDFConfig,
    CrpsForEnsemble,
    CrpsForEnsembleConfig,
    RankHistogram,
    RankHistogramConfig,
)

DEFAULT_SCORES: list[type[BaseScore] | type[BaseCategoricalScore]] = [
    RankHistogram,
    CrpsForEnsemble,
    ContinuousScores,
    CategoricalScores,
]
