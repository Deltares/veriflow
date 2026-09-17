"""The various verification scores that can be requested to be applied to the datatree."""

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
from .spatial import SALScore, SALScoreConfig

DEFAULT_SCORES: list[type[BaseScore] | type[BaseCategoricalScore]] = [
    RankHistogram,
    CrpsForEnsemble,
    ContinuousScores,
    CategoricalScores,
    SALScore,
]
