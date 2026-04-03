"""The various verification scores that can be requested to be applied to the datamodel."""

from .base import BaseCategoricalScore, BaseScore
from .categorical import CategoricalScores
from .continuous import ContinuousScores
from .probabilistic import CrpsForEnsemble, RankHistogram

DEFAULT_SCORES: list[type[BaseScore] | type[BaseCategoricalScore]] = [
    RankHistogram,
    CrpsForEnsemble,
    ContinuousScores,
    CategoricalScores,
]
