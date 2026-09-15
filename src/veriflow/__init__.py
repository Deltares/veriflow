"""
veriflow: a package to run a selection of verification functions on a dataset.

The data the functions apply to are observations and (a combination of multiple) forecasts.

"""

from .base import Base, BaseConfig
from .constants import (
    DataSinkKind,
    DataSourceKind,
    DataType,
    ScoreKind,
    SpatialType,
    StandardAttribute,
    StandardCoord,
    StandardDim,
)
from .datasources import (
    BaseDatasource,
    BaseDatasourceConfig,
    FewsNetCDF,
    FewsNetCDFConfig,
    FewsWebservice,
    FewsWebserviceConfig,
)
from .datatree import VeriflowAccessor, VeriflowDataTree
from .pipeline import run_pipeline
from .scores.base import BaseScore, BaseScoreConfig
from .scores.categorical import CategoricalScores, CategoricalScoresConfig
from .scores.continuous import ContinuousScores, ContinuousScoresConfig
from .scores.probabilistic import (
    CrpsCDF,
    CrpsCDFConfig,
    CrpsForEnsemble,
    CrpsForEnsembleConfig,
    RankHistogram,
    RankHistogramConfig,
)
from .scores.spatial import SALScore, SALScoreConfig
