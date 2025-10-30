"""Classes and functions to create a Config object from various types of configuration files."""

# The public interface
from dpyverification.configuration.base import BaseDatasourceConfig, Config, GeneralInfoConfig
from typing import Annotated

from pydantic import Field

from dpyverification.configuration.base import Config, GeneralInfoConfig
from dpyverification.configuration.default.datasinks import (
    CFCompliantNetCDFConfig,
    FewsNetCDFOutputConfig,
)
from dpyverification.configuration.default.datasources import (
    FewsNetCDFConfig,
    FewsWebserviceConfig,
    InternalDatasetConfig,
    ForecastRetrievalMethod,
    InternalDatasetConfig,
)
from dpyverification.configuration.default.scores import (
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    RankHistogramConfig,
    ReliabilityForEnsembleConfig,
)
from dpyverification.configuration.file import ConfigFile, ConfigKind
from dpyverification.configuration.utils import FewsWebserviceAuthConfig

DefaultDatasourceConfig = Annotated[
    FewsNetCDFConfig | FewsWebserviceConfig | InternalDatasetConfig,
    Field(discriminator="kind"),
]
DefaultScoreConfig = Annotated[
    CrpsCDFConfig | CrpsForEnsembleConfig | RankHistogramConfig,
    Field(discriminator="kind"),
]
DefaultSinkConfig = Annotated[
    CFCompliantNetCDFConfig,
    Field(discriminator="kind"),
]
