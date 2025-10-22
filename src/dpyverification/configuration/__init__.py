"""Classes and functions to create a Config object from various types of configuration files."""

# The public interface
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
    ForecastRetrievalMethod,
)
from dpyverification.configuration.default.scores import (
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    RankHistogramConfig,
)
from dpyverification.configuration.file import ConfigFile, ConfigKind
from dpyverification.configuration.utils import FewsWebserviceAuthConfig
