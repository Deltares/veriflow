"""Classes and functions to create a Config object from various types of configuration files."""

# The public interface
from typing import Annotated

from pydantic import Field

from veriflow.configuration.base import GeneralInfoConfig
from veriflow.configuration.config import Config
from veriflow.configuration.default.datasinks import (
    CFCompliantNetCDFConfig,
    FewsNetCDFOutputConfig,
)
from veriflow.configuration.default.datasources import (
    FewsNetCDFConfig,
    FewsWebserviceConfig,
    ForecastRetrievalMethod,
)
from veriflow.configuration.default.scores import (
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    RankHistogramConfig,
)
from veriflow.configuration.file import ConfigFile, ConfigKind
from veriflow.configuration.utils import FewsWebserviceAuthConfig
