"""A module for default implementation of datasinks."""

from typing import Annotated, Literal

from pydantic import Field

from dpyverification.configuration.config import BaseDatasinkConfig
from dpyverification.configuration.utils import LocalFile
from dpyverification.constants import NAME, DataSinkKind

# TODO(JB): Make output / internal datamodel CF compliant.  # noqa: FIX002
# https://github.com/Deltares-research/DPyVerification/issues/84


class BaseCFCompliantConfig(LocalFile, BaseDatasinkConfig):
    """A base cf-compliant NetCDF output config element."""

    conventions: Literal["CF-1.7"] = "CF-1.7"
    title: Annotated[
        str,
        Field(
            description="Value for the title attribute in the generated NetCDF."
            " A title will be generated if not provided",
        ),
    ] = f"Verification results created by {NAME}"
    institution: Annotated[
        str,
        Field(description="Value for the institution attribute in the generated NetCDF."),
    ]
    comment: Annotated[
        str,
        Field(description="Value for the comment attribute in the generated NetCDF."),
    ] = "Verification results created by {NAME}"


class FewsNetCDFOutputConfig(BaseCFCompliantConfig):
    """A fews NetCDF output config element."""

    export_adapter: Literal[DataSinkKind.fews_netcdf]


class CFCompliantNetCDFConfig(BaseCFCompliantConfig):
    """A cf-compliant NetCDF output config element."""

    export_adapter: Literal[DataSinkKind.cf_compliant_netcdf]
