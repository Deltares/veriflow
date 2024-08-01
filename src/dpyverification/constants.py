"""Collection of constant values definitions.

Both single values, and enum classes.
These values are can be used throughout DPyVerification.
"""

from enum import StrEnum


class DataSourceTypeEnum(StrEnum):
    """Enumeration of the supported input and / or output datasource types."""

    pixml = "pixml"
    fewsnetcdf = "fewsnetcdf"
    fewswebservice = "fewswebservice"


class SimObsType(StrEnum):
    """Enumeration of the supported types of input data."""

    sim = "sim"
    obs = "obs"
    combined = "combined"


class CalculationTypeEnum(StrEnum):
    """Enumeration of the implemented verification calculations."""

    simobspairs = "simobspair"
    pinscore = "pinscore"
