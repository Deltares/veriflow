"""The definition of the configuration settings.

This definition is used both as the schema for the configuration yaml file, and as the content of
the dpyverification configuration object.

To generate a yaml / json file with the json representation of this schema:
    import pathlib
    import yaml
    from dpyverification.configuration import ConfigSchema
    FILEPATH = pathlib.Path("YOUR_PATH_HERE")
    with FILEPATH.open("w") as myfile:
        yaml.dump(ConfigSchema.model_json_schema(), myfile)

Sidenote: It is also possible to go the other way around and generate a pydantic schema from a
yaml/json file, see datamodel_code_generator, for example from
https://docs.pydantic.dev/latest/integrations/datamodel_code_generator/ . Note that this can
generate a pydantic model that is not up-to-date with the latest pydantic / python, and might
need some modifications.
"""

# TODO(AU): Add pydantic Field with description, and maybe title, to all attributes. # noqa: FIX002
#   https://github.com/Deltares-research/DPyVerification/issues/9
#   Add pydantic Field with description, and maybe title, to approximately every attribute. To both
#   have a descriptive json schema when the json schema is generated from the pydantic objects, and
#   to document what the fields are for in the code. Maybe only for Literal attributes, the
#   description can be skipped. Do also add the description to private attributes, to document
#   their use.

# ruff: noqa: D101 Do not require class docstrings for the classes in this file

from datetime import datetime, timedelta
from typing import Annotated, Literal, TypeAlias

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from dpyverification.constants import (
    CalculationType,
    DataModelDims,
    DataSourceType,
    SimObsType,
    TimeUnits,
)


class DateTime(BaseModel):
    format: str = "%Y-%m-%dT%H:%M:%S%z"
    value: str

    @property
    def datetime64(self) -> np.datetime64:
        """As numpy datetime64."""
        return pd.to_datetime(self.value, format=self.format).to_numpy()

    @property
    def datetime(self) -> datetime:
        """As datetime datetime."""
        return pd.to_datetime(self.value, format=self.format)


class LeadTimes(BaseModel):
    unit: TimeUnits
    values: list[int]

    @property
    def timedelta64(self) -> list[np.timedelta64]:
        """As numpy timedelta64."""
        return [np.timedelta64(v, self.unit) for v in self.values]

    @property
    def timedelta(self) -> list[timedelta]:
        """As datetime timedelta."""

        def convert_to_timedelta(value: int) -> timedelta:
            return np.timedelta64(value, self.unit).astype(timedelta)  # type: ignore[no-any-return, misc]

        return [convert_to_timedelta(v) for v in self.values]


class TimePeriod(BaseModel):
    start: DateTime
    end: DateTime


class FewsWebservice(BaseModel):
    datasourcetype: Literal[DataSourceType.FEWSWEBSERVICE]
    url: str


class FewsWebserviceInput(FewsWebservice):
    location_ids: Annotated[list[str], Field(min_length=1)]
    parameter_ids: Annotated[list[str], Field(min_length=1)]
    module_instance_ids: Annotated[list[str], Field(min_length=1)]
    qualifier_ids: list[str] = []  # Note that no min_length, so empty list is ok.
    verificationperiod: TimePeriod | None = Field(
        None,
        description="Value from General verificationperiod used if not set.",
    )
    _document_format: Annotated[
        Literal["PI_XML"],
        Field(
            description=(
                "A private attribute with a default value:"
                " 1. A literal with a default value, as users are not expected to set it."
                " 2. Private, since the code only supports this one option for now, but it is"
                " likely that other values may be needed in the future, depending on what FEWS"
                " system this code interacts with. Then this value does need to be configurable,"
                " therefore do have it as a configuration setting already."
            ),
        ),
    ] = "PI_XML"
    _document_version: Annotated[
        Literal["1.32"],
        Field(
            description=(
                "A private attribute with a default value:"
                " 1. A literal with a default value, as users are not expected to set it."
                " 2. Private, since the code only supports this one option for now, but it is"
                " likely that other values may be needed in the future, depending on what FEWS"
                " system this code interacts with. Then this value does need to be configurable,"
                " therefore do have it as a configuration setting already."
            ),
        ),
    ] = "1.32"


class FewsWebserviceInputObs(FewsWebserviceInput):
    simobstype: Literal[SimObsType.OBS]


class FewsWebserviceInputSim(FewsWebserviceInput):
    simobstype: Literal[SimObsType.SIM]
    leadtimes: Annotated[
        LeadTimes | None,
        Field(
            description="Value from General leadtimes used if not set.",
        ),
    ] = None
    forecastcount: Annotated[
        int,
        Field(
            description=(
                "Number of forecast runs to retrieve."
                " When value is 0 (default), ALL matching forecast runs will be used."
            ),
        ),
    ] = 0
    ensemble_id: str | None = None


class FewsWebserviceOutput(FewsWebservice):
    pass


class LocalFile(BaseModel):
    directory: str
    filename: str


class FileInput(LocalFile):
    simobstype: SimObsType


class FileInputPixml(FileInput):
    datasourcetype: Literal[DataSourceType.PIXML]


class FileInputFewsnetcdf(FileInput):
    datasourcetype: Literal[DataSourceType.FEWSNETCDF]


class FewsNetcdfOutput(LocalFile):
    datasourcetype: Literal[DataSourceType.FEWSNETCDF]
    title: Annotated[
        str | None,
        Field(
            description=(
                "Value for the title attribute in the generated netcdf."
                " A title will be generated if not provided"
            ),
        ),
    ] = None
    institution: Annotated[
        str,
        Field(description="Value for the institution attribute in the generated netcdf."),
    ] = "Deltares"


DataSource: TypeAlias = (
    FewsWebserviceInputSim | FewsWebserviceInputObs | FileInputPixml | FileInputFewsnetcdf
)  # A Type Alias for the combination of data source schema classes

Output: TypeAlias = (
    FewsWebserviceOutput | FewsNetcdfOutput
)  # A Type Alias for the combination of output schema classes


class SimObsVariables(BaseModel):
    sim: str
    obs: str


class PinScore(BaseModel):
    calculationtype: Literal[CalculationType.PINSCORE]


class SimObsPairs(BaseModel):
    calculationtype: Literal[CalculationType.SIMOBSPAIRS]
    # One combination of list-of-leadtimes and list-of-variablepairs, use multiple SimObsPairs
    # to define more combinations
    leadtimes: Annotated[
        LeadTimes | None,
        Field(
            description="Value from General leadtimes used if not set.",
        ),
    ] = None
    variablepairs: list[SimObsVariables]


class RankHistogram(BaseModel):
    calculationtype: Literal[CalculationType.RANKHISTOGRAM]
    reduce_dims: Annotated[
        list[DataModelDims] | None,
        Field(
            description="""Dimension(s) over which to compute the histogram
            of ranks. Defaults to all dimensions.""",
        ),
    ] = None


class CRPSForEnsemble(BaseModel):
    calculationtype: Literal[CalculationType.CRPSForEnsemble]
    method: Annotated[
        Literal["ecdf", "fair"],
        Field(
            description="""Defaults to ecdf. See: https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_for_ensemble""",
            default="ecdf",
        ),
    ]
    reduce_dims: Annotated[
        list[DataModelDims] | None,
        Field(
            description="""Dimension(s) over which to compute the histogram
            of ranks. Defaults to all dimensions.""",
            default=None,
        ),
    ] = None


Calculation: TypeAlias = (
    SimObsPairs
    | PinScore
    | RankHistogram
    | CRPSForEnsemble  # A Type Alias for the combination of calculation schema classes
)


class GeneralInfo(BaseModel):
    verificationperiod: TimePeriod
    leadtimes: LeadTimes = LeadTimes(values=[0], unit=TimeUnits("h"))


class ConfigSchema(BaseModel):
    output: Annotated[list[Output], Field(min_length=1)]
    calculations: Annotated[list[Calculation], Field(min_length=1)]
    datasources: Annotated[list[DataSource], Field(min_length=1)]
    general: GeneralInfo
    fileversion: str
