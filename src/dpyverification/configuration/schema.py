"""The schema definition for the configuration yaml file.

To generate a yaml / json file with the json representation of this schema:
    with FILEPATH.open() as myfile:
        yaml.dump(ConfigSchema.model_json_schema(), myfile)

To generate a pydantic schema from a yaml/json file, see datamodel_code_generator,
for example from https://docs.pydantic.dev/latest/integrations/datamodel_code_generator/
Note that this can generate a pydantic model that is not up-to-date with the latest
pydantic / python, and might need some modifications.
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
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from dpyverification.constants import CalculationTypeEnum, DataSourceTypeEnum, SimObsType, TimeUnits


class DateTime(BaseModel):
    format: str | None = "%Y-%m-%dT%H:%M:%S%z"
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
    datasourcetype: Literal[DataSourceTypeEnum.fewswebservice]
    url: str


class FewsWebserviceInput(FewsWebservice):
    simobstype: Literal[SimObsType.obs, SimObsType.sim]
    location_ids: list[str]
    parameter_ids: list[str]
    module_instance_ids: list[str]
    qualifier_ids: list[str]
    document_format: Literal["PI_XML"]
    document_version: Literal["1.32"]  # What version we support
    leadtimes: LeadTimes | None = Field(None, description="Required for simulations.")

    @field_validator("leadtimes")
    @classmethod
    def check_field_leadtimes(cls, v: LeadTimes | None, info: ValidationInfo) -> LeadTimes | None:
        """Check if leadtimes defined, when simobstype is sim."""
        if info.data["simobstype"] == SimObsType.sim and v is None:  # type: ignore[misc]
            msg = "Lead times are required when simobstype is SimObsType.sim."
            raise ValueError(msg)
        return v


class FewsWebserviceOutput(FewsWebservice):
    pass


class LocalFile(BaseModel):
    directory: str
    filename: str


class FileInput(LocalFile):
    datasourcetype: Literal[DataSourceTypeEnum.pixml, DataSourceTypeEnum.fewsnetcdf]
    simobstype: SimObsType


class FewsNetcdfOutput(LocalFile):
    datasourcetype: Literal[DataSourceTypeEnum.fewsnetcdf]
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
    FewsWebserviceInput | FileInput
)  # A Type Alias for the combination of data source schema classes

Output: TypeAlias = (
    FewsWebserviceOutput | FewsNetcdfOutput
)  # A Type Alias for the combination of output schema classes


class SimObsPair(BaseModel):
    sim: str
    obs: str


class PinScore(BaseModel):
    calculationtype: Literal[CalculationTypeEnum.pinscore]


class SimObsPairs(BaseModel):
    calculationtype: Literal[CalculationTypeEnum.simobspairs]
    # One combination of list-of-leadtimes and list-of-variablepairs, use multiple SimObsPairs
    # to define more combinations
    leadtimes: LeadTimes | None = None  # Default from GeneralInfo
    variablepairs: list[SimObsPair]


Calculation: TypeAlias = (
    SimObsPairs | PinScore  # A Type Alias for the combination of calculation schema classes
)


class GeneralInfo(BaseModel):
    # Is this general info, or might it be different for different calculations?
    verificationperiod: TimePeriod
    leadtimes: LeadTimes | None = None


class ConfigSchema(BaseModel):
    output: Annotated[list[Output], Field(min_length=1)]
    calculations: Annotated[list[Calculation], Field(min_length=1)]
    datasources: Annotated[list[DataSource], Field(min_length=1)]
    general: GeneralInfo
    fileversion: str
