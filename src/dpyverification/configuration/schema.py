"""The schema definition for the configuration yaml file.

To generate a yaml / json file with the json representation of this schema:
    with FILEPATH.open() as myfile:
        yaml.dump(ConfigSchema.model_json_schema(), myfile)

To generate a pydantic schema from a yaml/json file, see datamodel_code_generator,
for example from https://docs.pydantic.dev/latest/integrations/datamodel_code_generator/
Note that this can generate a pydantic model that is not up-to-date with the latest
pydantic / python, and might need some modifications.
"""


# ruff: noqa: D101 Do not require class docstrings for the classes in this file

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from dpyverification.constants import CalculationTypeEnum, DataSourceTypeEnum, SimObsType, TimeUnits


class FewsWebservice(BaseModel):
    datasourcetype: Literal[DataSourceTypeEnum.fewswebservice]
    url: str


class FewsWebserviceInput(FewsWebservice):
    simobstype: SimObsType


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
    leadtimes: list[int] | None = (
        None  # Use GeneralInfo leadtimes when None, AND, only ok as subset of GeneralInfo leadtimes
    )
    leadtimesunit: Literal[TimeUnits.day, TimeUnits.hour, TimeUnits.minute, TimeUnits.second] = (
        TimeUnits.minute
    )
    variablepairs: list[SimObsPair]


Calculation: TypeAlias = (
    SimObsPairs | PinScore  # A Type Alias for the combination of calculation schema classes
)


class GeneralInfo(BaseModel):
    # Is this general info, or might it be different for different calculations?
    leadtimes: list[int] | None = None  # Do we need a default value, if it is optional? Yes
    leadtimesunit: Literal[TimeUnits.day, TimeUnits.hour, TimeUnits.minute, TimeUnits.second] = (
        TimeUnits.minute
    )


class ConfigSchema(BaseModel):
    output: Annotated[list[Output], Field(min_length=1)]
    calculations: Annotated[list[Calculation], Field(min_length=1)]
    datasources: Annotated[list[DataSource], Field(min_length=1)]
    general: GeneralInfo
    fileversion: str
