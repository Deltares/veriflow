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

from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field


class DataSourceTypeEnum(StrEnum):
    pixml = "pixml"
    fewsnetcdf = "fewsnetcdf"
    fewswebservice = "fewswebservice"


class SimObsType(StrEnum):
    sim = "sim"
    obs = "obs"
    combined = "combined"


class FewsWebservice(BaseModel):
    datasourcetype: Literal[DataSourceTypeEnum.fewswebservice]
    simobstype: SimObsType
    url: str


class LocalFile(BaseModel):
    datasourcetype: Literal[DataSourceTypeEnum.pixml, DataSourceTypeEnum.fewsnetcdf]
    simobstype: SimObsType
    directory: str
    filename: str


DataSource: TypeAlias = (
    FewsWebservice | LocalFile
)  # A Type Alias for the combination of data source schema classes


class ConfigSchema(BaseModel):
    datasources: Annotated[list[DataSource], Field(min_length=1)]
    fileversion: str
