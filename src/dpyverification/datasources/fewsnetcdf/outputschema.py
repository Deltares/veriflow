"""Pydantic schema definition for a fews compatible output netcdf file.

To generate a yaml / json file with the json representation of this schema:
    with FILEPATH.open() as myfile:
        yaml.dump(FewsNetcdfSchema.model_json_schema(), myfile)

To generate a pydantic schema from a yaml/json file, see datamodel_code_generator,
for example from https://docs.pydantic.dev/latest/integrations/datamodel_code_generator/
Note that this might generate a pydantic model that is not up-to-date with the latest
pydantic / python, and might need some modifications.

"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ruff: noqa: D101 Do not require class docstrings for the classes in this file


class FileAttrs(BaseModel):
    Conventions: Literal["CF-1.6"]
    title: str
    institution: str
    source: str


class TimeAttrs(BaseModel):
    standard_name: Literal["time"]
    long_name: Literal["time"]
    axis: Literal["T"]


class StationIdAttrs(BaseModel):
    # Does not require standard_name
    long_name: Literal["station identification code"]
    cf_role: Literal["timeseries_id"]


class Dims(BaseModel):
    time: int
    stations: int


class TimeEncoding(BaseModel):
    dtype: Literal["float64"]
    units: Literal["minutes since 1970-01-01 00:00:00.0 +0000"]


class TimeCoord(BaseModel):
    attrs: TimeAttrs
    encoding: TimeEncoding


class StationId(BaseModel):
    # How to validate dimension?
    dims: list[Literal["stations"]]
    attrs: StationIdAttrs


class CoordAttrs(BaseModel):
    standard_name: str
    long_name: str


class Coord(BaseModel):
    attrs: CoordAttrs


class Coords(BaseModel):
    time: TimeCoord
    lat: Coord
    lon: Coord
    station_id: StationId
    # station_names: Coord  # noqa: ERA001


class DataVarAttrs(BaseModel):
    standard_name: str
    long_name: str
    units: str


class DataVars(BaseModel):
    attrs: DataVarAttrs


class FewsNetcdfOutputSchema(BaseModel):
    attrs: FileAttrs
    dims: Dims
    coords: Coords
    data_vars: dict[Annotated[str, Field(pattern=r"[a-zA-Z_-].*")], DataVars]
