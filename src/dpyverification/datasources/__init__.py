"""The various datasources that can be used as input data."""

from .base import BaseDatasource
from .csv import Csv
from .fewsnetcdf import FewsNetCDF
from .fewswebservice import FewsWebservice
from .inputschemas import validate_input_data
from .netcdf import NetCDF

DEFAULT_DATASOURCES: list[type[BaseDatasource]] = [
    FewsNetCDF,
    FewsWebservice,
    NetCDF,
    Csv,
]
