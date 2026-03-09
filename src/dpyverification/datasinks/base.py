"""Module with the base class that all datasources should inherit from."""

from abc import abstractmethod

import xarray as xr

from dpyverification.base import Base
from dpyverification.configuration.config import BaseDatasinkConfig

__all__ = [
    "BaseDatasink",
    "BaseDatasinkConfig",
]


class BaseDatasink(Base):
    """Class to inherit from, defines the required methods and attributes."""

    kind = ""  # to be defined by subclasses
    config_class: type[BaseDatasinkConfig] = BaseDatasinkConfig  # to be defined by subclasses

    def __init__(self, config: BaseDatasinkConfig) -> None:
        self.config = config

    @abstractmethod
    def write_data(self, data: xr.Dataset) -> None:
        """Write output data for one verification pair to the datasource."""
