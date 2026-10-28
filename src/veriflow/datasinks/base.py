"""Module with the base class that all datasinks should inherit from."""

from abc import abstractmethod

from veriflow.base import Base
from veriflow.configuration.base import BaseDatasinkConfig
from veriflow.datatree.datatree import VeriflowDataTree

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
    def write_data(self, data: VeriflowDataTree) -> None:
        """Write output data for one verification pair to the datasource."""
