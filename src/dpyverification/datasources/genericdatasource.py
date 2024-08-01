"""Module with the base class that all datasources should inherit from."""

import pathlib
from typing import Self

import xarray

from dpyverification.configuration import DataSource
from dpyverification.constants import SimObsType


class GenericDatasource:
    """Class to inherit from, defines the required methods and attributes."""

    def __init__(self, dsconfig: DataSource) -> None:
        self.config = dsconfig
        self.simobstype = dsconfig.simobstype
        self.xarray = xarray.Dataset()

    @property
    def simobstype(self) -> SimObsType:
        """Whether the object represents sim or obs data."""
        return self._simobstype

    @simobstype.setter
    def simobstype(self, new_simobstype: SimObsType) -> None:
        if new_simobstype not in (SimObsType.sim, SimObsType.obs):
            # Even if the underlying file or service can contain combined data, the creation of the
            #  datasource objects should split those. This assumption can then be used in the
            #  creation of the data model.
            msg: str = (
                "The simpobstype of a " + self.__class__.__name__ + " can only be either sim or obs"
            )
            raise ValueError(msg)
        self._simobstype = new_simobstype

    @classmethod
    def get_data(cls, dsconfig: DataSource) -> list[Self]:
        """Get the data from the datasource, and return it in a predetermined format.

        Returns a list of datasource objects, one for each sim or obs
        """
        return [cls(dsconfig)]

    @classmethod
    def write_to_file(cls, path: pathlib.Path, dataset: xarray.Dataset) -> None:
        """Write the data in the xarray Dataset to the file at path.

        Details of how to write will need to be implemented in subclass.
        """
        if path.exists():
            msg = "File already exists"
            raise FileExistsError(msg)

        msg = (
            "Writing Dataset to file is dependent on the file type to write to,"
            " no generic implementation."
        )
        _ = dataset
        raise NotImplementedError(msg)
