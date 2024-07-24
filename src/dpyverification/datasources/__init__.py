"""The various datasources that can be used as input data."""

import xarray

from dpyverification.configuration import DataSource, SimObsType


class GenericDatasource:
    """Class to inherit from, defines the required methods and attributes."""

    def __init__(self, dsconfig: DataSource) -> None:
        self.config = dsconfig
        self.simobstype = dsconfig.simobstype

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

    @staticmethod
    def get_data(dsconfig: DataSource) -> xarray.DataArray:
        """Get the data from the datasource, and return it in a predetermined format."""
        _ = GenericDatasource(dsconfig)
        return xarray.DataArray()
