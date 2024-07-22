"""The various datasources that can be used as input data."""

import xarray

from dpyverification.configuration import DataSource


class GenericDatasource:
    """Class to inherit from, defines the required methods and attributes."""

    def __init__(self, dsconfig: DataSource) -> None:
        self.config = dsconfig

    @staticmethod
    def get_data(dsconfig: DataSource) -> xarray.DataArray:
        """Get the data from the datasource, and return it in a predetermined format."""
        _ = GenericDatasource(dsconfig)
        return xarray.DataArray()
