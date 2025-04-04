"""Module with the base class that all datasources should inherit from."""

import xarray

from dpyverification.configuration import DataSink, DataSource


class GenericDatasink:
    """Class to inherit from, defines the required methods and attributes."""

    def __init__(self, dsconfig: DataSource) -> None:
        self.config = dsconfig
        self.xarray = xarray.Dataset()

    @classmethod
    def write_data(cls, dsconfig: DataSink, dataset: xarray.Dataset) -> None:
        """Write the data in the xarray Dataset to the datasource.

        Details of how to write will need to be implemented in subclass.
        """
        msg = (
            "Writing Dataset to file / webservice is dependent on the file type to write to,"
            " no generic implementation."
        )
        _ = dsconfig
        _ = dataset
        raise NotImplementedError(msg)
