"""Read and write netcdf files in a fews compatible format."""

from pathlib import Path
from typing import Self

import xarray as xr

from dpyverification.configuration import DataSource, DataSourceTypeEnum, SimObsType
from dpyverification.datasources.genericdatasource import GenericDatasource

from .schema import FewsNetcdfSchema


class FewsNetcdfFile(GenericDatasource):
    """For reading data from, and writing data to, a fews netcdf file."""

    @staticmethod
    def _nc_to_xarray(path: Path, kind: SimObsType) -> xr.Dataset:
        """Read fews netcdf file and return xr.Dataset.

        Compatible with both observations and (ensemble) forecasts.

        Parameters
        ----------
        path : Path
            Path to the netcdf file
        kind : Literal["sim", "obs"]
            String indicating the kind. Should be either sim (for simulations)
             or obs (for observations).


        Returns
        -------
        xr.Dataset
            Dataset representation of the fews netcdf file.

        Raises
        ------
        TypeError
            Raised when pd.DataFrame.to_xarray() does not return xr.DataArray.
        """
        ds = xr.open_dataset(path)

        # Verify the structure of the dataset against known schema
        schema_like = ds.to_dict()  # type: ignore[misc] # Yes, the dict could have any content, it will be checked against the model
        # For now, assume the schema used for fewsnetcdf output, also holds for fewsnetcdf input
        # Probably, want to use a different schema at a later time for input
        # Assign to _, since the model will throw an error when not compliant
        _ = FewsNetcdfSchema(**schema_like)  # type: ignore[misc]

        if kind == SimObsType.obs and "ensemble_member" in ds.coords:
            # Can this happen? What to do? Squeeze it out like in pixml file?
            raise NotImplementedError

        raise NotImplementedError
        # From here on, may need to also adapt how datamodel uses an xarray
        # DataModel only tested for single parameter inputs for now, not yet multiple
        # Need to check what coordinates an obs, and a sim, can have, and if that matches the
        #  DataModel expectations on inputs
        return ds  # type: ignore[unreachable] # yes, for now this is unreachable, but do want to keep it

    @classmethod
    def get_data(cls, dsconfig: DataSource) -> list[Self]:
        """Retrieve fewsnetcdf content as an xarray DataArray."""
        if dsconfig.datasourcetype != DataSourceTypeEnum.fewsnetcdf:
            msg = "Input dsconfig does not have datasourcetype fewsnetcdf"
            raise TypeError(msg)
        if dsconfig.simobstype == SimObsType.combined:
            msg = "Cannot yet handle combined simobs data"
            raise NotImplementedError(msg)

        filepath = Path(dsconfig.directory) / dsconfig.filename
        fnf = cls(dsconfig)
        fnf.xarray = cls._nc_to_xarray(filepath, dsconfig.simobstype)
        return [fnf]

    @classmethod
    def write_to_file(cls, path: Path, dataset: xr.Dataset) -> None:
        """Write the data in the xarray Dataset to the file at path.

        Details of how to write will need to be implemented in subclass.
        """
        if path.exists():
            # To consider: add a forcing flag, to force an overwrite of the file
            msg = "File already exists: " + str(path)
            raise FileExistsError(msg)

        # Check the dataset against the required schema
        # For now, assume the schema used for fewsnetcdf input, also holds for fewsnetcdf output
        # Probably, want to use a different schema at a later time for output
        # Assign to _, since the model will throw an error when not compliant
        schema_like = dataset.to_dict()  # type: ignore[misc] # Yes, the dict could have any content, it will be checked against the FewsNetcdfSchema
        _ = FewsNetcdfSchema(**schema_like)  # type: ignore[misc] # See previous line

        dataset.to_netcdf(path)
