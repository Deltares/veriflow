"""Read and write netcdf files in a fews compatible format."""

from pathlib import Path
from typing import Self

import xarray as xr

from dpyverification.configuration import DataSource, FileInputFewsnetcdf
from dpyverification.constants import SimObsType
from dpyverification.datasources.genericdatasource import GenericDatasource

from .schema import FewsNetcdfFileInputSchema


class FewsNetcdfFileSource(GenericDatasource):
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
        _ = FewsNetcdfFileInputSchema(**schema_like)  # type: ignore[misc]

        if kind == SimObsType.OBS and "ensemble_member" in ds.coords:
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
        if not isinstance(dsconfig, FileInputFewsnetcdf):
            msg = "Input dsconfig does not have datasourcetype fewsnetcdf"
            raise TypeError(msg)
        if dsconfig.simobstype == SimObsType.COMBINED:
            msg = "Cannot yet handle combined simobs data"
            raise NotImplementedError(msg)

        filepath = Path(dsconfig.directory) / dsconfig.filename
        fnf = cls(dsconfig)
        fnf.xarray = cls._nc_to_xarray(filepath, dsconfig.simobstype)
        return [fnf]
