"""Read and write netcdf files in a fews compatible format."""

from pathlib import Path
from typing import Self

import xarray as xr

from dpyverification.configuration import FileInputFewsnetcdfConfig
from dpyverification.constants import SimObsKind
from dpyverification.datasources.base import BaseDatasource

from .schema import FewsNetcdfFileInputObsSchema, FewsNetcdfFileInputSimSchema


class FewsNetcdfFile(BaseDatasource):
    """For reading data from, and writing data to, a fews netcdf file."""

    kind = "fewsnetcdf"
    config_class = FileInputFewsnetcdfConfig

    def __init__(self, config: FileInputFewsnetcdfConfig) -> None:
        self.config: FileInputFewsnetcdfConfig = config

    @staticmethod
    def convert_obs_to_datamodel(ds: xr.Dataset) -> xr.Dataset:
        """Convert an obs file to match naming conventions of datamodel."""
        # Renames
        ds = ds.rename({"stations": "location_id", "station_id": "location_id"})  # type: ignore[misc]
        # Drop x, y, z
        return ds.drop_vars(["x", "y", "z", "station_names"])

    @staticmethod
    def convert_sim_to_datamodel(ds: xr.Dataset) -> xr.Dataset:
        """Convert an sim file to match naming conventions of datamodel."""
        # Set analysis_time coordinate for each data variable
        # this is a bit of a workaround, since the FEWS webservice
        # does not assign the anlysis_time as a coordinate on the
        # netcdf.
        ds = ds.assign_coords(analysis_time=("analysis_time", ds.analysis_time.data))  # type: ignore[misc]
        for da in ds.data_vars:
            ds[da] = ds[da].expand_dims(analysis_time=ds["analysis_time"])

        # Renames
        ds = ds.rename(
            {
                "stations": "location_id",
                "analysis_time": "simulation_starttime",
                "station_id": "location_id",
            },  # type: ignore[misc]
        )
        # Rename only when ensemble forecast
        if "realization" in ds:
            ds = ds.rename({"realization": "ensemble_member"})  # type: ignore[misc]
        # Drop coords
        if "x" in ds:
            ds = ds.drop_vars("x")
        if "y" in ds:
            ds = ds.drop_vars("y")
        if "z" in ds:
            ds = ds.drop_vars("z")
        if "station_names" in ds:
            ds = ds.drop_vars("station_names")
        return ds

    @staticmethod
    def nc_to_xarray(path: Path, kind: str) -> xr.Dataset:
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

        # Get the dataset as dict, to validate against schema
        schema_like = ds.to_dict()  # type: ignore[misc]

        if kind == SimObsKind.OBS:
            _ = FewsNetcdfFileInputObsSchema(**schema_like)  # type: ignore[misc]
            return FewsNetcdfFile.convert_obs_to_datamodel(ds)
        if kind == SimObsKind.SIM:
            _ = FewsNetcdfFileInputSimSchema(**schema_like)  # type: ignore[misc]
            return FewsNetcdfFile.convert_sim_to_datamodel(ds)

        msg = f"Kind is not valid: {kind}. Expected {SimObsKind.OBS} or {SimObsKind.SIM}"
        raise NotImplementedError(msg)

    def get_data(self) -> Self:
        """Retrieve fewsnetcdf content as an xarray DataArray."""
        if self.config.simobstype == SimObsKind.COMBINED:
            msg = "Cannot yet handle combined simobs data"
            raise NotImplementedError(msg)

        filepath = Path(self.config.directory) / self.config.filename
        self.xarray = self.nc_to_xarray(filepath, self.config.simobstype)
        return self
