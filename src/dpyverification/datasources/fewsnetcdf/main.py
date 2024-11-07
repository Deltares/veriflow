"""Read and write netcdf files in a fews compatible format."""

from pathlib import Path
from typing import TYPE_CHECKING, Self

import xarray as xr

from dpyverification.configuration import DataSource, FewsNetcdfOutput, Output
from dpyverification.constants import (
    DataModelAttributes,
    DataModelCoords,
    DataModelDims,
    DataSourceType,
    SimObsType,
)
from dpyverification.datasources.genericdatasource import GenericDatasource

from .schema import FewsNetcdfSchema

if TYPE_CHECKING:
    from collections.abc import Hashable


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
        if dsconfig.datasourcetype != DataSourceType.FEWSNETCDF:
            msg = "Input dsconfig does not have datasourcetype fewsnetcdf"
            raise TypeError(msg)
        if dsconfig.simobstype == SimObsType.COMBINED:
            msg = "Cannot yet handle combined simobs data"
            raise NotImplementedError(msg)

        filepath = Path(dsconfig.directory) / dsconfig.filename
        fnf = cls(dsconfig)
        fnf.xarray = cls._nc_to_xarray(filepath, dsconfig.simobstype)
        return [fnf]

    @classmethod
    def write_data(cls, dsconfig: Output, dataset: xr.Dataset) -> None:
        """Write the data in the xarray Dataset to the file as specified in the output config.

        The input dataset is assumed to be the DataModel output.
        """
        if not isinstance(dsconfig, FewsNetcdfOutput):
            msg = "Input dsconfig does not have datasourcetype fewsnetcdf"
            raise TypeError(msg)
        filepath = Path(dsconfig.directory) / dsconfig.filename
        if filepath.exists():
            # To consider: add a forcing flag, to force an overwrite of the file
            msg = "File already exists: " + str(filepath)
            raise FileExistsError(msg)

        # Make a copy of the dataset, so we do not modify datamodel output:
        #   Do not do explicitly, the rename_dims will already return a new object

        # Renames from DPyVerification datamodel to FewsNetcdf compliance
        renames = {DataModelDims.simstart: "analysis_time"}
        dataset = dataset.rename_dims(renames)
        renames = {DataModelCoords.simstart.name: "analysis_time"}
        dataset = dataset.rename_vars(renames)

        # Remove attributes not usable / desired in the netcdf
        dataset.attrs.pop(DataModelAttributes.timestep)  # type: ignore[misc] # attrs is a dict[Any,Any]

        # Add any missing information required for CF compliance
        #   Note that most CF compliance will already be done in the creation of the xarray as
        #   part of the datamodel
        cls.add_global_attrs(dataset, dsconfig)
        cls.add_coord_attrs(dataset)
        cls.add_var_attrs(dataset)

        # Rename variable to only have letters, digits and underscores?

        # Check the dataset against the required schema
        # For now, assume the schema used for fewsnetcdf input, also holds for fewsnetcdf output
        # Probably, want to use a different schema at a later time for output
        # Assign to _, since the model will throw an error when not compliant
        schema_like = dataset.to_dict()  # type: ignore[misc] # Yes, the dict could have any content, it will be checked against the FewsNetcdfSchema
        _ = FewsNetcdfSchema(**schema_like)  # type: ignore[misc] # See previous line

        dataset.to_netcdf(filepath)

    @staticmethod
    def add_global_attrs(dataset: xr.Dataset, dsconfig: FewsNetcdfOutput) -> None:
        """Add required global attributes if missing."""
        global_attrs = {
            "Conventions": "CF-1.6",
        }
        # TODO(AU): What global attributes to use when writing a (fews) netcdf file? # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/30
        if "title" not in dataset.attrs:  # type: ignore[misc] # attrs is a dict[Any,Any]
            global_attrs["title"] = "Verification results from DPyVerification"
        if "institution" not in dataset.attrs:  # type: ignore[misc] # attrs is a dict[Any,Any]
            if dsconfig.institution:
                global_attrs["institution"] = dsconfig.institution
            else:
                global_attrs["institution"] = "Unknown"
        if "source" not in dataset.attrs:  # type: ignore[misc] # attrs is a dict[Any,Any]
            msg = "No source attribute should be impossible, programmer error?"
            raise RuntimeError(msg)
        dataset.attrs.update(global_attrs)  # type: ignore[misc] # attrs is a dict[Any,Any]

    @staticmethod
    def add_coord_attrs(dataset: xr.Dataset) -> None:
        """Add required coordinate attributes if missing."""
        # TODO(AU): Variable name mappings (including coordinate variables) # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/31
        for coord in dataset.coords:
            coord_attrs: dict[str, Hashable] = {}
            if coord == "leadtime":
                # TODO(AU): Add leadtime coordinate during initialization # noqa: FIX002
                #   https://github.com/Deltares-research/DPyVerification/issues/15
                coord_attrs["standard_name"] = "forecast_period"
            elif not any(k in dataset.coords[coord].attrs for k in ("standard_name", "long_name")):  # type: ignore[misc] # attrs is a dict[Any,Any]
                coord_attrs["long_name"] = coord
            dataset.coords[coord].attrs.update(coord_attrs)  # type: ignore[misc] # attrs is a dict[Any,Any]

    @staticmethod
    def add_var_attrs(dataset: xr.Dataset) -> None:
        """Add required variable attributes if missing."""
        for var in dataset.variables:
            var_attrs = {}
            if not any(k in dataset.variables[var].attrs for k in ("standard_name", "long_name")):  # type: ignore[misc] # attrs is a dict[Any,Any]
                var_attrs["long_name"] = var
            dataset.variables[var].attrs.update(var_attrs)  # type: ignore[misc] # attrs is a dict[Any,Any]
            # TODO(AU): Variable name mappings (including coordinate variables) # noqa: FIX002
            #   https://github.com/Deltares-research/DPyVerification/issues/31
