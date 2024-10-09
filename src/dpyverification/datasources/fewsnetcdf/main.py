"""Read and write netcdf files in a fews compatible format."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import xarray as xr

from dpyverification.configuration import DataSource, FewsNetcdfOutput, Output
from dpyverification.constants import (
    DataModelAttributes,
    DataModelCoords,
    DataModelDims,
    DataSourceTypeEnum,
    SimObsType,
)
from dpyverification.datasources.fewsnetcdf.inputschema import FewsNetcdfInputSchema
from dpyverification.datasources.fewsnetcdf.outputschema import FewsNetcdfOutputSchema
from dpyverification.datasources.genericdatasource import GenericDatasource

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
        _ = FewsNetcdfInputSchema(**schema_like)  # type: ignore[misc]

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

        # Check if dataset has ensemble dimension
        dataset_has_ensemble_dim = DataModelDims.ensemble in dataset.dims

        # Make a copy of the dataset, so we do not modify datamodel output:
        #   Do not do explicitly, the rename_dims will already return a new object

        # Renames from DPyVerification datamodel to FewsNetcdf compliance
        if dataset_has_ensemble_dim:
            renames = {DataModelDims.ensemble: "realization"}
            dataset = dataset.rename_dims(renames)
            renames = {DataModelCoords.ensemble.name: "realization"}
            dataset = dataset.rename_vars(renames)
        renames = {DataModelDims.location: "stations"}
        dataset = dataset.rename_dims(renames)
        renames = {DataModelCoords.location.name: "station_id"}
        dataset = dataset.rename_vars(renames)

        # Remove attributes not usable / desired in the netcdf
        dataset.attrs.pop(DataModelAttributes.timestep)  # type: ignore[misc] # attrs is a dict[Any,Any]

        # Remove dimensions not usable / desired in teh netcdf
        dataset = dataset.drop(DataModelDims.simstart)

        # Add any missing information required for CF compliance
        #   Note that most CF compliance will already be done in the creation of the xarray as
        #   part of the datamodel
        cls.add_global_attrs(dataset, dsconfig)
        cls.add_coord_attrs(dataset)
        cls.add_var_attrs(dataset)

        # Set the encoding for compliance with the Delft-FEWS Archive.
        cls.add_dim_encoding(dataset)

        # Rename variables to only have letters, digits and underscores.
        dataset = cls.check_and_update_var_names(dataset)

        # Flatten leadtime dimension into parameter name, only if ensemble in dataset dim
        if dataset_has_ensemble_dim:
            dataset = cls.flatten_dim_as_vars(dataset, dim=DataModelDims.leadtime)

        # Check the dataset against the required schema
        # Assign to _, since the model will throw an error when not compliant
        schema_like = dataset.to_dict(encoding=True)  # type: ignore[misc] # Yes, the dict could have any content, it will be checked against the FewsNetcdfSchema
        _ = FewsNetcdfOutputSchema(**schema_like)  # type: ignore[misc] # See previous line

        dataset.to_netcdf(filepath)

    @staticmethod
    def add_global_attrs(dataset: xr.Dataset, dsconfig: FewsNetcdfOutput) -> None:
        """Add required global attributes if missing."""
        global_attrs = {
            "Conventions": "CF-1.6",
        }
        if "title" not in dataset.attrs:  # type: ignore[misc] # attrs is a dict[Any,Any]
            # Create title at datamodel init, to include obs source info?
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
        # Where to get the default mappings for coordinate names?
        for coord in dataset.coords:
            coord_attrs: dict[str, Hashable] = {}
            if coord == "leadtime":
                # Temp, leadtime should be added in datamodel init already,
                #   and become a standard coordinate
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
                var_attrs["standard_name"] = var
                var_attrs["long_name"] = var
            dataset.variables[var].attrs.update(var_attrs)  # type: ignore[misc] # attrs is a dict[Any,Any]
            # What about standard_name, for instance water_volume_transport_in_river_channel
            #  for Q variables? How to determine, where to get from?

    @staticmethod
    def add_dim_encoding(dataset: xr.Dataset) -> None:
        """Set encoding for compliance with Delft-FEWS Archive."""
        dataset.time.encoding["dtype"] = "float64"  # type: ignore[misc]
        dataset.time.encoding["units"] = "minutes since 1970-01-01 00:00:00.0 +0000"  # type: ignore[misc]
        dataset.realization.encoding["dtype"] = "int32"  # type: ignore[misc]

    @staticmethod
    def var_name_valid(name: str) -> bool:
        """Check if string contains only letters, numbers and _ or -."""
        return re.match("^[A-Za-z0-9_]*$", name)  # type: ignore  # noqa: PGH003

    @staticmethod
    def check_and_update_var_names(dataset: xr.Dataset) -> xr.Dataset:
        """Check and update variable to only have letters, digits and underscores?."""
        for name in list(dataset.data_vars):  # type: ignore[misc]
            if not FewsNetcdfFile.var_name_valid(name):  # type: ignore[misc]
                new_name = name.replace(".", "_")  # type: ignore[misc]
                if FewsNetcdfFile.var_name_valid(new_name):  # type: ignore[misc]
                    # Rename the variable and attrs
                    dataset = dataset.rename_vars({name: new_name})  # type: ignore[misc]
                    dataset[new_name].attrs["long_name"] = new_name  # type: ignore[misc]
                else:
                    msg = f"Cannot convert variable name {name} to a valid name."  # type: ignore[misc]
                    raise NotImplementedError(msg)
        return dataset

    @staticmethod
    def flatten_dim_as_vars(dataset: xr.Dataset, dim: str) -> xr.Dataset:
        """Flatten a specific dimension into variable names.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to be flattened.
        dim : str
            The dimension that should be flattened and resolved in variable names.

        Returns
        -------
        xr.Dataset
            A new dataset without dim, but with new variables, containing a reference to the index
            value of the dimension.

        Raises
        ------
        NotImplementedError
            Raised when dim is other than class:`DataModelDims.leadtime`.
        ValueError
            Raised when Delft-FEWS compliant variable name cannot be found.
        """
        __allowed_dims = [DataModelDims.leadtime]

        if dim not in __allowed_dims:
            msg = f"Flattening dimension '{dim}' not supported."
            raise NotImplementedError(msg)

        dataset_slices = []
        for leadtime in dataset[dim].to_numpy():  # type: ignore[misc]
            dataset_slice = dataset.sel(**{dim: leadtime})  # type: ignore[misc]
            # Drop dimensions and coordinates
            dataset_slice = dataset_slice.drop([dim])  # type: ignore[misc]

            # Convert leadtime to number of hours
            # TODO(jurianbeunk): set renaming equal to general config lead time units.  # noqa: E501, FIX002
            # https://github.com/Deltares-research/DPyVerification/issues/18
            number_of_hours = leadtime.astype("timedelta64[h]") / np.timedelta64(1, "h")  # type: ignore[misc]
            if not number_of_hours.is_integer():  # type: ignore[misc]
                msg = f"Dimension '{dim}' cannot be converted to whole number in hours."
                raise ValueError(msg)
            number_of_hours_string = f"{int(number_of_hours):03d}"  # type: ignore[misc]
            dataset_slices.append(
                dataset_slice.rename(
                    {  # type: ignore[misc]
                        var: str(var) + f"_{number_of_hours_string}h"
                        for var in dataset_slice.data_vars
                    },
                ),
            )
        return xr.merge(dataset_slices)
