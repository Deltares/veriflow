"""Module with the dpyverification internal DataModel."""

from collections.abc import Sequence

import numpy as np
import xarray

from dpyverification.configuration import GeneralInfo
from dpyverification.constants import (
    NAME,
    VERSION_FULL,
    DataModelAttributes,
    DataModelCoords,
    DataModelDims,
    SimObsType,
)
from dpyverification.datasources.genericdatasource import GenericDatasource


class DataModel:
    """The dpyverification internal DataModel."""

    input: xarray.Dataset
    intermediate: xarray.Dataset
    # output is a @property with an explicit setter

    def __init__(
        self,
        datalist: Sequence[GenericDatasource],
        generalconfig: GeneralInfo,
    ) -> None:
        self.input, coords, time_step = self._construct_input_dataset(datalist, generalconfig)
        self.intermediate = self._create_intermediate_dataset(self.input, coords, time_step)
        self._output = self._initialize_output_dataset(coords, time_step)

    @property
    def output(self) -> xarray.Dataset:
        """The combined output of the verifications."""
        return self._output

    @output.setter
    def output(self, _: xarray.Dataset) -> None:
        msg: str = (
            "DataModel.output cannot be set directly, use the add_to_output method to specify"
            " how to combine the existing output and the new output."
        )
        raise AttributeError(msg)

    @staticmethod
    def _construct_input_dataset(
        datalist: Sequence[GenericDatasource],
        generalconfig: GeneralInfo,
    ) -> tuple[xarray.Dataset, xarray.Coordinates, np.timedelta64]:
        """
        Parse the list of datasources.

        Check whether the datasources form a compatible combination.
        Create an xarray with the combined input information.
        Assign the xarray dataset to self.input.
        """
        # Determine sizes and values of combined dimensions.
        obs_list: list[GenericDatasource] = []
        sim_list: list[GenericDatasource] = []
        time_steps: list[np.timedelta64] = []
        time_starts: list[np.datetime64] = []
        time_ends: list[np.datetime64] = []
        locations_list: list[xarray.Coordinates] = []
        ensemble_list: list[int] = []
        simstart_list: list[np.datetime64] = []
        for ds in datalist:
            obs_list.append(ds) if ds.simobstype == SimObsType.OBS else sim_list.append(ds)

            DataModel._check_source_dims_and_coords(
                ds,
            )  # Method will raise an error when there is a problem
            step, start, end, location, ensemble_numbers, simstart_values = DataModel._parse_source(
                ds,
            )

            time_steps.append(step)
            time_starts.append(start)
            time_ends.append(end)
            locations_list.append(location)
            ensemble_list += ensemble_numbers
            simstart_list += simstart_values

        coords = xarray.Coordinates()
        # TODO(AU): Allow additional dimensions and coordinates, beyond the fixed set # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/10
        #   See also the note related to this in _check_source_dims_and_coords().
        #   The xarray.merge() has certain input flags that can be set, can we use those to trigger
        #   errors on merging empty datasets with a subselection of the dimensions / coordinates, to
        #   then provide as-specific-as-possible error messages to the user? Can do something, e.g.
        #   merge will indeed give error when e.g. loc1 and loc2 have switched lat/lon values, but
        #   it will be cryptic for the end user what the problem is. Therefore e.g. the locations
        #   merge here has a try-except to provide additional error information.

        # Add location coordinates to coords
        try:
            locations = xarray.merge(locations_list)
        except Exception as incompat:
            # TODO(AU): User-readable list of to be able to find the problem # noqa: FIX002
            #   https://github.com/Deltares-research/DPyVerification/issues/13
            msg = "Incompatible locations in combination of datasources"
            raise AttributeError(msg) from incompat
        coords = coords.assign(locations.coords)

        # Add time coordinate to coords
        time_coord, time_step = DataModel._create_time_coord(
            time_starts,
            time_ends,
            time_steps,
            datalist[0].xarray[DataModelCoords.time.name].coords,  # type: ignore[misc] # coords is a DataArrayCoordinates[Any]
        )
        coords = coords.assign(time_coord)

        # TODO(AU): Allow input datasets with leadtime already taken into account # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/11
        #   See issue for full description.
        #   Here, check that leadtime values in input datasets are subset of generalconfig.leadtimes
        #   And, the simstart coordinate needs to be completed with simstarts derived from
        #   the time + leadtime dimensions -> Actually, the simstart_values of  _parse_source()
        #   should be ok for that, just need to be calculated inside that method.

        # Add the other coordinates to get the full set
        leadtimes = generalconfig.leadtimes.timedelta64
        ensemble_list = list(set(ensemble_list))
        simstart_list = list(set(simstart_list))
        additional_coords = {
            DataModelCoords.leadtime.name: leadtimes,
            DataModelCoords.ensemble.name: ensemble_list,
            DataModelCoords.simstart.name: simstart_list,
        }
        coords = coords.assign(additional_coords)

        inputdataset = xarray.Dataset(coords=coords)
        # TODO(AU): Add more checks on the combination before merge() # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/14
        obs_sets = [obs.xarray for obs in obs_list]
        sim_sets = [sim.xarray for sim in sim_list]
        merge_set = [inputdataset, *obs_sets, *sim_sets]
        inputdataset = xarray.merge(merge_set)
        # Register the timestep as an attribute, for easy access
        inputdataset.attrs.update({DataModelAttributes.timestep: time_step})  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.

        # Return the constructed coords, without any changes that the xarray.merge() might have
        #  caused to the self.input coordinates
        return inputdataset, coords, time_step

    @staticmethod
    def _create_intermediate_dataset(
        inputdataset: xarray.Dataset,
        coords: xarray.Coordinates,
        time_step: np.timedelta64,
    ) -> xarray.Dataset:
        # Construct time coordinate for intermediate dataset based on simstart and leadtime values
        leadtimes = coords[DataModelCoords.leadtime.name].data  # type: ignore[misc]  # Yes, data is a Any array, we assume it is compatible with np.min and np.max
        simstarts = coords[DataModelCoords.simstart.name].data  # type: ignore[misc]  # Yes, data is a Any array, we assume it is compatible with np.min and np.max
        time_start: np.datetime64 = np.min(simstarts) + np.min(leadtimes)  # type: ignore[misc] # Yes, simstarts and leadtimes are a Any array, we assume it is compatible with np.min and np.max
        time_end: np.datetime64 = np.max(simstarts) + np.max(leadtimes)  # type: ignore[misc] # Yes, simstarts and leadtimes are a Any array, we assume it is compatible with np.min and np.max
        time_values = np.arange(
            time_start,
            time_end + time_step,
            time_step,
            dtype=np.datetime64,
        )

        # Check there are no additional coordinate variables that use the time dimension, and would
        #   need to be adapted, e.g. additional coordinates inherited from the datasources
        for coordname in coords:
            if (
                DataModelCoords.time.name != coordname  # Except for time coordinate itself
                and DataModelDims.time
                in coords[coordname].dims  # No other coordinate should have the time dimension
            ):
                msg = (
                    f"Coordinate {coordname} uses the {DataModelDims.time} dimension, creating"
                    " an intermediate dataset in this situation is not implemented yet."
                )
                raise NotImplementedError(msg)

        update_coords = {
            DataModelCoords.time.name: time_values,
        }
        coords = coords.assign(update_coords)
        intermediatedataset = xarray.Dataset(coords=coords)

        # For data variables with a simstart dimension, extract only specific values
        # For data variables with neither a simstart nor a leadtime dimension, extract values at
        #   intermediate dataset time locations
        # For data variables with a leadtime dimension, extract values at intermediate dataset time
        #   locations (?)
        # TODO(AU): Allow input datasets with leadtime already taken into account # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/11
        #   See issue for full description.
        #   Here, for variables with a leadtime dimension, extract values at intermediate dataset
        #     time locations (?)

        def transform_to_intermediate_data_variable(datavar: xarray.DataArray) -> xarray.DataArray:
            """Transform a variable to intermediate datavariable."""
            if DataModelDims.simstart in datavar.dims and DataModelDims.leadtime in datavar.dims:
                msg = (
                    f"Data variables are expected to have at maximum one of the"
                    f" {DataModelDims.leadtime} and {DataModelDims.simstart} dimensions. Use of"
                    f" variables that have both of these dimensions is not supported."
                )
                raise ValueError(msg)
            if (
                DataModelDims.simstart not in datavar.dims
                and DataModelDims.leadtime not in datavar.dims
            ):
                select_at = {DataModelCoords.time.name: time_values}
                return datavar.sel(select_at)
            if DataModelDims.leadtime in datavar.dims:
                msg = f"Data variables with {DataModelDims.leadtime} dimension not supported yet."
                raise NotImplementedError(msg)

            leadtime: np.timedelta64
            for index, leadtime in enumerate(leadtimes):  # type: ignore[misc]  # Yes, leadtimes is a Any array, we assume it is a compatible with np.min and np.max
                # Select all values at specific simstart - time combinations
                #   For each simstart, since inside loop for specific leadtime, want only values
                #   for one specific time.
                #   Based on http://xarray.pydata.org/en/stable/indexing.html#more-advanced-indexing,
                #   pointwise indexing can be done by creating DataArrays for indexing,
                #   including what resulting dimension / coordinates the values map to.
                select_at = {
                    DataModelCoords.time.name: list(  # type: ignore[dict-item]
                        inputdataset[DataModelCoords.simstart.name].data + leadtime,  # type: ignore[misc]
                    ),
                    DataModelCoords.simstart.name: xarray.DataArray(  # type: ignore[dict-item]
                        inputdataset[DataModelCoords.simstart.name].data,  # type: ignore[misc]
                        dims=DataModelDims.time,
                    ),
                }
                is_first_iteration = not index
                if is_first_iteration:
                    intermediatedataset[datavar.name] = datavar.sel(select_at).expand_dims(
                        dim={"leadtime": [leadtime]},
                        axis=len(datavar.dims) - 1,
                    )
                else:
                    intermediatedataset[datavar.name] = intermediatedataset[
                        datavar.name
                    ].combine_first(
                        datavar.sel(select_at).expand_dims(
                            dim={"leadtime": [leadtime]},
                            axis=len(datavar.dims) - 1,
                        ),
                    )
            return intermediatedataset[varname]

        for varname in inputdataset.data_vars:
            intermediatedataset[varname] = transform_to_intermediate_data_variable(
                inputdataset.data_vars[varname],
            )
        return intermediatedataset

    @staticmethod
    def _initialize_output_dataset(
        coords: xarray.Coordinates,
        time_step: np.timedelta64,
    ) -> xarray.Dataset:
        """Initialize the output dataset with coordinates and attributes."""
        # TODO(AU): Add leadtime dimension and coordinate during initialization # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/15

        outputdataset = xarray.Dataset(coords=coords)
        # TODO(AU): Refactor the _output.attrs update calls here # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/17

        # Register the timestep as an attribute, for easy access
        outputdataset.attrs.update({DataModelAttributes.timestep: time_step})  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        # Register how this output was created
        source_str = NAME + " version " + VERSION_FULL
        outputdataset.attrs.update({DataModelAttributes.source: source_str})  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        outputdataset.attrs.update({DataModelAttributes.featuretype: "timeSeries"})  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        # Make sure the location_id variable (a string array in python) is encoded as NC_CHAR in
        #   netcdf export, to be CF compliant
        to_char = {"dtype": "S1"}
        outputdataset[DataModelCoords.location.name].encoding.update(to_char)  # type: ignore[misc]  # Yes, encoding is een any-any dict, however here we only add to it.
        # Update all coordinates with (CF compliancy) attributes
        outputdataset[DataModelCoords.time.name].attrs.update(DataModelCoords.time.attributes)  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        outputdataset[DataModelCoords.location.name].attrs.update(  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
            DataModelCoords.location.attributes,
        )
        outputdataset[DataModelCoords.lat.name].attrs.update(DataModelCoords.lat.attributes)  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        outputdataset[DataModelCoords.lon.name].attrs.update(DataModelCoords.lon.attributes)  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        outputdataset[DataModelCoords.ensemble.name].attrs.update(  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
            DataModelCoords.ensemble.attributes,
        )
        outputdataset[DataModelCoords.simstart.name].attrs.update(  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
            DataModelCoords.simstart.attributes,
        )

        return outputdataset

    @staticmethod
    def _check_source_dims_and_coords(ds: GenericDatasource) -> None:
        # all ds should have these dimensions
        obs_dims = frozenset(
            [
                DataModelDims.time,
                DataModelDims.location,
            ],
        )
        obs_coords = frozenset(
            [
                DataModelCoords.time.name,
                DataModelCoords.location.name,
                DataModelCoords.lat.name,
                DataModelCoords.lon.name,
            ],
        )
        # sim ds are allowed to have these dimensions
        # TODO(AU): Allow input datasets with leadtime already taken into account # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/11
        #   See issue for full description.
        #   Here, need to have simstart, or can do without? Will depend on whether leadtime already
        #   taken into account in the ds? So need either simstart or leadtime? Can have both?
        sim_dims = [DataModelDims.ensemble, DataModelDims.simstart, *obs_dims]
        sim_coords = [
            DataModelCoords.ensemble.name,
            DataModelCoords.simstart.name,
            *obs_coords,
        ]

        # TODO(AU): Allow additional dimensions and coordinates, beyond the fixed set # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/10
        #   In addition to the known dimensions and coordinates, data might be provided that has
        #   for instance a temperature profile over a set of heights/depths. Should every possible
        #   dimension/coordinate become a hardcoded option, or can we allow generic extras? What
        #   should then be the demands on those extra dimensions/coordinates. And, what if
        #   dimensions without a coordinate are used, how to know that two datasets mean the same
        #   thing with the dimension, if there are no coordinates at all that use the dimension?
        #   Will require adaptation both here, and additional checks on the combination of the
        #   datasets.

        if ds.simobstype == SimObsType.OBS:
            if frozenset(ds.xarray.sizes) != obs_dims:
                msg = "For Observations data, the exact required dimensions are: " + str(
                    obs_dims,
                )
                raise ValueError(msg)
            if frozenset(ds.xarray.coords) != obs_coords:
                msg = "For Observations data, the exact required coordinates are: " + str(
                    obs_dims,
                )
                raise ValueError(msg)
        else:
            # For sim data, need at least obs dims and coords, and can have additional entries
            # from the list of allowed sim dims and coords
            if not all(x in ds.xarray.dims for x in obs_dims):
                msg = "For Simulations data, the minimum required dimensions are: " + str(
                    obs_dims,
                )
                raise ValueError(msg)
            if any(x not in sim_dims for x in ds.xarray.dims):
                msg = "For Simulations data, the only allowed dimensions are: " + str(sim_dims)
                raise ValueError(msg)
            if not all(x in ds.xarray.coords for x in obs_coords):
                msg = "For Simulations data, the minimum required coordinates are: " + str(
                    obs_coords,
                )
                raise ValueError(msg)
            if any(x not in sim_coords for x in ds.xarray.coords):
                msg = "For Simulations data, the only allowed coordinates are: " + str(sim_coords)
                raise ValueError(msg)

    @staticmethod
    def _parse_source(
        ds: GenericDatasource,
    ) -> tuple[
        np.timedelta64,
        np.datetime64,
        np.datetime64,
        xarray.Coordinates,
        list[int],
        list[np.datetime64],
    ]:
        if not ds.xarray.sizes[DataModelCoords.time.name] > 1:
            # Not inside _check_source_dims_and_coords, because might want to allow this, in that
            # case directly related to the code line following
            msg = "Scalar time dimension not supported"
            raise ValueError(msg)
        time_coord: xarray.DataArray = ds.xarray.time

        # register the start, end and timestep of the time dimension
        start: np.datetime64 = min(time_coord.data)  # type: ignore[misc] # Due to the time_coord numpy array
        end: np.datetime64 = max(time_coord.data)  # type: ignore[misc] # Due to the time_coord numpy array
        timediffs: list[np.timedelta64] = list(np.unique(np.diff(time_coord.data)))  # type: ignore[misc] # Due to the time_coord numpy array
        if len(timediffs) > 1:
            msg = "Time dimension should be uniformly increasing"
            raise ValueError(msg)

        # This will return a Coordinates object, that holds has the lat and lon coordinates (i.e.
        #  all coordinates for the dimensions of ds.xarray[DataModelCoords.location.name])
        location = ds.xarray[DataModelCoords.location.name].coords  # type: ignore[misc] # coords is a DataArrayCoordinates[Any]

        # Note: in the following statements, use list(X) since that conserves numpy datatype, using
        #  X.tolist() would convert to python type

        if DataModelCoords.ensemble.name in ds.xarray.coords:
            # TODO(AU): Add more checks on the the datasources before merge() # noqa: FIX002
            #   https://github.com/Deltares-research/DPyVerification/issues/14
            # Here, check that ensemble_members are indeed int. Or already
            #  in _check_source_dims_and_coords?
            ens: list[int] = list(ds.xarray[DataModelCoords.ensemble.name].data)  # type: ignore[misc]
        else:
            ens = []

        if DataModelCoords.simstart.name in ds.xarray.coords:
            # TODO(AU): Add more checks on the the datasources before merge() # noqa: FIX002
            #   https://github.com/Deltares-research/DPyVerification/issues/14
            # Here, check that simstart are indeed np.datetime64. Or already
            #  in _check_source_dims_and_coords?
            simstart: list[np.datetime64] = list(ds.xarray[DataModelCoords.simstart.name].data)  # type: ignore[misc]
        else:
            simstart = []

        return (
            timediffs[0],
            start,
            end,
            location,  # type: ignore[misc] # coords is a DataArrayCoordinates[Any]
            ens,
            simstart,
        )

    @staticmethod
    def _create_time_coord(
        time_starts: list[np.datetime64],
        time_ends: list[np.datetime64],
        time_steps: list[np.timedelta64],
        default_time_array: xarray.Coordinates,
    ) -> tuple[xarray.Coordinates, np.timedelta64]:
        if len(set(time_steps)) == 1 and len(set(time_starts)) == 1 and len(set(time_ends)) == 1:
            time_coord = default_time_array
            time_step = time_steps[0]
        else:
            if len(set(time_steps)) != 1:
                # This will require quite some thought:
                #   Use smallest timestep and fill in intermediate missing steps?
                #   Or use largest timestep, and resample the others?
                #   Or require user to specify timestep, and how to handle?
                # Note that this method is used twice, check that both uses have same requirements
                msg = (
                    "Time dimensions of the input data sources do not all have"
                    " the same timestep."
                )
                raise NotImplementedError(msg)
            time_step = time_steps[0]
            time_start = np.min(time_starts)
            time_end = np.max(time_ends)
            time_values = np.arange(
                time_start,
                time_end + time_step,
                time_step,
                dtype=np.datetime64,
            )
            if not all(x in time_values for x in time_starts) or not all(
                x in time_values for x in time_ends
            ):
                msg = (
                    "Time dimensions have the same timestep, but with a timeshift for some"
                    " of the input data sources."
                )

                raise NotImplementedError(msg)
            coord_dict = {DataModelCoords.time.name: time_values}
            time_coord = xarray.Coordinates(coord_dict)
        return time_coord, time_step

    def add_to_output(self, new_output: xarray.Dataset | xarray.DataArray) -> None:
        """Add the result of a specific verification to the datamodel output."""
        # Perform various checks on the combination
        # Merge together if the checks pass

        # TODO(AU): Output data content requirements are not complete # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/26
        #   Here, check that the to-be-added output does not overwrite any existing variables
        #   OR, allow appending to a certain dimension?
        #   OR, allow overwriting if only NaNs are overwritten (i.e. the var was created with only
        #   partial data)?
        if not isinstance(new_output, xarray.Dataset | xarray.DataArray):  # type: ignore[misc]
            msg = "Expected type xr.DataArray or xr.Dataset, got type(new_output)"  # type: ignore[unreachable] # mypy assumes the right type is always provided, but additional check is needed.
            raise TypeError(msg)

        # Check that the to-be-added output does not overwrite any existing variables
        if isinstance(new_output, xarray.DataArray):  # type: ignore[misc]
            # Check if DataArray has a name
            if new_output.name is None:
                msg = "DataArray has no name"
                raise ValueError(msg)
            a = [str(new_output.name)]
        else:
            a = [str(x) for x in new_output.data_vars]
        b = [str(x) for x in self.output.data_vars]
        match = any(var in b for var in a)
        if match:
            msg = (
                "Cannot add to output, variables with same name already present in output."
                " Existing: " + str(b) + " To be added: " + str(a)
            )
            raise RuntimeError(msg)

        # Check that dimensions and coordinates match (except time dimension)
        #
        # TODO(AU): Check dims and coords match when combining calculation outputs # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/27
        #   First, implement strictest check, entirely the same. Follow up with next TODO item.
        #
        # TODO(AU): Output data content requirements are not complete # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/26
        #   Here, check that dimensions and coordinates match
        #   OR, allow extending of dimensions

        # Check time dimension and coordinate
        #
        # TODO(AU): Output data content requirements are not complete # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/26
        #   Here, register the start, end and timestep of the time dimension
        #   xarray.merge will not complain about adding intermediate times, but we want to have
        #   a fixed timestep. OR, allow non-monotonic timeseries?

        # Validate time coordinates, but only if dim is present.
        if DataModelDims.time in new_output.dims:
            try:
                self._output[DataModelCoords.time.name].sel(
                    {DataModelCoords.time.name: new_output[DataModelCoords.time.name].data},  # type: ignore[misc] # data is Any, we assume np.datetime64 array
                )
            except KeyError as mismatch:
                # new times are not a subset of existing times, what to do?
                timestep: np.timedelta64 = self.output.attrs[DataModelAttributes.timestep]  # type: ignore[misc] # Due to the Any attrs
                timestart: np.datetime64 = min(self.output[DataModelCoords.time.name].data)  # type: ignore[misc] # Due to the numpy array .data
                new_time = new_output[DataModelCoords.time.name]
                new_start: np.datetime64 = min(new_time.data)  # type: ignore[misc] # Due to the numpy array .data
                new_diffs = np.unique(np.diff(new_time.data))  # type: ignore[misc] # Due to the time_coord numpy array
                # All timediffs should be larger and integer divisible with timestep of _output
                is_compatible = all(new_diffs % timestep == 0)  # type: ignore[misc] # Due to the numpy array new_diffs
                if is_compatible:
                    # Even though moduli within new_time are ok, might still be at an offset
                    # Since moduli ok, if offset of one new time ok, all new times ok
                    is_compatible = not ((timestart - new_start) % timestep)
                if not is_compatible:
                    msg = (
                        f"Timecoordinate values of new output are not compatible: not all values are"
                        f" at a position start_time {min(self.output[DataModelCoords.time.name].data)}"  # type: ignore[misc] # Due to the numpy arrays
                        f" +/- integer multiple of timestep {timestep}"
                    )
                    raise ValueError from mismatch
                # The times are compatible, however just doing an xarray merge does not guarantee a full
                #  monotonic time. Therefore, explicitly create new time coord values
                time_starts = [timestart, new_start]
                time_ends: list[np.datetime64] = [
                    max(self.output[DataModelCoords.time.name].data),  # type: ignore[misc] # Due to the numpy arrays
                    max(new_time.data),  # type: ignore[misc] # Due to the numpy arrays
                ]

                time_coord, time_step = self._create_time_coord(
                    time_starts,
                    time_ends,
                    [timestep],
                    self.output[DataModelCoords.time.name].coords,  # type: ignore[misc] # coords is a DataArrayCoordinates[Any]
                )
                self._output = self._output.merge(time_coord)

        # Final merge
        #   No conflicts between the to-be-added output and earlier created outputs
        #
        # TODO(AU): Output data content requirements are not complete # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/26
        #   Do we indeed want to use xarray.merge, without any qualifiers?
        self._output = self._output.merge(new_output)
