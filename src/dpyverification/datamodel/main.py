"""Module with the dpyverification internal DataModel."""

from collections.abc import Sequence

import numpy as np
import xarray

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

    def __init__(self, datalist: Sequence[GenericDatasource]) -> None:
        # unpack the list
        self._xarrays_from_inputs(datalist)

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

    def _xarrays_from_inputs(self, datalist: Sequence[GenericDatasource]) -> None:
        """
        Parse the list of datasources.

        Check whether the datasources form a compatible combination.
        Create an xarray with the combined input information.
        Initialize the output xarray.
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
            obs_list.append(ds) if ds.simobstype == SimObsType.obs else sim_list.append(ds)

            self._check_source_dims_and_coords(
                ds,
            )  # Method will raise an error when there is a problem
            step, start, end, location, ensemble_numbers, simstart_values = self._parse_source(ds)

            time_steps.append(step)
            time_starts.append(start)
            time_ends.append(end)
            locations_list.append(location)
            ensemble_list += ensemble_numbers
            simstart_list += simstart_values

        coords = xarray.Coordinates()

        # Add location coordinates to coords
        try:
            locations = xarray.merge(locations_list)
        except Exception as incompat:
            # STILL NEEDS: list of ids, lat, lon, ordered by id, to be able to find the problem
            msg = "Incompatible locations in combination of datasources"
            raise AttributeError(msg) from incompat
        coords = coords.assign(locations.coords)

        # Add time coordinate to coords
        time_coord, time_step = self._create_time_coord(
            time_starts,
            time_ends,
            time_steps,
            datalist[0].xarray[DataModelCoords.time.name].coords,  # type: ignore[misc] # coords is a DataArrayCoordinates[Any]
        )
        coords = coords.assign(time_coord)

        # SHOULD HERE CHECK If leadtimes defined, check that they are multiples of time_step

        # What if dimensions without a coordinate are used, how to know that two datasets mean
        # the same thing with the dimension, if there are no coordinates at all that use the
        # dimension?
        # The xarray.merge() has certain input flags that can be set, can we use those to trigger
        # errors on merging empty datasets with a subselection of the dimensions / coordinates, to
        # then provide as-specific-as-possible error messages to the user? -> Can do something, e.g.
        # merge will indeed give error when e.g. loc1 and loc2 have switched lat/lon values, but it
        # will be cryptic for the end user what the problem is.

        # When we allow datasets with leadtime already taken into account, cannot mix with simstart
        #  based datasets? In that case, need to parse the simstart datasets approximately HERE
        #  into leadtime datasets.

        # Add the other coordinates to get the full set
        ensemble_list = list(set(ensemble_list))
        simstart_list = list(set(simstart_list))
        additional_coords = {
            DataModelCoords.ensemble.name: ensemble_list,
            DataModelCoords.simstart.name: simstart_list,
        }
        coords = coords.assign(additional_coords)

        self.input = xarray.Dataset(coords=coords)
        # THERE REALLY HAS NOT BEEN ENOUGH CHECKING YET, e.g. on variable names being unique
        # And, what if different sims have different number of ensembles?
        obs_sets = [obs.xarray for obs in obs_list]
        sim_sets = [sim.xarray for sim in sim_list]
        merge_set = [self.input, *obs_sets, *sim_sets]
        self.input = xarray.merge(merge_set)
        # Register the timestep as an attribute, for easy access
        self.input.attrs.update({DataModelAttributes.timestep: time_step})  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.

        # Add extra output dimensions / coordinates here, e.g. leadtime
        # Do make sure to check that that does not affect the self.input
        # On leadtime: do we even want to allow different leadtimes for different calculations?
        #   Because, that would make the leadtime dimension very irregular, and introduce a lot of
        #   missing values, when parameters use different lead times, but the same leadtime coord.
        #   Could alternatively have a calculation-specific leadtime dimension, and only when the
        #   calculation uses different leadtimes than the general leadtimes?
        #   Update: calc specific leadtimes need to be a subset of the general leadtimes
        # Set units attribute on leadtime, and/or use timedelta64 for the leadtime coordinate?
        #   Depending on answer, also need to update simobspairs use of leadtime.

        self._output = xarray.Dataset(coords=coords)
        # Register the timestep as an attribute, for easy access
        self._output.attrs.update({DataModelAttributes.timestep: time_step})  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        # Register how this output was created
        source_str = NAME + " version " + VERSION_FULL
        self._output.attrs.update({DataModelAttributes.source: source_str})  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        self._output.attrs.update({DataModelAttributes.featuretype: "timeSeries"})  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        # Make sure the location_id variable (a string array in python) is encoded as NC_CHAR in
        #   netcdf export, to be CF compliant
        to_char = {"dtype": "S1"}
        self._output[DataModelCoords.location.name].encoding.update(to_char)  # type: ignore[misc]  # Yes, encoding is een any-any dict, however here we only add to it.
        # Update all coordinates with (CF compliancy) attributes
        self._output[DataModelCoords.time.name].attrs.update(DataModelCoords.time.attributes)  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        self._output[DataModelCoords.location.name].attrs.update(  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
            DataModelCoords.location.attributes,
        )
        self._output[DataModelCoords.lat.name].attrs.update(DataModelCoords.lat.attributes)  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        self._output[DataModelCoords.lon.name].attrs.update(DataModelCoords.lon.attributes)  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
        self._output[DataModelCoords.ensemble.name].attrs.update(  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
            DataModelCoords.ensemble.attributes,
        )
        self._output[DataModelCoords.simstart.name].attrs.update(  # type: ignore[misc]  # Yes, attrs is een any-any dict, however here we only add to it.
            DataModelCoords.simstart.attributes,
        )

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
        # DO THEY need to have simstart, or can do without? Will depend on whether leadtime already
        #   taken into account in the ds? So need either simstart or leadtime?
        sim_dims = [DataModelDims.ensemble, DataModelDims.simstart, *obs_dims]
        sim_coords = [DataModelCoords.ensemble.name, DataModelCoords.simstart.name, *obs_coords]

        if ds.simobstype == SimObsType.obs:
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
            # SHOULD CHECK that ensemble_members are indeed int
            ens: list[int] = list(ds.xarray[DataModelCoords.ensemble.name].data)  # type: ignore[misc]
        else:
            ens = []

        if DataModelCoords.simstart.name in ds.xarray.coords:
            # SHOULD CHECK that simstart are indeed np.datetime64
            # SHOULD CHECK that simstart is part of the time coord values (actually, just that it is
            #  at a valid timediff, does not need to be part of the time coord values)
            #  Because simstart+leadtime needs to be a potentially valid time coord value
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

    def add_to_output(self, new_output: xarray.Dataset) -> None:
        """Add the Dataset, with the result of a specific verification, to the datamodel output."""
        # check that the to-be-added output does not overwrite any existing variables
        # OR, allow appending to a certain dimension?
        # OR, allow overwriting if only NaNs are overwritten (i.e. the var was created with only
        #  partial data)?
        a = [str(x) for x in new_output.data_vars]
        b = [str(x) for x in self.output.data_vars]
        match = any(var in b for var in a)
        if match:
            msg = (
                "Cannot add to output, variables with same name already present in output."
                " Existing: " + str(b) + " To be added: " + str(a)
            )
            raise RuntimeError(msg)

        # check that dimensions and coordinates match
        # OR, allow extending of dimensions

        # register the start, end and timestep of the time dimension
        #   xarray.merge will not complain about adding intermediate times, but we want to have
        #   a fixed timestep
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

        # Ok to add
        # No conflicts between the to-be-added output and earlier created outputs
        # Do we indeed want to use merge here?
        self._output = self._output.merge(new_output)
