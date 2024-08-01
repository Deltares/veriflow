"""Module with the dpyverification internal DataModel."""

from collections.abc import Sequence

import numpy as np
import xarray

from dpyverification.constants import SimObsType
from dpyverification.datasources.genericdatasource import GenericDatasource


class DataModelCoords:
    """List of coordinate names.

    To avoid hardcoded strings in multiple places,
    have a single list with the names of known coordinates / dimensions.
    """

    time = "time"
    location = "location_id"
    ensemble = "ensemble_member"


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
        raise ValueError(msg)

    def _xarrays_from_inputs(self, datalist: Sequence[GenericDatasource]) -> None:
        time_coord: xarray.DataArray

        # Determine sizes and values of combined dimensions.
        time_starts: list[np.datetime64] = []
        time_ends: list[np.datetime64] = []
        time_steps: list[np.timedelta64] = []
        obs_list: list[GenericDatasource] = []
        sim_list: list[GenericDatasource] = []
        locations_list: list[str] = []
        ensemble_list: list[int] = []
        for ds in datalist:
            # all ds should have these dimensions
            obs_dims = frozenset([DataModelCoords.time, DataModelCoords.location])
            sim_dims = [DataModelCoords.ensemble, *obs_dims]
            if ds.simobstype == SimObsType.obs:
                if frozenset(ds.xarray.sizes) != obs_dims:
                    msg = "For Observations data, the exact required dimensions are: " + str(
                        obs_dims,
                    )
                    raise ValueError(msg)
                obs_list.append(ds)
            if ds.simobstype == SimObsType.sim:
                if not all(x in ds.xarray.dims for x in obs_dims):
                    msg = "For Simulations data, the minimum required dimensions are: " + str(
                        obs_dims,
                    )
                    raise ValueError(msg)
                if any(x not in sim_dims for x in ds.xarray.dims):
                    msg = "For Simulations data, the only allowed dimensions are: " + str(sim_dims)
                    raise ValueError(msg)
                sim_list.append(ds)

            if not ds.xarray.sizes[DataModelCoords.time] > 1:
                msg = "Scalar time dimension not supported"
                raise ValueError(msg)

            time_coord = ds.xarray.time

            # register the start, end and timestep of the time dimension
            timediffs = np.unique(np.diff(time_coord.data))  # type: ignore[misc] # Due to the time_coord numpy array
            if len(timediffs) > 1:  # type: ignore[misc] # Due to the time_coord numpy array
                msg = "Time dimension should be uniformly increasing"
                raise ValueError(msg)
            time_steps.append(timediffs[0])  # type: ignore[misc] # Due to the time_coord numpy array
            time_starts.append(min(time_coord.data))  # type: ignore[misc] # Due to the time_coord numpy array
            time_ends.append(max(time_coord.data))  # type: ignore[misc] # Due to the time_coord numpy array

            # SHOULD CHECK that location_ids are indeed strings
            l_temp: list[str] = ds.xarray[DataModelCoords.location].data.tolist()  # type: ignore[misc]
            locations_list += l_temp

            if DataModelCoords.ensemble in ds.xarray.dims:
                # SHOULD CHECK that ensemble_members are indeed int
                e_temp: list[int] = ds.xarray[DataModelCoords.ensemble].data.tolist()  # type: ignore[misc]
                ensemble_list += e_temp

        time_coord = self._create_time_coord(time_starts, time_ends, time_steps, datalist)
        unique_locations = list(set(locations_list))
        unique_ensembles = list(set(ensemble_list))

        coords = {
            DataModelCoords.time: time_coord.data,  # type: ignore[misc]  # Due to the numpy arrays
            DataModelCoords.location: unique_locations,
            DataModelCoords.ensemble: unique_ensembles,
        }

        self.input = xarray.Dataset(coords=coords)  # type: ignore[misc]  # Due to the numpy arrays
        # THERE REALLY HAS NOT BEEN ENOUGH CHECKING YET, e.g. on variable names being unique
        obs_sets = [obs.xarray for obs in obs_list]
        sim_sets = [sim.xarray for sim in sim_list]
        merge_set = [self.input, *obs_sets, *sim_sets]
        self.input = xarray.merge(merge_set)

        # Add extra output dimensions / coordinates here, e.g. leadtime

        self._output = xarray.Dataset(coords=coords)  # type: ignore[misc]  # Due to the numpy arrays

    @staticmethod
    def _create_time_coord(
        time_starts: list[np.datetime64],
        time_ends: list[np.datetime64],
        time_steps: list[np.timedelta64],
        datalist: Sequence[GenericDatasource],
    ) -> xarray.DataArray:
        time_coord: xarray.DataArray

        if len(set(time_steps)) == 1 and len(set(time_starts)) == 1 and len(set(time_ends)) == 1:
            time_coord = datalist[0].xarray.time
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
            time_coord = xarray.DataArray(
                np.arange(time_start, time_end + time_step, time_step, dtype=np.datetime64),  # type: ignore[misc] # Due to the numpy array
            )
            if not all(x in time_coord for x in time_starts) or not all(
                x in time_coord for x in time_ends
            ):
                msg = (
                    "Time dimensions have the same timestep, but with a timeshift for some"
                    " of the input data sources."
                )

                raise NotImplementedError(msg)
        return time_coord

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

        # Ok to add
        # No conflicts between the to-be-added output and earlier created outputs
        # Do we indeed want to use merge here?
        self._output = self._output.merge(new_output)
