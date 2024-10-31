"""Create simobspair for specific lead times."""

import xarray
from numpy import datetime64, timedelta64

from dpyverification.configuration import Calculation, Config
from dpyverification.constants import CalculationTypeEnum, DataModelCoords, DataModelDims
from dpyverification.datamodel import DataModel


def simobspairs(
    calcconfig: Calculation,
    data: DataModel,
    fullconfig: Config,
) -> xarray.Dataset:
    """Create pairs of obs and sim values, for the given leadtimes (default leadtime 0)."""
    if calcconfig.calculationtype != CalculationTypeEnum.simobspairs:
        msg = "Input calcconfig does not have datasourcetype simobspairs"
        raise TypeError(msg)
    # TODO(AU): Adapt config creation to be able to pass general items to other parts # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/21
    #   Here, should be able to use calcconfig.leadtimes unconditionally
    if calcconfig.leadtimes:
        # TODO(AU): Check that specific leadtimes are subset of general leadtimes # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/16
        #   Here, this if-elif should then no longer be needed.
        leadtimes = calcconfig.leadtimes.timedelta64
    elif fullconfig.content.general.leadtimes:
        leadtimes = fullconfig.content.general.leadtimes.timedelta64
    else:
        leadtimes = [timedelta64(0)]

    leadsets = []
    # TODO(AU): Allow input datasets with leadtime already taken into account # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/11
    #   See issue for full description.
    #   Here, adapt to use intermediate dataset as source.
    for leadtime in leadtimes:
        # TODO(AU): Add unit test on simobspair creation # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/33
        #   Here, make this a function? Have data.input as argument, instead of full data?
        leadset = data.input.coords.to_dataset()
        newtime: list[datetime64] = list(
            data.input[DataModelCoords.simstart.name].data + leadtime,  # type: ignore[misc] # Quite certain that data.input[DataModelCoords.time.name].data will be a 1D array of datetime64
        )
        new_coords = {DataModelCoords.time.name: newtime}
        leadset = leadset.assign_coords(new_coords)
        for pair in calcconfig.variablepairs:
            # Construct variable names:
            #   varnamegeneral_calctypename_varname
            # Where
            # - varnamegeneral is assumed equal to obsvar name
            # - calctypename is taken to be equal to enum string value
            outnamesim = f"{pair.obs}_{CalculationTypeEnum.simobspairs}_{pair.sim}"
            outnameobs = f"{pair.obs}_{CalculationTypeEnum.simobspairs}_{pair.obs}"

            # TODO(AU): Add unit test on simobspair creation # noqa: FIX002
            #   https://github.com/Deltares-research/DPyVerification/issues/33
            #   Have a unit test with leadtimes that are incompatible with the available data.
            #   Additional thoughts on that, from earlier:
            #   Wait for adaptation of example input files, when smaller, can use large leadtime
            #   to be beyond end. To test what if any newtime values are not part of the input time
            #   dimension? -> Will give KeyError. What do we want to do in that case? Skip leadtime
            #   entirely? Or do create, but fully empty? Truncate newtime at min and max of time can
            #   be a first step, to only get valid time values. But what if newtime is then empty?

            # Parse the obs values
            select_at = {
                DataModelCoords.time.name: leadset[DataModelCoords.time.name],
            }
            vals = data.input[pair.obs].sel(select_at)
            leadset[outnameobs] = vals.expand_dims(
                dim={"leadtime": [leadtime]},
                axis=len(vals.dims),
            )
            if "units" in data.input[pair.obs].attrs:  # type: ignore[misc] # attrs is a dict[Any,Any]
                leadset[outnameobs].attrs.update({"units": data.input[pair.obs].attrs["units"]})  # type: ignore[misc] # attrs is a dict[Any,Any]

            # Parse the sim values
            #
            # Select all sim values at specific simstart - time combinations
            #   For each simstart, since inside loop for specific leadtime, want only values for one
            #   specific time.
            #   Based on http://xarray.pydata.org/en/stable/indexing.html#more-advanced-indexing,
            #   pointwise indexing can be done by creating DataArrays for indexing, including what
            #   resulting dimension / coordinates the values map to.
            select_at[DataModelCoords.simstart.name] = xarray.DataArray(
                data.input[DataModelCoords.simstart.name].data,  # type: ignore[misc] # Quite certain that data.input[DataModelCoords.simstart.name].data will be a 1D array of datetime64
                dims=DataModelDims.time,
            )
            vals = data.input[pair.sim].sel(select_at)
            leadset[outnamesim] = vals.expand_dims(
                dim={"leadtime": [leadtime]},
                axis=len(vals.dims),
            )
            if "units" in data.input[pair.sim].attrs:  # type: ignore[misc] # attrs is a dict[Any,Any]
                leadset[outnamesim].attrs.update({"units": data.input[pair.sim].attrs["units"]})  # type: ignore[misc] # attrs is a dict[Any,Any]
        leadsets.append(leadset)
    # merge will expand time to cover all leadtimes
    return xarray.merge(leadsets)
