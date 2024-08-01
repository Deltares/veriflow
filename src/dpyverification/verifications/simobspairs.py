"""Create simobspair for specific lead times."""

import xarray
from numpy import datetime64, timedelta64

from dpyverification.configuration import Calculation, Config
from dpyverification.constants import CalculationTypeEnum
from dpyverification.datamodel import DataModel, DataModelCoords


def simobspairs(
    calcconfig: Calculation,
    data: DataModel,
    fullconfig: Config,
) -> xarray.Dataset:
    """Create pairs of obs and sim values, for the given leadtimes (default leadtime 0)."""
    if calcconfig.calculationtype != CalculationTypeEnum.simobspairs:
        msg = "Input calcconfig does not have datasourcetype simobspairs"
        raise TypeError(msg)
    if calcconfig.leadtimes:
        leadtimes = calcconfig.leadtimes
    elif fullconfig.general.leadtimes:
        leadtimes = fullconfig.general.leadtimes
    else:
        leadtimes = [0]
    leadsets = []
    for leadtime in leadtimes:
        leadset = data.input.coords.to_dataset()
        # need to document that leadtime is expected to be in minutes
        newtime: list[datetime64] = list(
            data.input[DataModelCoords.time].data + timedelta64(leadtime, "m"),  # type: ignore[misc] # Quite certain that data.input[DataModelCoords.time].data will be a 1D array of datetime64
        )
        newcoord = {DataModelCoords.time: newtime}
        leadset = leadset.assign_coords(newcoord)
        for pair in calcconfig.variablepairs:
            # varnamegeneral_calctypename_simvar_leadtime
            # Where
            # - varnamegeneral is assumed equal to obsvar name
            # - calctypename is taken to be equal to enum string value
            outnamesim = f"{pair.obs}_{CalculationTypeEnum.simobspairs}_{pair.sim}_{leadtime}"
            outnameobs = f"{pair.obs}_{CalculationTypeEnum.simobspairs}_{pair.obs}_{leadtime}"
            dims = data.input[pair.obs].dims
            leadset[outnameobs] = (dims, data.input[pair.obs].data)  # type: ignore[misc] # data.input[pair.obs].data is indeed Any, but no problem when assigning to a dataset again
            # sim can have an extra ensemble_member / realizations dimension,
            #  how to handle that in the sim-obs-pairs output?
            dims = data.input[pair.sim].dims
            leadset[outnamesim] = (dims, data.input[pair.sim].data)  # type: ignore[misc] # data.input[pair.obs].data is indeed Any, but no problem when assigning to a dataset again
        leadsets.append(leadset)
    # merge will expand time to cover all leadtimes
    return xarray.merge(leadsets)
