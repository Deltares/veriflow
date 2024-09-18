"""PI-XML support module."""

import math
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import xarray as xr
from fewsio.pi import (  # type: ignore[import-untyped] # See comment below imports
    Timeseries,
    TimeseriesId,
)

from dpyverification.configuration import DataSource
from dpyverification.constants import (
    DataModelCoords,
    DataModelDims,
    DataSourceTypeEnum,
    SimObsType,
)
from dpyverification.datasources.genericdatasource import GenericDatasource

if TYPE_CHECKING:
    import datetime


# Ignore import untyped on fewsio since no type stub available. Unfortunately, this also means
#  that almost all locations where types from fewsio are used, need to have a type: ignore[misc],
#  because those types are seen as Any


class PiXmlFile(GenericDatasource):
    """For reading data from a pixml file."""

    @staticmethod
    def pi_xml_to_xarray(path: Path, simobstype: SimObsType) -> xr.Dataset:
        """Convert pi-xml to an :py:class:`~xarray.Dataset`.

        Parameters
        ----------
        path : Path
            Path to file.
        simobstype : SimObsType
            Indicator for what type of data is contained in the file.

        Returns
        -------
        xr.Dataset
            :py:class:`~xarray.Dataset` representation of the pi-xml file.
        """
        # Load  pi-xml file
        pi_series: Timeseries = Timeseries(path, binary=False)  # type: ignore[misc] # Timeseries has no type hinting, so pi_series is Any
        times: list[datetime.datetime] = pi_series.times  # type: ignore[misc] # pi_series is Any
        variables: set[str] = {k.parameter_id for k, _ in pi_series.items()}  # type: ignore[misc]  # pi_series is Any, and the keys from the items are Any, we are assuming str
        if len(variables) != 1:
            msg = "More than one parameter found."
            raise NotImplementedError(msg)
        variable_name = variables.pop()
        data_arrays = []

        def get_location_info(
            pi_series: Timeseries,
            timeseries_id: TimeseriesId,
        ) -> tuple[str, float, float]:
            location_info = pi_series.get_location(timeseries_id.location_id)  # type: ignore[misc] # location_info is Any
            lat = float(location_info.lat)  # type: ignore[misc] # lat is Any, we are assuming float convertable
            lon = float(location_info.lon)  # type: ignore[misc] # lat is Any, we are assuming float convertable
            if not math.isfinite(lat) or not math.isfinite(lon):
                msg = (
                    f"Lat ({lat}) and lon ({lon}) must be finite, from file {pi_series.path.name}."  # type: ignore[misc] # pi_series is Any
                )
                raise ValueError(msg)
            return str(timeseries_id.location_id), lat, lon  # type: ignore[misc] # timeseries_id is Any

        if simobstype == SimObsType.sim:
            simulation_starttime: datetime.datetime = pi_series.forecast_datetime  # type: ignore[misc]  # pi_series is Any
            ensemble_member: int
            for ensemble_member in range(pi_series.ensemble_size):  # type: ignore[misc]  # pi_series is Any
                timeseries_id: TimeseriesId
                for timeseries_id, data in pi_series.items(  # type: ignore[misc] # pi_series and data are Any
                    ensemble_member=ensemble_member,
                ):
                    location_id, lat, lon = get_location_info(pi_series, timeseries_id)  # type: ignore[misc]  # pi_series is Any
                    coords = {  # separate variable for readability and type hinting
                        DataModelCoords.time: times,
                        DataModelCoords.location: [location_id],
                        DataModelCoords.ensemble: [ensemble_member],
                        DataModelCoords.lat: ([DataModelDims.location], [lat]),
                        DataModelCoords.lon: ([DataModelDims.location], [lon]),
                        DataModelCoords.simstart: [simulation_starttime],
                    }
                    da = xr.DataArray(
                        data=np.expand_dims(data, axis=(1, 2, 3)),  # type: ignore[misc] # data and ndarray are Any
                        dims=[
                            DataModelDims.time,
                            DataModelDims.location,
                            DataModelDims.ensemble,
                            DataModelDims.simstart,
                        ],
                        coords=coords,
                    )
                    da.name = variable_name
                    data_arrays.append(da)

        elif simobstype == SimObsType.obs:
            for timeseries_id, data in pi_series.items():  # type: ignore[misc] # pi_series and data are Any
                location_id, lat, lon = get_location_info(pi_series, timeseries_id)  # type: ignore[misc]  # pi_series is Any
                coords = {
                    DataModelCoords.time: times,
                    DataModelCoords.location: [location_id],
                    DataModelCoords.lat: ([DataModelDims.location], [lat]),
                    DataModelCoords.lon: ([DataModelDims.location], [lon]),
                }
                da = xr.DataArray(
                    data=np.expand_dims(data, axis=(1)),  # type: ignore[misc] # data and ndarray are Any
                    dims=[
                        DataModelDims.time,
                        DataModelDims.location,
                    ],
                    coords=coords,
                )
                da.name = variable_name
                data_arrays.append(da)
        else:
            msg = f"{simobstype} not supported."
            raise NotImplementedError(msg)
        return xr.merge(data_arrays)

    @classmethod
    def get_data(cls, dsconfig: DataSource) -> list[Self]:
        """Retrieve pixml content as an xarray DataArray."""
        if dsconfig.datasourcetype != DataSourceTypeEnum.pixml:
            msg = "Input dsconfig does not have datasourcetype pixml"
            raise TypeError(msg)
        if dsconfig.simobstype == SimObsType.combined:
            msg = "Cannot yet handle combined simobs data"
            raise NotImplementedError(msg)

        filepath = Path(dsconfig.directory) / dsconfig.filename
        pif = cls(dsconfig)
        pif.xarray = cls.pi_xml_to_xarray(filepath, dsconfig.simobstype)
        return [pif]

    # classmethod write_to_file remains explicitly not implemented for pi xml
