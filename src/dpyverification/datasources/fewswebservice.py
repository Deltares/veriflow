"""PI-XML support module."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Self

import requests

from dpyverification.configuration import DataSource
from dpyverification.configuration.schema import FewsWebserviceInput, GeneralInfo
from dpyverification.constants import DataSourceTypeEnum, SimObsType
from dpyverification.datasources.genericdatasource import GenericDatasource
from dpyverification.datasources.pixml import PiXmlFile

# Ignore import untyped on fewsio since no type stub available. Unfortunately, this also means
#  that almost all locations where types from fewsio are used, need to have a type: ignore[misc],
#  because those types are seen as Any


class FewsWebService(GenericDatasource):
    """For downloading data using a Delft-FEWS webservice."""

    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    timeout = 10

    @staticmethod
    def get_timeseries(dsconfig: FewsWebserviceInput, giconfig: GeneralInfo) -> requests.Response:
        """Docstring."""
        endpoint = "timeseries"
        url = dsconfig.url + "/" + endpoint
        start = giconfig.verificationperiod.start.datetime
        end = giconfig.verificationperiod.end.datetime

        if dsconfig.simobstype == SimObsType.sim:
            # Work out the correct forecastStartTime and forecastEndTime
            # so that all forecasts overlapping with the verification period
            # defined by start_time and end_time will be requested from the web service.
            if giconfig.leadtimes is None:
                msg = "No lead times specified for simulation."
                raise ValueError(msg)
            start_forecast_time = start - max(giconfig.leadtimes.timedelta)
            end_forecast_time = end
            params = {
                "locationIds": dsconfig.location_ids,
                "parameterIds": dsconfig.parameter_ids,
                "moduleInstanceIds": dsconfig.module_instance_ids,
                "qualifierIds": dsconfig.qualifier_ids,
                "startTime": datetime.strftime(start, FewsWebService.datetime_format),
                "endTime": datetime.strftime(end, FewsWebService.datetime_format),
                "startForecastTime": datetime.strftime(
                    start_forecast_time,
                    FewsWebService.datetime_format,
                ),
                "endForecastTime": datetime.strftime(
                    end_forecast_time,
                    FewsWebService.datetime_format,
                ),
                "documentFormat": dsconfig.document_format,
                "documentVersion": dsconfig.document_version,
            }
        if dsconfig.simobstype == SimObsType.obs:
            params = {
                "locationIds": dsconfig.location_ids,
                "parameterIds": dsconfig.parameter_ids,
                "moduleInstanceIds": dsconfig.module_instance_ids,
                "qualifierIds": dsconfig.qualifier_ids,
                "startTime": datetime.strftime(start, FewsWebService.datetime_format),
                "endTime": datetime.strftime(end, FewsWebService.datetime_format),
                "documentFormat": dsconfig.document_format,
                "documentVersion": dsconfig.document_version,
            }
        response = requests.get(url=url, params=params, timeout=FewsWebService.timeout)
        response.raise_for_status()
        return response

    @classmethod
    def get_data(cls, dsconfig: DataSource, giconfig: GeneralInfo | None = None) -> list[Self]:
        """Retrieve :py::class`~xarray.Dataset` from Delft-FEWS Webservice."""
        if dsconfig.datasourcetype != DataSourceTypeEnum.fewswebservice:
            msg = "Input dsconfig does not have datasourcetype fewswebservice"
            raise TypeError(msg)
        if giconfig is None:
            msg = "giconfig cannot be None. General Info is required."
            raise TypeError(msg)
        fws = cls(dsconfig)
        response = cls.get_timeseries(dsconfig=dsconfig, giconfig=giconfig)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(response.content)
        fws.xarray = PiXmlFile.pi_xml_to_xarray(Path(temp_file_path), dsconfig.simobstype)

        return [fws]

    # classmethod write_to_file remains explicitly not implemented for pi xml
