"""Module for reading from and writing to a fews webservice."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Self

import requests

from dpyverification.configuration import DataSource
from dpyverification.configuration.schema import FewsWebserviceInput, FewsWebserviceInputSim
from dpyverification.datasources.genericdatasource import GenericDatasource
from dpyverification.datasources.pixml import PiXmlFile


class FewsWebService(GenericDatasource):
    """For downloading data using a Delft-FEWS webservice."""

    # TODO(AU): Fix and document timezone information in fewswebservice requests # noqa: FIX002
    #   A hardcoded Z is added at the end, that cannot be right? See the issue for details:
    #   https://github.com/Deltares-research/DPyVerification/issues/43
    #
    # The datetime format that is used to pass datetimes to the fewswebservice
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    timeout = 30

    @staticmethod
    def get_timeseries_xml_string(dsconfig: FewsWebserviceInput) -> requests.Response:
        """Perform a REST GET to retrieve timeseries data as a pi-xml string."""
        ########## DOING: update de comments, per de PR comment
        endpoint = "timeseries"
        url = dsconfig.url + "/" + endpoint
        if dsconfig.verificationperiod is None:
            msg = "No verificationperiod specified."
            raise ValueError(msg)
        start = dsconfig.verificationperiod.start.datetime
        end = dsconfig.verificationperiod.end.datetime

        params = {
            "locationIds": dsconfig.location_ids,
            "parameterIds": dsconfig.parameter_ids,
            "moduleInstanceIds": dsconfig.module_instance_ids,
            "qualifierIds": dsconfig.qualifier_ids,
            "startTime": datetime.strftime(start, FewsWebService.datetime_format),
            "endTime": datetime.strftime(end, FewsWebService.datetime_format),
            "documentFormat": dsconfig._document_format,  # noqa: SLF001 # This config private member is meant to be used directly
            "documentVersion": dsconfig._document_version,  # noqa: SLF001 # This config private member is meant to be used directly
        }

        if isinstance(dsconfig, FewsWebserviceInputSim):
            if dsconfig.leadtimes is None:
                msg = "No lead times specified for simulation."
                raise ValueError(msg)
            if dsconfig.forecastcount != 1:
                # TODO(AU): Issues 44 and 45 # noqa: FIX002
                #   See the more detailed split up in the get_data() for more info, and
                #   https://github.com/Deltares-research/DPyVerification/issues/44
                #   https://github.com/Deltares-research/DPyVerification/issues/45
                msg = (
                    "Retrieving ALL forecasts within a period not yet implemented,"
                    " specify a (very large) forecastcount value for now."
                )
                raise NotImplementedError(msg)

            # Work out the correct forecastStartTime and forecastEndTime
            # so that all forecasts overlapping with the verification period
            # defined by start_time and end_time will be requested from the web service.
            start_forecast_time = start - max(dsconfig.leadtimes.timedelta)
            end_forecast_time = end

            params.update(
                {
                    "startForecastTime": datetime.strftime(
                        start_forecast_time,
                        FewsWebService.datetime_format,
                    ),
                    "endForecastTime": datetime.strftime(
                        end_forecast_time,
                        FewsWebService.datetime_format,
                    ),
                    "forecastCount": str(dsconfig.forecastcount),
                },
            )

        response = requests.get(url=url, params=params, timeout=FewsWebService.timeout)
        response.raise_for_status()
        return response

    @classmethod
    def get_data(cls, dsconfig: DataSource) -> list[Self]:
        """Retrieve :py::class`~xarray.Dataset` from Delft-FEWS Webservice."""
        if not isinstance(dsconfig, FewsWebserviceInput):
            msg = "Input dsconfig does not have datasourcetype FewsWebserviceInput"
            raise TypeError(msg)
        if isinstance(dsconfig, FewsWebserviceInputSim) and dsconfig.forecastcount != 1:
            if dsconfig.forecastcount == 0:
                # TODO(AU): Implement ability to retrieve all forecastruns in period # noqa: FIX002
                #   First, issue 44 should be fixed. Then, see this issue for details and solution
                #   direction
                #   https://github.com/Deltares-research/DPyVerification/issues/45
                msg = (
                    "Retrieving ALL forecasts within a period not yet implemented,"
                    " specify a (very large) forecastcount value for now."
                )
            else:
                # TODO(AU): Implement ability to retrieve more than one forecastrun # noqa: FIX002
                #   See issue for details and solution direction
                #   https://github.com/Deltares-research/DPyVerification/issues/44
                msg = (
                    "Retrieving more than one forecast within a period not yet implemented,"
                    " due to fews-io package limitation in converting pixml files."
                )
            raise NotImplementedError(msg)
        fws = cls(dsconfig)
        response = cls.get_timeseries_xml_string(dsconfig=dsconfig)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(response.content)
        fws.xarray = PiXmlFile.pi_xml_to_xarray(Path(temp_file_path), dsconfig.simobstype)

        return [fws]

    # classmethod write_to_file remains explicitly not implemented for pi xml
