"""A python interface to the Delft-FEWS Webservices."""

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Union

import requests
import requests.auth
from pydantic_core import Url


class TimeseriesType(StrEnum):
    """The available timeseries types."""

    EXTERNAL_HISTORICAL = "EXTERNAL_HISTORICAL"
    EXTERNAL_FORECASTING = "EXTERNAL_FORECASTING"
    SIMULATED_HISTORICAL = "SIMULATED_HISTORICAL"
    SIMULATED_FORECASTING = "SIMULATED_FORECASTING"


class DocumentFormat(StrEnum):
    """The available document formats."""

    PI_NETCDF = "PI_NETCDF"
    PI_JSON = "PI_JSON"


JSONValue = Union[str, int, float, bool, None, "JSONDict", list["JSONValue"]]
JSONDict = dict[str, JSONValue]


class FewsWebserviceClient:
    """An interface to the GET_TIMESERIES endpoint."""

    datetime_format = "%Y-%m-%dT%H:%M:%SZ"

    def __init__(self, url: str | Url, username: str | None, password: str | None) -> None:
        self.url = url
        self.session = requests.Session()
        if username is not None and password is not None:
            self.session.auth = requests.auth.HTTPBasicAuth(username=username, password=password)
        elif username is None and password is not None or username is not None and password is None:
            msg = (
                f"Authorization of {FewsWebserviceClient.__name__} requires either both a ",
                "username and password, or no username and password.",
            )
            raise ValueError(msg)

    def format_datetime(self, time: datetime | None) -> str | None:
        """Format datetime to string."""
        if isinstance(time, datetime):
            return time.strftime(self.datetime_format)
        return time

    def timedelta_to_milliseconds(self, td: timedelta | None) -> int | None:
        """Format datetime to string."""
        if isinstance(td, timedelta):
            return int(td.total_seconds() * 1000)  # milliseconds
        return td

    def format_list_of_datetime(
        self,
        datetime_list: list[datetime | None] | None,
    ) -> list[str | None] | None:
        """Format list of datetime."""
        if datetime_list is not None:
            return [self.format_datetime(t) for t in datetime_list if datetime_list]
        return None

    def get_timeseries(  # noqa: PLR0913
        self,
        location_ids: list[str],
        parameter_ids: list[str],
        module_instance_ids: list[str] | str,
        qualifier_ids: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        start_forecast_time: datetime | None = None,
        end_forecast_time: datetime | None = None,
        lead_time: timedelta | None = None,
        ensemble_id: str | None = None,
        ensemble_member_id: int | None = None,
        forecast_count: int | None = None,
        external_forecast_times: list[datetime | None] | None = None,
        export_id_map: str | None = None,
        timeseries_type: TimeseriesType | None = None,
        document_format: DocumentFormat = DocumentFormat.PI_NETCDF,
        only_headers: bool | None = None,
    ) -> requests.Response:
        """Get a timeseries from the Delft-FEWS webservice."""
        params = {
            "locationIds": location_ids,
            "parameterIds": parameter_ids,
            "moduleInstanceIds": module_instance_ids,
            "qualifierIds": qualifier_ids,
            "startTime": self.format_datetime(start_time),
            "endTime": self.format_datetime(end_time),
            "startForecastTime": self.format_datetime(start_forecast_time),
            "endForecastTime": self.format_datetime(end_forecast_time),
            "leadTime": self.timedelta_to_milliseconds(lead_time),
            "ensembleId": ensemble_id,
            "ensembleMemberId": ensemble_member_id,
            "forecastCount": forecast_count,
            "externalForecastTimes": self.format_list_of_datetime(external_forecast_times),
            "exportIdMap": export_id_map,
            "timeSeriesType": timeseries_type,
            "documentFormat": document_format,
            "onlyHeaders": only_headers,
        }

        if document_format == DocumentFormat.PI_NETCDF:
            headers = {
                "Accept-Encoding": "identity",  # Disable automatic gzip/deflate decoding
            }
        else:
            headers = {}

        response = self.session.get(url=f"{self.url}/timeseries", params=params, headers=headers)  # type:ignore[arg-type]
        response.raise_for_status()
        return response

    def get_netcdf_storage_forecast_reference_times(
        self,
        start_time: datetime,
        end_time: datetime,
        module_instance_ids: list[str] | str,
    ) -> list[datetime]:
        """Get forecastTimes from external netcdf storage."""
        if isinstance(module_instance_ids, str):
            module_instance_ids = [module_instance_ids]

        params = {
            "startTime": self.format_datetime(start_time),
            "endTime": self.format_datetime(end_time),
            "requestedAttributes": "module_instance_id",
            "documentFormat": "PI_JSON",
        }
        response = self.session.get(
            url=f"{self.url}/archive/netcdfstorageforecasts",
            params=params,
        )
        response.raise_for_status()

        def get_forecast_time_on_matching_module_instance_id(
            item: dict[str, str],
            module_instance_ids: list[str],
        ) -> datetime | None:
            """Return the forecastTime when the module instance id matches."""
            if "attributes" not in item:
                return None
            for attr in item["attributes"]:
                if attr["value"] in module_instance_ids:  # type:ignore[index]
                    iso_time: str = item["forecastTime"]
                    return datetime.fromisoformat(iso_time)
            return None

        response_json: dict[str, list[dict[str, str]]] = response.json()

        forecast_reference_times = []
        for json_item in response_json["externalNetCDFStorageForecasts"]:
            value = get_forecast_time_on_matching_module_instance_id(
                json_item,
                module_instance_ids=module_instance_ids,
            )
            if value is not None:
                forecast_reference_times.append(value)

        return forecast_reference_times

    @staticmethod
    def parse_forecast_reference_times_from_json_headers(
        json_dict: JSONDict,
        module_instance_id: str,
    ) -> list[datetime]:
        """Parse the forecast reference times from timeseries headers."""

        def _parse_forecast_date_from_header(
            header: dict[str, dict[str, dict[str, str]]],
            module_instance_id: str,
        ) -> datetime | None:
            """Parse one element."""
            if "forecastDate" in header:
                date = header["forecastDate"]["date"]
                time = header["forecastDate"]["time"]
                if header["moduleInstanceId"] == module_instance_id:
                    return datetime.fromisoformat(f"{date}T{time}")
            return None

        forecast_dates: list[datetime] = []
        for timeseries in json_dict["timeSeries"]:  # type:ignore[union-attr, misc]
            forecast_date = _parse_forecast_date_from_header(
                timeseries["header"],  # type:ignore[misc, index, call-overload, arg-type]
                module_instance_id=module_instance_id,
            )
            if forecast_date is not None:
                forecast_dates.append(forecast_date)
        return forecast_dates
