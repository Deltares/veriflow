"""A python interface to the Delft-FEWS Webservices."""

from datetime import datetime
from enum import StrEnum

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


class FewsWebserviceClient:
    """An interface to the GET_TIMESERIES endpoint."""

    datetime_format = "%Y-%m-%dT%H:%M:%SZ"

    def __init__(self, url: str | Url, username: str | None, password: str | None) -> None:
        self.url = url
        self.session = requests.Session()
        if username and password:
            self.session.auth = requests.auth.HTTPBasicAuth(username=username, password=password)

    def format_datetime(self, time: datetime | None) -> str | None:
        """Format datetime to string."""
        if isinstance(time, datetime):
            return time.strftime(self.datetime_format)
        return time

    def format_list_of_datetime(
        self,
        datetime_list: list[datetime] | None,
    ) -> list[str | None] | None:
        """Format list of datetime."""
        if datetime_list is not None:
            return [self.format_datetime(t) for t in datetime_list if datetime_list]
        return None

    def get_timeseries(  # noqa: PLR0913
        self,
        location_ids: list[str],
        parameter_ids: list[str],
        module_instance_ids: list[str],
        qualifier_ids: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        start_forecast_time: datetime | None = None,
        end_forecast_time: datetime | None = None,
        lead_time: int | None = None,
        ensemble_id: str | None = None,
        ensemble_member_id: int | None = None,
        forecast_count: int | None = None,
        external_forecast_times: list[datetime] | None = None,
        timeseries_type: TimeseriesType = TimeseriesType.EXTERNAL_HISTORICAL,
        document_format: DocumentFormat = DocumentFormat.PI_NETCDF,
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
            "leadTime": lead_time,
            "ensembleId": ensemble_id,
            "ensembleMemberId": ensemble_member_id,
            "forecastCount": forecast_count,
            "externalForecastTimes": self.format_list_of_datetime(external_forecast_times),
            "timeSeriesType": timeseries_type,
            "documentFormat": document_format,
        }

        if document_format == DocumentFormat.PI_NETCDF:
            headers = {
                "Accept-Encoding": "identity",  # Disable automatic gzip/deflate decoding
            }
        else:
            headers = {}  # type: ignore[unreachable] # yes, unreachable for now

        response = self.session.get(url=f"{self.url}/timeseries?", params=params, headers=headers)  # type: ignore[arg-type]
        response.raise_for_status()
        return response

    def get_netcdf_storage_forecasts_forecast_reference_times(
        self,
        start_time: datetime,
        end_time: datetime,
        module_instance_ids: list[str],
    ) -> list[datetime]:
        """Get forecastTimes from external netcdf storage."""
        params = {
            "startTime": self.format_datetime(start_time),
            "endTime": self.format_datetime(end_time),
            "requestedAttributes": "module_instance_id",
            "documentFormat": "PI_JSON",
        }
        response = self.session.get(
            url=f"{self.url}/archive/netcdfstorageforecasts?",
            params=params,
        )
        response.raise_for_status()

        def get_forecast_time_on_matching_module_instance_id(
            item: dict[str, str],
            module_instance_ids: list[str],
        ) -> datetime | None:
            """Return the forecastTime when the module instance id matches."""
            if "attributes" in item:
                for attr in item["attributes"]:
                    if any(attr["value"] == mid for mid in module_instance_ids):  # type:ignore[index]
                        iso_time: str = item["forecastTime"]
                        return datetime.fromisoformat(iso_time)
            return None

        response_json: dict[str, list[dict[str, str]]] = response.json()

        forecast_reference_time_list = []
        for json_item in response_json["externalNetCDFStorageForecasts"]:
            value = get_forecast_time_on_matching_module_instance_id(
                json_item,
                module_instance_ids=module_instance_ids,
            )
            if value is not None:
                forecast_reference_time_list.append(value)

        return forecast_reference_time_list
