"""A python interface to the Delft-FEWS Webservices."""

from datetime import datetime
from enum import StrEnum

import requests
import requests.auth
import xarray as xr


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

    def __init__(self, url: str, username: str, password: str) -> None:
        self.url = url
        self.session = requests.Session()
        self.session.auth = requests.auth.HTTPBasicAuth(username=username, password=password)

    def get_timeseries(  # noqa: PLR0913
        self,
        location_ids: list[str],
        parameter_ids: list[str],
        module_instance_ids: list[str],
        qualifier_ids: list[str],
        start_time: datetime,
        end_time: datetime,
        start_forecast_time: datetime | None = None,
        end_forecast_time: datetime | None = None,
        ensemble_id: str | None = None,
        ensemble_member_id: int | None = None,
        forecast_count: int | None = None,
        timeseries_type: TimeseriesType = TimeseriesType.EXTERNAL_HISTORICAL,
        document_format: DocumentFormat = DocumentFormat.PI_NETCDF,
    ) -> requests.Response:
        """Get a timeseries from the Delft-FEWS webservice."""
        params = {
            "locationIds": location_ids,
            "parameterIds": parameter_ids,
            "moduleInstanceIds": module_instance_ids,
            "qualifierIds": qualifier_ids,
            "startTime": start_time.strftime(self.datetime_format),
            "endTime": end_time.strftime(self.datetime_format),
            "startForecastTime": start_forecast_time,
            "endForecastTime": end_forecast_time,
            "ensembleId": ensemble_id,
            "ensembleMemberId": ensemble_member_id,
            "forecastCount": forecast_count,
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

    def get_archive_netcdfstorage_forecasts(self) -> requests.Response:
        """Get all metadata from available forecasts in external storage archive."""
        # Request
        # Convert response in readable format
        return requests.Response()

    def get_all_forecasts_in_verification_period(self) -> xr.Dataset:
        """Get all available forecasts from NetCDF archive."""
        # get metadata
        # get one forecast for each available analysis_time
        # write each forecast to disk
        # merge forecasts into dataset from disk
        return xr.Dataset()
