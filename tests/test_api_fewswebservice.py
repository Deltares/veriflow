"""Test the fewswebservice module of the dpyverification.datasources package."""

# mypy: ignore-errors

from datetime import datetime

from dpyverification.datasources.fewswebservice import (
    FewsWebserviceClient,
)


def test_parsing_forecast_reference_times_from_timeseries_headers(
    fews_webservice_timeseries_headers_only: dict,
) -> None:
    """Test the parsing of forecast reference times."""
    result = FewsWebserviceClient.parse_forecast_reference_times_from_json_headers(
        fews_webservice_timeseries_headers_only,
        module_instance_id="WASIM_Alpenrhein_ICONCH1_Con",
    )
    assert len(result) == 1
    assert isinstance(result, list)
    assert isinstance(result[0], datetime)
