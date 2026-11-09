"""Test the fewswebservice module of the veriflow.datasources package."""

# mypy: ignore-errors

from datetime import datetime

from veriflow.api.fewswebservice import (
    FewsWebserviceClient,
)
from veriflow.datasources.fewswebservice import (
    FewsWebserviceClient as FewsWebserviceClientDatasource,
)


def test_format_boolean() -> None:
    """Test the format_boolean method converts Python booleans to lowercase strings."""
    client = FewsWebserviceClient(
        url="http://example.com",
        username=None,
        password=None,
    )

    assert client.format_boolean(value=True) == "true"
    assert client.format_boolean(value=False) == "false"
    assert client.format_boolean(value=None) is None


def test_parsing_forecast_reference_times_from_timeseries_headers(
    fews_webservice_timeseries_headers_only: dict,
) -> None:
    """Test the parsing of forecast reference times."""
    result = FewsWebserviceClientDatasource.parse_forecast_reference_times_from_json_headers(
        fews_webservice_timeseries_headers_only,
        module_instance_id="WASIM_Alpenrhein_ICONCH1_Con",
    )
    assert len(result) == 1
    assert isinstance(result, list)
    assert isinstance(result[0], datetime)
