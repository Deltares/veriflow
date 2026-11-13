"""Test the preprocessors."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from veriflow.configuration.base import GeneralInfoConfig
from veriflow.configuration.config import Config
from veriflow.configuration.default.preprocessors import TimeAggregationConfig
from veriflow.configuration.utils import TimeDelta
from veriflow.constants import PreprocessorKind, StandardDim, TimeAggregationMethod, TimeUnits
from veriflow.datasources.netcdf import NetCDF
from veriflow.preprocessors import DEFAULT_PREPROCESSORS
from veriflow.preprocessors.base import BasePreprocessor
from veriflow.preprocessors.time_aggregation import TimeAggregation

# mypy: disable-error-code="misc"

SOURCE_ID = "observation_source"
OUTPUT_ID = "observation_processed"
N_DAYS = 10  # The observations fixture holds 10 days of hourly data


def make_config(**kwargs: object) -> TimeAggregationConfig:
    """Create a time aggregation config for the observation source, with sensible defaults."""
    options = {
        "preprocessor_adapter": PreprocessorKind.time_aggregation,
        "source_id": SOURCE_ID,
        "output_source_id": OUTPUT_ID,
        "frequency": TimeDelta(value=1, unit=TimeUnits.day),
        "method": TimeAggregationMethod.mean,
        **kwargs,
    }
    return TimeAggregationConfig(**options)  # type:ignore[arg-type]


@pytest.fixture
def observed_10min(xarray_observed_historical: xr.Dataset) -> xr.Dataset:
    """Observations with a 10-minute time step, starting at 00:10, covering three full hours."""
    ds = xarray_observed_historical.isel(time=slice(0, 18))
    return ds.assign_coords(time=pd.date_range("2025-01-01T00:10", periods=18, freq="10min"))


def test_time_delta_conversions() -> None:
    """A TimeDelta is configured with a value and a unit."""
    delta = TimeDelta(value=90, unit=TimeUnits.minute)
    assert delta.timedelta64 == np.timedelta64(90, "m")
    assert delta.stdlib_timedelta == pd.Timedelta(minutes=90).to_pytimedelta()


@pytest.mark.parametrize("unit", [TimeUnits.year, TimeUnits.month])
def test_time_delta_rejects_units_without_fixed_length(unit: TimeUnits) -> None:
    """Calendar units have no fixed length, so cannot be used as duration."""
    with pytest.raises(ValidationError, match="no fixed length"):
        TimeDelta(value=1, unit=unit)


@pytest.mark.parametrize("value", [0, -1])
def test_time_delta_rejects_non_positive_values(value: int) -> None:
    """A duration must be positive."""
    with pytest.raises(ValidationError):
        TimeDelta(value=value, unit=TimeUnits.hour)


def test_time_aggregation_config_from_nested_dict() -> None:
    """The config can be created from plain data, as read from a yaml file."""
    config = TimeAggregationConfig.model_validate(
        {
            "preprocessor_adapter": "time_aggregation",
            "source_id": SOURCE_ID,
            "output_source_id": OUTPUT_ID,
            "frequency": {"value": 1, "unit": "h"},
            "method": "max",
        },
    )
    assert config.output_source_id == OUTPUT_ID
    assert config.frequency == TimeDelta(value=1, unit=TimeUnits.hour)
    assert config.method == TimeAggregationMethod.max
    assert config.closed == "right"
    assert config.label == "right"
    assert config.min_count is None


def test_time_aggregation_is_registered_as_default() -> None:
    """The time aggregation is available to the pipeline by its kind."""
    assert TimeAggregation in DEFAULT_PREPROCESSORS
    assert TimeAggregation.kind == PreprocessorKind.time_aggregation


@pytest.mark.parametrize("method", list(TimeAggregationMethod))
def test_time_aggregation_daily_matches_pandas(
    xarray_observed_historical: xr.Dataset,
    method: TimeAggregationMethod,
) -> None:
    """Aggregation gives the same result as the equivalent pandas resampling."""
    config = make_config(method=method, closed="left", label="left")
    result = TimeAggregation(config).preprocess(xarray_observed_historical)

    for var in xarray_observed_historical.data_vars:
        hourly = xarray_observed_historical[var].transpose(StandardDim.time, StandardDim.station)
        resampled = pd.DataFrame(
            hourly.to_numpy(),
            index=xarray_observed_historical[StandardDim.time].to_numpy(),
        ).resample("1D", closed="left", label="left")
        expected = getattr(resampled, method.value)()
        np.testing.assert_allclose(
            result[var].transpose(StandardDim.time, StandardDim.station).to_numpy(),
            expected.to_numpy(),
            rtol=1e-5,
        )

    # 10 days of hourly data results in 10 daily values
    assert result.sizes[StandardDim.time] == N_DAYS
    assert result[StandardDim.time][0].to_numpy() == np.datetime64("2025-01-01T00:00")


def test_time_aggregation_past_period_mean(observed_10min: xr.Dataset) -> None:
    """The default (closed and labelled right) gives the mean over the past hour."""
    config = make_config(frequency=TimeDelta(value=1, unit=TimeUnits.hour))
    result = TimeAggregation(config).preprocess(observed_10min)

    expected_times = pd.date_range("2025-01-01T01:00", periods=3, freq="h")
    np.testing.assert_array_equal(result[StandardDim.time].to_numpy(), expected_times)

    # The value at 01:00 is the mean of the 6 values in (00:00, 01:00]
    var = next(iter(observed_10min.data_vars))
    first_hour = observed_10min[var].isel(time=slice(0, 6)).mean(StandardDim.time)
    np.testing.assert_allclose(
        result[var].sel(time="2025-01-01T01:00").to_numpy(),
        first_hour.to_numpy(),
        rtol=1e-5,
    )


def test_time_aggregation_min_count_masks_incomplete_intervals(
    xarray_observed_historical: xr.Dataset,
) -> None:
    """Intervals with too few values (here at the data edges) become NaN."""
    config = make_config(min_count=24)  # Default (right, right) has partial first and last day
    result = TimeAggregation(config).preprocess(xarray_observed_historical)

    var = next(iter(result.data_vars))
    values = result[var].transpose(StandardDim.time, StandardDim.station).to_numpy()
    # Partial first and last day, in addition to the 10 full days
    assert values.shape[0] == N_DAYS + 1
    assert np.isnan(values[0]).all()
    assert np.isnan(values[-1]).all()
    assert not np.isnan(values[1:-1]).any()


def test_time_aggregation_gap_is_not_zero_for_sum(observed_10min: xr.Dataset) -> None:
    """An interval without any data is missing, also for a sum (which would otherwise be 0)."""
    # Keep the first and the last hour, leaving the second hour empty
    gap = observed_10min.isel(time=slice(0, 6)).combine_first(
        observed_10min.isel(time=slice(12, 18)),
    )
    config = make_config(
        frequency=TimeDelta(value=1, unit=TimeUnits.hour),
        method=TimeAggregationMethod.sum,
    )
    result = TimeAggregation(config).preprocess(gap.assign_attrs(observed_10min.attrs))

    var = next(iter(result.data_vars))
    assert np.isnan(result[var].sel(time="2025-01-01T02:00").to_numpy()).all()
    assert not np.isnan(result[var].sel(time="2025-01-01T01:00").to_numpy()).any()


def test_time_aggregation_keeps_metadata(xarray_observed_historical: xr.Dataset) -> None:
    """Dataset attrs, variable attrs, dims order and coordinates are kept."""
    result = TimeAggregation(make_config(method=TimeAggregationMethod.max)).preprocess(
        xarray_observed_historical,
    )

    assert result.attrs == xarray_observed_historical.attrs
    for var in xarray_observed_historical.data_vars:
        assert result[var].dims == xarray_observed_historical[var].dims
        assert result[var].attrs["units"] == xarray_observed_historical[var].attrs["units"]
        assert result[var].attrs["cell_methods"] == "time: max"
        assert result[var].dtype == xarray_observed_historical[var].dtype
    for coord in xarray_observed_historical.coords:
        if coord != StandardDim.time:
            xr.testing.assert_equal(result[coord], xarray_observed_historical[coord])
    assert result[StandardDim.time].attrs == xarray_observed_historical[StandardDim.time].attrs


def test_time_aggregation_does_not_modify_input(xarray_observed_historical: xr.Dataset) -> None:
    """The input dataset is left untouched."""
    original = xarray_observed_historical.copy(deep=True)
    TimeAggregation(make_config()).preprocess(xarray_observed_historical)
    xr.testing.assert_identical(xarray_observed_historical, original)


def test_time_aggregation_rejects_forecast_data(
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    """Only historical data is supported."""
    with pytest.raises(ValueError, match="only supported for historical data types"):
        TimeAggregation(make_config(source_id="simulation_ensemble_source")).preprocess(
            xarray_simulated_forecast_ensemble,
        )


def test_time_aggregation_rejects_dataset_without_data_variables(
    xarray_observed_historical: xr.Dataset,
) -> None:
    """A dataset without data variables cannot be aggregated."""
    empty = xarray_observed_historical.drop_vars(list(xarray_observed_historical.data_vars))
    with pytest.raises(ValueError, match="no data variables"):
        TimeAggregation(make_config()).preprocess(empty)


def test_time_aggregation_init_keeps_config() -> None:
    """The preprocessor can be created from a raw config dict, as done by the pipeline."""
    config = make_config(method=TimeAggregationMethod.min)
    preprocessor = TimeAggregation.from_config(config.model_dump())
    assert isinstance(preprocessor.config, TimeAggregationConfig)
    assert preprocessor.config == config


def test_base_preprocessor_preprocess_is_abstract(xarray_observed_historical: xr.Dataset) -> None:
    """The base class does not implement preprocess."""

    class _Dummy(BasePreprocessor):
        kind = "dummy"

    with pytest.raises(NotImplementedError):
        _Dummy(make_config()).preprocess(xarray_observed_historical)


def test_config_accepts_preprocessor_for_configured_source(
    xarray_general_info_config_historical: GeneralInfoConfig,
    xarray_observed_historical_datasource: NetCDF,
) -> None:
    """A preprocessor can be configured for a configured datasource."""
    config = Config(
        general=xarray_general_info_config_historical,
        datasources=[xarray_observed_historical_datasource.config],
        preprocessors=[make_config()],
    )
    assert config.preprocessors is not None
    assert len(config.preprocessors) == 1


def test_config_rejects_preprocessor_for_unknown_source(
    xarray_general_info_config_historical: GeneralInfoConfig,
    xarray_observed_historical_datasource: NetCDF,
) -> None:
    """A preprocessor must refer to a configured datasource."""
    with pytest.raises(ValidationError, match="not configured in datasources"):
        Config(
            general=xarray_general_info_config_historical,
            datasources=[xarray_observed_historical_datasource.config],
            preprocessors=[make_config(source_id="unknown_source")],
        )


def test_output_source_id_is_required() -> None:
    """A preprocessor config without an output_source_id is not allowed."""
    with pytest.raises(ValidationError, match="output_source_id"):
        TimeAggregationConfig.model_validate(
            {
                "preprocessor_adapter": "time_aggregation",
                "source_id": SOURCE_ID,
                "frequency": {"value": 1, "unit": "h"},
                "method": "max",
            },
        )


def test_config_rejects_preprocessor_without_output_source_id(
    xarray_general_info_config_historical: GeneralInfoConfig,
    xarray_observed_historical_datasource: NetCDF,
) -> None:
    """The (base) preprocessor config in Config, as loaded from yaml, also requires an output id."""
    with pytest.raises(ValidationError, match="output_source_id"):
        Config(
            general=xarray_general_info_config_historical,
            datasources=[xarray_observed_historical_datasource.config],
            preprocessors=[
                {  # type:ignore[list-item]
                    "preprocessor_adapter": "time_aggregation",
                    "source_id": SOURCE_ID,
                    "frequency": {"value": 1, "unit": "h"},
                    "method": "max",
                },
            ],
        )


def test_config_accepts_chained_preprocessors_with_output_ids(
    xarray_general_info_config_historical: GeneralInfoConfig,
    xarray_observed_historical_datasource: NetCDF,
) -> None:
    """A preprocessor can use the output of an earlier preprocessor as its source."""
    preprocessors = [
        make_config(output_source_id="observation_daily"),
        make_config(source_id="observation_daily", output_source_id="observation_daily_max"),
    ]
    config = Config(
        general=xarray_general_info_config_historical,
        datasources=[xarray_observed_historical_datasource.config],
        preprocessors=preprocessors,
    )
    assert config.preprocessors is not None
    assert len(config.preprocessors) == len(preprocessors)


def test_config_rejects_preprocessor_using_output_of_later_preprocessor(
    xarray_general_info_config_historical: GeneralInfoConfig,
    xarray_observed_historical_datasource: NetCDF,
) -> None:
    """Preprocessors are applied in order, so an output can only be used by later ones."""
    with pytest.raises(ValidationError, match="not configured in datasources"):
        Config(
            general=xarray_general_info_config_historical,
            datasources=[xarray_observed_historical_datasource.config],
            preprocessors=[
                make_config(source_id="observation_daily"),
                make_config(output_source_id="observation_daily"),
            ],
        )


@pytest.mark.parametrize(
    "output_ids",
    [
        [SOURCE_ID],  # same as the id of a datasource
        ["observation_daily", "observation_daily"],  # same as the output of another preprocessor
    ],
)
def test_config_rejects_non_unique_output_source_id(
    xarray_general_info_config_historical: GeneralInfoConfig,
    xarray_observed_historical_datasource: NetCDF,
    output_ids: list[str],
) -> None:
    """An output_source_id must be unique."""
    with pytest.raises(ValidationError, match="must be unique"):
        Config(
            general=xarray_general_info_config_historical,
            datasources=[xarray_observed_historical_datasource.config],
            preprocessors=[make_config(output_source_id=output_id) for output_id in output_ids],
        )
