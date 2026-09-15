"""Test the functions in the pipeline module."""

from typing import TYPE_CHECKING, cast

import pytest
from pytest_lazy_fixtures import lf

from veriflow.configuration.base import GeneralInfoConfig
from veriflow.configuration.config import SupportedSchemaVersion
from veriflow.configuration.default.scores import (
    BaseScoreConfig,
    CategoricalScoresConfig,
    ContinuousScoresConfig,
    EventOperator,
    ThresholdEvent,
)
from veriflow.configuration.file import Config
from veriflow.constants import (
    DataType,
    ScoreKind,
    StandardDim,
    SupportedCategoricalScores,
    SupportedContinuousScore,
)
from veriflow.datasinks.cf_compliant_netcdf import CFCompliantNetCDF
from veriflow.datasources.csv import Csv
from veriflow.datasources.fewsnetcdf import FewsNetCDF
from veriflow.datasources.netcdf import NetCDF
from veriflow.pipeline import run_pipeline

if TYPE_CHECKING:
    import xarray as xr


@pytest.mark.parametrize(
    "score_config",
    [lf("score_config_crps"), lf("score_config_rank_histogram")],
)
def test_pipeline_fewsnetcdf(
    fews_netcdf_observed_historical: FewsNetCDF,
    fews_netcdf_simulated_forecast_ensemble_frt: FewsNetCDF,
    score_config: BaseScoreConfig,
    datasink_cf_compliant_netcdf: CFCompliantNetCDF,
) -> None:
    """Full integration tests of the pipeline."""
    config = Config(
        version=SupportedSchemaVersion.V0,
        general=fews_netcdf_observed_historical.config.general,
        datasources=[
            fews_netcdf_observed_historical.config,
            fews_netcdf_simulated_forecast_ensemble_frt.config,
        ],
        scores=[score_config],
        datasinks=[datasink_cf_compliant_netcdf.config],
    )
    _ = run_pipeline(config)


def test_pipeline_xarray_categorical_scores(
    xarray_general_info_config: GeneralInfoConfig,
    xarray_observed_historical_datasource: NetCDF,
    xarray_observed_forecast_single_datasource: NetCDF,
    xarray_thresholds: Csv,
    datasink_cf_compliant_netcdf: CFCompliantNetCDF,
) -> None:
    """Full integration tests of the pipeline."""
    categorical_score_config = CategoricalScoresConfig(
        general=xarray_general_info_config,
        score_adapter=ScoreKind.categorical_scores,
        scores=[SupportedCategoricalScores.accuracy, SupportedCategoricalScores.false_alarm_rate],
        events=[ThresholdEvent(threshold="warn_1", operator=EventOperator.GREATER_THAN)],
        reduce_dims=[],
    )

    config = Config(
        version=SupportedSchemaVersion.V0,
        general=xarray_general_info_config,
        datasources=[
            xarray_observed_historical_datasource.config,
            xarray_observed_forecast_single_datasource.config,
            xarray_thresholds.config,
        ],
        scores=[categorical_score_config],
        datasinks=[datasink_cf_compliant_netcdf.config],
    )
    _ = run_pipeline(config)


def test_pipeline_historical_data_only(
    xarray_general_info_config_historical: GeneralInfoConfig,
    xarray_observed_historical_datasource: NetCDF,
    datasink_cf_compliant_netcdf: CFCompliantNetCDF,
) -> None:
    """Full integration tests of the pipeline."""
    # Test running with only historical data

    continuous_score_config = ContinuousScoresConfig(
        general=xarray_general_info_config_historical,
        score_adapter=ScoreKind.continuous_scores,
        scores=[SupportedContinuousScore.rmse, SupportedContinuousScore.mae],
        reduce_dims=[StandardDim.time],
    )

    # Create a dummy datasource, by copying the observed historical
    # For testing purposes, set the data_type to "simulated_historical"
    dummy_config = xarray_observed_historical_datasource.config.model_copy()
    dummy_config.source_id = "simulation_ensemble_source"
    dummy_config.data_type = DataType.simulated_historical

    config = Config(
        version=SupportedSchemaVersion.V0,
        general=xarray_general_info_config_historical,
        datasources=[
            xarray_observed_historical_datasource.config,
            dummy_config,
        ],
        scores=[continuous_score_config],
        datasinks=[datasink_cf_compliant_netcdf.config],
    )
    output_datatree = run_pipeline(config)
    pair_id = next(iter(output_datatree.veriflow.verification_pairs))
    path_in_dt = f"{pair_id}/output/{continuous_score_config.score_adapter}"
    score_dataset = cast("xr.DataTree", output_datatree[path_in_dt])

    # Load into memory
    score_dataset.load()

    # Test wether the pipeline runs successfully and produces the expected output dataset.
    # Since the two datasources are identical, we expect perfect scores.
    assert "rmse" in score_dataset.data_vars
    assert score_dataset["rmse"].dims == ("station",)
    assert all(score_dataset["rmse"] == 0)
    assert "mae" in score_dataset.data_vars
    assert score_dataset["mae"].dims == ("station",)
    assert all(score_dataset["mae"] == 0)
