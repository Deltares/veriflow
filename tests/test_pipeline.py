"""Test the functions in the pipeline module."""

import pytest
from pytest_lazy_fixtures import lf

from dpyverification.configuration.base import GeneralInfoConfig
from dpyverification.configuration.default.scores import (
    BaseScoreConfig,
    CategoricalScoresConfig,
    EventOperator,
    ThresholdEvent,
)
from dpyverification.configuration.file import Config
from dpyverification.constants import ScoreKind, SupportedCategoricalScores
from dpyverification.datasinks.cf_compliant_netdf import CFCompliantNetCDF
from dpyverification.datasources.csv import Csv
from dpyverification.datasources.fewsnetcdf import FewsNetCDF
from dpyverification.datasources.netcdf import NetCDF
from dpyverification.pipeline import execute_pipeline


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
        fileversion="0.0.1",
        general=fews_netcdf_observed_historical.config.general,
        datasources=[
            fews_netcdf_observed_historical.config,
            fews_netcdf_simulated_forecast_ensemble_frt.config,
        ],
        scores=[score_config],
        datasinks=[datasink_cf_compliant_netcdf.config],
    )
    _ = execute_pipeline(config)


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
        fileversion="0.0.1",
        general=xarray_general_info_config,
        datasources=[
            xarray_observed_historical_datasource.config,
            xarray_observed_forecast_single_datasource.config,
            xarray_thresholds.config,
        ],
        scores=[categorical_score_config],
        datasinks=[datasink_cf_compliant_netcdf.config],
    )
    _ = execute_pipeline(config)
