"""Test the functions in the pipeline module."""

import pytest
from pytest_lazy_fixtures import lf

from dpyverification.configuration.default.scores import BaseScoreConfig
from dpyverification.configuration.file import Config
from dpyverification.datasinks.cf_compliant_netcdf import CFCompliantNetCDF
from dpyverification.datasources.fewsnetcdf import FewsNetCDF
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
