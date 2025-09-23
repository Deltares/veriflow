"""Test the functions in the pipeline module."""

import pytest
from dpyverification.configuration.base import GeneralInfoConfig
from dpyverification.configuration.default.scores import BaseScoreConfig
from dpyverification.configuration.file import Config
from dpyverification.datasinks.cf_compliant_netdf import CFCompliantNetCDF
from dpyverification.datasources.fewsnetcdf import FewsNetCDFFile
from dpyverification.pipeline import execute_pipeline
from pytest_lazy_fixtures import lf


@pytest.mark.parametrize(
    "score_config",
    [lf("score_config_crps"), lf("score_config_rank_histogram")],
)
def test_pipeline_fewsnetcdf(
    general_info_config_fewsnetcdf: GeneralInfoConfig,
    datasource_fewsnetcdf_obs: FewsNetCDFFile,
    datasource_fewsnetcdf_sim: FewsNetCDFFile,
    score_config: BaseScoreConfig,
    datasink_cf_compliant_netcdf: CFCompliantNetCDF,
) -> None:
    """Full integration tests of the pipeline."""
    config = Config(
        fileversion="0.0.1",
        general=general_info_config_fewsnetcdf,
        datasources=[datasource_fewsnetcdf_obs.config, datasource_fewsnetcdf_sim.config],
        scores=[score_config],
        datasinks=[datasink_cf_compliant_netcdf.config],
    )
    _ = execute_pipeline(config)
