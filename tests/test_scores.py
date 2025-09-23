"""Module to test the available scores."""

from dpyverification.configuration.default.scores import CrpsForEnsembleConfig, RankHistogramConfig
from dpyverification.datamodel.main import SimObsDataset
from dpyverification.scores.probabilistic import CrpsForEnsemble, RankHistogram


def test_crps(
    score_config_crps: CrpsForEnsembleConfig,
    simobsdataset_fews_netcdf_data: SimObsDataset,
) -> None:
    """Test CRPS."""
    result = CrpsForEnsemble(score_config_crps).compute(data=simobsdataset_fews_netcdf_data)
    assert result.name == score_config_crps.kind


def test_rank_histogram(
    score_config_rank_histogram: RankHistogramConfig,
    simobsdataset_fews_netcdf_data: SimObsDataset,
) -> None:
    """Test CRPS."""
    result = RankHistogram(score_config_rank_histogram).compute(data=simobsdataset_fews_netcdf_data)
    assert result.name == score_config_rank_histogram.kind
