"""Module to test the available scores."""

from dpyverification.configuration.default.scores import CrpsForEnsembleConfig, RankHistogramConfig
from dpyverification.datamodel.main import InputDataset
from dpyverification.scores.probabilistic import CrpsForEnsemble, RankHistogram


def test_crps(
    score_config_crps: CrpsForEnsembleConfig,
    input_dataset_fews_netcdf_data: InputDataset,
) -> None:
    """Test CRPS."""
    result = CrpsForEnsemble(score_config_crps).compute(data=input_dataset_fews_netcdf_data)
    assert result.name == score_config_crps.kind


def test_rank_histogram(
    score_config_rank_histogram: RankHistogramConfig,
    input_dataset_fews_netcdf_data: InputDataset,
) -> None:
    """Test CRPS."""
    result = RankHistogram(score_config_rank_histogram).compute(data=input_dataset_fews_netcdf_data)
    assert result.name == score_config_rank_histogram.kind
