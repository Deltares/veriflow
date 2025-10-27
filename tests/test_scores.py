"""Module to test the available scores."""

from copy import deepcopy

import xarray as xr
from dpyverification.configuration.default.scores import (
    ContinuousScoresConfig,
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    RankHistogramConfig,
)
from dpyverification.constants import TimeseriesKind
from dpyverification.datamodel.main import InputDataset
from dpyverification.datasources.fewsnetcdf import FewsNetCDF
from dpyverification.scores.continuous import ContinuousScores
from dpyverification.scores.probabilistic import CrpsCDF, CrpsForEnsemble, RankHistogram


def test_ensemble_crps(
    score_config_crps: CrpsForEnsembleConfig,
    input_dataset_fews_netcdf_simulated_forecast_ensemble: InputDataset,
) -> None:
    """Test CRPS."""
    result = CrpsForEnsemble(score_config_crps).compute(
        data=input_dataset_fews_netcdf_simulated_forecast_ensemble,
    )
    assert result.name == score_config_crps.kind


def test_ensemble_rank_histogram(
    score_config_rank_histogram: RankHistogramConfig,
    input_dataset_fews_netcdf_simulated_forecast_ensemble: InputDataset,
) -> None:
    """Test CRPS."""
    result = RankHistogram(score_config_rank_histogram).compute(
        data=input_dataset_fews_netcdf_simulated_forecast_ensemble,
    )
    assert result.name == score_config_rank_histogram.kind


def test_probabilistic_crps_cdf(
    score_config_crps_cdf: CrpsCDFConfig,
    fews_netcdf_simulated_forecast_probabilistic_fp: FewsNetCDF,
) -> None:
    """Test CRPS."""
    sim = fews_netcdf_simulated_forecast_probabilistic_fp.get_data().data_array

    # Synthetic obs
    mean_sim = sim.threshold.mean()  # type:ignore[misc]
    obs_dummy = xr.full_like(sim.mean(["threshold", "forecast_period"]), mean_sim)  # type:ignore[misc]
    obs_dummy.name = "source_observation"
    obs_dummy.attrs.update({"timeseries_kind": TimeseriesKind.observed_historical})  # type:ignore[misc]

    input_dataset = InputDataset([obs_dummy, sim])

    config_instance = deepcopy(score_config_crps_cdf.model_dump())  # type:ignore[misc]
    conf = config_instance  # type:ignore[misc]
    conf["general"]["verification_pairs"][0].update(  # type:ignore[misc]
        {"id": "pair1", "obs": "source_observation", "sim": "source_probabilistic"},
    )

    result = CrpsCDF(CrpsCDFConfig(**conf)).compute(  # type:ignore[misc]
        data=input_dataset,
    )
    assert result.name == score_config_crps_cdf.kind


def test_single_continuous_scores(
    score_config_continuous: ContinuousScoresConfig,
    fews_netcdf_simulated_forecast_single_fp: FewsNetCDF,
) -> None:
    """Test CRPS."""
    sim = fews_netcdf_simulated_forecast_single_fp.get_data().data_array

    # Synthetic obs
    obs = sim.isel(forecast_period=0)
    obs.name = "observed"
    obs.attrs.update({"timeseries_kind": TimeseriesKind.observed_historical})  # type:ignore[misc]

    input_dataset = InputDataset([obs, sim])

    result = ContinuousScores(score_config_continuous).compute(
        data=input_dataset,
    )
    assert isinstance(result, list)
