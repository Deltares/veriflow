"""Module to test the available scores."""

from copy import deepcopy

import xarray as xr

from dpyverification.configuration.default.scores import (
    CategoricalScoresConfig,
    ContinuousScoresConfig,
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    RankHistogramConfig,
)
from dpyverification.constants import DataType
from dpyverification.datamodel.main import InputDataset
from dpyverification.datasources.fewsnetcdf import FewsNetCDF
from dpyverification.scores.categorical import CategoricalScores
from dpyverification.scores.continuous import ContinuousScores
from dpyverification.scores.probabilistic import CrpsCDF, CrpsForEnsemble, RankHistogram


def test_ensemble_crps(
    score_config_crps: CrpsForEnsembleConfig,
    xarray_observed_historical: xr.DataArray,
    xarray_simulated_forecast_ensemble: xr.DataArray,
) -> None:
    """Test CRPS."""
    obs = xarray_observed_historical
    sim = xarray_simulated_forecast_ensemble
    obs_reprojected = InputDataset.map_historical_into_forecast_space(obs, sim)

    result = CrpsForEnsemble(score_config_crps).validate_and_compute(
        obs=obs_reprojected,
        sim=sim,
    )
    assert result.name == score_config_crps.score_adapter  # type:ignore[misc]


def test_ensemble_rank_histogram(
    score_config_rank_histogram: RankHistogramConfig,
    xarray_observed_historical: xr.DataArray,
    xarray_simulated_forecast_ensemble: xr.DataArray,
) -> None:
    """Test CRPS."""
    obs = xarray_observed_historical
    sim = xarray_simulated_forecast_ensemble
    obs_reprojected = InputDataset.map_historical_into_forecast_space(obs, sim)

    result = RankHistogram(score_config_rank_histogram).validate_and_compute(
        obs=obs_reprojected,
        sim=sim,
    )
    assert result.name == "histogram_rank"  # type:ignore[misc]


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
    obs_dummy.attrs.update({"data_type": DataType.observed_historical})  # type:ignore[misc]

    config_instance = deepcopy(score_config_crps_cdf.model_dump())  # type:ignore[misc]
    conf = config_instance  # type:ignore[misc]
    conf["general"]["verification_pairs"][0].update(  # type:ignore[misc]
        {"id": "pair1", "obs": "source_observation", "sim": "source_probabilistic"},
    )

    score = CrpsCDF(CrpsCDFConfig(**conf))  # type:ignore[misc]
    result = score.validate_and_compute(obs=obs_dummy, sim=sim)
    assert score_config_crps_cdf.score_adapter in result


def test_single_continuous_scores(
    score_config_continuous: ContinuousScoresConfig,
    xarray_observed_historical: xr.DataArray,
    xarray_simulated_forecast_single: xr.DataArray,
) -> None:
    """Test CRPS."""
    obs = xarray_observed_historical
    sim = xarray_simulated_forecast_single
    obs_reprojected = InputDataset.map_historical_into_forecast_space(obs, sim)

    result = ContinuousScores(score_config_continuous).validate_and_compute(
        obs=obs_reprojected,
        sim=sim,
    )
    assert isinstance(result, xr.Dataset)  # type:ignore[misc]
    assert "mae" in result
    assert "rmse" in result


def test_categorical_scores(
    score_config_categorical: CategoricalScoresConfig,
    xarray_observed_historical: xr.DataArray,
    xarray_simulated_forecast_single: xr.DataArray,
    xarray_thresholds: xr.DataArray,
) -> None:
    """Test the categorical scores config."""
    instance = CategoricalScores(config=score_config_categorical)
    instance.validate_and_compute(
        obs=xarray_observed_historical,
        sim=xarray_simulated_forecast_single,
        thresholds=xarray_thresholds.data_array,  # type:ignore[misc]
    )
