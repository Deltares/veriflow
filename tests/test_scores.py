"""Module to test the available scores."""

import importlib
import sys
from copy import deepcopy

import pytest
import xarray as xr

from veriflow.configuration.base import GeneralInfoConfig
from veriflow.configuration.default.scores import (
    CategoricalScoresConfig,
    ContinuousScoresConfig,
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    EventOperator,
    RankHistogramConfig,
    SALScoreConfig,
    ThresholdEvent,
)
from veriflow.configuration.utils import LeadTimes, VerificationPair, VerificationPeriod
from veriflow.constants import DataType, ScoreKind, SpatialType, StandardDim
from veriflow.datasources.fewsnetcdf import FewsNetCDF
from veriflow.datatree.datatree import VeriflowAccessor
from veriflow.scores.categorical import CategoricalScores
from veriflow.scores.continuous import ContinuousScores
from veriflow.scores.probabilistic import CrpsCDF, CrpsForEnsemble, RankHistogram
from veriflow.scores.spatial import SALScore

# mypy: disable-error-code="misc"
# mypy: disable-error-code="attr-defined"
# mypy: disable-error-code="list-item"
# mypy: disable-error-code="no-untyped-call"

pysteps_available = importlib.util.find_spec("pysteps") is not None

_SKIP_PYSTEPS = pysteps_available is False


def test_ensemble_crps(
    score_config_crps: CrpsForEnsembleConfig,
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    """Test CRPS."""
    variable = "var_0"
    obs = xarray_observed_historical[variable]
    sim = xarray_simulated_forecast_ensemble[variable]
    obs.attrs["data_type"] = xarray_observed_historical.attrs["data_type"]
    sim.attrs["data_type"] = xarray_simulated_forecast_ensemble.attrs["data_type"]
    obs_reprojected = VeriflowAccessor.map_historical_into_forecast_space(obs, sim)

    result = CrpsForEnsemble(score_config_crps).validate_and_compute(
        obs=obs_reprojected,
        sim=sim,
    )
    assert result.name == score_config_crps.score_adapter


def test_ensemble_rank_histogram(
    score_config_rank_histogram: RankHistogramConfig,
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    """Test CRPS."""
    variable = "var_0"
    obs = xarray_observed_historical[variable]
    sim = xarray_simulated_forecast_ensemble[variable]
    obs.attrs["data_type"] = xarray_observed_historical.attrs["data_type"]
    sim.attrs["data_type"] = xarray_simulated_forecast_ensemble.attrs["data_type"]
    obs_reprojected = VeriflowAccessor.map_historical_into_forecast_space(obs, sim)

    result = RankHistogram(score_config_rank_histogram).validate_and_compute(
        obs=obs_reprojected,
        sim=sim,
    )
    assert result.name == "histogram_rank"


def test_probabilistic_crps_cdf(
    score_config_crps_cdf: CrpsCDFConfig,
    fews_netcdf_simulated_forecast_probabilistic_fp: FewsNetCDF,
) -> None:
    """Test CRPS."""
    sim_ds = fews_netcdf_simulated_forecast_probabilistic_fp.get_data().dataset
    variable = next(iter(sim_ds.data_vars))
    sim = sim_ds[variable]
    sim.attrs["data_type"] = sim_ds.attrs["data_type"]

    # Synthetic obs
    mean_sim = sim.threshold.mean()
    obs_dummy = xr.full_like(sim.mean(["threshold", "lead_time"]), mean_sim)
    obs_dummy.name = "source_observation"
    obs_dummy.attrs.update({"data_type": DataType.observed_historical})

    config_instance = deepcopy(score_config_crps_cdf.model_dump())
    conf = config_instance
    conf["general"]["verification_pairs"][0].update(
        {
            "id": "pair1",
            "obs": "source_observation",
            "sim": "source_probabilistic",
            "variable": variable,
        },
    )

    score = CrpsCDF(CrpsCDFConfig(**conf))
    result = score.validate_and_compute(obs=obs_dummy, sim=sim)
    assert score_config_crps_cdf.score_adapter in result


def test_single_continuous_scores(
    score_config_continuous: ContinuousScoresConfig,
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_single: xr.Dataset,
) -> None:
    """Test CRPS."""
    variable = "var_0"
    obs = xarray_observed_historical[variable]
    sim = xarray_simulated_forecast_single[variable]
    obs.attrs["data_type"] = xarray_observed_historical.attrs["data_type"]
    sim.attrs["data_type"] = xarray_simulated_forecast_single.attrs["data_type"]
    obs_reprojected = VeriflowAccessor.map_historical_into_forecast_space(obs, sim)

    result = ContinuousScores(score_config_continuous).validate_and_compute(
        obs=obs_reprojected,
        sim=sim,
    )
    assert isinstance(result, xr.Dataset)
    assert "mae" in result
    assert "rmse" in result
    assert "nse" in result
    assert "kge" in result


class TestCategoricalScores:
    """Test suite for categorical scores."""

    def generate_score_config(self, general: GeneralInfoConfig) -> CategoricalScoresConfig:
        """Generate a categorical scores configuration based on the provided general info."""
        return CategoricalScoresConfig(
            score_adapter=ScoreKind.categorical_scores,
            general=general,
            scores=["accuracy", "false_alarm_rate"],
            events=[ThresholdEvent(threshold="warn_2", operator=EventOperator.GREATER_THAN)],
            reduce_dims=[StandardDim.forecast_reference_time],
        )

    def test_simulated_forecast_single_point(
        self,
        xarray_verification_period: VerificationPeriod,
        xarray_lead_times: LeadTimes,
        xarray_observed_historical: xr.Dataset,
        xarray_simulated_forecast_single: xr.Dataset,
        xarray_thresholds: xr.DataArray,
    ) -> None:
        """Test categorical scores for simulated forecast single point."""
        variable = "var_1"
        obs = xarray_observed_historical[variable]
        sim = xarray_simulated_forecast_single[variable]

        general_info_config = GeneralInfoConfig(
            verification_period=xarray_verification_period,
            lead_times=xarray_lead_times,
            verification_pairs=[
                VerificationPair(
                    id="dummy_id",
                    observations_source_id="observation_source",
                    simulations_source_id="simulation_single_source",
                    variable=variable,
                ),
            ],
        )
        obs.attrs["data_type"] = xarray_observed_historical.attrs["data_type"]
        sim.attrs["data_type"] = xarray_simulated_forecast_single.attrs["data_type"]
        instance = CategoricalScores(
            config=self.generate_score_config(general_info_config),
        )
        instance.validate_and_compute(
            obs=obs,
            sim=sim,
            thresholds=xarray_thresholds.dataset[variable],
        )

    def test_simulated_forecast_ensemble_point(
        self,
        xarray_verification_period: VerificationPeriod,
        xarray_lead_times: LeadTimes,
        xarray_observed_historical: xr.Dataset,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        xarray_thresholds: xr.DataArray,
    ) -> None:
        """Test categorical scores for simulated forecast ensemble point."""
        variable = "var_1"
        obs = xarray_observed_historical[variable]
        sim = xarray_simulated_forecast_ensemble[variable]

        general_info_config = GeneralInfoConfig(
            verification_period=xarray_verification_period,
            lead_times=xarray_lead_times,
            verification_pairs=[
                VerificationPair(
                    id="dummy_id",
                    observations_source_id="observation_source",
                    simulations_source_id="simulation_ensemble_source",
                    variable=variable,
                ),
            ],
        )
        obs.attrs["data_type"] = xarray_observed_historical.attrs["data_type"]
        sim.attrs["data_type"] = xarray_simulated_forecast_ensemble.attrs["data_type"]
        instance = CategoricalScores(
            config=self.generate_score_config(general_info_config),
        )
        result = instance.validate_and_compute(
            obs=obs,
            sim=sim,
            thresholds=xarray_thresholds.dataset[variable],
        )
        assert "realization" in result.dims
        for realization in sim.realization.values:
            member_sim = sim.sel(realization=realization, drop=True)
            member_result = instance.validate_and_compute(
                obs=obs,
                sim=member_sim,
                thresholds=xarray_thresholds.dataset[variable],
            )
            xr.testing.assert_equal(
                result.sel(realization=realization, drop=True),
                member_result,
            )

    def test_simulated_forecast_single_gridded(
        self,
        xarray_verification_period: VerificationPeriod,
        xarray_lead_times: LeadTimes,
        xarray_observed_historical_gridded: xr.Dataset,
        xarray_simulated_forecast_single_gridded: xr.Dataset,
        xarray_thresholds: xr.DataArray,
    ) -> None:
        """Test categorical scores for simulated forecast single gridded."""
        variable = "var_1"
        obs = xarray_observed_historical_gridded[variable]
        sim = xarray_simulated_forecast_single_gridded[variable]

        general_info_config = GeneralInfoConfig(
            verification_period=xarray_verification_period,
            lead_times=xarray_lead_times,
            verification_pairs=[
                VerificationPair(
                    id="dummy_id",
                    observations_source_id="observation_source",
                    simulations_source_id="simulation_single_source",
                    variable=variable,
                ),
            ],
        )
        obs.attrs["data_type"] = xarray_observed_historical_gridded.attrs["data_type"]
        sim.attrs["data_type"] = xarray_simulated_forecast_single_gridded.attrs["data_type"]
        instance = CategoricalScores(
            config=self.generate_score_config(general_info_config),
        )
        result = instance.validate_and_compute(
            obs=obs,
            sim=sim,
            thresholds=xarray_thresholds.dataset[variable],
        )
        assert "realization" not in result.dims

    def test_simulated_forecast_ensemble_gridded(
        self,
        xarray_verification_period: VerificationPeriod,
        xarray_lead_times: LeadTimes,
        xarray_observed_historical_gridded: xr.Dataset,
        xarray_simulated_forecast_ensemble_gridded: xr.Dataset,
        xarray_thresholds: xr.DataArray,
    ) -> None:
        """Test categorical scores for simulated forecast ensemble gridded."""
        variable = "var_1"
        obs = xarray_observed_historical_gridded[variable]
        sim = xarray_simulated_forecast_ensemble_gridded[variable]

        general_info_config = GeneralInfoConfig(
            verification_period=xarray_verification_period,
            lead_times=xarray_lead_times,
            verification_pairs=[
                VerificationPair(
                    id="dummy_id",
                    observations_source_id="observation_source",
                    simulations_source_id="simulation_ensemble_source",
                    variable=variable,
                ),
            ],
        )
        obs.attrs["data_type"] = xarray_observed_historical_gridded.attrs["data_type"]
        sim.attrs["data_type"] = xarray_simulated_forecast_ensemble_gridded.attrs["data_type"]
        instance = CategoricalScores(
            config=self.generate_score_config(general_info_config),
        )
        result = instance.validate_and_compute(
            obs=obs,
            sim=sim,
            thresholds=xarray_thresholds.dataset[variable],
        )
        assert "realization" in result.dims


@pytest.mark.skipif(
    _SKIP_PYSTEPS,
    reason="pysteps not installed or Python version >= 3.13 (pysteps pip install build fails on "
    "Windows for Python 3.13/3.14)",
)
@pytest.mark.pysteps
@pytest.mark.parametrize("ensemble", [False, True])
def test_sal_score_computes(
    xarray_simulated_forecast_single_gridded: xr.Dataset,
    xarray_general_info_config: object,
    *,
    ensemble: bool,
) -> None:
    """SAL preserves non-spatial dimensions and evaluates each member independently."""
    pytest.importorskip("pysteps")

    variable = "var_0"
    sim = xarray_simulated_forecast_single_gridded[variable]
    obs = xarray_simulated_forecast_single_gridded["var_1"]
    if ensemble:
        sim = xr.concat([sim, sim * 2], dim=xr.IndexVariable("realization", [3, 7]))  # type:ignore[assignment]
    sim.attrs.update(
        {
            "data_type": (
                DataType.simulated_forecast_ensemble
                if ensemble
                else DataType.simulated_forecast_single
            ),
            "spatial_type": SpatialType.gridded,
        },
    )
    obs.attrs.update(
        {"data_type": DataType.observed_historical, "spatial_type": SpatialType.gridded},
    )

    config = SALScoreConfig(
        score_adapter=ScoreKind.sal,
        general=xarray_general_info_config.model_dump(),
    )
    score = SALScore(config)
    result = score.validate_and_compute(obs=obs, sim=sim)

    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"structure", "amplitude", "location"}
    expected_dims = tuple(dim for dim in sim.dims if dim not in (StandardDim.y, StandardDim.x))
    for component in result.data_vars.values():
        assert component.dims == expected_dims
        for dim in expected_dims:
            xr.testing.assert_equal(component[dim], sim[dim])

    if ensemble:
        for realization in sim.realization.values:
            member = sim.sel(realization=realization, drop=True)
            member.attrs["data_type"] = DataType.simulated_forecast_single
            expected = score.validate_and_compute(obs=obs, sim=member)
            xr.testing.assert_allclose(result.sel(realization=realization, drop=True), expected)
        assert not result.amplitude.sel(realization=3, drop=True).equals(
            result.amplitude.sel(realization=7, drop=True),
        )


def test_ensemble_crps_gridded(
    xarray_simulated_forecast_ensemble_gridded: xr.Dataset,
    xarray_general_info_config: object,
) -> None:
    """Test CRPS for gridded ensemble forecast (dims: frt, lead_time, realization, y, x)."""
    variable = "var_0"

    # Create obs from sim by averaging over realization (deterministic)
    # Then re-grid it to match sim's grid structure
    sim = xarray_simulated_forecast_ensemble_gridded[variable]
    obs = sim.mean(dim="realization")  # Average ensemble members

    sim.attrs.update(
        {"data_type": DataType.simulated_forecast_ensemble, "spatial_type": SpatialType.gridded},
    )
    obs.attrs.update(
        {"data_type": DataType.observed_historical, "spatial_type": SpatialType.gridded},
    )

    # Create config with empty reduce_dims for gridded data (preserve all dims)
    config = CrpsForEnsembleConfig(
        score_adapter=ScoreKind.crps_for_ensemble,
        general=xarray_general_info_config.model_dump(),
        reduce_dims=[],  # Don't reduce any dimensions for gridded
        method="ecdf",
    )

    result = CrpsForEnsemble(config).validate_and_compute(
        obs=obs,
        sim=sim,
    )

    # Check that output is a DataArray
    assert isinstance(result, xr.DataArray)
    # Check output dimensions: (forecast_reference_time, lead_time, y, x)
    assert set(result.dims) == {"forecast_reference_time", "lead_time", "y", "x"}
    # Check output coordinates preserved
    assert "x" in result.coords
    assert "y" in result.coords
    # Check no NaN values (assuming valid input)
    assert not result.isnull().any()


def test_ensemble_rank_histogram_gridded(
    xarray_simulated_forecast_ensemble_gridded: xr.Dataset,
    xarray_general_info_config: object,
) -> None:
    """Test rank histogram for gridded ensemble forecast."""
    variable = "var_0"

    # Create obs from sim by averaging over realization
    sim = xarray_simulated_forecast_ensemble_gridded[variable]
    obs = sim.mean(dim="realization")  # Average ensemble members

    sim.attrs.update(
        {"data_type": DataType.simulated_forecast_ensemble, "spatial_type": SpatialType.gridded},
    )
    obs.attrs.update(
        {"data_type": DataType.observed_historical, "spatial_type": SpatialType.gridded},
    )

    # Create config with temporal reduce_dims for gridded data
    # For gridded data, reduce over temporal dimensions to preserve spatial dims
    config = RankHistogramConfig(
        score_adapter=ScoreKind.rank_histogram,
        general=xarray_general_info_config.model_dump(),
        reduce_dims=[
            StandardDim.forecast_reference_time,
            StandardDim.lead_time,
        ],  # Reduce temporal dims, preserve spatial
    )

    result = RankHistogram(config).validate_and_compute(
        obs=obs,
        sim=sim,
    )

    # Check that output is a DataArray
    assert isinstance(result, xr.DataArray)
    # Check output has rank dimension
    assert "rank" in result.dims or "realization" in result.dims
    # Check spatial dimensions preserved
    assert "x" in result.dims or "x" in result.coords
    assert "y" in result.dims or "y" in result.coords
