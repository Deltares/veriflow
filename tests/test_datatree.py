"""Test the veriflow datatree."""

from typing import cast

import pytest
import xarray as xr

from veriflow.configuration.base import BaseDatasinkConfig, GeneralInfoConfig
from veriflow.configuration.utils import VerificationPair
from veriflow.datasources.csv import Csv
from veriflow.datasources.fewsnetcdf import FewsNetCDF
from veriflow.datatree.datatree import DataTreeNode, VeriflowAccessor, VeriflowDataTree

# mypy: disable-error-code="misc"


def test_output_datatree_accessor(output_datatree_without_scores: VeriflowDataTree) -> None:
    """Test the output_dataset initializes successfully with lead time (fp) input."""
    assert output_datatree_without_scores is not None
    assert isinstance(output_datatree_without_scores, xr.DataTree)
    assert hasattr(output_datatree_without_scores, "veriflow")
    accessor = output_datatree_without_scores.veriflow
    assert hasattr(accessor, "verification_pairs")
    assert isinstance(accessor.verification_pairs, list)
    assert hasattr(accessor, "add_staged_input_data")
    assert hasattr(accessor, "add_score")


def test_input_staged_subtree_valid(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
) -> None:
    """Test the input data structure of the input dataset within the output datatree."""
    dt = output_datatree_without_scores
    assert dt is not None
    assert isinstance(dt, xr.DataTree)
    assert hasattr(dt, "veriflow")
    accessor = dt.veriflow
    assert hasattr(accessor, "get_aligned_input")
    assert callable(accessor.get_aligned_input)
    pair = dt.veriflow.verification_pairs[0]
    aligned_input_dt = accessor.get_aligned_input(pair)
    assert isinstance(aligned_input_dt, xr.DataTree)
    assert DataTreeNode.OBSERVATIONS in aligned_input_dt.children
    assert DataTreeNode.SIMULATIONS in aligned_input_dt.children
    reference_ds = aligned_input_dt[DataTreeNode.OBSERVATIONS].to_dataset()
    evaluation_ds = aligned_input_dt[DataTreeNode.SIMULATIONS].to_dataset()
    assert isinstance(reference_ds, xr.Dataset)
    assert isinstance(evaluation_ds, xr.Dataset)
    # Always expect exactly one data variable in both reference and evaluation datasets.
    #  because each verification pair applied to one variable.
    assert len(reference_ds.data_vars) == 1
    assert set(reference_ds.data_vars) == set(evaluation_ds.data_vars)
    # Check that indeed, the source_id of the reference and evaluation datasets exist.
    assert (
        reference_ds[next(iter(reference_ds.data_vars))].attrs["source_id"]
        == fake_verification_pair.observations_source_id
    )
    assert (
        evaluation_ds[next(iter(evaluation_ds.data_vars))].attrs["source_id"]
        == fake_verification_pair.simulations_source_id
    )


def test_add_score_to_output_dataset(
    output_datatree_without_scores: VeriflowDataTree,
    xarray_fake_score_result: xr.DataArray,
    fake_verification_pair: VerificationPair,
) -> None:
    """Test adding a score to the output dataset."""
    dt = output_datatree_without_scores

    # Add score
    dt.veriflow.add_score(
        fake_verification_pair,
        xarray_fake_score_result,
        name=str(xarray_fake_score_result.name),
    )
    assert isinstance(dt, xr.DataTree)

    # Verify that the score was correctly added to the DataTree at the expected path.
    path_in_tree = f"{fake_verification_pair.id}/output/{xarray_fake_score_result.name}"
    assert dt[path_in_tree] == xarray_fake_score_result


def test_add_score_dataset_result(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
) -> None:
    """Test adding a score whose result is already a Dataset (not a DataArray)."""
    dt = output_datatree_without_scores
    result = xr.Dataset({"fake_score": ("station", [1, 2, 3])})

    dt.veriflow.add_score(fake_verification_pair, result, name="fake_score")

    path_in_tree = f"{fake_verification_pair.id}/output/fake_score"
    assert result.identical(dt[path_in_tree].dataset)


def test_input_and_output(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
    xarray_fake_score_result: xr.DataArray,
) -> None:
    """Test the input() and output() accessor methods."""
    dt = output_datatree_without_scores
    dt.veriflow.add_score(
        fake_verification_pair,
        xarray_fake_score_result,
        name=str(xarray_fake_score_result.name),
    )

    aligned_input_dt = dt.veriflow.get_aligned_input(fake_verification_pair.id)
    assert isinstance(aligned_input_dt, xr.DataTree)
    assert (
        fake_verification_pair.variable
        in aligned_input_dt[DataTreeNode.OBSERVATIONS].to_dataset().data_vars
    )
    assert (
        fake_verification_pair.variable
        in aligned_input_dt[DataTreeNode.SIMULATIONS].to_dataset().data_vars
    )

    outputs_datatree = dt.veriflow.get_outputs(fake_verification_pair.id)
    assert isinstance(outputs_datatree, xr.DataTree)
    assert "fake_score" in dt.veriflow.list_scores(fake_verification_pair.id)


def test_list_scores(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
    xarray_fake_score_result: xr.DataArray,
) -> None:
    """Test listing the available scores for a verification pair."""
    dt = output_datatree_without_scores
    dt.veriflow.add_score(
        fake_verification_pair,
        xarray_fake_score_result,
        name="fake_score",
    )

    assert dt.veriflow.list_scores(fake_verification_pair.id) == ["fake_score"]


def test_path_exists_in_dt(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
) -> None:
    """Test path_exists_in_dt for both existing and non-existing paths."""
    dt = output_datatree_without_scores
    assert (
        dt.veriflow._path_exists_in_dt(f"{fake_verification_pair.id}/{DataTreeNode.ALIGNED_INPUT}")
        is True
    )
    assert dt.veriflow._path_exists_in_dt("does_not_exist") is False


def test_validate_path_does_not_exist_raises(
    output_datatree_without_scores: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
    input_data_datatree: VeriflowDataTree,
) -> None:
    """Test that adding input data twice for the same pair raises ValueError."""
    dt = output_datatree_without_scores
    obs, sim = input_data_datatree.veriflow.get_pair(fake_verification_pair)

    with pytest.raises(ValueError, match="already exists"):
        dt.veriflow.add_staged_input_data(
            verification_pair=fake_verification_pair,
            obs=obs,
            sim=sim,
        )


def test_add_input_data_simulated_forecast_ensemble(
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    """Test add_input_data accepts observations and an ensemble forecast (based on frt)."""
    dt = cast("VeriflowDataTree", xr.DataTree(name="veriflow_output"))
    dt.veriflow.add_input_data([xarray_observed_historical, xarray_simulated_forecast_ensemble])

    input_data_node = dt[DataTreeNode.INPUT_DATA]
    assert set(input_data_node.children) == {
        xarray_observed_historical.verification.source_id,
        xarray_simulated_forecast_ensemble.verification.source_id,
    }


def test_add_input_data_duplicate_source_id_raises(
    xarray_observed_historical: xr.Dataset,
) -> None:
    """Test that adding two datasets with the same source_id raises ValueError."""
    dt = cast("VeriflowDataTree", xr.DataTree(name="veriflow_output"))

    with pytest.raises(ValueError, match="already exists"):
        dt.veriflow.add_input_data([xarray_observed_historical, xarray_observed_historical])


def test_map_historical_into_forecast_space(
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_ensemble: xr.Dataset,
) -> None:
    """Test the function that maps observations into forecast space."""
    variable = "var_0"
    obs = xarray_observed_historical[variable]
    sim = xarray_simulated_forecast_ensemble[variable]

    # Map the obs into forecast space
    obs_reprojected = VeriflowAccessor.map_historical_into_forecast_space(obs, sim)

    # Get a subset of obs and sim
    sim_subset = obs_reprojected.isel(station=0, lead_time=0)
    obs_subset = obs.isel(station=0).sel(time=sim_subset.forecast_reference_time)

    # We expect all values at forecast_reference_time=0 to match the observed values
    assert all(sim_subset.to_numpy() == obs_subset.to_numpy())


def test_add_input_data_simulated_forecast_single(
    xarray_observed_historical: xr.Dataset,
    xarray_simulated_forecast_single: xr.Dataset,
) -> None:
    """Test add_input_data accepts observations and a single forecast (based on frt)."""
    dt = cast("VeriflowDataTree", xr.DataTree(name="veriflow_output"))
    dt.veriflow.add_input_data([xarray_observed_historical, xarray_simulated_forecast_single])


def test_add_input_data_fewsnetcdf(
    fews_netcdf_observed_historical: FewsNetCDF,
    fews_netcdf_simulated_forecast_ensemble_frt: FewsNetCDF,
) -> None:
    """Test the fewsnetcdf datasets are accepted by add_input_data."""
    dt = cast("VeriflowDataTree", xr.DataTree(name="veriflow_output"))
    dt.veriflow.add_input_data(
        [
            fews_netcdf_observed_historical.get_data().dataset,
            fews_netcdf_simulated_forecast_ensemble_frt.get_data().dataset,
        ],
    )


def test_add_input_data_thresholds_and_get_thresholds_array(
    xarray_thresholds: Csv,
) -> None:
    """Test thresholds data is accepted and retrievable via get_thresholds_array."""
    dt = cast("VeriflowDataTree", xr.DataTree(name="veriflow_output"))
    dt.veriflow.add_input_data([xarray_thresholds.dataset])

    variable = str(next(iter(xarray_thresholds.dataset.data_vars)))
    thresholds = dt.veriflow.get_thresholds_array(variable)
    assert thresholds.identical(xarray_thresholds.dataset[variable])


def test_get_thresholds_array_raises_when_missing(
    xarray_observed_historical: xr.Dataset,
) -> None:
    """Test get_thresholds_array raises when no thresholds dataset was added."""
    dt = cast("VeriflowDataTree", xr.DataTree(name="veriflow_output"))
    dt.veriflow.add_input_data([xarray_observed_historical])

    with pytest.raises(ValueError, match="No thresholds dataset found"):
        dt.veriflow.get_thresholds_array("var_0")


@pytest.fixture
def full_datatree(
    input_data_datatree: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
    xarray_fake_score_result: xr.DataArray,
) -> VeriflowDataTree:
    """Build a datatree containing input_data, aligned_input and output for one pair."""
    obs, sim = input_data_datatree.veriflow.get_pair(fake_verification_pair)
    input_data_datatree.veriflow.add_staged_input_data(
        verification_pair=fake_verification_pair,
        obs=obs,
        sim=sim,
    )
    input_data_datatree.veriflow.add_score(
        verification_pair=fake_verification_pair,
        result=xarray_fake_score_result,
        name="fake_score",
    )
    return input_data_datatree


@pytest.mark.parametrize(
    "flags",
    [
        {"include_input_data": True, "include_aligned_input_data": True, "include_output": True},
        {"include_input_data": False, "include_aligned_input_data": False, "include_output": False},
        {"include_input_data": True, "include_aligned_input_data": False, "include_output": False},
        {"include_input_data": False, "include_aligned_input_data": True, "include_output": False},
        {"include_input_data": False, "include_aligned_input_data": False, "include_output": True},
        {"include_input_data": True, "include_aligned_input_data": True, "include_output": False},
        {"include_input_data": True, "include_aligned_input_data": False, "include_output": True},
        {"include_input_data": False, "include_aligned_input_data": True, "include_output": True},
    ],
)
def test_get_output_for_datasink_filters_combination(
    full_datatree: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
    xarray_general_info_config: GeneralInfoConfig,
    flags: dict[str, bool],
) -> None:
    """Test get_output_for_datasink includes/excludes each node for every combination of flags."""
    config = BaseDatasinkConfig(
        export_adapter="fake",
        general=xarray_general_info_config,
        include_input_data=flags["include_input_data"],
        include_aligned_input_data=flags["include_aligned_input_data"],
        include_output=flags["include_output"],
    )
    result = full_datatree.veriflow.filter_nodes(
        include_input_data=flags["include_input_data"],
        include_aligned_input_data=flags["include_aligned_input_data"],
        include_output=flags["include_output"],
    )
    assert isinstance(result, xr.DataTree)

    source_ids = {
        config.general.verification_pairs[0].observations_source_id,
        config.general.verification_pairs[0].simulations_source_id,
    }
    if flags["include_input_data"]:
        assert (DataTreeNode.INPUT_DATA in result.children) is True
        assert set(result[DataTreeNode.INPUT_DATA].children) == source_ids
    else:
        assert (DataTreeNode.INPUT_DATA in result.children) is False

    pair_id = fake_verification_pair.id
    if flags["include_aligned_input_data"] or flags["include_output"]:
        assert pair_id in result.children
        pair_node = result[pair_id]
        assert (DataTreeNode.ALIGNED_INPUT in pair_node.children) is flags[
            "include_aligned_input_data"
        ]
        assert (DataTreeNode.OUTPUT in pair_node.children) is flags["include_output"]
    else:
        assert pair_id not in result.children


def test_get_output_for_datasink_preserves_data(
    full_datatree: VeriflowDataTree,
    fake_verification_pair: VerificationPair,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Test get_output_for_datasink keeps the underlying data unchanged (filters, not mutates)."""
    config = BaseDatasinkConfig(
        export_adapter="fake",
        general=xarray_general_info_config,
        include_input_data=True,
    )
    result = full_datatree.veriflow.filter_nodes(
        include_input_data=config.include_input_data,
        include_aligned_input_data=config.include_aligned_input_data
        if hasattr(config, "include_aligned_input_data")
        else False,
        include_output=config.include_output if hasattr(config, "include_output") else False,
    )

    pair_id = fake_verification_pair.id
    aligned_path = f"{pair_id}/{DataTreeNode.ALIGNED_INPUT}/{DataTreeNode.OBSERVATIONS}"
    assert result[aligned_path].to_dataset().identical(full_datatree[aligned_path].to_dataset())

    output_path = f"{pair_id}/{DataTreeNode.OUTPUT}/fake_score"
    assert result[output_path].to_dataset().identical(full_datatree[output_path].to_dataset())

    input_data_path = f"{DataTreeNode.INPUT_DATA}/{fake_verification_pair.observations_source_id}"
    assert (
        result[input_data_path]
        .to_dataset()
        .identical(
            full_datatree[input_data_path].to_dataset(),
        )
    )


def test_get_output_for_datasink_multiple_pairs(
    input_data_datatree: VeriflowDataTree,
    xarray_fake_score_result: xr.DataArray,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Test get_output_for_datasink filters aligned_input/output per verification pair."""
    dt = input_data_datatree
    for pair_id in ("test_pair_1", "test_pair_2"):
        verification_pair = VerificationPair(
            observations_source_id="observation_source",
            simulations_source_id="simulation_ensemble_source",
            id=pair_id,
            variable="var_0",
        )
        obs, sim = dt.veriflow.get_pair(verification_pair)
        dt.veriflow.add_staged_input_data(verification_pair=verification_pair, obs=obs, sim=sim)
        dt.veriflow.add_score(
            verification_pair=verification_pair,
            result=xarray_fake_score_result,
            name="fake_score",
        )

    config = BaseDatasinkConfig(
        export_adapter="fake",
        general=xarray_general_info_config,
        include_input_data=False,
        include_aligned_input_data=True,
        include_output=False,
    )
    result = dt.veriflow.filter_nodes(
        include_input_data=config.include_input_data,
        include_aligned_input_data=config.include_aligned_input_data
        if hasattr(config, "include_aligned_input_data")
        else False,
        include_output=config.include_output if hasattr(config, "include_output") else False,
    )

    assert DataTreeNode.INPUT_DATA not in result.children
    for pair_id in ("test_pair_1", "test_pair_2"):
        pair_node = result[pair_id]
        assert DataTreeNode.ALIGNED_INPUT in pair_node.children
        assert DataTreeNode.OUTPUT not in pair_node.children
