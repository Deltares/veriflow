"""Test the main module of the veriflow.configuration package."""

from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import Literal

import pytest
import xarray as xr
import yaml
from pydantic import BaseModel

from veriflow.configuration import Config
from veriflow.configuration.base import IdMap, IdMappingConfig
from veriflow.configuration.default.scores import (
    ContinuousScoresConfig,
    CrpsForEnsembleConfig,
    ReduceDims,
)
from veriflow.configuration.utils import (
    FewsWebserviceAuthConfig,
    LeadTimes,
    Range,
    TimeUnits,
    VerificationPair,
)
from veriflow.constants import SCHEMA_VERSION, StandardDim


@pytest.fixture  # type:ignore[misc] # has type overloaded function
def _mock_env(monkeypatch: Generator[pytest.MonkeyPatch, None, None]) -> None:
    """Create a mock environment for testing secret env vars."""
    monkeypatch.setenv("FEWSWEBSERVICE_URL", "https://fixture_url.test")  # type: ignore  # noqa: PGH003
    monkeypatch.setenv("FEWSWEBSERVICE_USERNAME", "fixture_user")  # type: ignore  # noqa: PGH003
    monkeypatch.setenv("FEWSWEBSERVICE_PASSWORD", "fixture_pass")  # type: ignore  # noqa: PGH003


@pytest.mark.usefixtures("_mock_env")
def test_auth_config_from_fixture() -> None:
    """Test authorization configuration."""
    config = FewsWebserviceAuthConfig()
    assert str(config.url) == "https://fixture_url.test/"
    assert config.username.get_secret_value() == "fixture_user"
    assert config.password.get_secret_value() == "fixture_pass"


def test_schema_up_to_date(tmp_path: Path) -> None:
    """Check that the schema for our config is up to date."""
    file_path_schema = (
        Path(__file__).parent.parent / "schemas" / f"{SCHEMA_VERSION}" / "config.schema.json"
    )
    assert file_path_schema.exists()

    tmp_file_path = tmp_path / "schema.json"
    Config.write_schema(tmp_file_path)

    # Check that the generated schema is identical to the one in the repository
    with file_path_schema.open("r", encoding="utf-8") as f:
        schema_in_repo = yaml.safe_load(f)  # type:ignore[misc]
    with tmp_file_path.open("r", encoding="utf-8") as f:
        generated_schema = yaml.safe_load(f)  # type:ignore[misc]
    assert schema_in_repo == generated_schema  # type:ignore[misc]


def test_schema_jsonable(tmp_path: Path) -> None:
    """Check that the schema for our config is jsonable.

    This so we can be sure it will generate correctly for the documentation of our configuration.
    """
    tmpfile = tmp_path / "config.json"

    Config.write_schema(tmpfile)

    assert tmpfile.exists()

    # TODO(AU): Additional tests on the configuration schema # noqa: FIX002
    #   https://github.com/Deltares/veriflow/issues/37
    #   When adding documentation, can add the json schema in the doc. Then, also compare the
    #   version in the documentation with the current version as per this test.


def test_schema_dump_with_user_models(tmp_path: Path) -> None:
    """Check that the schema for our config is jsonable.

    This so we can be sure it will generate correctly for the documentation of our configuration.
    """
    if not tmp_path.exists():
        tmp_path.mkdir()

    tmpfile = tmp_path / "config.json"

    class UserDatasourceConfig(BaseModel):
        import_adapter: Literal["userdatasource"]
        some_other_property: list[int]

    class UserScoreconfig(BaseModel):
        score_adapter: Literal["userscore"]
        some_other_property: list[int]

    class UserDataSinkConfig(BaseModel):
        export_adapter: Literal["userdatasink"]
        some_other_property: list[int]

    Config.write_schema(
        tmpfile,
        user_datasources_config=[UserDatasourceConfig],  # type:ignore[list-item]
        users_scores_config=[UserScoreconfig],  # type:ignore[list-item]
        user_datasinks_config=[UserDataSinkConfig],  # type:ignore[list-item]
    )
    assert tmpfile.exists()

    with tmpfile.open("r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)  # type:ignore[misc]

        # Check that the user-provided configuration is part of the schema
        assert (
            "userdatasource"
            in schema["properties"]["datasources"]["items"]["discriminator"]["mapping"]  # type:ignore[misc]
        )
        assert "userscore" in schema["properties"]["scores"]["items"]["discriminator"]["mapping"]  # type:ignore[misc]
        assert (
            "userdatasink"
            in schema["properties"]["datasinks"]["anyOf"][0]["items"]["discriminator"]["mapping"]  # type:ignore[misc]
        )


def test_lead_time_config() -> None:
    """Check lead times config identical when using list or range."""
    list_instance = LeadTimes(unit=TimeUnits.hour, values=[1, 2, 3])
    range_instance = LeadTimes(
        unit=TimeUnits.hour,
        values=Range(start=1, end=3, step=1).to_list(),
    )
    assert list_instance == range_instance
    assert list_instance.timedelta64 == range_instance.timedelta64
    assert list_instance.stdlib_timedelta == range_instance.stdlib_timedelta
    assert list_instance.max == range_instance.max
    assert list_instance.min == range_instance.min


def test_single_id_map_get_mapping() -> None:
    """Test id mapping get_mapping method."""
    config = IdMap({"intId1": {"sourceA": "extId1"}})
    assert config.get_external_to_internal_mapping("sourceA") == {"extId1": "intId1"}


def test_id_mapping_rename_dataset(xarray_observed_historical: xr.Dataset) -> None:
    """Test partial renaming of stations on dummy dataset."""
    config = IdMappingConfig(
        station=IdMap({"newstation1": {"observation_source": "station_0"}}),
    )
    new_ds = config.apply(xarray_observed_historical)
    assert next(iter(new_ds.station.to_numpy())) == "newstation1"  # type:ignore[misc]


def test_id_mapping_rename_dataset_fails_on_invalid_source(
    xarray_observed_historical: xr.Dataset,
) -> None:
    """Test partial renaming of stations on dummy dataset."""
    config = IdMappingConfig(
        station=IdMap({"newstation1": {"invalid_observation_source": "station0"}}),
    )
    with pytest.raises(ValueError, match="No IdMapping found for source"):
        config.apply(xarray_observed_historical)


def test_score_config_with_invalid_pair_reference(
    score_config_crps: CrpsForEnsembleConfig,
) -> None:
    """Test CRPS."""
    modified_config = deepcopy(score_config_crps.model_dump())  # type:ignore[misc]
    modified_config["verification_pair_ids"] = ["invalid_id"]  # type:ignore[misc]
    with pytest.raises(ValueError, match="Pair id"):
        _ = CrpsForEnsembleConfig(**modified_config)  # type:ignore[misc]


def test_reduce_dims_forecast_validation() -> None:
    """Test that reduce_dims validation works as expected."""
    # Valid cases
    valid_cases: list[
        list[
            Literal[
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
                StandardDim.time,
            ]
        ]
    ] = [
        [StandardDim.station, StandardDim.time],
        [StandardDim.station],
        [StandardDim.station, StandardDim.forecast_reference_time],
        [StandardDim.station, StandardDim.lead_time],
        [StandardDim.station, StandardDim.forecast_reference_time, StandardDim.lead_time],
    ]

    for reduce_dims in valid_cases:
        config = ReduceDims(reduce_dims=reduce_dims)  # type:ignore[arg-type]
        assert config.reduce_dims == reduce_dims

    # Invalid case: both historical and forecast dimensions would be filtered
    # at runtime by compute_reduce_and_preserve_dims(), so configuration validation
    # no longer rejects this. The unified ReduceDims accepts all dimension combinations
    # and filters them based on actual data dimensions.
    # This test is removed since validation now happens at runtime via the utility function.


def test_score_config_with_nse_and_no_reduce_dims_raises_validation_error(
    score_config_continuous: ContinuousScoresConfig,
) -> None:
    """Test that if nse is in scores, reduce_dims is not empty."""
    score_config_continuous_copy = deepcopy(score_config_continuous.model_dump())  # type:ignore[misc]
    score_config_continuous_copy["reduce_dims"] = []  # type:ignore[misc]

    match_str = "NSE: need at least one dimension to be reduced."
    with pytest.raises(ValueError, match=match_str):
        _ = ContinuousScoresConfig(**score_config_continuous_copy)  # type:ignore[misc]


def test_reduce_dims_validates_spatial_dimensions() -> None:
    """Test that ReduceDims validates x and y must be together and not with station."""
    # Valid: x and y together
    config = ReduceDims(reduce_dims=[StandardDim.x, StandardDim.y])
    assert config.reduce_dims == [StandardDim.x, StandardDim.y]

    # Invalid: x without y
    with pytest.raises(
        ValueError,
        match="Both x and y dimensions must be configured together",
    ):
        _ = ReduceDims(reduce_dims=[StandardDim.x])

    # Invalid: station with spatial dimensions
    with pytest.raises(
        ValueError,
        match="Cannot configure both spatial dimensions \\(x, y\\) and station together",
    ):
        _ = ReduceDims(reduce_dims=[StandardDim.station, StandardDim.x, StandardDim.y])


def test_not_accepted_verification_pair_id_raised() -> None:
    """Test that a not accepted verification pair ID raises an error."""
    with pytest.raises(
        ValueError,
        match="input_data' is a reserved id and cannot be used as a verification pair id",
    ):
        VerificationPair(
            id="input_data",
            observations_source_id="some_source",
            simulations_source_id="some_source",
            variable="some_variable",
        )
