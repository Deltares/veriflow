"""Test the zarr datasource."""

from pathlib import Path

import pytest
import xarray as xr

from veriflow.configuration.base import GeneralInfoConfig
from veriflow.configuration.default.datasources import ZarrConfig
from veriflow.configuration.utils import S3AuthConfig
from veriflow.constants import DataSourceKind, DataType
from veriflow.datasources.zarr import Zarr

# mypy: disable-error-code=misc


def _make_zarr_config(
    *,
    general: GeneralInfoConfig,
    path: str,
    auth_config: S3AuthConfig | None = None,
    storage_options: dict[str, str] | None = None,
    data_type: DataType = DataType.observed_historical,
) -> ZarrConfig:
    return ZarrConfig(
        general=general,
        source_id=general.verification_pairs[0].observations_source_id,
        data_type=data_type,
        import_adapter=DataSourceKind.ZARR,
        path=path,
        auth_config=auth_config,
        storage_options=storage_options,
    )


def test_fetch_data_local_store(
    tmp_path: Path,
    xarray_general_info_config: GeneralInfoConfig,
    xarray_observed_historical: xr.Dataset,
) -> None:
    """Reading a local zarr store yields the original dataset and sets data_type."""
    store_path = tmp_path / "obs.zarr"
    xarray_observed_historical.to_zarr(store_path)

    datasource = Zarr(
        config=_make_zarr_config(
            general=xarray_general_info_config,
            path=str(store_path),
        ),
    )
    datasource.fetch_data()

    assert datasource.dataset.attrs["data_type"] == DataType.observed_historical
    assert datasource.dataset.attrs["source_id"] == "observation_source"

    xr.testing.assert_equal(
        datasource.dataset.drop_attrs(),
        xarray_observed_historical.drop_attrs(),
    )


def test_unsupported_data_type_raises(
    tmp_path: Path,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Unsupported data_type values are rejected by the datasource."""
    config = _make_zarr_config(
        general=xarray_general_info_config,
        path=str(tmp_path / "obs.zarr"),
    )

    with pytest.raises(NotImplementedError):
        Zarr(config=config).data_type = DataType.simulated_historical


def test_is_remote_path() -> None:
    """The remote-path heuristic detects scheme-style URLs."""
    assert Zarr._is_remote_path("s3://bucket/key/store.zarr")
    assert Zarr._is_remote_path("gs://bucket/key/store.zarr")
    assert not Zarr._is_remote_path("/tmp/store.zarr")  # noqa: S108
    assert not Zarr._is_remote_path(r"C:\Users\me\store.zarr")


def test_build_storage_options_local_returns_none(
    tmp_path: Path,
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """For local paths, no storage_options should be passed to xr.open_zarr."""
    config = _make_zarr_config(
        general=xarray_general_info_config,
        path=str(tmp_path / "obs.zarr"),
        auth_config=S3AuthConfig(anon=True),
        storage_options={"foo": "bar"},
    )
    assert Zarr(config=config)._build_storage_options() is None


def test_build_storage_options_remote_merges(
    xarray_general_info_config: GeneralInfoConfig,
) -> None:
    """Storage options from auth_config and storage_options dict are merged."""
    auth = S3AuthConfig(anon=True, region_name="eu-west-1", endpoint_url="https://s3.dummy.com")  # type: ignore[arg-type]
    config = _make_zarr_config(
        general=xarray_general_info_config,
        path="s3://bucket/key/store.zarr",
        auth_config=auth,
        storage_options={"requester_pays": "true", "endpoint_url": "https://s3.dummy.com"},
    )
    options = Zarr(config=config)._build_storage_options()
    assert options is not None
    assert options["anon"] is True
    assert options["requester_pays"] == "true"


def test_s3_auth_config_to_storage_options_unwraps_secrets() -> None:
    """SecretStr fields are unwrapped to plain strings for s3fs."""
    auth = S3AuthConfig(
        access_key_id="AKIA",  # type: ignore[arg-type]
        secret_access_key="secret",  # type: ignore[arg-type]  # noqa: S106
        session_token="token",  # type: ignore[arg-type]  # noqa: S106
        endpoint_url="https://s3.example.com",  # type: ignore[arg-type]
        region_name="us-east-1",
    )
    options = auth.to_storage_options()
    assert options["key"] == "AKIA"
    assert options["secret"] == "secret"  # noqa: S105
    assert options["token"] == "token"  # noqa: S105
    assert options["client_kwargs"] == {
        "endpoint_url": "https://s3.example.com/",
        "region_name": "us-east-1",
    }
    assert options["anon"] is False
