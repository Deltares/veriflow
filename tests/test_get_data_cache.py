"""Integration tests for ``BaseDatasource.get_data`` cache behaviour.

Uses an in-test ``FakeDatasource`` (registered via ``fake_seed_registry``) backed by
the existing ``xarray_simulated_forecast_ensemble`` and ``xarray_observed_historical``
fixtures. The cache itself writes to a real local Zarr store under ``cache_dir``.
"""

# mypy: ignore-errors

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from tests.conftest import FakeDatasource, FakeDatasourceConfig
from veriflow.cache.cache import ZarrCache
from veriflow.cache.config import ZarrCacheConfig
from veriflow.configuration.base import GeneralInfoConfig
from veriflow.configuration.utils import (
    LeadTimes,
    TimeUnits,
)
from veriflow.constants import DataType, StandardDim

_EXPECTED_LEAD_TIMES_SPLIT = 2
_EXPECTED_LEAD_TIMES_TOTAL = 4

FakeSeedRegistry = Callable[[str, DataType, xr.Dataset], None]
FakeFetchSpy = list[dict[str, object]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_forecast_config(
    general: GeneralInfoConfig,
    *,
    variables: list[str],
    stations: list[str],
    source: str = "source_single",
) -> FakeDatasourceConfig:
    """Build a ``FakeDatasourceConfig`` for forecast data."""
    return FakeDatasourceConfig(
        general=general,
        source_id=source,
        data_type=DataType.simulated_forecast_ensemble,
        variables=variables,
        stations=stations,
    )


def _make_fake_historical_config(
    general: GeneralInfoConfig,
    *,
    variables: list[str],
    stations: list[str],
    source: str = "observed",
) -> FakeDatasourceConfig:
    """Build a ``FakeDatasourceConfig`` for historical data."""
    return FakeDatasourceConfig(
        general=general,
        source_id=source,
        data_type=DataType.observed_historical,
        variables=variables,
        stations=stations,
    )


def _zarr_store_path(cache_dir: str) -> Path:
    """Return the conventional path to the on-disk Zarr cache store."""
    return Path(cache_dir) / "veriflow-cache.zarr"


def _zarr_store_has_data(cache_dir: str) -> bool:
    """Return ``True`` if the zarr store contains actual data (not just an empty dir)."""
    p = _zarr_store_path(cache_dir)
    if not p.exists():
        return False
    return any(p.iterdir())


def _ts(day: int) -> datetime:
    """Build a tz-naive ``2025-01-<day>`` timestamp matching the seed's tz-naive coordinates."""
    return datetime(2025, 1, day)  # noqa: DTZ001


# ---------------------------------------------------------------------------
# Forecast scenarios
# ---------------------------------------------------------------------------


class TestForecastNoCache:
    """Behaviour when no cache is configured at all."""

    def test_no_cache_configured(
        self,
        xarray_general_info_config: GeneralInfoConfig,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
        cache_dir_local: str,
    ) -> None:
        """Verify ``get_data`` fetches and skips the cache when none is configured."""
        fake_seed_registry(
            "source_single",
            DataType.simulated_forecast_ensemble,
            xarray_simulated_forecast_ensemble,
        )
        # No cache attached
        assert xarray_general_info_config.cache is None
        cfg = _make_fake_forecast_config(
            xarray_general_info_config,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds = FakeDatasource(cfg).get_data()
        assert len(fake_fetch_spy) == 1
        assert set(ds.dataset.data_vars) == {"var_0", "var_1"}
        # No zarr store written
        assert not _zarr_store_has_data(cache_dir_local)


class TestForecastCacheMiss:
    """Behaviour on a cache miss."""

    def test_miss_with_write_mode_creates_store(
        self,
        general_info_config_with_cache: GeneralInfoConfig,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
        cache_dir_local: str,
    ) -> None:
        """Verify a writable cache miss persists data to the on-disk store."""
        fake_seed_registry(
            "source_single",
            DataType.simulated_forecast_ensemble,
            xarray_simulated_forecast_ensemble,
        )
        cfg = _make_fake_forecast_config(
            general_info_config_with_cache,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds = FakeDatasource(cfg)
        ds.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds.get_data()
        assert len(fake_fetch_spy) == 1
        assert _zarr_store_has_data(cache_dir_local)
        assert set(ds.dataset.data_vars) == {"var_0", "var_1"}

    def test_miss_with_read_only_does_not_write(  # noqa: PLR0913, PLR0917
        self,
        xarray_general_info_config: GeneralInfoConfig,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        cache_zarr_config_readonly: ZarrCacheConfig,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
        cache_dir_local: str,
    ) -> None:
        """Verify a read-only cache miss leaves the on-disk store empty."""
        fake_seed_registry(
            "source_single",
            DataType.simulated_forecast_ensemble,
            xarray_simulated_forecast_ensemble,
        )
        general = xarray_general_info_config.model_copy(deep=True)
        general.cache = cache_zarr_config_readonly
        cfg = _make_fake_forecast_config(
            general,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        FakeDatasource(cfg).get_data()
        assert len(fake_fetch_spy) == 1
        assert not _zarr_store_has_data(cache_dir_local)


class TestForecastCacheFullHit:
    """Behaviour on a full cache hit."""

    def test_full_hit_skips_fetch(
        self,
        general_info_config_with_cache: GeneralInfoConfig,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
    ) -> None:
        """Verify a full cache hit serves data without re-fetching."""
        fake_seed_registry(
            "source_single",
            DataType.simulated_forecast_ensemble,
            xarray_simulated_forecast_ensemble,
        )
        cfg = _make_fake_forecast_config(
            general_info_config_with_cache,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        # First call populates cache
        ds = FakeDatasource(cfg)
        ds.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds.get_data()
        assert len(fake_fetch_spy) == 1
        fake_fetch_spy.clear()
        # Second call with the same config is a full cache hit — no fetch.
        cfg2 = _make_fake_forecast_config(
            general_info_config_with_cache,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds2 = FakeDatasource(cfg2)
        ds2.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds2.get_data()
        assert len(fake_fetch_spy) == 0
        assert set(ds2.dataset.data_vars) == {"var_0", "var_1"}


class TestForecastPartialHit:
    """Behaviour on a partial cache hit (split fetch)."""

    def test_missing_variables_only_fetches_missing(
        self,
        general_info_config_with_cache: GeneralInfoConfig,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
    ) -> None:
        """Verify only missing variables are re-fetched on a partial hit."""
        fake_seed_registry(
            "source_single",
            DataType.simulated_forecast_ensemble,
            xarray_simulated_forecast_ensemble,
        )
        # Pre-populate cache with only var_0
        cfg_a = _make_fake_forecast_config(
            general_info_config_with_cache,
            variables=["var_0"],
            stations=["station_0", "station_1"],
        )
        ds_a = FakeDatasource(cfg_a)
        ds_a.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds_a.get_data()
        assert len(fake_fetch_spy) == 1
        fake_fetch_spy.clear()
        # Now request both var_0 and var_1 — should fetch only var_1
        cfg_b = _make_fake_forecast_config(
            general_info_config_with_cache,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds = FakeDatasource(cfg_b)
        ds.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds.get_data()
        assert len(fake_fetch_spy) == 1
        assert fake_fetch_spy[0]["variables"] == ["var_1"]
        assert set(ds.dataset.data_vars) == {"var_0", "var_1"}

    def test_missing_stations_only_fetches_missing(
        self,
        general_info_config_with_cache: GeneralInfoConfig,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
    ) -> None:
        """Verify only missing stations are re-fetched on a partial hit."""
        fake_seed_registry(
            "source_single",
            DataType.simulated_forecast_ensemble,
            xarray_simulated_forecast_ensemble,
        )
        # Pre-populate cache with only station_0
        cfg_a = _make_fake_forecast_config(
            general_info_config_with_cache,
            variables=["var_0", "var_1"],
            stations=["station_0"],
        )
        ds_a = FakeDatasource(cfg_a)
        ds_a.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds_a.get_data()
        fake_fetch_spy.clear()
        # Request station_0 + station_1
        cfg_b = _make_fake_forecast_config(
            general_info_config_with_cache,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds_b = FakeDatasource(cfg_b)
        ds_b.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds_b.get_data()
        assert len(fake_fetch_spy) == 1
        assert fake_fetch_spy[0]["stations"] == ["station_1"]
        assert set(ds_b.dataset[StandardDim.station].values) == {"station_0", "station_1"}

    def test_missing_lead_times_only_fetches_missing(
        self,
        general_info_config_with_cache: GeneralInfoConfig,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
    ) -> None:
        """Verify only missing lead-times are re-fetched on a partial hit."""
        fake_seed_registry(
            "source_single",
            DataType.simulated_forecast_ensemble,
            xarray_simulated_forecast_ensemble,
        )
        # Pre-populate cache with lead_times [1, 2] (in hours, since seed uses hourly fp_step)
        # We need lead_times that exist in the seed.
        # The seed has lead_times = pd.timedelta_range(0, periods=fp_n, freq="h")
        # i.e. [0h, 1h, 2h, ..., 119h]. Use hour units.
        general_lt12 = general_info_config_with_cache.model_copy(deep=True)
        general_lt12.lead_times = LeadTimes(unit=TimeUnits.hour, values=[1, 2])
        cfg_a = _make_fake_forecast_config(
            general_lt12,
            variables=["var_0"],
            stations=["station_0"],
        )
        ds_a = FakeDatasource(cfg_a)
        ds_a.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds_a.get_data()
        fake_fetch_spy.clear()
        # Now request lead-times one through four.
        general_lt1234 = general_info_config_with_cache.model_copy(deep=True)
        general_lt1234.lead_times = LeadTimes(unit=TimeUnits.hour, values=[1, 2, 3, 4])
        cfg_b = _make_fake_forecast_config(
            general_lt1234,
            variables=["var_0"],
            stations=["station_0"],
        )
        ds_b = FakeDatasource(cfg_b)
        ds_b.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds_b.get_data()
        assert len(fake_fetch_spy) == 1
        # Spy records lead_times.values - after split, missing should be [3, 4] in some unit
        # The split converts to nanoseconds. Verify by length of timedelta64.
        spy_lt = fake_fetch_spy[0]["config"].lead_times
        assert spy_lt is not None
        assert len(spy_lt.timedelta64) == _EXPECTED_LEAD_TIMES_SPLIT
        # Resulting dataset should have all 4
        assert ds_b.dataset[StandardDim.lead_time].size == _EXPECTED_LEAD_TIMES_TOTAL

    def test_missing_frt_period_only_fetches_missing(
        self,
        general_info_config_with_cache: GeneralInfoConfig,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
    ) -> None:
        """Verify a rolling forecast-reference-time window only fetches missing FRTs and merges.

        Operational scenario: a previous run cached forecasts issued up to "yesterday"; the current
        run requests a window rolled one day forward ("last week until now"), so only the most
        recent forecast reference times are missing and must be fetched, then merged with the
        cached remainder.
        """
        fake_seed_registry(
            "source_single",
            DataType.simulated_forecast_ensemble,
            xarray_simulated_forecast_ensemble,
        )

        # First run caches forecasts with reference times in [01-05 .. 01-08].
        general_cached = general_info_config_with_cache.model_copy(deep=True)
        general_cached.verification_period.start = _ts(5)
        general_cached.verification_period.end = _ts(8)
        cfg_a = _make_fake_forecast_config(
            general_cached,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        _ = FakeDatasource(cfg_a)
        _.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        _.get_data()
        assert len(fake_fetch_spy) == 1
        fake_fetch_spy.clear()

        # Second run rolls the window forward to [01-06 .. 01-09]: the most recent forecast
        # reference time day (01-08 .. 01-09) is missing from the cache.
        general_now = general_info_config_with_cache.model_copy(deep=True)
        general_now.verification_period.start = _ts(6)
        general_now.verification_period.end = _ts(9)
        cfg_b = _make_fake_forecast_config(
            general_now,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds_b = FakeDatasource(cfg_b)
        ds_b.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds_b.get_data()

        # Only the missing FRT tail is fetched from the datasource.
        assert len(fake_fetch_spy) == 1
        assert fake_fetch_spy[0]["vp_start"] == _ts(8)
        assert fake_fetch_spy[0]["vp_end"] == _ts(9)

        # The merged result covers all requested reference times with no gaps (proper merge).
        merged = ds_b.dataset.sortby(StandardDim.forecast_reference_time).load()
        frts = merged[StandardDim.forecast_reference_time].values
        assert frts.min() == np.datetime64("2025-01-06")
        assert frts.max() == np.datetime64("2025-01-09")
        for var in ("var_0", "var_1"):
            assert not merged[var].isnull().any().item()

        # Values match the seed over the requested FRT window (cached + fetched merged correctly).
        expected = xarray_simulated_forecast_ensemble.sel(
            station=["station_0", "station_1"],
            forecast_reference_time=slice(
                np.datetime64("2025-01-06"),
                np.datetime64("2025-01-09"),
            ),
            lead_time=merged[StandardDim.lead_time].values,
        ).sortby(StandardDim.forecast_reference_time)
        order = (
            StandardDim.station,
            StandardDim.forecast_reference_time,
            StandardDim.lead_time,
            StandardDim.realization,
        )
        for var in ("var_0", "var_1"):
            np.testing.assert_allclose(
                merged[var].transpose(*order).values,
                expected[var].transpose(*order).values,
            )

    def test_missing_earlier_frt_period_reads_back_sorted(
        self,
        general_info_config_with_cache: GeneralInfoConfig,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
    ) -> None:
        """Verify an earlier (backward) window works despite the append leaving the axis unsorted.

        Incremental ``append_dim`` writes new forecast reference times at the end of the store, so
        requesting an *earlier* window makes the store's coordinate non-monotonic. The read-back
        must sort first for the slice selection to succeed.
        """
        fake_seed_registry(
            "source_single",
            DataType.simulated_forecast_ensemble,
            xarray_simulated_forecast_ensemble,
        )

        # First run caches forecasts with reference times in [01-05 .. 01-08].
        general_cached = general_info_config_with_cache.model_copy(deep=True)
        general_cached.verification_period.start = _ts(5)
        general_cached.verification_period.end = _ts(8)
        cfg_a = _make_fake_forecast_config(
            general_cached,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds_a = FakeDatasource(cfg_a)
        ds_a.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds_a.get_data()
        fake_fetch_spy.clear()

        # Second run requests an EARLIER window [01-03 .. 01-06] (left overlap): 01-03/01-04 are
        # missing and get appended at the end of the store, leaving it out of order.
        general_now = general_info_config_with_cache.model_copy(deep=True)
        general_now.verification_period.start = _ts(3)
        general_now.verification_period.end = _ts(6)
        cfg_b = _make_fake_forecast_config(
            general_now,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds = FakeDatasource(cfg_b)
        ds.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds.get_data()

        # Only the missing (earlier) reference times are fetched.
        assert len(fake_fetch_spy) == 1
        assert fake_fetch_spy[0]["vp_start"] == _ts(3)
        assert fake_fetch_spy[0]["vp_end"] == _ts(5)

        # The read-back sorts the store, so the full requested window is returned with no gaps.
        merged = ds.dataset.sortby(StandardDim.forecast_reference_time).load()
        frts = merged[StandardDim.forecast_reference_time].values
        assert frts.min() == np.datetime64("2025-01-03")
        assert frts.max() == np.datetime64("2025-01-06")
        for var in ("var_0", "var_1"):
            assert not merged[var].isnull().any().item()

    def test_multi_missing_dims_full_refetch(
        self,
        general_info_config_with_cache: GeneralInfoConfig,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
    ) -> None:
        """Verify multi-dim misses trigger a full re-fetch (no split)."""
        fake_seed_registry(
            "source_single",
            DataType.simulated_forecast_ensemble,
            xarray_simulated_forecast_ensemble,
        )
        # Pre-populate cache with subset of variables AND subset of stations
        cfg_a = _make_fake_forecast_config(
            general_info_config_with_cache,
            variables=["var_0"],
            stations=["station_0"],
        )
        ds_a = FakeDatasource(cfg_a)
        ds_a.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds_a.get_data()
        fake_fetch_spy.clear()
        # Request both extensions
        cfg_b = _make_fake_forecast_config(
            general_info_config_with_cache,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds = FakeDatasource(cfg_b)
        ds.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds.get_data()
        # Multi-missing → split_config returns None → full re-fetch with original config
        assert len(fake_fetch_spy) == 1
        assert fake_fetch_spy[0]["variables"] == ["var_0", "var_1"]
        assert fake_fetch_spy[0]["stations"] == ["station_0", "station_1"]
        assert set(ds.dataset.data_vars) == {"var_0", "var_1"}


# ---------------------------------------------------------------------------
# Historical scenarios
# ---------------------------------------------------------------------------


class TestHistoricalCache:
    """Behaviour for historical (non-forecast) datasources."""

    def test_miss_with_write_mode_creates_store(
        self,
        general_info_config_historical_with_cache: GeneralInfoConfig,
        xarray_observed_historical: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
        cache_dir_local: str,
    ) -> None:
        """Verify a historical cache miss persists data to the on-disk store."""
        fake_seed_registry(
            "observed",
            DataType.observed_historical,
            xarray_observed_historical,
        )
        cfg = _make_fake_historical_config(
            general_info_config_historical_with_cache,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds = FakeDatasource(cfg)
        ds.cache = ZarrCache(general_info_config_historical_with_cache.cache)  # type: ignore[arg-type]
        ds.get_data()
        assert len(fake_fetch_spy) == 1
        assert _zarr_store_path(cache_dir_local).exists()
        assert set(ds.dataset.data_vars) == {"var_0", "var_1"}

    def test_missing_recent_time_period_is_fetched_and_merged(
        self,
        general_info_config_historical_with_cache: GeneralInfoConfig,
        xarray_observed_historical: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
    ) -> None:
        """Verify a rolling time window reuses cached observations and fetches only the new tail.

        Operational scenario: a previous run cached observations up to "yesterday"; the current
        run requests a window rolled one day forward, so only the most recent day is missing from
        the cache and must be fetched, then merged with the cached remainder.
        """
        fake_seed_registry(
            "observed",
            DataType.observed_historical,
            xarray_observed_historical,
        )

        # First run caches observations for [01-02 .. 01-09].
        general_cached = general_info_config_historical_with_cache.model_copy(deep=True)
        general_cached.verification_period.start = _ts(2)
        general_cached.verification_period.end = _ts(9)
        cfg_a = _make_fake_historical_config(
            general_cached,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds_a = FakeDatasource(cfg_a)
        ds_a.cache = ZarrCache(general_info_config_historical_with_cache.cache)  # type: ignore[arg-type]
        ds_a.get_data()
        assert len(fake_fetch_spy) == 1
        fake_fetch_spy.clear()

        # Second run rolls the window forward to [01-03 .. 01-10]: the most recent day
        # (01-09 .. 01-10) is missing from the cache.
        general_now = general_info_config_historical_with_cache.model_copy(deep=True)
        general_now.verification_period.start = _ts(3)
        general_now.verification_period.end = _ts(10)
        cfg_b = _make_fake_historical_config(
            general_now,
            variables=["var_0", "var_1"],
            stations=["station_0", "station_1"],
        )
        ds = FakeDatasource(cfg_b)
        ds.cache = ZarrCache(general_info_config_historical_with_cache.cache)  # type: ignore[arg-type]
        ds.get_data()

        # Only the missing tail is fetched from the datasource.
        assert len(fake_fetch_spy) == 1
        assert fake_fetch_spy[0]["vp_start"] == _ts(9)
        assert fake_fetch_spy[0]["vp_end"] == _ts(10)

        # The merged result spans the full requested window with no gaps (proper merge).
        merged = ds.dataset.sortby(StandardDim.time).load()
        assert merged[StandardDim.time].min().values == np.datetime64("2025-01-03")
        assert merged[StandardDim.time].max().values == np.datetime64("2025-01-10")
        for var in ("var_0", "var_1"):
            assert not merged[var].isnull().any().item()

        # Values match the seed over the requested window (cached + fetched merged correctly).
        expected = xarray_observed_historical.sel(
            station=["station_0", "station_1"],
            time=slice(np.datetime64("2025-01-03"), np.datetime64("2025-01-10")),
        ).sortby(StandardDim.time)
        for var in ("var_0", "var_1"):
            np.testing.assert_allclose(
                merged[var].transpose(StandardDim.time, StandardDim.station).values,
                expected[var].transpose(StandardDim.time, StandardDim.station).values,
            )


# ---------------------------------------------------------------------------
# Round-trip: cache append after partial fetch
# ---------------------------------------------------------------------------


class TestCacheWriteBack:
    """Behaviour of the cache write-back after a partial fetch."""

    def test_partial_fetch_appends_only_new_data(
        self,
        general_info_config_with_cache: GeneralInfoConfig,
        xarray_simulated_forecast_ensemble: xr.Dataset,
        fake_seed_registry: FakeSeedRegistry,
        fake_fetch_spy: FakeFetchSpy,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify only newly-fetched data is appended to the cache."""
        fake_seed_registry(
            "source_single",
            DataType.simulated_forecast_ensemble,
            xarray_simulated_forecast_ensemble,
        )
        # First call populates cache with var_0 only
        cfg_a = _make_fake_forecast_config(
            general_info_config_with_cache,
            variables=["var_0"],
            stations=["station_0"],
        )
        ds_a = FakeDatasource(cfg_a)
        ds_a.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds_a.get_data()
        fake_fetch_spy.clear()

        # Spy on cache.append
        appended: list[dict[str, object]] = []
        original_append = ZarrCache.append

        def append_spy(
            self: ZarrCache,
            new_dataset: xr.Dataset,
            source: str,
            append_dim: Literal[
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
                StandardDim.time,
                "variable",
            ],
        ) -> None:
            appended.append(
                {
                    "vars": list(new_dataset.data_vars),
                    "source": source,
                    "dim": append_dim,
                },
            )
            return original_append(self, new_dataset, source, append_dim)

        monkeypatch.setattr(ZarrCache, "append", append_spy)

        # Request var_0 + var_1 — should fetch only var_1, append only var_1
        cfg_b = _make_fake_forecast_config(
            general_info_config_with_cache,
            variables=["var_0", "var_1"],
            stations=["station_0"],
        )
        ds_b = FakeDatasource(cfg_b)
        ds_b.cache = ZarrCache(general_info_config_with_cache.cache)  # type: ignore[arg-type]
        ds_b.get_data()
        assert len(appended) == 1
        assert appended[0]["vars"] == ["var_1"]
