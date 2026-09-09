"""Unit tests for the veriflow cache primitives."""

# mypy: ignore-errors

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from veriflow.cache.cache import (
    CacheRequest,
    DataRequest,
    ZarrCache,
)
from veriflow.cache.config import ReadWriteMode, ZarrCacheConfig
from veriflow.cache.utils import combine_cached_and_fetched_data
from veriflow.configuration.base import GeneralInfoConfig
from veriflow.configuration.default.datasources import NetCDFConfig
from veriflow.configuration.utils import LeadTimes, S3AuthConfig, TimePeriod
from veriflow.constants import DataSourceKind, DataType, StandardDim, TimeUnits
from veriflow.datasources.netcdf import NetCDF

_EXPECTED_MISSING_COUNT_MULTI = 2
_EXPECTED_COMBINED_TIME_SIZE = 3
_EXPECTED_FRT_SIZE = 4

# ---------------------------------------------------------------------------
# DataRequest.get_missing_and_available
# ---------------------------------------------------------------------------


class TestDataRequestSets:
    """Set semantics for variables/stations DataRequest."""

    def test_disjoint_returns_full_missing(self) -> None:
        """Verify disjoint sets report all requested as missing."""
        req = DataRequest(requested={"a", "b"}, cached={"c", "d"})
        missing, available = req.get_missing_and_available
        assert missing == {"a", "b"}
        assert available is None

    def test_identical_returns_empty_missing(self) -> None:
        """Verify identical sets report nothing missing."""
        req = DataRequest(requested={"a", "b"}, cached={"a", "b"})
        missing, available = req.get_missing_and_available
        assert missing is None
        assert available == {"a", "b"}

    def test_partial_overlap(self) -> None:
        """Verify partial overlap splits into missing/available."""
        req = DataRequest(requested={"a", "b", "c"}, cached={"b", "c", "d"})
        missing, available = req.get_missing_and_available
        assert missing == {"a"}
        assert available == {"b", "c"}

    def test_empty_cache(self) -> None:
        """Verify an empty cache reports the full request as missing."""
        req = DataRequest(requested={"a"}, cached=set())
        missing, available = req.get_missing_and_available
        assert missing == {"a"}
        assert available is None

    def test_mismatched_types_raises(self) -> None:
        """Verify mismatched requested/cached types raise ``ValueError``."""
        with pytest.raises(ValueError, match="same type"):
            DataRequest(
                requested={"a"},
                cached=TimePeriod(
                    start=datetime(2020, 1, 1, tzinfo=UTC),
                    end=datetime(2020, 1, 2, tzinfo=UTC),
                ),
            )


class TestDataRequestTimePeriod:
    """All 6 time-period arrangements + identical."""

    @staticmethod
    def _tp(start: str, end: str) -> TimePeriod:
        return TimePeriod(start=datetime.fromisoformat(start), end=datetime.fromisoformat(end))

    def test_requested_before_cached(self) -> None:
        """Verify a request entirely before the cache is fully missing."""
        # RRRR    CCCC
        req = DataRequest(
            requested=self._tp("2020-01-01", "2020-01-02"),
            cached=self._tp("2020-01-05", "2020-01-06"),
        )
        missing, available = req.get_missing_and_available
        assert missing == self._tp("2020-01-01", "2020-01-02")
        assert available is None

    def test_requested_after_cached(self) -> None:
        """Verify a request entirely after the cache is fully missing."""
        # CCCC    RRRR
        req = DataRequest(
            requested=self._tp("2020-01-05", "2020-01-06"),
            cached=self._tp("2020-01-01", "2020-01-02"),
        )
        missing, available = req.get_missing_and_available
        assert missing == self._tp("2020-01-05", "2020-01-06")
        assert available is None

    def test_requested_envelopes_cached(self) -> None:
        """Verify a request that envelopes the cache is fully missing."""
        # RRRR  vs  CC
        req = DataRequest(
            requested=self._tp("2020-01-01", "2020-01-10"),
            cached=self._tp("2020-01-03", "2020-01-05"),
        )
        missing, available = req.get_missing_and_available
        assert missing == self._tp("2020-01-01", "2020-01-10")
        assert available is None

    def test_cached_envelopes_requested(self) -> None:
        """Verify a request inside the cache is fully available."""
        # RRRR inside CCCCCC
        req = DataRequest(
            requested=self._tp("2020-01-03", "2020-01-05"),
            cached=self._tp("2020-01-01", "2020-01-10"),
        )
        missing, available = req.get_missing_and_available
        assert missing is None
        assert available == self._tp("2020-01-03", "2020-01-05")

    def test_requested_left_overlap(self) -> None:
        """Verify a request overlapping the cache on the left splits correctly."""
        # RRRR overlapping left of CCCC
        req = DataRequest(
            requested=self._tp("2020-01-01", "2020-01-04"),
            cached=self._tp("2020-01-03", "2020-01-06"),
        )
        missing, available = req.get_missing_and_available
        assert missing == self._tp("2020-01-01", "2020-01-03")
        assert available == self._tp("2020-01-03", "2020-01-04")

    def test_requested_right_overlap(self) -> None:
        """Verify a request overlapping the cache on the right splits correctly."""
        # CCCC then RRRR overlapping right
        req = DataRequest(
            requested=self._tp("2020-01-04", "2020-01-08"),
            cached=self._tp("2020-01-01", "2020-01-06"),
        )
        missing, available = req.get_missing_and_available
        assert missing == self._tp("2020-01-06", "2020-01-08")
        assert available == self._tp("2020-01-04", "2020-01-06")


class TestDataRequestLeadTimes:
    """LeadTimes semantics."""

    @staticmethod
    def _lt(values: list[int]) -> LeadTimes:
        return LeadTimes(unit=TimeUnits.day, values=values)

    def test_identical(self) -> None:
        """Verify identical lead-time lists are fully available."""
        req = DataRequest(requested=self._lt([1, 2]), cached=self._lt([1, 2]))
        missing, available = req.get_missing_and_available
        assert missing is None
        assert available == self._lt([1, 2])

    def test_fully_missing(self) -> None:
        """Verify disjoint lead-time lists are fully missing."""
        req = DataRequest(requested=self._lt([5, 6]), cached=self._lt([1, 2]))
        missing, available = req.get_missing_and_available
        assert missing == self._lt([5, 6])
        assert available is None

    def test_partial(self) -> None:
        """Verify partial lead-time overlap splits into missing/available."""
        req = DataRequest(requested=self._lt([1, 2, 3, 4]), cached=self._lt([1, 2]))
        missing, available = req.get_missing_and_available
        assert isinstance(missing, LeadTimes)
        assert isinstance(available, LeadTimes)
        # Values are converted to nanoseconds
        ns_per_day = 24 * 3600 * 10**9
        assert missing.values == [3 * ns_per_day, 4 * ns_per_day]
        assert available.values == [1 * ns_per_day, 2 * ns_per_day]


# ---------------------------------------------------------------------------
# CacheRequest
# ---------------------------------------------------------------------------


def _all_match_request() -> CacheRequest:
    """Build a historical request where everything matches the cache.

    Cache strictly envelopes the requested period (cache logic doesn't
    handle exact-equality periods).
    """
    requested = TimePeriod(
        start=datetime(2020, 1, 2, tzinfo=UTC),
        end=datetime(2020, 1, 9, tzinfo=UTC),
    )
    cached = TimePeriod(
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 1, 10, tzinfo=UTC),
    )
    return CacheRequest(
        variables=DataRequest(requested={"a"}, cached={"a"}),
        stations=DataRequest(requested={"s"}, cached={"s"}),
        time_period=DataRequest(requested=requested, cached=cached),
    )


class TestHistoricalCacheRequest:
    """Behaviour of historical ``CacheRequest.missing_count``/``missing_dims``."""

    def test_no_missing(self) -> None:
        """Verify a fully-cached request reports zero missing."""
        req = _all_match_request()
        assert req.missing_count == 0
        assert req.missing_dims == []

    def test_missing_variables(self) -> None:
        """Verify a missing variable bumps ``missing_count`` and ``missing_dims``."""
        req = _all_match_request()
        req.variables = DataRequest(requested={"a", "b"}, cached={"a"})
        assert req.missing_count == 1
        assert req.missing_dims == ["variable"]

    def test_missing_stations(self) -> None:
        """Verify a missing station bumps ``missing_count`` and ``missing_dims``."""
        req = _all_match_request()
        req.stations = DataRequest(requested={"s1", "s2"}, cached={"s1"})
        assert req.missing_count == 1
        assert req.missing_dims == [StandardDim.station]

    def test_missing_time_period(self) -> None:
        """Verify a missing time period bumps ``missing_count`` and ``missing_dims``."""
        req = _all_match_request()
        # Requested period extends past cached envelope on the right.
        req.time_period = DataRequest(
            requested=TimePeriod(
                start=datetime(2020, 1, 5, tzinfo=UTC),
                end=datetime(2020, 2, 1, tzinfo=UTC),
            ),
            cached=TimePeriod(
                start=datetime(2020, 1, 1, tzinfo=UTC),
                end=datetime(2020, 1, 10, tzinfo=UTC),
            ),
        )
        assert req.missing_count == 1
        assert req.missing_dims == [StandardDim.time]

    def test_multi_missing_returns_no_split(
        self,
        xarray_general_info_config_historical: GeneralInfoConfig,
    ) -> None:
        """Verify multi-dim misses skip the single-dim split shortcut."""
        config = NetCDFConfig(
            general=xarray_general_info_config_historical,
            source_id=xarray_general_info_config_historical.verification_pairs[
                0
            ].observations_source_id,
            data_type=DataType.observed_historical,
            directory=".",
            filename_glob="*.nc",
            import_adapter=DataSourceKind.NETCDF,
            stations=["s1", "s2"],
            variables=["a", "b"],
        )
        datasource = NetCDF(config)
        req = _all_match_request()
        req.variables = DataRequest(requested={"a", "b"}, cached={"a"})
        req.stations = DataRequest(requested={"s1", "s2"}, cached={"s1"})
        assert req.missing_count == _EXPECTED_MISSING_COUNT_MULTI
        assert req.split_config(datasource) is None


class TestForecastCacheRequest:
    """Behaviour of forecast ``CacheRequest.missing_count``/``missing_dims``."""

    @staticmethod
    def _empty() -> CacheRequest:
        # Cache strictly envelopes requested period.
        requested_period = TimePeriod(
            start=datetime(2020, 1, 2, tzinfo=UTC),
            end=datetime(2020, 1, 9, tzinfo=UTC),
        )
        cached_period = TimePeriod(
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=datetime(2020, 1, 10, tzinfo=UTC),
        )
        lt = LeadTimes(unit=TimeUnits.day, values=[1, 2])
        return CacheRequest(
            variables=DataRequest(requested={"a"}, cached={"a"}),
            stations=DataRequest(requested={"s"}, cached={"s"}),
            forecast_reference_time_period=DataRequest(
                requested=requested_period,
                cached=cached_period,
            ),
            lead_times=DataRequest(requested=lt, cached=lt),
        )

    def test_no_missing(self) -> None:
        """Verify a fully-cached forecast request reports zero missing."""
        req = self._empty()
        assert req.missing_count == 0
        assert req.missing_dims == []

    def test_missing_lead_times(self) -> None:
        """Verify a missing lead time bumps ``missing_count`` and ``missing_dims``."""
        req = self._empty()
        req.lead_times = DataRequest(
            requested=LeadTimes(unit=TimeUnits.day, values=[1, 2, 3]),
            cached=LeadTimes(unit=TimeUnits.day, values=[1, 2]),
        )
        assert req.missing_count == 1
        assert req.missing_dims == [StandardDim.lead_time]

    def test_missing_frt(self) -> None:
        """Verify a missing FRT period bumps ``missing_count`` and ``missing_dims``."""
        req = self._empty()
        req.forecast_reference_time_period = DataRequest(
            requested=TimePeriod(
                start=datetime(2020, 1, 5, tzinfo=UTC),
                end=datetime(2020, 2, 1, tzinfo=UTC),
            ),
            cached=TimePeriod(
                start=datetime(2020, 1, 1, tzinfo=UTC),
                end=datetime(2020, 1, 10, tzinfo=UTC),
            ),
        )
        assert req.missing_count == 1
        assert req.missing_dims == [StandardDim.forecast_reference_time]


# ---------------------------------------------------------------------------
# combine_cached_and_fetched_data
# ---------------------------------------------------------------------------


def _ds_one_var(name: str = "v1") -> xr.Dataset:
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    stations = ["s1", "s2"]
    arr = np.arange(4, dtype="float32").reshape(2, 2)
    return xr.Dataset(
        {name: (("time", "station"), arr)},
        coords={"time": times, "station": stations},
        attrs={"data_type": DataType.observed_historical},
    )


def _forecast_ds(frt_dates: list[str], name: str = "v1") -> xr.Dataset:
    """Build a small forecast dataset indexed by forecast reference time, lead time and station."""
    frts = np.array(frt_dates, dtype="datetime64[ns]")
    lead_times = np.array([0, 3600], dtype="timedelta64[s]").astype("timedelta64[ns]")
    stations = ["s1", "s2"]
    shape = (len(frts), len(lead_times), len(stations))
    arr = np.arange(int(np.prod(shape)), dtype="float32").reshape(shape)
    dims = (StandardDim.forecast_reference_time, StandardDim.lead_time, StandardDim.station)
    return xr.Dataset(
        {name: (dims, arr)},
        coords={
            StandardDim.forecast_reference_time: frts,
            StandardDim.lead_time: lead_times,
            StandardDim.station: stations,
        },
        attrs={"data_type": DataType.simulated_forecast_single},
    )


class TestCombineCachedAndFetched:
    """Behaviour of ``combine_cached_and_fetched_data``."""

    def test_combine_along_variable(self) -> None:
        """Verify combining along the ``variable`` dim merges data variables."""
        a = _ds_one_var("v1")
        b = _ds_one_var("v2")
        result = combine_cached_and_fetched_data(a, b, "variable")
        assert set(result.data_vars) == {"v1", "v2"}

    def test_combine_along_station(self) -> None:
        """Verify combining along ``station`` extends the station coordinate."""
        a = _ds_one_var("v1")
        b_arr = np.arange(2, dtype="float32").reshape(2, 1)
        b = xr.Dataset(
            {"v1": (("time", "station"), b_arr)},
            coords={"time": a["time"].values, "station": ["s3"]},
        )
        result = combine_cached_and_fetched_data(a, b, StandardDim.station)
        assert list(result["station"].values) == ["s1", "s2", "s3"]

    def test_combine_along_time(self) -> None:
        """Verify combining along ``time`` extends the time coordinate."""
        a = _ds_one_var("v1")
        b_arr = np.arange(2, dtype="float32").reshape(1, 2)
        b = xr.Dataset(
            {"v1": (("time", "station"), b_arr)},
            coords={
                "time": np.array(["2020-01-03"], dtype="datetime64[ns]"),
                "station": ["s1", "s2"],
            },
        )
        result = combine_cached_and_fetched_data(a, b, StandardDim.time)
        assert result["time"].size == _EXPECTED_COMBINED_TIME_SIZE

    def test_unsupported_dim_raises(self) -> None:
        """Verify an unknown dim raises ``ValueError``."""
        a = _ds_one_var()
        with pytest.raises(ValueError, match="Unsupported dimension"):
            combine_cached_and_fetched_data(a, a, "bogus")


# ---------------------------------------------------------------------------
# ZarrCache round-trip
# ---------------------------------------------------------------------------


class TestZarrCache:
    """Round-trip behaviour of the local-disk ``ZarrCache``."""

    def test_get_dataset_missing_returns_none(self, cache_dir_local: str) -> None:
        """Verify ``get_dataset`` on an unknown source returns an empty dataset."""
        cfg = ZarrCacheConfig(
            path=str(Path(cache_dir_local) / "store.zarr"),
            read_write_mode=ReadWriteMode.read_write,
        )
        cache = ZarrCache(cfg)
        result = cache.get_dataset(source="not_there")
        assert result is None

    def test_append_then_get_round_trip(self, cache_dir_local: str) -> None:
        """Verify ``append`` followed by ``get_dataset`` returns the appended data."""
        cfg = ZarrCacheConfig(
            path=str(Path(cache_dir_local) / "store.zarr"),
            read_write_mode=ReadWriteMode.read_write,
        )
        cache = ZarrCache(cfg)
        ds = _ds_one_var("v1")
        cache.append(ds, source="src1")
        result = cache.get_dataset(source="src1")
        assert "v1" in result.data_vars
        assert result["v1"].shape == ds["v1"].shape

    def test_is_writable(self, cache_dir_local: str) -> None:
        """Verify ``is_writable`` reflects the configured read/write mode."""
        cfg_w = ZarrCacheConfig(
            path=str(Path(cache_dir_local) / "store_w.zarr"),
            read_write_mode=ReadWriteMode.read_write,
        )
        cfg_r = ZarrCacheConfig(
            path=str(Path(cache_dir_local) / "store_r.zarr"),
            read_write_mode=ReadWriteMode.read,
        )
        assert ZarrCache(cfg_w).is_writable
        assert not ZarrCache(cfg_r).is_writable

    # ------------------------------------------------------------------
    # Remote (https / s3) cache configuration
    # ------------------------------------------------------------------

    def test_remote_is_remote_true(self, cache_dir_remote: str) -> None:
        """Verify ``is_remote`` returns True for an HTTPS S3 URL."""
        cfg = ZarrCacheConfig(path=cache_dir_remote)
        assert ZarrCache(cfg).is_remote

    def test_remote_validator_does_not_create_directory(
        self,
        cache_dir_remote: str,
        tmp_path: Path,
    ) -> None:
        """Verify the path validator skips ``mkdir`` for remote URLs."""
        # Sanity: tmp_path exists, cache_dir_remote is not a local path that
        # would have been created by the validator.
        cfg = ZarrCacheConfig(path=cache_dir_remote)
        assert cfg.is_remote_path()
        assert not Path(cache_dir_remote).exists()
        # tmp_path is unrelated; just verifies the fixture did not interfere.
        assert tmp_path.exists()

    def test_remote_storage_options_merges_auth_and_extra(
        self,
        cache_dir_remote: str,
    ) -> None:
        """Verify auth_config and storage_options are merged for remote stores."""
        cfg = ZarrCacheConfig(
            path=cache_dir_remote,
            auth_config=S3AuthConfig(
                anon=True,
                region_name="eu-west-1",
                endpoint_url="https://s3.dummy.com",
            ),
            storage_options={"requester_pays": "true"},
        )
        options = ZarrCache(cfg).storage_options
        assert options is not None
        assert options["anon"] is True
        assert options["requester_pays"] == "true"
        assert options["client_kwargs"] == {
            "region_name": "eu-west-1",
            "endpoint_url": "https://s3.dummy.com/",
        }

    def test_remote_read_only_is_not_writable(self, cache_dir_remote: str) -> None:
        """Verify a remote cache defaults to read-only and reports not writable."""
        cfg = ZarrCacheConfig(path=cache_dir_remote)
        cache = ZarrCache(cfg)
        assert cache.is_remote
        assert not cache.is_writable

    def test_double_round_trip_sorts_forecast_reference_time(
        self,
        cache_dir_local: str,
    ) -> None:
        """Verify appending older forecasts after newer ones yields a sorted, sliceable frt axis.

        Simulates two runs: the first caches the most recent forecast reference times, the
        second (later) run fetches and appends older forecast reference times. Because Zarr
        appends place new coordinate values at the end of the store, the frt axis is left
        non-monotonic on disk; ``get_dataset`` must return it sorted so that ``slice`` based
        selection works.
        """
        cfg = ZarrCacheConfig(
            path=str(Path(cache_dir_local) / "store_frt.zarr"),
            read_write_mode=ReadWriteMode.read_write,
        )
        cache = ZarrCache(cfg)

        # First run: cache the most recent forecasts.
        recent = _forecast_ds(["2020-01-05", "2020-01-06"])
        cache.append(recent, source="fc", append_dim=StandardDim.forecast_reference_time)

        # Second (later) run: fetch and append older forecasts.
        older = _forecast_ds(["2020-01-01", "2020-01-02"])
        cache.append(older, source="fc", append_dim=StandardDim.forecast_reference_time)

        result = cache.get_dataset(source="fc")

        # All four forecast reference times are present and sorted strictly ascending.
        frt = result[StandardDim.forecast_reference_time].to_numpy()
        assert frt.size == _EXPECTED_FRT_SIZE
        assert (np.diff(frt) > np.timedelta64(0)).all()
        np.testing.assert_array_equal(
            frt,
            np.array(
                ["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-06"],
                dtype="datetime64[ns]",
            ),
        )

        # A slice selection over the (now monotonic) axis works and spans both runs.
        selected = result.sel(
            {
                StandardDim.forecast_reference_time: slice(
                    np.datetime64("2020-01-02", "ns"),
                    np.datetime64("2020-01-05", "ns"),
                ),
            },
        )
        np.testing.assert_array_equal(
            selected[StandardDim.forecast_reference_time].to_numpy(),
            np.array(["2020-01-02", "2020-01-05"], dtype="datetime64[ns]"),
        )
