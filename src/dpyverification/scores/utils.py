"""Utility functions shared across the scores module."""

from typing import Protocol

import xarray as xr

from dpyverification.configuration.base import BaseConfig, BaseScoreConfig
from dpyverification.configuration.default.scores import ContinuousScoresConfig
from dpyverification.constants import StandardCoord, StandardDim, TimeseriesKind
from dpyverification.datamodel import InputDataset


def set_data_array_attributes(
    da: xr.DataArray,
    long_name: str,
    units: str,
    standard_name: str | None = None,
    config: BaseConfig | None = None,
) -> xr.DataArray:
    """Set configuration attributes on xr.DataArray."""
    cf_attrs: dict[str, str] = {
        "long_name": long_name,
        "units": units,
    }

    if standard_name is not None:
        cf_attrs.update({"standard_name": standard_name})

    union: dict[str, str]

    if config is not None:
        config_attrs: dict[str, str] = config.model_dump()
        union = cf_attrs | config_attrs
    else:
        union = cf_attrs

    return da.assign_attrs(union)


def reassign_station_auxiliary_coords(
    result: xr.DataArray,
    sim: xr.DataArray,
) -> xr.DataArray:
    """Reassign auxiliary coordinates on dimension station.

    These typically include, station_id, station_name, lat, lon, x, y, z.
    """
    for coord in sim.coords:  # type:ignore[misc]
        # Reassign only coords with dim station
        if sim[coord].dims == (StandardDim.station,):
            result = result.assign_coords({coord: sim[coord]})  # type:ignore[misc]
    return result


class ScoreFunc(Protocol):
    """Callable score taking two DataArrays and returning a DataArray."""

    def __call__(  # noqa: D102
        self,
        first: xr.DataArray,
        second: xr.DataArray,
        **kwargs: object,
    ) -> xr.DataArray | xr.Dataset: ...


class WrappedScoreFunc(Protocol):
    """Callable that consumes dataset and config and returns a Dataset."""

    def __call__(  # noqa: D102
        self,
        data: InputDataset,
        config: BaseScoreConfig,
        **kwargs: object,
    ) -> xr.DataArray: ...


def loop_verification_pairs(func: ScoreFunc) -> WrappedScoreFunc:
    """Loop over verification pairs.

    A helper function that can be re-used for scores to avoid duplicate code.
    """

    def wrapper(data: InputDataset, config: BaseScoreConfig, **kwargs: object) -> xr.DataArray:
        results: list[xr.DataArray] = []
        for pair in config.verification_pairs:
            obs, sim = data.get_verification_pair(pair)

            # Timer series kind
            simulation_timeseries_kind = data.get_simulated_timeseries_kind_from_pair(pair)

            if simulation_timeseries_kind == TimeseriesKind.simulated_forecast_ensemble:
                # Broadcast obs like sim
                obs = obs.broadcast_like(sim.isel({StandardDim.realization: 0}))  # type:ignore[misc]

            # Function call
            result: xr.DataArray | xr.Dataset = func(sim, obs, **kwargs)

            # Determine the function name
            if hasattr(func, "__qualname__"):
                function_name: str = func.__qualname__
            else:
                msg = "Could not determine the function name."
                raise AttributeError(msg)

            # If a score returns a Dataset, get the first variable
            if isinstance(result, xr.Dataset):  # type:ignore[misc]
                if len(result.data_vars) != 1:
                    msg = f"Computation of {function_name} created more than one variable."
                    raise ValueError(msg)
                result = result[next(iter(result.data_vars))]

            # Set verification_pair dim
            result = result.expand_dims({"verification_pair": 1})

            # Assign auxiliary coords on dim, indicating the obs source and sim source
            result = result.assign_coords(
                {
                    "verification_pair": ("verification_pair", [pair.id]),  # type:ignore[misc]
                    "obs_source": ("verification_pair", [pair.obs]),
                    "sim_source": ("verification_pair", [pair.sim]),
                },
            )

            score_name = (
                str(config.kind)
                if not isinstance(config, ContinuousScoresConfig)
                else function_name
            )

            # Set variable name on xr.DataArray
            result.name = score_name

            # Set attributes on data array
            result = set_data_array_attributes(
                result,
                long_name=score_name,
                units=sim[StandardCoord.units.name].to_numpy()[0],  # type:ignore[misc]
            )

            results.append(result)

        return xr.merge(results)[score_name]

    return wrapper
