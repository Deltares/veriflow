"""Standalone Plotly figures for a single verification pair.

This module provides the low-level builders (``score_plot``, ``scatter_pair_plot``,
``rank_histogram_pair_plot``, ``rank_histogram_3d_pair_plot``, ``reanalysis_pair_plot``,
``forecast_pair_plot``, ``forecast_scroller``): each takes one pre-sliced ``xr.Dataset`` for a
single verification pair and returns a standalone, ready-to-show ``go.Figure``. Any selection
- station, lead time - is applied by slicing the dataset with ``ds.sel(...)`` before calling a
builder. There is no Dash app here, but ``forecast_scroller`` does embed a native Plotly
slider/play button so a whole sequence of forecasts can be scrolled through within the one
figure, without a separate server.

See :mod:`tree_plots` for the companion high-level builders that operate on a whole
``xr.DataTree`` (composing one of these standalone figures per verification pair into a
single grid/overlay figure).

Typical usage::

    from veriflow import run_pipeline
    from tree_plots import get_pair_dataset

    dt = run_pipeline(...)
    pair_id = dt.veriflow.verification_pairs[0]
    ds = get_pair_dataset(dt, pair_id)

    sliced = ds.sel(station="some-station", lead_time=ds.coords["lead_time"][0])
    variable = ds.attrs["variable"]
    fig = scatter_pair_plot(sliced, obs_var=f"{variable}_obs", sim_var=f"{variable}_sim")
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr
from plotly.subplots import make_subplots
from pydantic import BaseModel

# Serialize Plotly figures as self-contained HTML (with plotly.js loaded from the
# CDN) instead of only the ``application/vnd.plotly.v1+json`` MIME bundle. The MIME
# bundle renders in JupyterLab/VS Code but shows up as
# "Data type cannot be displayed: application/vnd.plotly.v1+json" in the static HTML
# produced by nbsphinx/nbconvert for the documentation site.
pio.renderers.default = "notebook_connected"


class Theme(BaseModel):
    """Theme settings for figure-wide styling tokens."""

    font_family: str = "Segoe UI, Arial, sans-serif"
    font_color: str = "#022B6D"
    card_bg: str = "#FFFFFF"
    accent: str = "#7A491C"
    accent_muted: str = "#61BD4D"
    reference: str = "#818181"


THEME = Theme()
MISSING_VALUE_MARKER = -999

PLOT_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font={"family": THEME.font_family, "color": THEME.font_color},
        paper_bgcolor=THEME.card_bg,
        plot_bgcolor=THEME.card_bg,
    ),
)


def lead_time_hours(ds: xr.Dataset) -> np.ndarray:
    """Return the ``lead_time`` coordinate of ``ds`` expressed in hours."""
    values = ds.coords["lead_time"].values
    return np.array([float(v / np.timedelta64(1, "h")) for v in np.atleast_1d(values)])


def _format_lead_time_label(value: object) -> str:
    """Return a human-readable lead-time label in hours for a scalar ``lead_time`` value.

    Returns an empty string if ``value`` is not a scalar timedelta (e.g. an array left after
    slicing). Whole-hour values are rendered without a decimal part.
    """
    array = np.atleast_1d(np.asarray(value))
    if array.size != 1:
        return ""
    hours = float(array[0] / np.timedelta64(1, "h"))
    hours_text = f"{int(hours)}" if hours.is_integer() else f"{hours:g}"
    return f"lead time {hours_text} h"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a ``#RRGGBB`` string to a Plotly ``rgba(...)`` string."""
    cleaned = hex_color.lstrip("#")
    red, green, blue = (int(cleaned[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _squeeze_to_single_station(ds: xr.Dataset) -> xr.Dataset:
    """Return ``ds`` with the ``station`` dimension dropped.

    If ``station`` is not present the dataset is returned unchanged. If it is present
    with more than one entry a ``ValueError`` is raised, since the time-series plot can
    only render one station at a time.
    """
    if "station" not in ds.dims:
        return ds
    if ds.sizes["station"] != 1:
        msg = (
            "forecast_timeseries_plot expects a single station; pass station=... or "
            "pre-select one with ds.sel(station=...) before calling."
        )
        raise ValueError(msg)
    return ds.isel(station=0)


def _observed_series(obs: xr.DataArray, time_coord: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the observed array against the ``time`` coordinate into a unique series.

    The observed variable is stored on the ``(forecast_reference_time, lead_time)`` grid,
    so the same valid time appears in several forecasts. The values are flattened, sorted
    by valid time and de-duplicated to build one continuous historical series.
    """
    times = np.ravel(np.asarray(time_coord.values))
    values = np.ravel(np.asarray(obs.values, dtype=float))
    values[values == MISSING_VALUE_MARKER] = np.nan

    order = np.argsort(times)
    times = times[order]
    values = values[order]

    unique_times, first_idx = np.unique(times, return_index=True)
    return unique_times, values[first_idx]


def _as_clean_float_array(data: xr.DataArray) -> np.ndarray:
    """Return ``data`` values as floats with the configured missing marker replaced by NaN."""
    values = np.asarray(data.values, dtype=float)
    values[values == MISSING_VALUE_MARKER] = np.nan
    return values


def _iter_station_slices(
    ds: xr.Dataset,
) -> list[tuple[object | None, xr.Dataset]]:
    """Return datasets split by station, or the original dataset when stationless."""
    if "station" not in ds.dims:
        return [(None, ds)]

    return [
        (station_value, ds.sel(station=station_value))
        for station_value in np.atleast_1d(ds.coords["station"].values)
    ]


def find_crps_variable(ds: xr.Dataset, *, exclude_vars: tuple[str, ...] = ()) -> str:
    """Return the name of a CRPS-like score variable in ``ds``.

    A score variable is any data variable that has a ``lead_time`` dimension and is not
    one of ``exclude_vars`` (typically the obs/sim input variables). CRPS-like names are
    preferred; if none match, the first score candidate is returned.
    """
    excluded = set(exclude_vars)
    score_candidates = [
        str(name)
        for name, data_array in ds.data_vars.items()
        if str(name) not in excluded and "lead_time" in data_array.dims
    ]
    if not score_candidates:
        msg = "No score variable with a 'lead_time' dimension found in the dataset."
        raise ValueError(msg)
    crps_candidates = [name for name in score_candidates if "crps" in name.lower()]
    return (crps_candidates or score_candidates)[0]


def _score_variables_to_plot(
    ds: xr.Dataset,
    *,
    score_var: str | None,
    score_vars: str | list[str] | tuple[str, ...] | None,
    exclude_vars: tuple[str, ...],
) -> list[str]:
    """Return score variables requested for a lead-time score plot."""
    if score_var is not None and score_vars is not None:
        msg = "Pass either score_var or score_vars, not both."
        raise ValueError(msg)

    if score_vars is None:
        variables = [score_var or find_crps_variable(ds, exclude_vars=exclude_vars)]
    elif isinstance(score_vars, str):
        variables = [score_vars]
    else:
        variables = list(score_vars)

    if not variables:
        msg = "At least one score variable must be provided."
        raise ValueError(msg)

    missing = [variable for variable in variables if variable not in ds.data_vars]
    if missing:
        msg = (
            f"Score variable(s) {missing} not found in the dataset. "
            f"Available variables: {sorted(ds.data_vars)}."
        )
        raise ValueError(msg)

    return variables


def scatter_pair_plot(
    ds: xr.Dataset,
    *,
    obs_var: str,
    sim_var: str,
    legend_state: dict[str, bool] | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build a standalone observation-vs-simulation scatter figure for one verification pair.

    ``ds`` must already be sliced to a single station (or have no ``station`` dimension) and,
    for forecast data, to a single lead time. The simulated variable may carry a
    ``realization`` dimension (an ensemble) or not (a deterministic/historical simulation);
    the ensemble case additionally shows the ensemble mean. Pass a shared ``legend_state`` dict
    (as :func:`tree_plots.scatter_plot` does) so repeated calls contribute only one legend
    entry per trace type; when not given, a fresh state is used so the figure is fully
    self-contained.

    Points are drawn with partial opacity so that overlapping markers accumulate into a
    visibly darker/denser cloud, giving a sense of point density without a separate
    heatmap/contour layer.
    """
    if legend_state is None:
        legend_state = {
            "members": True,
            "mean": True,
            "simulation": True,
            "reference": True,
        }

    ds = _squeeze_to_single_station(ds)
    obs = ds[obs_var]
    sim = ds[sim_var]
    is_ensemble = "realization" in sim.dims

    obs_values = obs.values

    if is_ensemble:
        sim_mean = sim.mean(dim="realization").values
        # Ensure realization is the last axis so the flattened simulation values line up
        # with the observations repeated per realization.
        sim_ordered = sim.transpose(*[d for d in sim.dims if d != "realization"], "realization")
        n_realization = sim_ordered.sizes["realization"]
        obs_flat = np.repeat(np.ravel(obs_values), n_realization)
        sim_flat = sim_ordered.values.flatten()
    else:
        sim_mean = sim.values
        obs_flat = np.ravel(obs_values)
        sim_flat = np.ravel(sim.values)

    valid_mask = (sim_flat != MISSING_VALUE_MARKER) & np.isfinite(obs_flat) & np.isfinite(sim_flat)
    obs_flat = obs_flat[valid_mask]
    sim_flat = sim_flat[valid_mask]

    if obs_flat.size and sim_flat.size:
        axis_values = np.concatenate([obs_flat, sim_flat])
    else:
        fallback_values = np.concatenate([np.ravel(obs_values), np.ravel(sim_mean)])
        axis_values = fallback_values[np.isfinite(fallback_values)]

    if axis_values.size:
        axis_min = float(np.min(axis_values))
        axis_max = float(np.max(axis_values))
    else:
        axis_min, axis_max = 0.0, 1.0

    span = axis_max - axis_min
    padding = max(abs(axis_min), 1.0) * 0.05 if span == 0 else span * 0.05

    fig = go.Figure()

    if is_ensemble:
        fig.add_trace(
            go.Scatter(
                x=obs_flat,
                y=sim_flat,
                mode="markers",
                name="ensemble members",
                legendgroup="members",
                # Low opacity so overlapping members accumulate into a visibly darker cloud.
                marker={
                    "size": 5,
                    "color": THEME.accent_muted,
                    "symbol": "circle",
                    "opacity": 0.6,
                },
                showlegend=legend_state["members"],
            ),
        )
        legend_state["members"] = False
        fig.add_trace(
            go.Scatter(
                x=np.ravel(obs_values),
                y=np.ravel(sim_mean),
                mode="markers",
                name="ensemble mean",
                legendgroup="mean",
                marker={"size": 4, "color": THEME.accent, "symbol": "circle", "opacity": 0.6},
                showlegend=legend_state["mean"],
            ),
        )
        legend_state["mean"] = False
    else:
        fig.add_trace(
            go.Scatter(
                x=np.ravel(obs_values),
                y=np.ravel(sim_mean),
                mode="markers",
                name="simulation",
                legendgroup="simulation",
                marker={"size": 6, "color": THEME.accent, "symbol": "circle", "opacity": 0.5},
                showlegend=legend_state["simulation"],
            ),
        )
        legend_state["simulation"] = False

    fig.add_trace(
        go.Scatter(
            x=[axis_min - padding, axis_max + padding],
            y=[axis_min - padding, axis_max + padding],
            mode="lines",
            name="1:1 line",
            legendgroup="reference",
            line={"color": THEME.reference, "dash": "dash", "width": 1},
            showlegend=legend_state["reference"],
        ),
    )
    legend_state["reference"] = False

    axis_range = [axis_min - padding, axis_max + padding]
    fig.update_xaxes(title_text=obs_var, range=axis_range)
    fig.update_yaxes(title_text=sim_var, range=axis_range, scaleanchor="x", scaleratio=1)
    fig.update_layout(template=template, hovermode="closest")
    return fig


def score_plot(
    ds: xr.Dataset,
    *,
    score_var: str | None = None,
    score_vars: str | list[str] | tuple[str, ...] | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build a standalone score-vs-lead-time figure for one verification pair.

    ``ds`` is a single pair's score dataset (e.g. from ``tree_plots.get_score_dataset``) and
    must retain its ``lead_time`` dimension. The score variable is located automatically
    (CRPS-like names preferred); pass ``score_var`` to use a specific variable name instead, or
    ``score_vars`` to plot multiple score variables, for example ``score_vars=["crps", "mae"]``.

    If a ``station`` dimension is present, one line is drawn per station; otherwise a single
    line per score variable is drawn. Any remaining dimensions other than ``lead_time`` and
    ``station`` are averaged out.
    """
    variables = _score_variables_to_plot(
        ds,
        score_var=score_var,
        score_vars=score_vars,
        exclude_vars=(),
    )
    single_variable = len(variables) == 1

    fig = go.Figure()
    hours = lead_time_hours(ds)

    for variable in variables:
        score_data = ds[variable]
        if "lead_time" not in score_data.dims:
            msg = f"Score variable '{variable}' must have a 'lead_time' dimension."
            raise ValueError(msg)

        dims_to_reduce = [
            dim_name for dim_name in score_data.dims if dim_name not in {"lead_time", "station"}
        ]
        if dims_to_reduce:
            score_data = score_data.mean(dim=dims_to_reduce, skipna=True)

        if "station" in score_data.dims:
            for station in score_data.coords["station"].values:
                station_values = np.ravel(score_data.sel(station=station).values)
                valid = np.isfinite(station_values)
                name = str(station) if single_variable else f"{variable} - {station}"
                fig.add_trace(
                    go.Scatter(
                        x=hours[valid],
                        y=station_values[valid],
                        mode="lines+markers",
                        name=name,
                        legendgroup=variable,
                    ),
                )
        else:
            score_values = np.ravel(score_data.values)
            valid = np.isfinite(score_values)
            fig.add_trace(
                go.Scatter(
                    x=hours[valid],
                    y=score_values[valid],
                    mode="lines+markers",
                    name=variable,
                    legendgroup=variable,
                ),
            )

    yaxis_title = variables[0].upper() if single_variable else "Score"
    fig.update_layout(
        xaxis={
            "title": "Lead Time (h)",
            "tickmode": "array",
            "tickvals": hours,
            "ticktext": [f"{int(h)} h" for h in hours],
        },
        yaxis_title=yaxis_title,
        template=template,
        hovermode="x unified",
    )
    return fig


def rank_histogram_pair_plot(
    ds: xr.Dataset,
    *,
    rank_var: str = "histogram_rank",
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build a standalone rank-histogram bar chart for one verification pair.

    ``ds`` must already be sliced so that ``rank_var`` is one-dimensional over the ``rank``
    coordinate (pre-select ``station``/``lead_time`` with ``ds.sel(...)`` before calling).
    """
    hist = ds[rank_var]
    fig = go.Figure(
        go.Bar(
            x=hist.coords["rank"].values,
            y=np.ravel(hist.values),
            marker_color=THEME.accent,
            showlegend=False,
        ),
    )
    fig.update_xaxes(title_text="Rank")
    fig.update_yaxes(title_text="Count")
    fig.update_layout(template=template)
    return fig


def rank_histogram_3d_pair_plot(
    ds: xr.Dataset,
    *,
    rank_var: str = "histogram_rank",
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build a standalone 3D rank-histogram surface for one verification pair.

    This is a three-dimensional version of :func:`rank_histogram_pair_plot`. ``ds`` must
    already be sliced so that ``rank_var`` reduces to two dimensions (``lead_time`` by
    ``rank``); pre-select ``station`` with ``ds.sel(station=...)`` before calling if needed.
    """
    hist = ds[rank_var]
    if "lead_time" not in hist.dims or "rank" not in hist.dims:
        msg = (
            "rank_histogram_3d_pair_plot expects the rank variable to have both 'lead_time' "
            f"and 'rank' dimensions; got dimensions {tuple(hist.dims)}."
        )
        raise ValueError(msg)
    extra_dims = set(hist.dims) - {"lead_time", "rank"}
    if extra_dims:
        msg = (
            "rank_histogram_3d_pair_plot expects the rank variable to reduce to two dimensions "
            f"(lead_time x rank); got extra dimensions {sorted(extra_dims)}. Pass 'station' "
            "to select a single station."
        )
        raise ValueError(msg)
    # Orient as (lead_time, rank) so the surface z-grid matches y=lead_time, x=rank.
    hist = hist.transpose("lead_time", "rank")

    fig = go.Figure(
        go.Surface(
            x=hist.coords["rank"].values,
            y=lead_time_hours(ds),
            z=np.asarray(hist.values, dtype=float),
            colorscale="Earth",
            colorbar={"title": "Count"},
            hovertemplate=("Rank: %{x}<br>Lead time: %{y} h<br>Count: %{z}<extra></extra>"),
        ),
    )
    fig.update_scenes(
        xaxis_title_text="Rank",
        yaxis_title_text="Lead time (h)",
        zaxis_title_text="Count",
    )
    fig.update_layout(template=template)
    return fig


def reanalysis_pair_plot(
    ds: xr.Dataset,
    *,
    obs_var: str,
    sim_var: str,
    error_var: str | None = None,
    legend_state: dict[str, bool] | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build a standalone observed/simulated historical time-series figure for one pair.

    This is intended for reanalysis or other historical verification runs where ``ds`` uses a
    direct ``time`` dimension instead of the forecast ``(forecast_reference_time, lead_time)``
    grid. By default, all stations are plotted with translucent traces; pre-select a station
    with ``ds.sel(station=...)`` before calling to plot a single one. Pass ``error_var`` to draw
    one score/error variable, such as ``"mean_error"`` or ``"mae"``, on a secondary y-axis.
    Pass a shared ``legend_state`` dict (as :func:`tree_plots.reanalysis_timeseries_plot` does)
    so repeated calls contribute only one legend entry per trace type; when not given, a fresh
    state is used so the figure is fully self-contained.
    """
    if "time" not in ds.coords:
        msg = "reanalysis_pair_plot expects a dataset with a 'time' coordinate."
        raise ValueError(msg)
    if error_var is not None and error_var not in ds.data_vars:
        msg = (
            f"Error variable '{error_var}' not found in the dataset. "
            f"Available variables: {sorted(ds.data_vars)}."
        )
        raise ValueError(msg)

    if legend_state is None:
        legend_state = {}
    legend_state.setdefault("observed", True)
    legend_state.setdefault(sim_var, True)
    if error_var is not None:
        legend_state.setdefault(error_var, True)

    obs = ds[obs_var]
    time_values = np.ravel(ds.coords["time"].values)
    station_slices = _iter_station_slices(ds)
    multi_station = len(station_slices) > 1

    fig = make_subplots(specs=[[{"secondary_y": error_var is not None}]])

    for station_value, station_ds in station_slices:
        station_label = "" if station_value is None else str(station_value)
        hover_station = "station: %{customdata}<br>" if station_value is not None else ""
        customdata = np.full(time_values.shape, station_label, dtype=object)
        obs_values = np.ravel(_as_clean_float_array(station_ds[obs_var]))
        sim_values = np.ravel(_as_clean_float_array(station_ds[sim_var]))

        obs_valid = np.isfinite(obs_values)
        fig.add_trace(
            go.Scatter(
                x=time_values[obs_valid],
                y=obs_values[obs_valid],
                customdata=customdata[obs_valid],
                mode="lines+markers",
                name="observed",
                legendgroup="observed",
                line={"color": THEME.font_color, "width": 1.5},
                marker={"size": 5},
                opacity=0.45 if multi_station else 1.0,
                showlegend=legend_state["observed"],
                hovertemplate=(
                    f"{hover_station}time: %{{x}}<br>{obs_var}: %{{y}}<extra>observed</extra>"
                ),
            ),
        )
        legend_state["observed"] = False

        sim_valid = np.isfinite(sim_values)
        fig.add_trace(
            go.Scatter(
                x=time_values[sim_valid],
                y=sim_values[sim_valid],
                customdata=customdata[sim_valid],
                mode="lines+markers",
                name=sim_var,
                legendgroup=sim_var,
                line={"color": THEME.accent, "width": 1.5},
                marker={"size": 5},
                opacity=0.65 if multi_station else 1.0,
                showlegend=legend_state[sim_var],
                hovertemplate=(
                    f"{hover_station}time: %{{x}}<br>{sim_var}: %{{y}}<extra>{sim_var}</extra>"
                ),
            ),
        )
        legend_state[sim_var] = False

        if error_var is None:
            continue

        error_values = np.ravel(_as_clean_float_array(station_ds[error_var]))
        error_valid = np.isfinite(error_values)
        fig.add_trace(
            go.Scatter(
                x=time_values[error_valid],
                y=error_values[error_valid],
                customdata=customdata[error_valid],
                mode="lines+markers",
                name=error_var,
                legendgroup=error_var,
                line={"color": THEME.reference, "width": 1.5, "dash": "dot"},
                marker={"size": 5, "symbol": "diamond"},
                opacity=0.7 if multi_station else 1.0,
                showlegend=legend_state[error_var],
                hovertemplate=(
                    f"{hover_station}time: %{{x}}<br>{error_var}: %{{y}}<extra>{error_var}</extra>"
                ),
            ),
            secondary_y=True,
        )
        legend_state[error_var] = False

    units = str(obs.attrs.get("units", ""))
    y_title = f"{obs_var} / {sim_var}" + (f" ({units})" if units else "")
    fig.update_yaxes(title_text=y_title, secondary_y=False)
    if error_var is not None:
        error_units = str(ds[error_var].attrs.get("units", ""))
        error_title = error_var + (f" ({error_units})" if error_units else "")
        fig.update_yaxes(title_text=error_title, secondary_y=True)
    fig.update_xaxes(title_text="Time")
    fig.update_layout(template=template, hovermode="closest")
    return fig


def forecast_pair_plot(
    ds: xr.Dataset,
    *,
    obs_var: str,
    sim_var: str,
    show_members: bool = False,
    show_spread: bool = True,
    spread_quantiles: tuple[float, float] = (0.1, 0.9),
    legend_state: dict[str, bool] | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build a standalone historical-observed/forecast-trajectories figure for one pair.

    Both variables are stored on the ``(forecast_reference_time, lead_time)`` grid and share
    an auxiliary ``time(forecast_reference_time, lead_time)`` coordinate giving each value's
    valid time. The observed values are flattened against ``time`` into one continuous line,
    while each ``forecast_reference_time`` produces one forecast trajectory drawn along its
    valid times. ``ds`` must already be sliced to a single station (or have no ``station``
    dimension).

    The simulated variable may carry a ``realization`` dimension (an ensemble) or not
    (deterministic):

    - Ensemble: the ensemble mean is drawn per forecast; with ``show_spread`` a shaded band
      between ``spread_quantiles`` is added, and with ``show_members`` every member line is
      drawn faintly.
    - Deterministic: a single line is drawn per forecast.

    Pass a shared ``legend_state`` dict (as :func:`tree_plots.forecast_timeseries_plot` does)
    so repeated calls contribute only one legend entry per trace type; when not given, a fresh
    state is used so the figure is fully self-contained.
    """
    if legend_state is None:
        legend_state = {}
    for key in ("observed", "member", "spread", "mean", "forecast"):
        legend_state.setdefault(key, True)

    ds = _squeeze_to_single_station(ds)
    obs = ds[obs_var]
    sim = ds[sim_var]
    time_coord = ds.coords["time"]

    fig = go.Figure()

    # Continuous historical observed series.
    obs_times, obs_values = _observed_series(obs, time_coord)
    obs_valid = np.isfinite(obs_values)
    fig.add_trace(
        go.Scatter(
            x=obs_times[obs_valid],
            y=obs_values[obs_valid],
            mode="lines",
            name="observed",
            legendgroup="observed",
            line={"color": THEME.font_color, "width": 2},
            showlegend=legend_state["observed"],
        ),
    )
    legend_state["observed"] = False

    is_ensemble = "realization" in sim.dims
    reference_times = np.atleast_1d(ds.coords["forecast_reference_time"].values)

    spread_label = (
        f"ensemble spread ({int(spread_quantiles[0] * 100)}-{int(spread_quantiles[1] * 100)}%)"
    )

    for reference_time in reference_times:
        sim_f = sim.sel(forecast_reference_time=reference_time)
        valid_time = np.ravel(time_coord.sel(forecast_reference_time=reference_time).values)

        if is_ensemble:
            sim_ordered = sim_f.transpose(
                *[dim for dim in sim_f.dims if dim != "realization"],
                "realization",
            )
            members = np.asarray(sim_ordered.values, dtype=float).reshape(len(valid_time), -1)
            members[members == MISSING_VALUE_MARKER] = np.nan

            if show_members:
                for column in range(members.shape[1]):
                    member = members[:, column]
                    finite = np.isfinite(member)
                    fig.add_trace(
                        go.Scatter(
                            x=valid_time[finite],
                            y=member[finite],
                            mode="lines",
                            line={"color": THEME.accent_muted, "width": 1},
                            opacity=0.25,
                            legendgroup="members",
                            name="ensemble members",
                            showlegend=legend_state["member"],
                            hoverinfo="skip",
                        ),
                    )
                    legend_state["member"] = False

            if show_spread:
                lower = np.nanquantile(members, spread_quantiles[0], axis=1)
                upper = np.nanquantile(members, spread_quantiles[1], axis=1)
                finite = np.isfinite(lower) & np.isfinite(upper)
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([valid_time[finite], valid_time[finite][::-1]]),
                        y=np.concatenate([upper[finite], lower[finite][::-1]]),
                        fill="toself",
                        fillcolor=_hex_to_rgba(THEME.accent, 0.15),
                        line={"color": "rgba(0, 0, 0, 0)"},
                        legendgroup="spread",
                        name=spread_label,
                        showlegend=legend_state["spread"],
                        hoverinfo="skip",
                    ),
                )
                legend_state["spread"] = False

            mean = np.nanmean(members, axis=1)
            finite = np.isfinite(mean)
            fig.add_trace(
                go.Scatter(
                    x=valid_time[finite],
                    y=mean[finite],
                    mode="lines",
                    line={"color": THEME.accent, "width": 1.5},
                    legendgroup="mean",
                    name="ensemble mean forecast",
                    showlegend=legend_state["mean"],
                ),
            )
            legend_state["mean"] = False
        else:
            values = np.asarray(np.ravel(sim_f.values), dtype=float)
            values[values == MISSING_VALUE_MARKER] = np.nan
            finite = np.isfinite(values)
            fig.add_trace(
                go.Scatter(
                    x=valid_time[finite],
                    y=values[finite],
                    mode="lines",
                    line={"color": THEME.accent, "width": 1},
                    legendgroup="forecast",
                    name="forecast",
                    showlegend=legend_state["forecast"],
                ),
            )
            legend_state["forecast"] = False

    units = str(obs.attrs.get("units", ""))
    y_title = f"{obs_var} / {sim_var}" + (f" ({units})" if units else "")
    fig.update_yaxes(title_text=y_title)
    fig.update_layout(template=template, hovermode="x unified")
    return fig


def _forecast_scroller_frame(
    *,
    lead_hours: np.ndarray,
    sim_values: np.ndarray,
    obs_rel_hours: np.ndarray,
    obs_values: np.ndarray,
    is_ensemble: bool,
    show_spread: bool,
    show_members: bool,
    spread_quantiles: tuple[float, float],
    spread_label: str,
) -> list[go.Scatter]:
    """Build the traces for one forecast_reference_time of a scroller.

    Both the forecast (``lead_hours``) and observed (``obs_rel_hours``) x-values are hours
    relative to that forecast's own reference time, rather than absolute dates: since
    ``lead_hours`` is the same array for every forecast, the forecast trace sits at the exact
    same x-position in every frame, so it never visibly shifts as you scroll or play through
    the animation.
    """
    obs_valid = np.isfinite(obs_values)
    traces: list[go.Scatter] = [
        go.Scatter(
            x=obs_rel_hours[obs_valid],
            y=obs_values[obs_valid],
            mode="lines+markers",
            name="observed",
            legendgroup="observed",
            line={"color": THEME.font_color, "width": 2},
            marker={"size": 4},
        ),
    ]

    if is_ensemble:
        members = sim_values
        if show_spread:
            lower = np.nanquantile(members, spread_quantiles[0], axis=1)
            upper = np.nanquantile(members, spread_quantiles[1], axis=1)
            finite = np.isfinite(lower) & np.isfinite(upper)
            traces.append(
                go.Scatter(
                    x=np.concatenate([lead_hours[finite], lead_hours[finite][::-1]]),
                    y=np.concatenate([upper[finite], lower[finite][::-1]]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(THEME.accent, 0.15),
                    line={"color": "rgba(0, 0, 0, 0)"},
                    legendgroup="spread",
                    name=spread_label,
                    hoverinfo="skip",
                ),
            )

        mean = np.nanmean(members, axis=1)
        finite = np.isfinite(mean)
        traces.append(
            go.Scatter(
                x=lead_hours[finite],
                y=mean[finite],
                mode="lines+markers",
                line={"color": THEME.accent, "width": 2},
                marker={"size": 5},
                legendgroup="mean",
                name="ensemble mean forecast",
            ),
        )

        if show_members:
            for column in range(members.shape[1]):
                member = members[:, column]
                finite = np.isfinite(member)
                traces.append(
                    go.Scatter(
                        x=lead_hours[finite],
                        y=member[finite],
                        mode="lines",
                        line={"color": THEME.accent_muted, "width": 1},
                        opacity=0.3,
                        legendgroup="members",
                        name="ensemble members",
                        showlegend=column == 0,
                        hoverinfo="skip",
                    ),
                )
    else:
        finite = np.isfinite(sim_values)
        traces.append(
            go.Scatter(
                x=lead_hours[finite],
                y=sim_values[finite],
                mode="lines+markers",
                line={"color": THEME.accent, "width": 2},
                marker={"size": 5},
                legendgroup="forecast",
                name="forecast",
            ),
        )

    return traces


def _forecast_scroller_slider(reference_times: np.ndarray) -> list[dict]:
    """Return the Plotly slider definition used to scrub through forecast_scroller frames."""
    steps = [
        {
            "args": [
                [str(index)],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": np.datetime_as_string(reference_time, unit="h"),
            "method": "animate",
        }
        for index, reference_time in enumerate(reference_times)
    ]
    return [
        {
            "active": 0,
            "currentvalue": {"prefix": "Forecast issued: "},
            "pad": {"t": 40},
            "steps": steps,
        },
    ]


def _forecast_scroller_playback_buttons(play_speed: float) -> list[dict]:
    """Return the Play/Pause button definitions for the forecast_scroller animation.

    ``play_speed`` scales a 700 ms base frame duration (2.0 plays twice as fast, 0.5 half as
    fast). The transition duration is kept at 0: each frame shows a different forecast/window
    of observations rather than a continuous deformation of the same data, so instantly
    swapping frames (instead of interpolating between them) looks cleaner and is faster.
    """
    frame_duration = round(700 / play_speed)
    return [
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.0,
            "y": 1.15,
            "buttons": [
                {
                    "label": "\u25b6 Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "\u23f8 Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        },
    ]


def forecast_scroller(
    ds: xr.Dataset,
    *,
    obs_var: str,
    sim_var: str,
    obs_before: float | None = None,
    obs_after: float | None = None,
    show_members: bool = False,
    show_spread: bool = True,
    spread_quantiles: tuple[float, float] = (0.1, 0.9),
    play_speed: float = 1.0,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build an interactive figure to scroll through individual forecasts over time.

    ``ds`` must already be sliced to a single station (or have no ``station`` dimension). Both
    variables are stored on the ``(forecast_reference_time, lead_time)`` grid and share an
    auxiliary ``time(forecast_reference_time, lead_time)`` coordinate giving each value's valid
    time. Unlike :func:`forecast_pair_plot`, which overlays every forecast at once, this draws
    a single forecast trajectory together with a window of observed data around it, and lets
    the user scroll between forecasts with a slider (or a play button, to animate through
    them) - so one forecast can be inspected against its surrounding context without the
    clutter of every other forecast in the record.

    ``obs_var`` is stored on the same ``(forecast_reference_time, lead_time)`` grid as the
    forecast, so its ``lead_time=0`` slice (assumed to be the first entry of the ``lead_time``
    coordinate) already gives one observed value per ``forecast_reference_time``, valid at that
    same reference time; this is used directly to build the scrolling observed window, instead
    of re-deriving it from every forecast's valid-time grid (as :func:`forecast_pair_plot` does
    for its continuous historical line).

    ``obs_before``/``obs_after`` set how many hours of observed data are shown before/after
    each forecast's own valid-time span; when not given, both default to the forecast's own
    horizon length, so the observed context is as wide as the forecast itself on each side. Both
    the forecast and the observed window are plotted in hours relative to each forecast's own
    reference time (rather than absolute dates), so the forecast always sits at the exact same
    x-position and never appears to move as you scroll or play through forecasts; the actual
    date is shown in the slider label and on hover. ``play_speed`` scales the animation speed
    of the play button (2.0 plays twice as fast, 0.5 half as fast).

    The simulated variable may carry a ``realization`` dimension (an ensemble) or not
    (deterministic), as in :func:`forecast_pair_plot`: the ensemble mean is drawn, with an
    optional shaded ``spread_quantiles`` band and, with ``show_members``, faint individual
    member lines.
    """
    ds = _squeeze_to_single_station(ds)
    ds = ds.sortby("forecast_reference_time")
    obs = ds[obs_var]
    sim = ds[sim_var]
    time_coord = ds.coords["time"]
    is_ensemble = "realization" in sim.dims

    reference_times = np.atleast_1d(ds.coords["forecast_reference_time"].values)
    n_forecasts = reference_times.size
    if n_forecasts == 0:
        msg = "forecast_scroller requires at least one forecast_reference_time in the dataset."
        raise ValueError(msg)

    # obs shares the forecast's (forecast_reference_time, lead_time) grid, so its lead_time=0
    # slice already gives one observation per forecast_reference_time (valid at that time).
    obs_at_reference = _as_clean_float_array(obs.isel(lead_time=0))
    obs_time_at_reference = np.ravel(np.asarray(time_coord.isel(lead_time=0).values))

    lead_hours = lead_time_hours(ds)
    horizon_hours = float(lead_hours.max() - lead_hours.min()) if lead_hours.size else 0.0
    before_hours = obs_before if obs_before is not None else horizon_hours
    after_hours = obs_after if obs_after is not None else horizon_hours
    window_start_offset = np.timedelta64(round(lead_hours.min() * 3600), "s") - np.timedelta64(
        round(before_hours * 3600),
        "s",
    )
    window_end_offset = np.timedelta64(round(lead_hours.max() * 3600), "s") + np.timedelta64(
        round(after_hours * 3600),
        "s",
    )

    # Pre-extract plain numpy arrays and window bounds for every forecast at once: this makes
    # building each frame pure numpy indexing, instead of repeating a comparatively expensive
    # xarray label-based ds.sel() lookup per forecast_reference_time.
    if is_ensemble:
        sim_values = sim.transpose("forecast_reference_time", "lead_time", "realization").values
    else:
        sim_values = sim.transpose("forecast_reference_time", "lead_time").values
    sim_values = np.asarray(sim_values, dtype=float)
    sim_values[sim_values == MISSING_VALUE_MARKER] = np.nan

    window_starts = reference_times + window_start_offset
    window_ends = reference_times + window_end_offset
    obs_slice_starts = np.searchsorted(obs_time_at_reference, window_starts, side="left")
    obs_slice_ends = np.searchsorted(obs_time_at_reference, window_ends, side="right")

    spread_label = (
        f"ensemble spread ({int(spread_quantiles[0] * 100)}-{int(spread_quantiles[1] * 100)}%)"
    )

    frames = []
    for index in range(n_forecasts):
        lo, hi = obs_slice_starts[index], obs_slice_ends[index]
        obs_rel_hours = (obs_time_at_reference[lo:hi] - reference_times[index]) / np.timedelta64(
            1,
            "h",
        )
        traces = _forecast_scroller_frame(
            lead_hours=lead_hours,
            sim_values=sim_values[index],
            obs_rel_hours=obs_rel_hours,
            obs_values=obs_at_reference[lo:hi],
            is_ensemble=is_ensemble,
            show_spread=show_spread,
            show_members=show_members,
            spread_quantiles=spread_quantiles,
            spread_label=spread_label,
        )
        issued = np.datetime_as_string(reference_times[index], unit="h")
        frames.append(
            go.Frame(
                data=traces,
                name=str(index),
                layout=go.Layout(title=f"Forecast issued {issued}"),
            ),
        )

    units = str(obs.attrs.get("units", ""))
    y_title = f"{obs_var} / {sim_var}" + (f" ({units})" if units else "")
    lead_hours_min = float(lead_hours.min()) if lead_hours.size else 0.0
    lead_hours_max = float(lead_hours.max()) if lead_hours.size else 0.0

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=frames[0].layout.title,
            xaxis={
                "title": "Hours relative to forecast issue time",
                "range": [lead_hours_min - before_hours, lead_hours_max + after_hours],
            },
            yaxis={"title": y_title},
            template=template,
            hovermode="x unified",
        ),
        frames=frames,
    )
    fig.update_layout(
        sliders=_forecast_scroller_slider(reference_times),
        updatemenus=_forecast_scroller_playback_buttons(play_speed),
    )
    return fig
