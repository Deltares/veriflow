"""Multi-pair Plotly figures composed from a whole verification ``xr.DataTree``.

This module provides ``get_pair_dataset``/``get_score_dataset`` (plot-ready ``xr.Dataset``
builders for a single verification pair) and the high-level figure builders (``crps_plot``,
``scatter_plot``, ``rank_histogram_plot``, ``rank_histogram_3d_plot``,
``reanalysis_timeseries_plot``, ``forecast_timeseries_plot``): each figure builder takes the
``xr.DataTree`` itself (see ``veriflow.datamodel.output.VeriflowAccessor``), fetches every
verification pair's dataset via ``get_pair_dataset``/``get_score_dataset``, builds one
standalone figure per pair with the corresponding low-level builder from :mod:`pair_plots`, and
composes them into a single grid/overlay figure.

There is no interactive layer (no Dash): any selection that used to be an interactive
control - station, lead time - is applied via keyword arguments on these functions, which
forward the resulting ``.sel(...)`` slice to the low-level builder.

Typical usage::

    from veriflow import run_pipeline

    dt = run_pipeline(...)
    pair_id = dt.veriflow.verification_pairs[0]
    ds = get_pair_dataset(dt, pair_id)

    scatter_plot(dt, lead_time=ds.coords["lead_time"][0])
    crps_plot(dt, stations=["station-a", "station-b"])
    reanalysis_timeseries_plot(dt, station="some-station", error_var="mean_error")
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from pair_plots import (
    PLOT_TEMPLATE,
    _format_lead_time_label,
    forecast_pair_plot,
    rank_histogram_3d_pair_plot,
    rank_histogram_pair_plot,
    reanalysis_pair_plot,
    scatter_pair_plot,
    score_plot,
)
from plotly.subplots import make_subplots

from veriflow.datamodel.output import DataTreeNode, VeriflowDataTree


def get_score_dataset(
    dt: VeriflowDataTree,
    verification_pair_id: str,
    score_name: str,
) -> xr.Dataset:
    """Get a single score's result as a ``xr.Dataset`` for a specific verification pair."""
    outputs = dt.veriflow.get_outputs(verification_pair_id)
    return cast("xr.Dataset", outputs[score_name].to_dataset())


def get_pair_dataset(
    dt: VeriflowDataTree,
    verification_pair_id: str,
    *,
    obs_suffix: str = "_obs",
    sim_suffix: str = "_sim",
    include_scores: Sequence[str] = (),
) -> xr.Dataset:
    """Get a single, plot-ready ``xr.Dataset`` for a specific verification pair.

    The ``aligned_input`` observations and simulations share the same data variable name
    (the verification pair's ``variable``), so they are renamed to ``<variable><obs_suffix>``
    and ``<variable><sim_suffix>`` before being merged into one dataset. Pass
    ``include_scores`` to additionally merge in one or more named ``output`` scores (e.g. to
    plot an error score alongside the observations and simulations).
    """
    aligned_input = dt.veriflow.get_aligned_input(verification_pair_id)
    observations_dataset = aligned_input[DataTreeNode.OBSERVATIONS].to_dataset()
    variable = str(next(iter(observations_dataset.data_vars)))

    obs_ds = cast(
        "xr.Dataset",
        observations_dataset.rename({variable: f"{variable}{obs_suffix}"}),  # type:ignore[misc]
    )
    sim_ds = cast(
        "xr.Dataset",
        aligned_input[DataTreeNode.SIMULATIONS]
        .to_dataset()
        .rename({variable: f"{variable}{sim_suffix}"}),  # type:ignore[misc]
    )

    merged: xr.Dataset = xr.merge(  # type:ignore[call-overload]
        [
            obs_ds,
            sim_ds,
            *(get_score_dataset(dt, verification_pair_id, name) for name in include_scores),
        ],
        compat="override",
        join="outer",
    )
    merged.attrs["variable"] = variable  # type:ignore[misc]
    merged.attrs["verification_pair_id"] = verification_pair_id  # type:ignore[misc]
    return merged


def _axis_id(fig: go.Figure, axis: go.layout.XAxis | go.layout.YAxis, letter: str) -> str:
    """Return the on-figure axis id (e.g. ``"x3"``) for a live axis object from ``get_subplot``."""
    prefix = f"{letter}axis"
    for key in fig.layout:
        if str(key).startswith(prefix) and fig.layout[key] is axis:
            return f"{letter}{str(key)[len(prefix) :]}"
    msg = f"Could not resolve the on-figure id of the given {letter}-axis."
    raise ValueError(msg)


def _place_pair_figure(
    pair_fig: go.Figure,
    grid: go.Figure,
    *,
    row: int,
    col: int,
    secondary_y: bool = False,
    match_scaleanchor: bool = False,
) -> None:
    """Copy one standalone per-pair figure's traces and axis styling into one cell of ``grid``.

    ``pair_fig`` is expected to be a single-cell figure (as built by the low-level ``*_plot``
    functions), optionally with a secondary y-axis. Axis titles/ranges are copied onto the
    corresponding subplot axes of ``grid``; with ``match_scaleanchor`` the y-axis' 1:1
    ``scaleanchor`` (used by :func:`pair_plots.scatter_pair_plot`) is remapped to the grid's own
    x-axis id for this cell.
    """
    secondary_ys = [trace.yaxis == "y2" for trace in pair_fig.data] if secondary_y else None
    grid.add_traces(
        pair_fig.data,
        rows=[row] * len(pair_fig.data),
        cols=[col] * len(pair_fig.data),
        secondary_ys=secondary_ys,
    )

    target_x = grid.get_subplot(row, col).xaxis
    if pair_fig.layout.xaxis.title.text:
        target_x.title = pair_fig.layout.xaxis.title.text
    if pair_fig.layout.xaxis.range is not None:
        target_x.range = pair_fig.layout.xaxis.range

    for is_secondary in {True, False} if secondary_y else {False}:
        source_y = pair_fig.layout.yaxis2 if is_secondary else pair_fig.layout.yaxis
        target_y = grid.get_subplot(row, col, secondary_y=is_secondary).yaxis
        if source_y.title.text:
            target_y.title = source_y.title.text
        if source_y.range is not None:
            target_y.range = source_y.range
        if match_scaleanchor and source_y.scaleanchor is not None:
            target_y.scaleanchor = _axis_id(grid, target_x, "x")
            target_y.scaleratio = source_y.scaleratio


def scatter_plot(
    dt: VeriflowDataTree,
    *,
    pair_ids: Sequence[str] | None = None,
    lead_time: object | None = None,
    station: object | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build observation-vs-simulation scatter subplots for every verification pair.

    One subplot is drawn per verification pair (``pair_ids``, or every pair in ``dt`` if not
    given), laid out in a grid of at most two columns (filling left-to-right, then
    top-to-bottom). For each pair the dataset is obtained with ``get_pair_dataset``
    and the observed/simulated variable names are derived from its ``variable`` attribute (see
    :func:`pair_plots.scatter_pair_plot`, which builds the standalone figure for a single pair).

    If ``lead_time`` is given and a dataset has a ``lead_time`` dimension, that dataset is
    sliced with ``ds.sel(lead_time=lead_time)`` before plotting. Datasets without a
    ``lead_time`` dimension (e.g. historical simulations) are left untouched, so the same
    call works for both forecast and historical setups.

    The simulated variable may carry a ``realization`` dimension (an ensemble) or not (a
    deterministic/historical simulation); the ensemble case additionally shows the ensemble
    mean. Each pair's dataset must be sliced to a single station (or have no ``station``
    dimension).
    """
    pairs = list(pair_ids) if pair_ids is not None else dt.veriflow.verification_pairs
    if not pairs:
        msg = "The DataTree contains no verification pairs to plot."
        raise ValueError(msg)

    n_cols = min(2, len(pairs))
    n_rows = -(-len(pairs) // n_cols)  # ceil division

    # Pre-build (and lead-time slice) each pair's standalone figure so legend visibility is
    # shared across all of them before they are composed into the grid below.
    pair_figures: list[go.Figure] = []
    titles: list[str] = []
    legend_state = {
        "members": True,
        "mean": True,
        "simulation": True,
        "reference": True,
    }
    for pair_id in pairs:
        ds = get_pair_dataset(dt, pair_id)
        title = str(pair_id)
        if lead_time is not None and "lead_time" in ds.dims:
            ds = ds.sel(lead_time=lead_time)
            label = _format_lead_time_label(ds.coords["lead_time"].values)
            if label:
                title = f"{title} ({label})"
        if station is not None:
            ds = ds.sel(station=station)
            title = f"{title} - station {station}"

        variable = str(ds.attrs["variable"])
        pair_figures.append(
            scatter_pair_plot(
                ds,
                obs_var=f"{variable}_obs",
                sim_var=f"{variable}_sim",
                legend_state=legend_state,
                template=template,
            ),
        )
        titles.append(title)

    grid = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=min(0.12, 0.3 / n_cols),
        vertical_spacing=min(0.12, 0.6 / n_rows),
    )
    for index, pair_fig in enumerate(pair_figures):
        row = index // n_cols + 1
        col = index % n_cols + 1
        _place_pair_figure(pair_fig, grid, row=row, col=col, match_scaleanchor=True)

    grid.update_layout(
        template=template,
        hovermode="closest",
        height=420 * n_rows,
    )
    return grid


def crps_plot(
    dt: VeriflowDataTree,
    *,
    pair_ids: Sequence[str] | None = None,
    score_name: str = "crps_for_ensemble",
    stations: list[object] | None = None,
    score_var: str | None = None,
    score_vars: str | list[str] | tuple[str, ...] | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build a score-vs-lead-time figure with one line per verification pair.

    One line is drawn per verification pair (``pair_ids``, or every pair in ``dt`` if not
    given), overlaid on a single shared figure. For each pair the dataset is obtained with
    ``get_score_dataset(dt, pair_id, score_name)`` (see :func:`pair_plots.score_plot`,
    which builds the standalone figure for a single pair); pass ``score_var``/``score_vars`` to
    select which score variable(s) to plot, as in :func:`pair_plots.score_plot`.

    Each pair's dataset must retain its ``lead_time`` dimension. If a ``station`` dimension is
    present, one line is drawn per station (labelled ``"<pair id> - <station>"`` when more than
    one pair is plotted); otherwise a single line labelled by the pair id is drawn. Any
    remaining dimensions other than ``lead_time`` and ``station`` are averaged out.
    """
    pairs = list(pair_ids) if pair_ids is not None else dt.veriflow.verification_pairs
    if not pairs:
        msg = "The DataTree contains no verification pairs to plot."
        raise ValueError(msg)

    multi_pair = len(pairs) > 1
    pair_figures: list[go.Figure] = []
    all_hours: list[np.ndarray] = []
    for pair_id in pairs:
        ds = get_score_dataset(dt, pair_id, score_name)
        if stations is not None and "station" in ds.dims:
            ds = ds.sel(station=stations)

        pair_fig = score_plot(ds, score_var=score_var, score_vars=score_vars, template=template)
        if multi_pair:
            for trace in pair_fig.data:
                trace.name = f"{pair_id} - {trace.name}"
        pair_figures.append(pair_fig)
        all_hours.append(np.asarray(pair_fig.layout.xaxis.tickvals))

    grid = make_subplots(rows=1, cols=1)
    for pair_fig in pair_figures:
        _place_pair_figure(pair_fig, grid, row=1, col=1)

    tick_hours = np.unique(np.concatenate(all_hours)) if all_hours else np.array([])
    y_titles = {str(pair_fig.layout.yaxis.title.text) for pair_fig in pair_figures}
    yaxis_title = next(iter(y_titles)) if len(y_titles) == 1 else "Score"

    grid.update_layout(
        xaxis={
            "title": "Lead Time (h)",
            "tickmode": "array",
            "tickvals": tick_hours,
            "ticktext": [f"{int(h)} h" for h in tick_hours],
        },
        yaxis_title=yaxis_title,
        template=template,
        hovermode="x unified",
        title=f"{yaxis_title} vs Lead Time for {len(pairs)} Verification Pair(s)",
    )
    return grid


def rank_histogram_plot(
    dt: VeriflowDataTree,
    *,
    pair_ids: Sequence[str] | None = None,
    score_name: str = "rank_histogram",
    rank_var: str = "histogram_rank",
    station: object | None = None,
    lead_time: object | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build rank-histogram subplots for every verification pair.

    One subplot is drawn per verification pair (``pair_ids``, or every pair in ``dt`` if not
    given), laid out in a grid of at most two columns (filling left-to-right, then
    top-to-bottom). For each pair the dataset is obtained with
    ``get_score_dataset(dt, pair_id, score_name)`` and ``rank_var`` is plotted (see
    :func:`pair_plots.rank_histogram_pair_plot`, which builds the standalone figure for a
    single pair).

    ``station`` and ``lead_time`` select the slice to plot: when given (and present on the
    dataset) the dataset is sliced with ``ds.sel(station=...)`` and/or ``ds.sel(lead_time=...)``
    so that ``rank_var`` is one-dimensional over the ``rank`` coordinate. The chosen station and
    lead time are shown in each subplot title.
    """
    pairs = list(pair_ids) if pair_ids is not None else dt.veriflow.verification_pairs
    if not pairs:
        msg = "The DataTree contains no verification pairs to plot."
        raise ValueError(msg)

    n_cols = min(2, len(pairs))
    n_rows = -(-len(pairs) // n_cols)  # ceil division

    # Pre-build (and slice) each pair's standalone figure so subplot titles can carry
    # station/lead time.
    pair_figures: list[go.Figure] = []
    titles: list[str] = []
    for pair_id in pairs:
        ds = get_score_dataset(dt, pair_id, score_name)
        if station is not None and "station" in ds.dims:
            ds = ds.sel(station=station)
        if lead_time is not None and "lead_time" in ds.dims:
            ds = ds.sel(lead_time=lead_time)

        title_parts = [str(pair_id)]
        if "station" in ds.coords:
            title_parts.append(f"station {np.atleast_1d(ds.coords['station'].values)[0]}")
        if "lead_time" in ds.coords:
            label = _format_lead_time_label(ds.coords["lead_time"].values)
            if label:
                title_parts.append(label)

        pair_figures.append(rank_histogram_pair_plot(ds, rank_var=rank_var, template=template))
        titles.append(", ".join(title_parts))

    grid = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=min(0.12, 0.3 / n_cols),
        vertical_spacing=min(0.2, 1.0 / n_rows),
    )
    for index, pair_fig in enumerate(pair_figures):
        row = index // n_cols + 1
        col = index % n_cols + 1
        _place_pair_figure(pair_fig, grid, row=row, col=col)

    grid.update_layout(
        template=template,
        height=380 * n_rows,
        title=f"Rank Histograms for {len(pairs)} Verification Pair(s)",
    )
    return grid


def rank_histogram_3d_plot(
    dt: VeriflowDataTree,
    *,
    pair_ids: Sequence[str] | None = None,
    score_name: str = "rank_histogram",
    rank_var: str = "histogram_rank",
    station: object | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build 3D rank-histogram surfaces for every verification pair.

    This is a three-dimensional version of :func:`rank_histogram_plot`. Instead of collapsing
    the dataset to a single lead time, the ``lead_time`` axis is kept and shown as a third
    dimension: for each verification pair a 3D surface is drawn with ``rank`` on the x-axis,
    ``lead_time`` (in hours) on the y-axis, and the rank count on the z-axis. This shows all
    rank histograms across lead times in one figure, so the evolution of the histogram shape
    with increasing lead time becomes visible.

    One 3D subplot is drawn per verification pair (``pair_ids``, or every pair in ``dt`` if not
    given), laid out in a grid of at most two columns (filling left-to-right, then
    top-to-bottom). ``station`` selects the station to plot when the dataset has a ``station``
    dimension; it must reduce ``rank_var`` to two dimensions (``lead_time`` by ``rank``). The
    chosen station is shown in each subplot title. See
    :func:`pair_plots.rank_histogram_3d_pair_plot`, which builds the standalone figure for a
    single pair.
    """
    pairs = list(pair_ids) if pair_ids is not None else dt.veriflow.verification_pairs
    if not pairs:
        msg = "The DataTree contains no verification pairs to plot."
        raise ValueError(msg)

    n_cols = min(2, len(pairs))
    n_rows = -(-len(pairs) // n_cols)  # ceil division

    # Pre-build (and slice) each pair's standalone figure so subplot titles can carry station.
    pair_figures: list[go.Figure] = []
    titles: list[str] = []
    for pair_id in pairs:
        ds = get_score_dataset(dt, pair_id, score_name)
        if station is not None and "station" in ds.dims:
            ds = ds.sel(station=station)

        title_parts = [str(pair_id)]
        if "station" in ds.coords:
            title_parts.append(f"station {np.atleast_1d(ds.coords['station'].values)[0]}")

        pair_figures.append(rank_histogram_3d_pair_plot(ds, rank_var=rank_var, template=template))
        titles.append(", ".join(title_parts))

    grid = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        specs=[[{"type": "surface"} for _ in range(n_cols)] for _ in range(n_rows)],
        horizontal_spacing=min(0.12, 0.3 / n_cols),
        vertical_spacing=min(0.2, 1.0 / n_rows),
    )
    for index, pair_fig in enumerate(pair_figures):
        row = index // n_cols + 1
        col = index % n_cols + 1
        # Only the first subplot's colorbar is shown, to avoid cluttering the grid.
        grid.add_trace(pair_fig.data[0].update(showscale=index == 0), row=row, col=col)

    grid.update_scenes(
        xaxis_title_text="Rank",
        yaxis_title_text="Lead time (h)",
        zaxis_title_text="Count",
    )
    grid.update_layout(
        template=template,
        height=480 * n_rows,
        title=f"3D Rank Histograms for {len(pairs)} Verification Pair(s)",
    )
    return grid


def reanalysis_timeseries_plot(
    dt: VeriflowDataTree,
    *,
    pair_ids: Sequence[str] | None = None,
    station: object | None = None,
    error_var: str | None = None,
    error_score_name: str | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Plot observed and simulated historical time series for every verification pair.

    This is intended for reanalysis or other historical verification runs where the pair
    datasets use a direct ``time`` dimension instead of the forecast
    ``(forecast_reference_time, lead_time)`` grid.

    One stacked subplot is drawn per verification pair (``pair_ids``, or every pair in ``dt``
    if not given). By default, all stations are plotted with translucent traces. Pass a scalar
    station id or a list of station ids to pre-select the station dimension before plotting.
    Pass ``error_var`` to draw one score/error variable, such as ``"mean_error"`` or ``"mae"``,
    on a secondary y-axis; ``error_score_name`` selects the ``output`` score this comes from
    (defaults to ``error_var`` itself). See :func:`pair_plots.reanalysis_pair_plot`, which
    builds the standalone figure for a single pair.
    """
    pairs = list(pair_ids) if pair_ids is not None else dt.veriflow.verification_pairs
    if not pairs:
        msg = "The DataTree contains no verification pairs to plot."
        raise ValueError(msg)

    include_scores = [error_score_name or error_var] if error_var is not None else []

    # Pre-build (and station-slice) each pair's standalone figure so legend visibility is
    # shared across all of them before they are composed into the grid below.
    pair_figures: list[go.Figure] = []
    titles: list[str] = []
    legend_state: dict[str, bool] = {}
    for pair_id in pairs:
        ds = get_pair_dataset(dt, pair_id, include_scores=include_scores)
        if station is not None:
            if "station" not in ds.dims:
                msg = "Cannot select station because this dataset has no 'station' dimension."
                raise ValueError(msg)
            ds = ds.sel(station=station)

        title_parts = [str(pair_id)]
        if "station" in ds.coords and "station" not in ds.dims:
            title_parts.append(f"station {np.asarray(ds.coords['station'].values).item()}")
        elif "station" in ds.dims and station is not None:
            title_parts.append(f"{ds.sizes['station']} stations")

        variable = str(ds.attrs["variable"])
        pair_figures.append(
            reanalysis_pair_plot(
                ds,
                obs_var=f"{variable}_obs",
                sim_var=f"{variable}_sim",
                error_var=error_var,
                legend_state=legend_state,
                template=template,
            ),
        )
        titles.append(", ".join(title_parts))

    grid = make_subplots(
        rows=len(pairs),
        cols=1,
        shared_xaxes=False,
        subplot_titles=titles,
        vertical_spacing=min(0.12, 1.0 / len(pairs)),
        specs=[[{"secondary_y": error_var is not None}] for _ in pairs],
    )

    for index, pair_fig in enumerate(pair_figures, start=1):
        # Give each row's traces their own legendgroup, so toggling one pair's legend entry
        # does not also hide the same trace type in another pair's subplot.
        for trace in pair_fig.data:
            trace.legendgroup = f"{index}-{trace.legendgroup}"
        _place_pair_figure(pair_fig, grid, row=index, col=1, secondary_y=error_var is not None)

    grid.update_xaxes(title_text="Time", row=len(pairs), col=1)
    for index in range(1, len(pairs) + 1):
        grid.update_yaxes(matches="y", row=index, col=1, secondary_y=False)
    grid.update_layout(
        template=template,
        hovermode="closest",
        height=340 * len(pairs),
    )
    return grid


def forecast_timeseries_plot(
    dt: VeriflowDataTree,
    *,
    pair_ids: Sequence[str] | None = None,
    station: object | None = None,
    show_members: bool = False,
    show_spread: bool = True,
    spread_quantiles: tuple[float, float] = (0.1, 0.9),
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Plot historical observed series and forecast trajectories for every pair.

    One stacked subplot is drawn per verification pair (``pair_ids``, or every pair in ``dt``
    if not given). For each pair the dataset is obtained with ``get_pair_dataset``
    and the observed/simulated variable names are derived from its ``variable`` attribute (see
    :func:`pair_plots.forecast_pair_plot`, which builds the standalone figure for a single
    pair).

    Both variables are stored on the ``(forecast_reference_time, lead_time)`` grid and share
    an auxiliary ``time(forecast_reference_time, lead_time)`` coordinate giving each value's
    valid time. The observed values are flattened against ``time`` into one continuous line,
    while each ``forecast_reference_time`` produces one forecast trajectory drawn along its
    valid times.

    The simulated variable may carry a ``realization`` dimension (an ensemble) or not
    (deterministic):

    - Ensemble: the ensemble mean is drawn per forecast; with ``show_spread`` a shaded band
      between ``spread_quantiles`` is added, and with ``show_members`` every member line is
      drawn faintly.
    - Deterministic: a single line is drawn per forecast.

    Pass ``station`` to select one station from multi-station datasets. If ``station`` is not
    given, each pair's dataset must already be sliced to a single station (or have no
    ``station`` dimension).
    """
    pairs = list(pair_ids) if pair_ids is not None else dt.veriflow.verification_pairs
    if not pairs:
        msg = "The DataTree contains no verification pairs to plot."
        raise ValueError(msg)

    # Pre-build (and station-slice) each pair's standalone figure so legend visibility is
    # shared across all of them before they are composed into the grid below.
    pair_figures: list[go.Figure] = []
    titles: list[str] = []
    legend_state = {
        "observed": True,
        "member": True,
        "spread": True,
        "mean": True,
        "forecast": True,
    }
    for pair_id in pairs:
        ds = get_pair_dataset(dt, pair_id)
        if station is not None:
            if "station" not in ds.dims:
                msg = "Cannot select station because this dataset has no 'station' dimension."
                raise ValueError(msg)
            ds = ds.sel(station=station)

        title_parts = [str(pair_id)]
        if "station" in ds.coords and "station" not in ds.dims:
            title_parts.append(f"station {np.asarray(ds.coords['station'].values).item()}")

        variable = str(ds.attrs["variable"])
        pair_figures.append(
            forecast_pair_plot(
                ds,
                obs_var=f"{variable}_obs",
                sim_var=f"{variable}_sim",
                show_members=show_members,
                show_spread=show_spread,
                spread_quantiles=spread_quantiles,
                legend_state=legend_state,
                template=template,
            ),
        )
        titles.append(", ".join(title_parts))

    grid = make_subplots(
        rows=len(pairs),
        cols=1,
        shared_xaxes=False,
        subplot_titles=titles,
        vertical_spacing=min(0.12, 1.0 / len(pairs)),
    )
    for index, pair_fig in enumerate(pair_figures, start=1):
        # Give each row's traces their own legendgroup, so toggling one pair's legend entry
        # does not also hide the same trace type in another pair's subplot.
        for trace in pair_fig.data:
            trace.legendgroup = f"{index}-{trace.legendgroup}"
        _place_pair_figure(pair_fig, grid, row=index, col=1)

    # Only label the x-axis of the bottom subplot so the "Time" label does not collide
    # with the title of the subplot below it.
    grid.update_xaxes(title_text="Time", row=len(pairs), col=1)

    # Share a single y-axis scale across all subplots so values are directly comparable.
    grid.update_yaxes(matches="y")

    grid.update_layout(
        template=template,
        hovermode="x unified",
        height=340 * len(pairs),
    )
    return grid
