import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from pydantic import BaseModel

from dpyverification.datamodel import OutputDataset


class Theme(BaseModel):
    """Theme settings for app-wide styling tokens."""

    font_family: str = "Segoe UI, Arial, sans-serif"
    font_color: str = "#022B6D"
    page_bg: str = "#F8FAFC"
    card_bg: str = "#FFFFFF"
    card_border: str = "#E2E8F0"
    accent: str = "#7A491C"
    accent_muted: str = "#61BD4D"
    reference: str = "#818181"


THEME = Theme()
MISSING_VALUE_MARKER = -999

STYLES = {
    "page": {
        "maxWidth": "1600px",
        "margin": "24px auto",
        "padding": "16px",
        "fontFamily": THEME.font_family,
        "color": THEME.font_color,
        "backgroundColor": THEME.page_bg,
    },
    "filters": {
        "display": "flex",
        "gap": "12px",
        "flexWrap": "wrap",
        "marginBottom": "12px",
        "backgroundColor": THEME.card_bg,
        "border": f"1px solid {THEME.card_border}",
        "borderRadius": "10px",
        "padding": "12px",
    },
    "card": {
        "backgroundColor": THEME.card_bg,
        "border": f"1px solid {THEME.card_border}",
        "borderRadius": "10px",
        "padding": "12px",
    },
}


PLOT_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font={"family": THEME.font_family, "color": THEME.font_color},
        paper_bgcolor=THEME.card_bg,
        plot_bgcolor=THEME.card_bg,
    ),
)


def create_app(output_dataset: OutputDataset) -> Dash:
    """Create a Dash app with sidebar navigation for scatter and CRPS views."""
    verification_pairs = output_dataset.verification_pairs
    pair_lookup = {pair.id: pair for pair in verification_pairs}

    if not verification_pairs:
        msg = "No verification pairs found in the output dataset."
        raise ValueError(msg)

    default_left_pair_id = verification_pairs[0].id
    default_right_pair_id = (
        verification_pairs[1].id if len(verification_pairs) > 1 else verification_pairs[0].id
    )

    def get_pair_dataset(pair_id):
        return output_dataset.get(pair_lookup[pair_id])

    def get_forecast_period_labels(ds):
        values = list(ds.coords["forecast_period"].values)
        labels = [f"{int(v / np.timedelta64(1, 'h'))} h" for v in values]
        return values, labels

    def get_scatter_controls(pair_id):
        ds = get_pair_dataset(pair_id)
        stations = [str(v) for v in ds.coords["station"].values]
        variables = [str(v) for v in ds.coords["variable"].values]
        _, forecast_labels = get_forecast_period_labels(ds)
        return stations, variables, forecast_labels

    def get_crps_score_variables(pair_id):
        pair = pair_lookup[pair_id]
        ds = get_pair_dataset(pair_id)
        input_vars = {str(pair.obs), str(pair.sim)}
        score_candidates = [
            var_name
            for var_name, data_array in ds.data_vars.items()
            if var_name not in input_vars and "forecast_period" in data_array.dims
        ]
        crps_candidates = [name for name in score_candidates if "crps" in name.lower()]
        return crps_candidates or score_candidates

    def get_crps_controls(pair_id):
        ds = get_pair_dataset(pair_id)
        stations = [str(v) for v in ds.coords["station"].values]
        variables = [str(v) for v in ds.coords["variable"].values]
        return stations, variables

    scatter_left_stations, scatter_left_variables, scatter_left_forecast = get_scatter_controls(
        default_left_pair_id,
    )
    scatter_right_stations, scatter_right_variables, scatter_right_forecast = get_scatter_controls(
        default_right_pair_id,
    )
    crps_left_stations, crps_left_variables = get_crps_controls(
        default_left_pair_id,
    )
    crps_right_stations, crps_right_variables = get_crps_controls(
        default_right_pair_id,
    )

    app = Dash(__name__)

    def make_scatter_panel(panel_key, default_pair_id, stations, variables, forecast_labels):
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Verification Pair"),
                                dcc.Dropdown(
                                    id=f"scatter-verification-pair-dropdown-{panel_key}",
                                    options=[
                                        {
                                            "label": f"{pair.id} ({pair.obs} vs {pair.sim})",
                                            "value": pair.id,
                                        }
                                        for pair in verification_pairs
                                    ],
                                    value=default_pair_id,
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "280px"},
                        ),
                        html.Div(
                            [
                                html.Label("Station"),
                                dcc.Dropdown(
                                    id=f"scatter-station-dropdown-{panel_key}",
                                    options=[{"label": s, "value": s} for s in stations],
                                    value=stations[0],
                                    clearable=False,
                                    searchable=True,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "260px"},
                        ),
                        html.Div(
                            [
                                html.Label("Forecast Period"),
                                dcc.Dropdown(
                                    id=f"scatter-forecast-period-dropdown-{panel_key}",
                                    options=[{"label": p, "value": p} for p in forecast_labels],
                                    value=forecast_labels[0],
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "220px"},
                        ),
                        html.Div(
                            [
                                html.Label("Variable"),
                                dcc.Dropdown(
                                    id=f"scatter-variable-dropdown-{panel_key}",
                                    options=[{"label": v, "value": v} for v in variables],
                                    value=variables[0],
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "180px"},
                        ),
                    ],
                    style=STYLES["filters"],
                ),
                dcc.Graph(
                    id=f"scatter-plot-{panel_key}",
                    style={"height": "620px", **STYLES["card"]},
                ),
            ],
            style={
                "flex": "1 1 560px",
                "minWidth": "460px",
                "display": "flex",
                "flexDirection": "column",
                "gap": "8px",
            },
        )

    def make_crps_panel(panel_key, default_pair_id, stations, variables):
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Verification Pair"),
                                dcc.Dropdown(
                                    id=f"crps-verification-pair-dropdown-{panel_key}",
                                    options=[
                                        {
                                            "label": f"{pair.id} ({pair.obs} vs {pair.sim})",
                                            "value": pair.id,
                                        }
                                        for pair in verification_pairs
                                    ],
                                    value=default_pair_id,
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "280px"},
                        ),
                        html.Div(
                            [
                                html.Label("Variable"),
                                dcc.Dropdown(
                                    id=f"crps-variable-dropdown-{panel_key}",
                                    options=[{"label": v, "value": v} for v in variables],
                                    value=variables[0],
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "180px"},
                        ),
                        html.Div(
                            [
                                html.Label("Stations"),
                                dcc.Dropdown(
                                    id=f"crps-station-dropdown-{panel_key}",
                                    options=[{"label": s, "value": s} for s in stations],
                                    value=[stations[0]],
                                    multi=True,
                                    clearable=False,
                                    searchable=True,
                                ),
                            ],
                            style={"flex": "2", "minWidth": "320px"},
                        ),
                    ],
                    style=STYLES["filters"],
                ),
                dcc.Graph(
                    id=f"crps-plot-{panel_key}",
                    style={"height": "620px", **STYLES["card"]},
                ),
            ],
            style={
                "flex": "1 1 560px",
                "minWidth": "460px",
                "display": "flex",
                "flexDirection": "column",
                "gap": "8px",
            },
        )

    views_bar = html.Div(
        [
            dcc.Tabs(
                id="view-selector",
                children=[
                    dcc.Tab(label="Scatter", value="scatter"),
                    dcc.Tab(label="CRPS", value="crps"),
                ],
                value="scatter",
                parent_style={"margin": "0"},
            ),
        ],
        style={
            **STYLES["card"],
            "marginBottom": "12px",
        },
    )

    scatter_view = html.Div(
        [
            html.Div(
                [
                    make_scatter_panel(
                        panel_key="left",
                        default_pair_id=default_left_pair_id,
                        stations=scatter_left_stations,
                        variables=scatter_left_variables,
                        forecast_labels=scatter_left_forecast,
                    ),
                    make_scatter_panel(
                        panel_key="right",
                        default_pair_id=default_right_pair_id,
                        stations=scatter_right_stations,
                        variables=scatter_right_variables,
                        forecast_labels=scatter_right_forecast,
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
        ],
        id="scatter-view",
    )

    crps_view = html.Div(
        [
            html.Div(
                [
                    make_crps_panel(
                        panel_key="left",
                        default_pair_id=default_left_pair_id,
                        stations=crps_left_stations,
                        variables=crps_left_variables,
                    ),
                    make_crps_panel(
                        panel_key="right",
                        default_pair_id=default_right_pair_id,
                        stations=crps_right_stations,
                        variables=crps_right_variables,
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
        ],
        id="crps-view",
        style={"display": "none"},
    )

    app.layout = html.Div(
        [
            views_bar,
            html.Div(
                [
                    scatter_view,
                    crps_view,
                ],
                style={"minWidth": "0"},
            ),
        ],
        style=STYLES["page"],
    )

    @app.callback(
        Output("scatter-view", "style"),
        Output("crps-view", "style"),
        Input("view-selector", "value"),
    )
    def switch_view(selected_view):
        if selected_view == "crps":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    def make_scatter_figure(
        selected_pair_id,
        selected_station,
        selected_forecast_period,
        selected_variable,
    ):
        selected_pair = pair_lookup[selected_pair_id]
        ds = get_pair_dataset(selected_pair_id)
        pair_forecast_period_values, pair_forecast_period_labels = get_forecast_period_labels(ds)
        pair_forecast_period_lookup = dict(
            zip(pair_forecast_period_labels, pair_forecast_period_values, strict=True),
        )
        forecast_period = pair_forecast_period_lookup[selected_forecast_period]

        obs_selected = ds[str(selected_pair.obs)].sel(
            station=selected_station,
            variable=selected_variable,
            forecast_period=forecast_period,
            drop=True,
        )
        sim_selected = ds[str(selected_pair.sim)].sel(
            station=selected_station,
            variable=selected_variable,
            forecast_period=forecast_period,
            drop=True,
        )

        obs_values = obs_selected.values
        sim_mean = sim_selected.mean(dim="realization").values

        obs_flat = np.repeat(obs_values, sim_selected.sizes["realization"])
        sim_flat = sim_selected.values.flatten()

        valid_mask = (
            (sim_flat != MISSING_VALUE_MARKER) & np.isfinite(obs_flat) & np.isfinite(sim_flat)
        )
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
        if span == 0:
            padding = max(abs(axis_min), 1.0) * 0.05
        else:
            padding = span * 0.05

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=obs_flat,
                y=sim_flat,
                mode="markers",
                name=f"{selected_pair.sim} ensemble members",
                marker={"size": 5, "color": THEME.accent_muted, "symbol": "circle"},
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=obs_values,
                y=sim_mean,
                mode="markers",
                name=f"{selected_pair.sim} ensemble mean",
                marker={"size": 7, "color": THEME.accent, "symbol": "circle"},
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[axis_min - padding, axis_max + padding],
                y=[axis_min - padding, axis_max + padding],
                mode="lines",
                name="1:1 line",
                line={"color": THEME.reference, "dash": "dash", "width": 1},
                showlegend=True,
            ),
        )
        fig.update_layout(
            xaxis_title=str(selected_pair.obs),
            yaxis_title=str(selected_pair.sim),
            xaxis={"range": [axis_min - padding, axis_max + padding]},
            yaxis={
                "range": [axis_min - padding, axis_max + padding],
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            template=PLOT_TEMPLATE,
            hovermode="closest",
        )
        return fig

    def make_crps_figure(selected_pair_id, selected_stations, selected_variable):
        selected_pair = pair_lookup[selected_pair_id]
        ds = get_pair_dataset(selected_pair_id)
        score_variables = get_crps_score_variables(selected_pair_id)

        if not score_variables:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No CRPS-like score variable found for pair: {selected_pair.id}",
                showarrow=False,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
            )
            fig.update_layout(
                template=PLOT_TEMPLATE,
            )
            return fig

        score_name = score_variables[0]
        score_data = ds[score_name]

        if "variable" in score_data.dims:
            score_data = score_data.sel(variable=selected_variable, drop=True)

        available_stations = [str(v) for v in ds.coords["station"].values]
        stations = selected_stations or [available_stations[0]]

        if "station" in score_data.dims:
            score_data = score_data.sel(station=stations)

        dims_to_reduce = [
            dim_name
            for dim_name in score_data.dims
            if dim_name not in {"forecast_period", "station"}
        ]
        if dims_to_reduce:
            score_data = score_data.mean(dim=dims_to_reduce, skipna=True)

        forecast_period_values, forecast_period_labels = get_forecast_period_labels(ds)
        forecast_hours = np.array(
            [float(v / np.timedelta64(1, "h")) for v in forecast_period_values],
        )

        fig = go.Figure()
        if "station" in score_data.dims:
            for station in score_data.coords["station"].values:
                station_name = str(station)
                station_values = np.ravel(score_data.sel(station=station).values)
                valid = np.isfinite(station_values)
                fig.add_trace(
                    go.Scatter(
                        x=forecast_hours[valid],
                        y=station_values[valid],
                        mode="lines+markers",
                        name=station_name,
                    ),
                )
        else:
            score_values = np.ravel(score_data.values)
            valid = np.isfinite(score_values)
            fig.add_trace(
                go.Scatter(
                    x=forecast_hours[valid],
                    y=score_values[valid],
                    mode="lines+markers",
                    name="CRPS",
                ),
            )

        fig.update_layout(
            xaxis={
                "title": "Forecast period (h)",
                "tickmode": "array",
                "tickvals": forecast_hours,
                "ticktext": forecast_period_labels,
            },
            yaxis_title=score_name,
            template=PLOT_TEMPLATE,
            hovermode="x unified",
        )
        return fig

    def register_scatter_panel_callbacks(panel_key):
        @app.callback(
            Output(f"scatter-station-dropdown-{panel_key}", "options"),
            Output(f"scatter-station-dropdown-{panel_key}", "value"),
            Output(f"scatter-forecast-period-dropdown-{panel_key}", "options"),
            Output(f"scatter-forecast-period-dropdown-{panel_key}", "value"),
            Output(f"scatter-variable-dropdown-{panel_key}", "options"),
            Output(f"scatter-variable-dropdown-{panel_key}", "value"),
            Input(f"scatter-verification-pair-dropdown-{panel_key}", "value"),
        )
        def update_scatter_controls(selected_pair_id):
            stations, variables, forecast_labels = get_scatter_controls(selected_pair_id)
            return (
                [{"label": station, "value": station} for station in stations],
                stations[0],
                [{"label": label, "value": label} for label in forecast_labels],
                forecast_labels[0],
                [{"label": variable, "value": variable} for variable in variables],
                variables[0],
            )

        @app.callback(
            Output(f"scatter-plot-{panel_key}", "figure"),
            Input(f"scatter-verification-pair-dropdown-{panel_key}", "value"),
            Input(f"scatter-station-dropdown-{panel_key}", "value"),
            Input(f"scatter-forecast-period-dropdown-{panel_key}", "value"),
            Input(f"scatter-variable-dropdown-{panel_key}", "value"),
        )
        def update_scatter_figure(
            selected_pair_id,
            selected_station,
            selected_forecast_period,
            selected_variable,
        ):
            return make_scatter_figure(
                selected_pair_id,
                selected_station,
                selected_forecast_period,
                selected_variable,
            )

    def register_crps_panel_callbacks(panel_key):
        @app.callback(
            Output(f"crps-station-dropdown-{panel_key}", "options"),
            Output(f"crps-station-dropdown-{panel_key}", "value"),
            Output(f"crps-variable-dropdown-{panel_key}", "options"),
            Output(f"crps-variable-dropdown-{panel_key}", "value"),
            Input(f"crps-verification-pair-dropdown-{panel_key}", "value"),
        )
        def update_crps_controls(selected_pair_id):
            stations, variables = get_crps_controls(selected_pair_id)
            return (
                [{"label": station, "value": station} for station in stations],
                [stations[0]],
                [{"label": variable, "value": variable} for variable in variables],
                variables[0],
            )

        @app.callback(
            Output(f"crps-plot-{panel_key}", "figure"),
            Input(f"crps-verification-pair-dropdown-{panel_key}", "value"),
            Input(f"crps-station-dropdown-{panel_key}", "value"),
            Input(f"crps-variable-dropdown-{panel_key}", "value"),
        )
        def update_crps_figure(
            selected_pair_id,
            selected_stations,
            selected_variable,
        ):
            return make_crps_figure(
                selected_pair_id,
                selected_stations,
                selected_variable,
            )

    @app.callback(
        Output("crps-station-dropdown-left", "value"),
        Output("crps-station-dropdown-right", "value"),
        Input("crps-station-dropdown-left", "value"),
        Input("crps-station-dropdown-right", "value"),
        State("crps-station-dropdown-left", "options"),
        State("crps-station-dropdown-right", "options"),
        prevent_initial_call=True,
    )
    def sync_crps_stations(left_values, right_values, left_options, right_options):
        left_values = left_values or []
        right_values = right_values or []
        left_available = {option["value"] for option in left_options}
        right_available = {option["value"] for option in right_options}

        if ctx.triggered_id == "crps-station-dropdown-left":
            shared = [station for station in left_values if station in right_available]
            if shared:
                return shared, shared
            return no_update, no_update

        if ctx.triggered_id == "crps-station-dropdown-right":
            shared = [station for station in right_values if station in left_available]
            if shared:
                return shared, shared
            return no_update, no_update

        return no_update, no_update

    register_scatter_panel_callbacks("left")
    register_scatter_panel_callbacks("right")
    register_crps_panel_callbacks("left")
    register_crps_panel_callbacks("right")

    return app
