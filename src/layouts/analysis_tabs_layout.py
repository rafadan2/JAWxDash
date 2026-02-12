from dash import dcc, html
import dash_bootstrap_components as dbc


from src import ids
from src.layouts.stat_table_layout import stat_table_layout


analysis_tabs_layout = dbc.Tabs(
    id=ids.Tabs.ANALYSIS,
    active_tab="summary_table",
    className="mt-3",
    children=[
        dbc.Tab(
            label="Summary Table",
            tab_id="summary_table",
            children=[
                html.Div(stat_table_layout, className="pt-2"),
            ],
        ),
        dbc.Tab(
            label="Distribution",
            tab_id="distribution",
            children=[
                html.Div(
                    [
                        html.Div(id=ids.Div.DISTRIBUTION_METRICS, className="p-2"),
                        dcc.Graph(id=ids.Graph.DISTRIBUTION_XY, style={"height": "560px"}),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id=ids.Graph.DISTRIBUTION_RESIDUALS,
                                        style={"height": "460px"},
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id=ids.Graph.DISTRIBUTION_RADIAL,
                                        style={"height": "460px"},
                                    ),
                                    width=6,
                                ),
                            ],
                            className="g-2 mt-1",
                        ),
                    ],
                    className="pt-2",
                )
            ],
        ),
        dbc.Tab(
            label="Gradient Violins",
            tab_id="gradient_violin",
            children=[
                html.Div(
                    [
                        dcc.Graph(id=ids.Graph.GRADIENT_VIOLIN),
                    ],
                    className="pt-2",
                )
            ],
        ),
        dbc.Tab(
            label="Spatial Binning",
            tab_id="spatial_binning",
            children=[
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Number of radial bins", className="mb-1"),
                                        dcc.Input(
                                            id=ids.Input.SPATIAL_BIN_RADIAL_COUNT,
                                            type="number",
                                            value=2,
                                            min=1,
                                            step=1,
                                            debounce=True,
                                            className="form-control",
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Number of angular bins", className="mb-1"),
                                        dcc.Input(
                                            id=ids.Input.SPATIAL_BIN_ANGULAR_COUNT,
                                            type="number",
                                            value=4,
                                            min=1,
                                            step=1,
                                            debounce=True,
                                            className="form-control",
                                        ),
                                    ],
                                    width=6,
                                ),
                            ],
                            className="g-2 align-items-end",
                        ),
                        html.Div(
                            "Radial boundaries are auto-calculated for near-equal point counts; angular bins are equal-angle sections.",
                            className="text-muted mt-2",
                        ),
                        html.Div(id=ids.Div.SPATIAL_BIN_COUNTS, className="p-2"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id=ids.Graph.SPATIAL_BIN_MAP, style={"height": "600px"}),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(id=ids.Graph.SPATIAL_BIN_TRENDS, style={"height": "600px"}),
                                    width=6,
                                ),
                            ],
                            className="g-2 mt-1",
                        ),
                    ],
                    className="pt-2",
                )
            ],
        ),
    ],
)
