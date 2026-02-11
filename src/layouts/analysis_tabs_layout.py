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
    ],
)
