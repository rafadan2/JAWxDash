from dash import Dash, dcc, html
import dash_bootstrap_components as dbc


# Local import
from src import ids
from src.logger_config import setup_logging
from src.layouts import edge_exclusion_layout, filemanager_layout, graph_layout, outline_layout, spot_layout, mappattern_layout, analysis_tabs_layout, stage_layout


from src.templates.settings_template import DEFAULT_SETTINGS


setup_logging()  # initiate logging module

app = Dash(
    __name__, 
    requests_pathname_prefix="/ellipsometer/", 
    routes_pathname_prefix="/ellipsometer/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=["https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"],
)
app.server.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 10  # Allow for large file uploads 10MB

server = app.server


app.layout = dbc.Container([
    # Initiating 'Store' for holding uploaded files i.e. *.txt and *.csv
    dcc.Store(id=ids.Store.UPLOADED_FILES, data={}, storage_type="session"),
    dcc.Store(id=ids.Store.DEFAULT_SETTINGS, data=DEFAULT_SETTINGS),
    dcc.Store(id=ids.Store.SETTINGS, data={}),
    dcc.Store(id=ids.Store.BATCH_MAIN_PLOTS_PAYLOAD, data={}),

    dbc.Row([
        # Left column
        dbc.Col([
            filemanager_layout,
        ], width=3),

        # Middle column
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    graph_layout,
                    analysis_tabs_layout,
                ])
            ], className="mt-1"),
        ], width=7),
        
        # Right column
        dbc.Col([
            spot_layout,
            outline_layout,
            stage_layout,
            mappattern_layout,
            edge_exclusion_layout,
        ], width=2),
    ]),

    # Bottom info panel
    dbc.Row([
        html.Div(
            id=ids.Div.INFO, 
            style={'border': '1px solid black', 'padding': '10px', 'marginTop': '20px'}
        ),
    ]),

], fluid=True)  #, className="app-container")


# Register callbacks
import src.callbacks.edge_exclusion_callbacks
import src.callbacks.filemanager_callbacks
import src.callbacks.graph_callbacks
import src.callbacks.distribution_callbacks
import src.callbacks.settings_callback
import src.callbacks.stat_table_callback
import src.callbacks.store_callbacks

server = app.server

if __name__ == '__main__':
    app.run(debug=True)
