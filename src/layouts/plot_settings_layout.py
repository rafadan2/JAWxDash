from dash import dcc, html
import dash_bootstrap_components as dbc
import logging

from src import ids


logger = logging.getLogger(__name__)


spot_layout = dbc.Card([
    dbc.CardHeader("Plot Settings"),
    dbc.CardBody([
        # Z data
        dbc.Row([
            dbc.Col(
                html.Label("Z-Data", className="mb-0 d-block text-body text-decoration-none"),
                width=5,
            ),
            dbc.Col(
                dcc.Dropdown(
                    id=ids.DropDown.Z_DATA,
                    options=[],
                    multi=False,
                    clearable=False,
                    className="mb-0",
                ),
                width=7,
            )
        ]),

        # Gradient mode
        dbc.Row([
            dbc.Col(
                html.Label("Gradient", className="mb-0 d-block text-body text-decoration-none"),
                width=5,
            ),
            dbc.Col(
                dcc.Dropdown(
                    id=ids.DropDown.GRADIENT_MODE,
                    options=[
                        {"label": "None", "value": "none"},
                        {"label": "|grad Z|", "value": "magnitude"},
                        {"label": "dZ/dX", "value": "dx"},
                        {"label": "dZ/dY", "value": "dy"},
                    ],
                    multi=False,
                    clearable=False,
                    className="mb-0",
                ),
                width=7,
            )
        ]),

        # Gradient grid mode
        dbc.Row([
            dbc.Col(
                html.Label("Grid mode", className="mb-0 d-block text-body text-decoration-none"),
                width=5,
            ),
            dbc.Col(
                dcc.RadioItems(
                    id=ids.RadioItems.GRADIENT_GRID_MODE,
                    options=[
                        {"label": "Auto", "value": "auto"},
                        {"label": "Manual", "value": "manual"},
                    ],
                    inline=True,
                    labelStyle={"margin-right": "15px"},
                    className="mb-0",
                ),
                width=7,
            )
        ]),

        # Gradient grid size
        dbc.Row(
            html.Label("Gradient grid size", style={"textAlign": "center"})
        ),
        dbc.Row(
            dcc.Slider(
                id=ids.Slider.GRADIENT_GRID_SIZE,
                min=24,
                max=180,
                step=1,
                marks={
                    24: "24",
                    32: "32",
                    48: "48",
                    64: "64",
                    96: "96",
                    128: "128",
                    160: "160",
                    180: "180",
                },
                className="mb-0",
            )
        ),

        # Gradient k-nearest
        dbc.Row(
            html.Label("Gradient k-nearest", style={"textAlign": "center"})
        ),
        dbc.Row(
            dcc.Slider(
                id=ids.Slider.GRADIENT_K_NEAREST,
                min=4,
                max=64,
                step=1,
                marks={
                    4: "4",
                    8: "8",
                    16: "16",
                    24: "24",
                    32: "32",
                    48: "48",
                    64: "64",
                },
                className="mb-0",
            )
        ),

        # Colormap
        dbc.Row([
            dbc.Col(
                html.Label("Colormap", className="mb-0 d-block text-body text-decoration-none"),
                width=5,
            ),
            dbc.Col(
                dcc.Dropdown(
                    id=ids.DropDown.COLORMAPS,
                    multi=False,
                    clearable=False,
                    className="mb-0",
                ),      
                width=7,          
            )
        ]),
        
        # Z scale min
        dbc.Row([
            dbc.Col(
                html.Label("Z-Scale Min", className="mb-0 d-block text-body text-decoration-none"),
                width=5,
            ),
            dbc.Col(
                dbc.InputGroup([
                    dcc.Input(
                        id=ids.Input.Z_SCALE_MIN,
                        type="number",
                        debounce=True,
                        placeholder="auto",
                        className="form-control",
                    ),
                    dbc.Button(
                        "Auto",
                        id=ids.Button.Z_SCALE_AUTO,
                        color="secondary",
                        size="sm",
                        n_clicks=0,
                    ),
                ], className="mb-0"),
                width=7,
            )
        ]),
        
        # Z scale max
        dbc.Row([
            dbc.Col(
                html.Label("Z-Scale Max", className="mb-0 d-block text-body text-decoration-none"),
                width=5,
            ),
            dbc.Col(
                dbc.InputGroup([
                    dcc.Input(
                        id=ids.Input.Z_SCALE_MAX,
                        type="number",
                        debounce=True,
                        placeholder="auto",
                        className="form-control",
                    ),
                    dbc.Button(
                        "2σ",
                        id=ids.Button.Z_SCALE_2SIGMA,
                        color="secondary",
                        size="sm",
                        n_clicks=0,
                    ),
                ], className="mb-0"),
                width=7,
            )
        ]),

        # Marker style
        dbc.Row([
            dbc.Col(
                html.Label("Marker style", className="mb-0 d-block text-body text-decoration-none"),
                width=5,
            ),
            dbc.Col(
                dcc.RadioItems(
                    id=ids.RadioItems.PLOT_STYLE,
                    options=[
                       {'label': 'PTS', 'value': 'point'},
                       {'label': 'ELL', 'value': 'ellipse'},
                    ],
                    inline=True,
                    labelStyle={"margin-right": "15px"},
                    className="mb-0",
                ),
                width=7,
            )
        ]),

        # Focus probes
        dbc.Row([
            dbc.Col(
                html.Label("Probes", className="mb-0 d-block text-body text-decoration-none"),
                width=5,
            ),
            dbc.Col(
                dcc.RadioItems(
                    id=ids.RadioItems.SPOT_SIZE,
                    options=[
                        {'label': 'ON', 'value': 0.03},
                        {'label': 'OFF', 'value': 0.3},
                    ],
                    inline=True,
                    labelStyle={"margin-right": "15px"},
                    className="mb-0",
                ),
                width=7,
            )
        ]),

        # Angle of incident
        dbc.Row(
            html.Label("Angle of incident (deg)", style={'textAlign': 'center'}),
        ),
        dbc.Row(
            dcc.Slider(
                id=ids.Slider.ANGLE_OF_INCIDENT, 
                min=45, 
                max=85, 
                step=1, 
                marks={i: str(i) for i in range(45, 86, 5)},
                className="mb-0",
            ),
        ),

        # Marker size
        dbc.Row(
            html.Label("Marker size", style={"textAlign": "center"})
        ),
        dbc.Row(
            dcc.Slider(
                id=ids.Slider.MARKER_SIZE,
                min=1,
                max=15,
                step=1,
                className="mb-0",
            )
        )

    ])
], className="mt-1")

