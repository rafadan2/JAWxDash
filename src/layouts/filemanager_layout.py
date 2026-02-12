from dash import html, dcc
import dash_bootstrap_components as dbc


from src import ids

filemanager_layout = dbc.Card([
    dbc.CardHeader("File Manager"),
    dbc.CardBody([
        # Drag-n-drop field
        dcc.Upload(
            id=ids.Upload.DRAG_N_DROP,
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '100px',
                'lineHeight': '100px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'backgroundColor': '#f8f9fa',
                'cursor': 'pointer'
            },            # Allow multiple files to be uploaded
            multiple=True,
            className="mb-1"
        ),
        
        # Vertical spacing
        #html.Div(style={"height": "10px"}),
        
        # File dropdown
        dcc.Dropdown(
            id=ids.DropDown.UPLOADED_FILES,
            options=[],
            value='',
            multi=False,
            clearable=False,
            className="mb-1",
        ),
    
        # File delete button
        dbc.Button(
                "Delete Selected",
                id=ids.Button.DELETE_SELECTED,
                color="warning",
                className="mb-1 w-100"
        ),

        # Clear file manager
        dbc.Button(
                "Clear File Manager",
                id=ids.Button.CLEAR_FILE_MANAGER,
                color="danger",
                className="mb-1 w-100"
        ),

        dbc.Button(
                "Batch Download Main Plots (PNG)",
                id=ids.Button.BATCH_MAIN_PLOTS,
                color="primary",
                className="mb-1 w-100",
                n_clicks=0,
        ),
    ])
], className="mt-1")

