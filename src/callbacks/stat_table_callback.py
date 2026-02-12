from dash import callback, dash_table, html, Output, Input, State
from dash.dash_table.Format import Format, Scheme
import logging


# Local imports
from src import ids
from src.ellipsometry_toolbox.ellipsometry import Ellipsometry
from src.ellipsometry_toolbox.masking import create_masked_file
from src.templates.settings_template import DEFAULT_SETTINGS


logger = logging.getLogger(__name__)


@callback(
    Output(ids.Div.STAT_TABLE, "children"),
    Input(ids.DropDown.UPLOADED_FILES, "value"),
    Input(ids.Store.SETTINGS, "data"),
    State(ids.Store.UPLOADED_FILES, "data"),
)
def update_stat_table(selected_file, settings, stored_files):

    if not selected_file or not stored_files or selected_file not in stored_files:
        return None
    
    file = Ellipsometry.from_path_or_stream(stored_files[selected_file])
    active_settings = {**DEFAULT_SETTINGS, **(settings or {})}

    if active_settings.get("ee_state"):
        file = create_masked_file(file, active_settings)

    if file.data.empty:
        return html.Div("No data available with current edge-exclusion settings.", className="text-muted")

    try:
        stats = file.statistics()
    except ValueError:
        return html.Div("No numeric columns available for statistics.", className="text-muted")

    for column in ("x", "y"):
        if column in stats.columns:
            stats.drop(columns=[column], inplace=True)
    if stats.shape[1] == 0:
        return html.Div("No parameter columns available for statistics.", className="text-muted")
    

    columns = [{"id": col, "name": col, "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)} for col in stats.columns]
    columns.insert(0, {"id": "stats", "name": "Stats"})
    
    stats.insert(0, "stats", stats.index.to_list())
    data = stats.to_dict("records")


    table = dash_table.DataTable(
        columns=columns,
        data=data,
        style_table={'overflowX': 'auto'},
        style_cell={'padding': '8px', 'textAlign': 'right'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
    )


    return table
