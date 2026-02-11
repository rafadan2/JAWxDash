from dash import callback, dash_table, Output, Input, State
from dash.dash_table.Format import Format, Scheme
import logging


# Local imports
from src import ids
from src.ellipsometry_toolbox.ellipsometry import Ellipsometry
from src.ellipsometry_toolbox.masking import create_masked_file
from src.utils.file_manager import get_file_path


logger = logging.getLogger(__name__)


@callback(
    Output(ids.Div.STAT_TABLE, "children"),
    Input(ids.DropDown.UPLOADED_FILES, "value"),
    Input(ids.Store.SETTINGS, "data"),
    State(ids.Store.UPLOADED_FILES, "data"),
)
def update_stat_table(selected_file, settings, stored_files):

    if not selected_file:
        return None
    
    file_path = get_file_path(stored_files, selected_file)
    if not file_path:
        return None

    file = Ellipsometry.from_path_or_stream(file_path)

    if settings["ee_state"]:
        file = create_masked_file(file, settings)


    stats = file.statistics()
    stats.drop(columns=["x", "y"], inplace=True)
    

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
    ),


    return table
