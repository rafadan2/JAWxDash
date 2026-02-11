from dash import callback, Output, Input, State, no_update
import logging
import numpy as np


from src import ids
from src.ellipsometry_toolbox.ellipsometry import Ellipsometry
from src.utils.file_manager import get_file_path

logger = logging.getLogger(__name__)


@callback(
    # Spot
    Output(ids.RadioItems.PLOT_STYLE, "value"),
    Output(ids.Slider.ANGLE_OF_INCIDENT, "value"),
    Output(ids.RadioItems.SPOT_SIZE, "value"),
    
    # Sample
    Output(ids.DropDown.COLORMAPS, "value"),
    Output(ids.DropDown.COLORMAPS, "options"),
    Output(ids.DropDown.SAMPLE_OUTLINE, "value"),
    Output(ids.DropDown.Z_DATA, "value"),
    Output(ids.Input.Z_SCALE_MIN, "value"),
    Output(ids.Input.Z_SCALE_MAX, "value"),
    
    # MapPattern offsets
    Output(ids.Input.MAPPATTERN_X, "value"),
    Output(ids.Input.MAPPATTERN_Y, "value"),
    Output(ids.Input.MAPPATTERN_THETA, "value"),
    
    # Sample offsets
    Output(ids.Input.SAMPLE_X, "value"),
    Output(ids.Input.SAMPLE_Y, "value"),
    Output(ids.Input.SAMPLE_THETA, "value"),
    Output(ids.Input.SAMPLE_RADIUS, "value"),
    Output(ids.Input.SAMPLE_WIDTH, "value"),
    Output(ids.Input.SAMPLE_HEIGHT, "value"),
    
    # Edge exclusion
    Output(ids.RadioItems.EDGE_EXCLUSION_STATE, "value"),
    Output(ids.RadioItems.EDGE_EXCLUSION_TYPE, "value"),
    Output(ids.Input.EDGE_EXCLUSION_DISTANCE, "value"),
    Output(ids.RadioItems.BATCH_PROCESSING, "value"),

    # Stage outline
    Output(ids.RadioItems.STAGE_STATE, "value"),

    # Input
    Input(ids.Store.DEFAULT_SETTINGS, "data"),

)
def load_default_settings(default_settings):

    return (
        # Spot
        default_settings["marker_type"],
        default_settings["angle_of_incident"],
        default_settings["spot_size"],
        
        # Sample
        default_settings["colormap_value"],
        default_settings["colormap_options"],
        default_settings["sample_outline"],
        default_settings["z_data_value"],
        default_settings["z_scale_min"],
        default_settings["z_scale_max"],
        
        # Mappattern offset
        default_settings["mappattern_x"],
        default_settings["mappattern_y"],
        default_settings["mappattern_theta"],

        # Sample offset
        default_settings["sample_x"],
        default_settings["sample_y"],
        default_settings["sample_theta"],
        default_settings["sample_radius"],
        default_settings["sample_width"],
        default_settings["sample_height"],

        # Edge exclusion
        default_settings["ee_state"],
        default_settings["ee_type"],
        default_settings["ee_distance"],
        default_settings["ee_batch_processing"],

        # Stage outline
        default_settings["stage_state"],
    )


@callback(
    Output(ids.Store.SETTINGS, "data"),
    
    # Spot
    Input(ids.RadioItems.PLOT_STYLE, "value"),
    Input(ids.Slider.ANGLE_OF_INCIDENT, "value"),
    Input(ids.RadioItems.SPOT_SIZE, "value"),

    # Sample
    Input(ids.DropDown.COLORMAPS, "value"),
    Input(ids.DropDown.SAMPLE_OUTLINE, "value"),
    Input(ids.DropDown.Z_DATA, "value"),
    Input(ids.Input.Z_SCALE_MIN, "value"),
    Input(ids.Input.Z_SCALE_MAX, "value"),
    
    # MapPattern offset
    Input(ids.Input.MAPPATTERN_X, "value"),
    Input(ids.Input.MAPPATTERN_Y, "value"),
    Input(ids.Input.MAPPATTERN_THETA, "value"),
    
    # Sample offset
    Input(ids.Input.SAMPLE_X, "value"),
    Input(ids.Input.SAMPLE_Y, "value"),
    Input(ids.Input.SAMPLE_THETA, "value"),
    Input(ids.Input.SAMPLE_RADIUS, "value"),
    Input(ids.Input.SAMPLE_WIDTH, "value"),
    Input(ids.Input.SAMPLE_HEIGHT, "value"),

    # Edge exclusion
    Input(ids.RadioItems.EDGE_EXCLUSION_STATE, "value"),
    Input(ids.RadioItems.EDGE_EXCLUSION_TYPE, "value"),
    Input(ids.Input.EDGE_EXCLUSION_DISTANCE, "value"),
    Input(ids.RadioItems.BATCH_PROCESSING, "value"),

    # Stage outline
    Input(ids.RadioItems.STAGE_STATE, "value"),

    # Store state
    State(ids.Store.SETTINGS, "data"),
)
def update_offset_setting_store(
    marker_type, angle_of_incident, spot_size,
    colormap_value, sample_outline, z_data_value, z_scale_min, z_scale_max,
    x_map, y_map, t_map, 
    x_sam, y_sam, t_sam, r_sam, w_sam, h_sam,
    ee_state, ee_type, ee_distance, batch_processing,
    stage_state,
    settings
    ):


    keys = (
        "marker_type", 
        "angle_of_incident", 
        "spot_size",
        "colormap_value", 
        "sample_outline", 
        "z_data_value",
        "z_scale_min",
        "z_scale_max",
        "mappattern_x",
        "mappattern_y",
        "mappattern_theta",
        "sample_x",
        "sample_y",
        "sample_theta",
        "sample_radius",
        "sample_width",
        "sample_height",
        "ee_state", 
        "ee_type", 
        "ee_distance", 
        "batch_processing",
        "stage_state",
    )
    values = (
        marker_type, angle_of_incident, spot_size,
        colormap_value, sample_outline, z_data_value,
        z_scale_min, z_scale_max,
        x_map, y_map, t_map,
        x_sam, y_sam, t_sam, r_sam, w_sam, h_sam,
        ee_state, ee_type, ee_distance, batch_processing,
        stage_state,
    )

    for key, value in zip(keys, values):
        settings[key] = value

    
    return settings


def _resolve_z_key(file, z_key):
    if z_key:
        return z_key if z_key in file.data else None

    columns = sorted(file.get_column_names())
    if len(columns) > 1:
        return columns[1]
    if columns:
        return columns[0]
    return None


def _calculate_z_min_max(selected_file, uploaded_files, z_key):
    if not selected_file or not uploaded_files or selected_file not in uploaded_files:
        return None

    file_path = get_file_path(uploaded_files, selected_file)
    if not file_path:
        return None

    file = Ellipsometry.from_path_or_stream(file_path)
    z_key = _resolve_z_key(file, z_key)
    if not z_key:
        return None

    z_data = file.data[z_key].to_numpy()
    zmin = float(np.nanmin(z_data))
    zmax = float(np.nanmax(z_data))

    if not np.isfinite(zmin) or not np.isfinite(zmax):
        return None

    return zmin, zmax


@callback(
    Output(ids.Input.Z_SCALE_MIN, "value", allow_duplicate=True),
    Output(ids.Input.Z_SCALE_MAX, "value", allow_duplicate=True),
    Input(ids.Button.Z_SCALE_2SIGMA, "n_clicks"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    State(ids.Store.SETTINGS, "data"),
    prevent_initial_call=True,
)
def set_z_scale_2sigma(n_clicks, selected_file, uploaded_files, settings):
    if not n_clicks or not selected_file:
        return no_update, no_update

    if not uploaded_files or selected_file not in uploaded_files:
        return no_update, no_update

    file_path = get_file_path(uploaded_files, selected_file)
    if not file_path:
        return no_update, no_update

    file = Ellipsometry.from_path_or_stream(file_path)
    z_key = settings.get("z_data_value") if settings else None
    z_key = _resolve_z_key(file, z_key)
    if not z_key:
        return no_update, no_update

    z_data = file.data[z_key].to_numpy()
    median = float(np.nanmedian(z_data))
    sigma = float(np.nanstd(z_data))

    if not np.isfinite(median) or not np.isfinite(sigma):
        return no_update, no_update

    return median - 2 * sigma, median + 2 * sigma


@callback(
    Output(ids.Input.Z_SCALE_MIN, "value", allow_duplicate=True),
    Output(ids.Input.Z_SCALE_MAX, "value", allow_duplicate=True),
    Input(ids.Button.Z_SCALE_AUTO, "n_clicks"),
    State(ids.DropDown.Z_DATA, "value"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    prevent_initial_call=True,
)
def set_z_scale_auto(n_clicks, z_key, selected_file, uploaded_files):
    if not n_clicks:
        return no_update, no_update

    result = _calculate_z_min_max(selected_file, uploaded_files, z_key)
    if not result:
        return no_update, no_update

    return result


@callback(
    Output(ids.Input.Z_SCALE_MIN, "value", allow_duplicate=True),
    Output(ids.Input.Z_SCALE_MAX, "value", allow_duplicate=True),
    Input(ids.DropDown.Z_DATA, "value"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    prevent_initial_call=True,
)
def update_z_scale_on_zdata_change(z_key, selected_file, uploaded_files):
    result = _calculate_z_min_max(selected_file, uploaded_files, z_key)
    if not result:
        return no_update, no_update

    return result
