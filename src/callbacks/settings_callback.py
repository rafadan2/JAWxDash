from dash import callback, Output, Input, State, no_update
import logging
import numpy as np


from src import ids
from src.callbacks.graph_callbacks import (
    _build_gradient_map as _build_gradient_map_with_axes,
    _build_value_map as _build_value_map_with_axes,
    _resolve_gradient_calc_grid_mode,
    _resolve_gradient_coordinate_mode,
    _resolve_heatmap_grid_mode,
    _resolve_render_mode,
)
from src.ellipsometry_toolbox.ellipsometry import Ellipsometry
from src.ellipsometry_toolbox.linear_translations import rotate, translate
from src.templates.settings_template import DEFAULT_SETTINGS

logger = logging.getLogger(__name__)


@callback(
    # Spot
    Output(ids.RadioItems.PLOT_STYLE, "value"),
    Output(ids.Slider.ANGLE_OF_INCIDENT, "value"),
    Output(ids.RadioItems.SPOT_SIZE, "value"),
    Output(ids.Slider.MARKER_SIZE, "value"),
    
    # Sample
    Output(ids.DropDown.COLORMAPS, "value"),
    Output(ids.DropDown.COLORMAPS, "options"),
    Output(ids.DropDown.SAMPLE_OUTLINE, "value"),
    Output(ids.DropDown.Z_DATA, "value"),
    Output(ids.RadioItems.RENDER_MODE, "value"),
    Output(ids.DropDown.GRADIENT_MODE, "value"),
    Output(ids.RadioItems.GRADIENT_COORDINATE_MODE, "value"),
    Output(ids.RadioItems.GRADIENT_GRID_MODE, "value"),
    Output(ids.Slider.GRADIENT_GRID_SIZE, "value"),
    Output(ids.Slider.GRADIENT_K_NEAREST, "value"),
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
    default_settings = {**DEFAULT_SETTINGS, **(default_settings or {})}

    return (
        # Spot
        default_settings["marker_type"],
        default_settings["angle_of_incident"],
        default_settings["spot_size"],
        default_settings["marker_size"],
        
        # Sample
        default_settings["colormap_value"],
        default_settings["colormap_options"],
        default_settings["sample_outline"],
        default_settings["z_data_value"],
        _resolve_render_mode(default_settings),
        default_settings["gradient_mode"],
        _resolve_gradient_coordinate_mode(default_settings),
        _resolve_heatmap_grid_mode(default_settings),
        default_settings["gradient_grid_size"],
        default_settings["gradient_k_nearest"],
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


def _merge_runtime_plot_settings(
    settings,
    render_mode=None,
    gradient_mode=None,
    gradient_coordinate_mode=None,
    heatmap_grid_mode=None,
    gradient_grid_size=None,
    gradient_k_nearest=None,
):
    active_settings = {**DEFAULT_SETTINGS, **(settings or {})}

    if render_mode in {"markers", "heatmap"}:
        active_settings["render_mode"] = render_mode
    if gradient_mode:
        active_settings["gradient_mode"] = gradient_mode
    if gradient_coordinate_mode in {"cartesian", "polar"}:
        active_settings["gradient_coordinate_mode"] = gradient_coordinate_mode
    if heatmap_grid_mode in {"auto", "manual"}:
        active_settings["heatmap_grid_mode"] = heatmap_grid_mode
    if gradient_grid_size is not None:
        active_settings["gradient_grid_size"] = gradient_grid_size
    if gradient_k_nearest is not None:
        active_settings["gradient_k_nearest"] = gradient_k_nearest

    # Keep the legacy setting in sync for callbacks that may still inspect it.
    active_settings["gradient_grid_mode"] = _resolve_gradient_calc_grid_mode(active_settings)
    return active_settings


@callback(
    Output(ids.Store.SETTINGS, "data"),
    
    # Spot
    Input(ids.RadioItems.PLOT_STYLE, "value"),
    Input(ids.Slider.ANGLE_OF_INCIDENT, "value"),
    Input(ids.RadioItems.SPOT_SIZE, "value"),
    Input(ids.Slider.MARKER_SIZE, "value"),

    # Sample
    Input(ids.DropDown.COLORMAPS, "value"),
    Input(ids.DropDown.SAMPLE_OUTLINE, "value"),
    Input(ids.DropDown.Z_DATA, "value"),
    Input(ids.RadioItems.RENDER_MODE, "value"),
    Input(ids.DropDown.GRADIENT_MODE, "value"),
    Input(ids.RadioItems.GRADIENT_COORDINATE_MODE, "value"),
    Input(ids.RadioItems.GRADIENT_GRID_MODE, "value"),
    Input(ids.Slider.GRADIENT_GRID_SIZE, "value"),
    Input(ids.Slider.GRADIENT_K_NEAREST, "value"),
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
    marker_type, angle_of_incident, spot_size, marker_size,
    colormap_value, sample_outline, z_data_value, render_mode, gradient_mode, gradient_coordinate_mode, gradient_grid_mode, gradient_grid_size, gradient_k_nearest, z_scale_min, z_scale_max,
    x_map, y_map, t_map, 
    x_sam, y_sam, t_sam, r_sam, w_sam, h_sam,
    ee_state, ee_type, ee_distance, batch_processing,
    stage_state,
    settings
    ):
    settings = settings or {}
    resolved_gradient_coordinate_mode = (
        gradient_coordinate_mode if gradient_coordinate_mode in {"cartesian", "polar"} else "cartesian"
    )
    resolved_heatmap_grid_mode = gradient_grid_mode if gradient_grid_mode in {"auto", "manual"} else "auto"
    legacy_gradient_grid_mode = (
        "polar_native"
        if resolved_gradient_coordinate_mode == "polar"
        else resolved_heatmap_grid_mode
    )

    keys = (
        "marker_type", 
        "angle_of_incident", 
        "spot_size",
        "marker_size",
        "colormap_value", 
        "sample_outline", 
        "z_data_value",
        "render_mode",
        "gradient_mode",
        "gradient_coordinate_mode",
        "heatmap_grid_mode",
        "gradient_grid_mode",
        "gradient_grid_size",
        "gradient_k_nearest",
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
        marker_type, angle_of_incident, spot_size, marker_size,
        colormap_value, sample_outline, z_data_value, render_mode, gradient_mode, resolved_gradient_coordinate_mode, resolved_heatmap_grid_mode, legacy_gradient_grid_mode, gradient_grid_size, gradient_k_nearest,
        z_scale_min, z_scale_max,
        x_map, y_map, t_map,
        x_sam, y_sam, t_sam, r_sam, w_sam, h_sam,
        ee_state, ee_type, ee_distance, batch_processing,
        stage_state,
    )

    for key, value in zip(keys, values):
        settings[key] = value

    
    return settings


@callback(
    Output(ids.Slider.GRADIENT_GRID_SIZE, "disabled"),
    Input(ids.RadioItems.GRADIENT_GRID_MODE, "value"),
)
def toggle_gradient_grid_size_disabled(grid_mode):
    return grid_mode != "manual"


@callback(
    Output(ids.Slider.GRADIENT_K_NEAREST, "disabled"),
    Input(ids.RadioItems.GRADIENT_COORDINATE_MODE, "value"),
    Input(ids.DropDown.GRADIENT_MODE, "value"),
)
def toggle_gradient_k_nearest_disabled(gradient_coordinate_mode, gradient_mode):
    return gradient_coordinate_mode == "polar" and gradient_mode != "none"


@callback(
    Output(ids.Input.MAPPATTERN_X, "value", allow_duplicate=True),
    Output(ids.Input.MAPPATTERN_Y, "value", allow_duplicate=True),
    Output(ids.Input.MAPPATTERN_THETA, "value", allow_duplicate=True),
    Output(ids.Input.SAMPLE_X, "value", allow_duplicate=True),
    Output(ids.Input.SAMPLE_Y, "value", allow_duplicate=True),
    Output(ids.Input.SAMPLE_THETA, "value", allow_duplicate=True),
    Input(ids.DropDown.SAMPLE_OUTLINE, "value"),
    prevent_initial_call=True,
)
def reset_offsets_for_circle_outline(sample_outline):
    if sample_outline != "circle":
        return no_update, no_update, no_update, no_update, no_update, no_update

    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def _resolve_z_key(file, z_key):
    columns = [column for column in sorted(file.get_column_names()) if str(column).lower() not in {"x", "y"}]
    if z_key:
        return z_key if z_key in columns else None

    if columns:
        return columns[0]
    return None


def _build_gradient_map(
    x_data,
    y_data,
    z_data,
    gradient_mode,
    grid_mode="auto",
    grid_size=None,
    k_nearest=None,
    polar_center=None,
):
    gradient_result = _build_gradient_map_with_axes(
        x_data,
        y_data,
        z_data,
        gradient_mode,
        grid_mode=grid_mode,
        grid_size=grid_size,
        k_nearest=k_nearest,
        polar_center=polar_center,
    )
    if not gradient_result:
        return None

    _, _, gradient_grid = gradient_result
    return gradient_grid


def _extract_active_plot_values(file, z_key, gradient_mode, settings):
    z_values = file.data[z_key].to_numpy()

    render_mode = _resolve_render_mode(settings)
    heatmap_grid_mode = _resolve_heatmap_grid_mode(settings)
    gradient_calc_grid_mode = _resolve_gradient_calc_grid_mode(settings)

    x_data = np.array(file.data["x"])
    y_data = np.array(file.data["y"])
    xy = rotate(np.vstack([x_data, y_data]), settings.get("mappattern_theta", 0.0))
    xy = translate(xy, [settings.get("mappattern_x", 0.0), settings.get("mappattern_y", 0.0)])
    x_data = xy[0, :]
    y_data = xy[1, :]

    if gradient_mode == "none":
        if render_mode == "heatmap":
            value_map = _build_value_map_with_axes(
                x_data,
                y_data,
                z_values,
                grid_mode=heatmap_grid_mode,
                grid_size=settings.get("gradient_grid_size"),
                k_nearest=settings.get("gradient_k_nearest"),
            )
            if value_map:
                _, _, value_grid = value_map
                return value_grid[np.isfinite(value_grid)]
        return z_values[np.isfinite(z_values)]

    gradient_grid = _build_gradient_map(
        x_data,
        y_data,
        z_values,
        gradient_mode,
        grid_mode=gradient_calc_grid_mode,
        grid_size=settings.get("gradient_grid_size"),
        k_nearest=settings.get("gradient_k_nearest"),
        polar_center=(settings.get("mappattern_x"), settings.get("mappattern_y")),
    )
    if gradient_grid is None:
        return z_values[np.isfinite(z_values)]

    return gradient_grid[np.isfinite(gradient_grid)]


def _calculate_z_min_max(selected_file, uploaded_files, z_key, gradient_mode, settings):
    if not selected_file or not uploaded_files or selected_file not in uploaded_files:
        return None

    file = Ellipsometry.from_path_or_stream(uploaded_files[selected_file])
    z_key = _resolve_z_key(file, z_key)
    if not z_key:
        return None

    active_settings = {**DEFAULT_SETTINGS, **(settings or {})}
    active_values = _extract_active_plot_values(file, z_key, gradient_mode, active_settings)
    if active_values.size == 0:
        return None

    zmin = float(np.nanmin(active_values))
    zmax = float(np.nanmax(active_values))

    if not np.isfinite(zmin) or not np.isfinite(zmax):
        return None

    return zmin, zmax


@callback(
    Output(ids.Input.Z_SCALE_MIN, "value", allow_duplicate=True),
    Output(ids.Input.Z_SCALE_MAX, "value", allow_duplicate=True),
    Input(ids.Button.Z_SCALE_2SIGMA, "n_clicks"),
    State(ids.DropDown.Z_DATA, "value"),
    State(ids.RadioItems.RENDER_MODE, "value"),
    State(ids.DropDown.GRADIENT_MODE, "value"),
    State(ids.RadioItems.GRADIENT_COORDINATE_MODE, "value"),
    State(ids.RadioItems.GRADIENT_GRID_MODE, "value"),
    State(ids.Slider.GRADIENT_GRID_SIZE, "value"),
    State(ids.Slider.GRADIENT_K_NEAREST, "value"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    State(ids.Store.SETTINGS, "data"),
    prevent_initial_call=True,
)
def set_z_scale_2sigma(
    n_clicks,
    z_key,
    render_mode,
    gradient_mode,
    gradient_coordinate_mode,
    gradient_grid_mode,
    gradient_grid_size,
    gradient_k_nearest,
    selected_file,
    uploaded_files,
    settings,
):
    if not n_clicks or not selected_file:
        return no_update, no_update

    if not uploaded_files or selected_file not in uploaded_files:
        return no_update, no_update

    settings = _merge_runtime_plot_settings(
        settings,
        render_mode=render_mode,
        gradient_mode=gradient_mode,
        gradient_coordinate_mode=gradient_coordinate_mode,
        heatmap_grid_mode=gradient_grid_mode,
        gradient_grid_size=gradient_grid_size,
        gradient_k_nearest=gradient_k_nearest,
    )
    file = Ellipsometry.from_path_or_stream(uploaded_files[selected_file])
    active_z_key = z_key or settings.get("z_data_value")
    active_z_key = _resolve_z_key(file, active_z_key)
    if not active_z_key:
        return no_update, no_update

    active_gradient_mode = gradient_mode or settings.get("gradient_mode", "none")
    active_values = _extract_active_plot_values(file, active_z_key, active_gradient_mode, settings)
    if active_values.size == 0:
        return no_update, no_update

    median = float(np.nanmedian(active_values))
    sigma = float(np.nanstd(active_values))
    zmin = float(np.nanmin(active_values))
    zmax = float(np.nanmax(active_values))

    if not np.isfinite(median) or not np.isfinite(sigma) or not np.isfinite(zmin) or not np.isfinite(zmax):
        return no_update, no_update

    lower = max(zmin, median - 2 * sigma)
    upper = min(zmax, median + 2 * sigma)

    if lower > upper:
        return zmin, zmax

    return lower, upper


@callback(
    Output(ids.Input.Z_SCALE_MIN, "value", allow_duplicate=True),
    Output(ids.Input.Z_SCALE_MAX, "value", allow_duplicate=True),
    Input(ids.Button.Z_SCALE_AUTO, "n_clicks"),
    State(ids.DropDown.Z_DATA, "value"),
    State(ids.RadioItems.RENDER_MODE, "value"),
    State(ids.DropDown.GRADIENT_MODE, "value"),
    State(ids.RadioItems.GRADIENT_COORDINATE_MODE, "value"),
    State(ids.RadioItems.GRADIENT_GRID_MODE, "value"),
    State(ids.Slider.GRADIENT_GRID_SIZE, "value"),
    State(ids.Slider.GRADIENT_K_NEAREST, "value"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    State(ids.Store.SETTINGS, "data"),
    prevent_initial_call=True,
)
def set_z_scale_auto(
    n_clicks,
    z_key,
    render_mode,
    gradient_mode,
    gradient_coordinate_mode,
    gradient_grid_mode,
    gradient_grid_size,
    gradient_k_nearest,
    selected_file,
    uploaded_files,
    settings,
):
    if not n_clicks:
        return no_update, no_update

    settings = _merge_runtime_plot_settings(
        settings,
        render_mode=render_mode,
        gradient_mode=gradient_mode,
        gradient_coordinate_mode=gradient_coordinate_mode,
        heatmap_grid_mode=gradient_grid_mode,
        gradient_grid_size=gradient_grid_size,
        gradient_k_nearest=gradient_k_nearest,
    )
    active_gradient_mode = gradient_mode or settings.get("gradient_mode", "none")
    result = _calculate_z_min_max(selected_file, uploaded_files, z_key, active_gradient_mode, settings)
    if not result:
        return no_update, no_update

    return result


@callback(
    Output(ids.Input.Z_SCALE_MIN, "value", allow_duplicate=True),
    Output(ids.Input.Z_SCALE_MAX, "value", allow_duplicate=True),
    Input(ids.DropDown.Z_DATA, "value"),
    Input(ids.RadioItems.RENDER_MODE, "value"),
    Input(ids.DropDown.GRADIENT_MODE, "value"),
    Input(ids.RadioItems.GRADIENT_COORDINATE_MODE, "value"),
    Input(ids.RadioItems.GRADIENT_GRID_MODE, "value"),
    Input(ids.Slider.GRADIENT_GRID_SIZE, "value"),
    Input(ids.Slider.GRADIENT_K_NEAREST, "value"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    State(ids.Store.SETTINGS, "data"),
    prevent_initial_call=True,
)
def update_z_scale_on_plot_selection_change(
    z_key,
    render_mode,
    gradient_mode,
    gradient_coordinate_mode,
    gradient_grid_mode,
    gradient_grid_size,
    gradient_k_nearest,
    selected_file,
    uploaded_files,
    settings,
):
    settings = _merge_runtime_plot_settings(
        settings,
        render_mode=render_mode,
        gradient_mode=gradient_mode,
        gradient_coordinate_mode=gradient_coordinate_mode,
        heatmap_grid_mode=gradient_grid_mode,
        gradient_grid_size=gradient_grid_size,
        gradient_k_nearest=gradient_k_nearest,
    )
    active_gradient_mode = gradient_mode or settings.get("gradient_mode", "none")
    result = _calculate_z_min_max(selected_file, uploaded_files, z_key, active_gradient_mode, settings)
    if not result:
        return no_update, no_update

    return result
