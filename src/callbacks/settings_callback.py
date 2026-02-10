from dash import callback, Output, Input, State, no_update
import logging
import numpy as np


from src import ids
from src.ellipsometry_toolbox.ellipsometry import Ellipsometry
from src.ellipsometry_toolbox.linear_translations import rotate, translate
from src.templates.settings_template import DEFAULT_SETTINGS

logger = logging.getLogger(__name__)
CRITICAL_COUNT = 500


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
    Output(ids.DropDown.GRADIENT_MODE, "value"),
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
        default_settings["marker_size"],
        
        # Sample
        default_settings["colormap_value"],
        default_settings["colormap_options"],
        default_settings["sample_outline"],
        default_settings["z_data_value"],
        default_settings["gradient_mode"],
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
    Input(ids.Slider.MARKER_SIZE, "value"),

    # Sample
    Input(ids.DropDown.COLORMAPS, "value"),
    Input(ids.DropDown.SAMPLE_OUTLINE, "value"),
    Input(ids.DropDown.Z_DATA, "value"),
    Input(ids.DropDown.GRADIENT_MODE, "value"),
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
    colormap_value, sample_outline, z_data_value, gradient_mode, z_scale_min, z_scale_max,
    x_map, y_map, t_map, 
    x_sam, y_sam, t_sam, r_sam, w_sam, h_sam,
    ee_state, ee_type, ee_distance, batch_processing,
    stage_state,
    settings
    ):
    settings = settings or {}

    keys = (
        "marker_type", 
        "angle_of_incident", 
        "spot_size",
        "marker_size",
        "colormap_value", 
        "sample_outline", 
        "z_data_value",
        "gradient_mode",
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
        colormap_value, sample_outline, z_data_value, gradient_mode,
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


def _interpolate_idw_grid(x_data, y_data, z_data):
    finite_mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(z_data)
    x = np.asarray(x_data)[finite_mask]
    y = np.asarray(y_data)[finite_mask]
    z = np.asarray(z_data)[finite_mask]

    point_count = z.size
    if point_count < 3:
        return None

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0 or y_span <= 0:
        return None

    grid_size = int(np.clip(np.sqrt(point_count) * 4, 40, 160))
    if point_count > 3 * CRITICAL_COUNT:
        grid_size = min(grid_size, 100)

    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xi, yi)

    flat_x = xx.ravel()
    flat_y = yy.ravel()
    z_interp = np.full(flat_x.shape[0], np.nan, dtype=float)
    nearest_dist = np.full(flat_x.shape[0], np.inf, dtype=float)

    eps = 1e-12
    chunk_size = 1024
    for start in range(0, flat_x.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_x.shape[0])

        dx = flat_x[start:stop, None] - x[None, :]
        dy = flat_y[start:stop, None] - y[None, :]
        dist2 = dx * dx + dy * dy

        nearest_idx = np.argmin(dist2, axis=1)
        row_idx = np.arange(dist2.shape[0])
        closest_dist2 = dist2[row_idx, nearest_idx]
        nearest_dist[start:stop] = np.sqrt(closest_dist2)

        exact_mask = closest_dist2 < eps
        chunk_values = np.empty(dist2.shape[0], dtype=float)
        if np.any(exact_mask):
            chunk_values[exact_mask] = z[nearest_idx[exact_mask]]

        non_exact_mask = ~exact_mask
        if np.any(non_exact_mask):
            non_exact_dist2 = dist2[non_exact_mask]
            weights = 1.0 / (non_exact_dist2 + eps)
            chunk_values[non_exact_mask] = (
                np.sum(weights * z[None, :], axis=1) / np.sum(weights, axis=1)
            )

        z_interp[start:stop] = chunk_values

    point_area = (x_span * y_span) / point_count
    spacing_estimate = np.sqrt(point_area) if point_area > 0 else 0.0
    mask_threshold = max(3.0 * spacing_estimate, 1e-6)
    z_interp[nearest_dist > mask_threshold] = np.nan

    return xi, yi, z_interp.reshape(grid_size, grid_size)


def _build_gradient_map(x_data, y_data, z_data, gradient_mode):
    interpolated = _interpolate_idw_grid(x_data, y_data, z_data)
    if not interpolated:
        return None

    xi, yi, z_grid = interpolated
    dz_dy, dz_dx = np.gradient(z_grid, yi, xi, edge_order=1)

    if gradient_mode == "dx":
        gradient_grid = dz_dx
    elif gradient_mode == "dy":
        gradient_grid = dz_dy
    else:
        gradient_grid = np.hypot(dz_dx, dz_dy)

    return gradient_grid


def _extract_active_plot_values(file, z_key, gradient_mode, settings):
    z_values = file.data[z_key].to_numpy()

    if gradient_mode == "none":
        return z_values[np.isfinite(z_values)]

    x_data = np.array(file.data["x"])
    y_data = np.array(file.data["y"])
    xy = rotate(np.vstack([x_data, y_data]), settings.get("mappattern_theta", 0.0))
    xy = translate(xy, [settings.get("mappattern_x", 0.0), settings.get("mappattern_y", 0.0)])
    x_data = xy[0, :]
    y_data = xy[1, :]

    gradient_grid = _build_gradient_map(x_data, y_data, z_values, gradient_mode)
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
    State(ids.DropDown.GRADIENT_MODE, "value"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    State(ids.Store.SETTINGS, "data"),
    prevent_initial_call=True,
)
def set_z_scale_2sigma(n_clicks, z_key, gradient_mode, selected_file, uploaded_files, settings):
    if not n_clicks or not selected_file:
        return no_update, no_update

    if not uploaded_files or selected_file not in uploaded_files:
        return no_update, no_update

    settings = {**DEFAULT_SETTINGS, **(settings or {})}
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

    if not np.isfinite(median) or not np.isfinite(sigma):
        return no_update, no_update

    return median - 2 * sigma, median + 2 * sigma


@callback(
    Output(ids.Input.Z_SCALE_MIN, "value", allow_duplicate=True),
    Output(ids.Input.Z_SCALE_MAX, "value", allow_duplicate=True),
    Input(ids.Button.Z_SCALE_AUTO, "n_clicks"),
    State(ids.DropDown.Z_DATA, "value"),
    State(ids.DropDown.GRADIENT_MODE, "value"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    State(ids.Store.SETTINGS, "data"),
    prevent_initial_call=True,
)
def set_z_scale_auto(n_clicks, z_key, gradient_mode, selected_file, uploaded_files, settings):
    if not n_clicks:
        return no_update, no_update

    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    active_gradient_mode = gradient_mode or settings.get("gradient_mode", "none")
    result = _calculate_z_min_max(selected_file, uploaded_files, z_key, active_gradient_mode, settings)
    if not result:
        return no_update, no_update

    return result


@callback(
    Output(ids.Input.Z_SCALE_MIN, "value", allow_duplicate=True),
    Output(ids.Input.Z_SCALE_MAX, "value", allow_duplicate=True),
    Input(ids.DropDown.Z_DATA, "value"),
    Input(ids.DropDown.GRADIENT_MODE, "value"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    State(ids.Store.SETTINGS, "data"),
    prevent_initial_call=True,
)
def update_z_scale_on_plot_selection_change(z_key, gradient_mode, selected_file, uploaded_files, settings):
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    active_gradient_mode = gradient_mode or settings.get("gradient_mode", "none")
    result = _calculate_z_min_max(selected_file, uploaded_files, z_key, active_gradient_mode, settings)
    if not result:
        return no_update, no_update

    return result
