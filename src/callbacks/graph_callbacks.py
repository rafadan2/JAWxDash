# Library imports
from dash import callback, Output, Input, State, no_update
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import logging
import os 


# Local imports
from src import ids
from src.ellipsometry_toolbox.ellipsometry import Ellipsometry
from src.utils.sample_outlines import generate_outline
from src.ellipsometry_toolbox.masking import radial_edge_exclusion_outline, uniform_edge_exclusion_outline
from src.utils.utilities import gen_spot
from src.ellipsometry_toolbox.linear_translations import rotate, translate
from src.utils.dxf import dxf_to_path
from src.templates.settings_template import DEFAULT_SETTINGS


from src.templates.graph_template import FIGURE_LAYOUT

logger = logging.getLogger(__name__)


CRITICAL_COUNT = 500
GRID_SIZE_MIN = 24
GRID_SIZE_MAX = 180
GRID_SIZE_AUTO_MIN = 40
GRID_SIZE_AUTO_MAX = 160
GRID_SIZE_AUTO_MAX_DENSE = 100
K_NEAREST_MIN = 4
K_NEAREST_MAX = 64


# Adding a placeholder for sample outline
# 8in wafer
r = 1*2.54


def _resolve_z_key(file, z_key):
    if z_key and z_key in file.data:
        return z_key

    columns = sorted(file.get_column_names())
    if len(columns) > 1:
        return columns[1]
    if columns:
        return columns[0]

    return None


def _gradient_label(gradient_mode, z_label):
    labels = {
        "magnitude": f"|grad({z_label})|",
        "dx": f"d({z_label})/dX",
        "dy": f"d({z_label})/dY",
        "laplacian": f"d2({z_label})/dX2 + d2({z_label})/dY2",
    }
    return labels.get(gradient_mode, f"|grad({z_label})|")


def _interpolate_idw_grid(x_data, y_data, z_data, grid_mode="auto", grid_size=None, k_nearest=None):
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

    if grid_mode == "manual":
        try:
            resolved_grid_size = int(grid_size)
        except (TypeError, ValueError):
            resolved_grid_size = DEFAULT_SETTINGS["gradient_grid_size"]
        resolved_grid_size = int(np.clip(resolved_grid_size, GRID_SIZE_MIN, GRID_SIZE_MAX))
    else:
        resolved_grid_size = int(np.clip(np.sqrt(point_count) * 4, GRID_SIZE_AUTO_MIN, GRID_SIZE_AUTO_MAX))
        if point_count > 3 * CRITICAL_COUNT:
            resolved_grid_size = min(resolved_grid_size, GRID_SIZE_AUTO_MAX_DENSE)

    try:
        resolved_k_nearest = int(k_nearest)
    except (TypeError, ValueError):
        resolved_k_nearest = DEFAULT_SETTINGS["gradient_k_nearest"]
    resolved_k_nearest = int(np.clip(resolved_k_nearest, K_NEAREST_MIN, K_NEAREST_MAX))
    k_eff = min(resolved_k_nearest, point_count)

    xi = np.linspace(x_min, x_max, resolved_grid_size)
    yi = np.linspace(y_min, y_max, resolved_grid_size)
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

        if k_eff < point_count:
            neighbor_idx = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
            selected_dist2 = np.take_along_axis(dist2, neighbor_idx, axis=1)
            selected_z = z[neighbor_idx]
        else:
            selected_dist2 = dist2
            selected_z = z[None, :]

        nearest_idx = np.argmin(selected_dist2, axis=1)
        row_idx = np.arange(selected_dist2.shape[0])
        closest_dist2 = selected_dist2[row_idx, nearest_idx]
        nearest_dist[start:stop] = np.sqrt(closest_dist2)

        exact_mask = closest_dist2 < eps
        chunk_values = np.empty(selected_dist2.shape[0], dtype=float)
        if np.any(exact_mask):
            chunk_values[exact_mask] = selected_z[row_idx[exact_mask], nearest_idx[exact_mask]]

        non_exact_mask = ~exact_mask
        if np.any(non_exact_mask):
            non_exact_dist2 = selected_dist2[non_exact_mask]
            non_exact_z = selected_z[non_exact_mask]
            weights = 1.0 / (non_exact_dist2 + eps)
            chunk_values[non_exact_mask] = (
                np.sum(weights * non_exact_z, axis=1) / np.sum(weights, axis=1)
            )

        z_interp[start:stop] = chunk_values

    point_area = (x_span * y_span) / point_count
    spacing_estimate = np.sqrt(point_area) if point_area > 0 else 0.0
    mask_threshold = max(3.0 * spacing_estimate, 1e-6)
    z_interp[nearest_dist > mask_threshold] = np.nan

    return xi, yi, z_interp.reshape(resolved_grid_size, resolved_grid_size)


def _build_gradient_map(x_data, y_data, z_data, gradient_mode, grid_mode="auto", grid_size=None, k_nearest=None):
    interpolated = _interpolate_idw_grid(
        x_data,
        y_data,
        z_data,
        grid_mode=grid_mode,
        grid_size=grid_size,
        k_nearest=k_nearest,
    )
    if not interpolated:
        return None

    xi, yi, z_grid = interpolated
    dz_dy, dz_dx = np.gradient(z_grid, yi, xi, edge_order=1)

    if gradient_mode == "dx":
        gradient_grid = dz_dx
    elif gradient_mode == "dy":
        gradient_grid = dz_dy
    elif gradient_mode == "laplacian":
        d2z_dy2 = np.gradient(dz_dy, yi, axis=0, edge_order=1)
        d2z_dx2 = np.gradient(dz_dx, xi, axis=1, edge_order=1)
        gradient_grid = d2z_dx2 + d2z_dy2
    else:
        gradient_grid = np.hypot(dz_dx, dz_dy)

    return xi, yi, gradient_grid

@callback(
        Output(ids.Graph.MAIN, "figure"),
        Output(ids.DropDown.Z_DATA, "options"),
        Input(ids.DropDown.UPLOADED_FILES, "value"),
        State(ids.Store.UPLOADED_FILES, "data"),
        Input(ids.Store.SETTINGS, "data")
)
def update_figure(selected_file:str, uploaded_files:dict, settings:dict):
    """
    Updates the figure, in accordance with:
    - Selected file
    - Spot style
        - Shape [point/ellipse]
        - Size [focus probe on/off]
        - Angle of incident
        - Colormap
    - Sample outline
    - Data 'channel'
    - MapPattern offset
        - x
        - y
        - theta
    - Sample offset
        - x
        - y
        - theta
    """
    
    # Setting up an empty figure
    figure = go.Figure(
        layout=go.Layout(
            FIGURE_LAYOUT  # Joining the template with the updates
        ),
    )
    settings = {**DEFAULT_SETTINGS, **(settings or {})}

    # If no file or z-data-value selected, return an empty figure
    if not selected_file:
        return no_update, no_update
    
    if not uploaded_files or selected_file not in uploaded_files:
        return figure, no_update
    
    # A sample has been selected, now let's unpack
    file = Ellipsometry.from_path_or_stream(uploaded_files[selected_file])

    # Resolve selected z-data for active file.
    z_label = _resolve_z_key(file, settings.get("z_data_value"))
    if not z_label:
        return figure, sorted(file.get_column_names())

    
    # exposing x,y,z data directly
    x_data = np.array(file.data["x"])
    y_data = np.array(file.data["y"])

    xy = rotate(np.vstack([x_data, y_data]), settings["mappattern_theta"])
    xy = translate(xy, [settings["mappattern_x"], settings["mappattern_y"]])
    x_data = xy[0,:]
    y_data = xy[1,:]
    
    z_data = file.data[z_label].to_numpy()

    gradient_mode = settings.get("gradient_mode", "none")
    gradient_grid_mode = settings.get("gradient_grid_mode", "auto")
    gradient_grid_size = settings.get("gradient_grid_size")
    gradient_k_nearest = settings.get("gradient_k_nearest")
    use_gradient_map = gradient_mode != "none"
    gradient_result = (
        _build_gradient_map(
            x_data,
            y_data,
            z_data,
            gradient_mode,
            grid_mode=gradient_grid_mode,
            grid_size=gradient_grid_size,
            k_nearest=gradient_k_nearest,
        )
        if use_gradient_map
        else None
    )

    if use_gradient_map and gradient_result:
        xi, yi, gradient_grid = gradient_result
        active_data = gradient_grid[np.isfinite(gradient_grid)]
        colorbar_label = _gradient_label(gradient_mode, z_label)
    else:
        use_gradient_map = False
        active_data = z_data[np.isfinite(z_data)]
        colorbar_label = z_label
    
    colorbar_title = dict(title=dict(text=colorbar_label, side="top"))
    
    z_scale_min = settings.get("z_scale_min")
    z_scale_max = settings.get("z_scale_max")
    if active_data.size:
        auto_min = float(np.nanmin(active_data))
        auto_max = float(np.nanmax(active_data))
    else:
        auto_min = np.nan
        auto_max = np.nan
    zmin = auto_min if z_scale_min is None else z_scale_min
    zmax = auto_max if z_scale_max is None else z_scale_max
    manual_scale = (z_scale_min is not None) or (z_scale_max is not None)

    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin >= zmax:
        zmin, zmax = auto_min, auto_max
        manual_scale = False


    # List for holding 'shapes'
    shapes = []

    if use_gradient_map and gradient_result:
        xi, yi, gradient_grid = gradient_result
        gradient_trace = dict(
            x=xi,
            y=yi,
            z=gradient_grid,
            colorscale=settings["colormap_value"],
            colorbar=colorbar_title,
            hovertemplate=f"x: %{{x}}<br>y: %{{y}}<br>{colorbar_label}: %{{z}}<extra></extra>",
            showscale=True,
            connectgaps=False,
        )
        if manual_scale:
            gradient_trace.update(zmin=zmin, zmax=zmax, zauto=False)

        figure.add_trace(go.Heatmap(**gradient_trace))
    else:
        is_ellipse = settings["marker_type"] == "ellipse"

        marker = dict(
            size=settings["marker_size"],
            opacity=0 if is_ellipse else 1,
            color=z_data,
            colorscale=settings["colormap_value"],  # set the colormap
            colorbar=colorbar_title,  # show selected Z-data name above color scale
            showscale=True  # show the color scale
        )
        if manual_scale:
            marker.update(cmin=zmin, cmax=zmax, cauto=False)

        primary_trace = dict(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=marker,
            hovertemplate="x: %{x}<br>y: %{y}<br>z: %{marker.color}<extra></extra>",
            showlegend=False,
        )

        figure.add_trace(go.Scatter(**primary_trace))


        if is_ellipse:
            # Making colors
            d_min, d_max = zmin, zmax

            if d_max > d_min:
                norm_zdata = (z_data - d_min) / (d_max - d_min)
                norm_zdata = np.clip(norm_zdata, 0, 1)
            else:
                norm_zdata = np.zeros_like(z_data, dtype=float)
            colors = px.colors.sample_colorscale(colorscale=settings["colormap_value"], samplepoints=norm_zdata)
            
            shapes.extend([gen_spot(x, y, c, settings["spot_size"], settings["angle_of_incident"]) for x, y, c in zip(x_data, y_data, colors)])
    

    # Adds outline if outline is selected
    if settings["sample_outline"]:
        # add sample outline to 'shapes'
        
        shapes.append(generate_outline(settings))
    

    # Add stage outline
    if settings["stage_state"]:
        CWD = os.getcwd()
        DXF_FILE = os.path.join(CWD, "src/assets/jaw_stage_outline.dxf")
        
        #dxf_filepath = r"src\assets\JAW stage outline.dxf"
        
        stage_outline = dxf_to_path(DXF_FILE)
        shapes.extend(stage_outline)
    
    # Add edge exclusion outline if selected
    if settings["sample_outline"] and settings["ee_state"]:

        ee = []
        if settings["ee_type"] == "radial":
            ee.append(radial_edge_exclusion_outline(settings))
        elif settings["ee_type"] == "uniform":
            ee.append(uniform_edge_exclusion_outline(settings))

        shapes.extend(ee)
    

    # Calculate 'zoom-window'
    xmin, xmax = min(x_data), max(x_data)
    ymin, ymax = min(y_data), max(y_data)
    scale_factor = 0.2
    scale_range = xmax - xmin

    #Adding shapes to the figure
    figure.update_layout(
        shapes=shapes,
        xaxis=dict(range=[xmin - scale_factor*scale_range, xmax + scale_factor*scale_range]),
        yaxis=dict(range=[ymin - scale_factor*scale_range, ymax + scale_factor*scale_range]),
    )


    return figure, sorted(file.get_column_names())
