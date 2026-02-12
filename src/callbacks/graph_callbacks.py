# Library imports
from dash import callback, Output, Input, State, clientside_callback, no_update
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import logging
import os 
import json
import re
import time


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
EXPORT_IMAGE_WIDTH = 1400
EXPORT_IMAGE_HEIGHT = 1000
EXPORT_IMAGE_SCALE = 1


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


def _resolve_z_limits(active_data, z_scale_min, z_scale_max, force_two_sigma=False):
    if active_data.size:
        auto_min = float(np.nanmin(active_data))
        auto_max = float(np.nanmax(active_data))
    else:
        auto_min = np.nan
        auto_max = np.nan

    if force_two_sigma:
        median = float(np.nanmedian(active_data)) if active_data.size else np.nan
        sigma = float(np.nanstd(active_data)) if active_data.size else np.nan
        if np.isfinite(median) and np.isfinite(sigma) and np.isfinite(auto_min) and np.isfinite(auto_max):
            lower = max(auto_min, median - 2 * sigma)
            upper = min(auto_max, median + 2 * sigma)
            if lower <= upper:
                return lower, upper, True

        return auto_min, auto_max, np.isfinite(auto_min) and np.isfinite(auto_max) and auto_min < auto_max

    zmin = auto_min if z_scale_min is None else z_scale_min
    zmax = auto_max if z_scale_max is None else z_scale_max
    manual_scale = (z_scale_min is not None) or (z_scale_max is not None)

    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin >= zmax:
        return auto_min, auto_max, False

    return zmin, zmax, manual_scale


def _clone_shapes(shapes):
    return [shape.copy() for shape in shapes]


def _resolve_batch_z_keys(file):
    return [column for column in sorted(file.get_column_names()) if column.lower() not in {"x", "y"}]


def _safe_filename_fragment(value):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "plot"


def _build_main_figure(file, settings, z_label=None, force_two_sigma=False, stage_outline=None):
    figure = go.Figure(
        layout=go.Layout(
            FIGURE_LAYOUT
        ),
    )
    settings = {**DEFAULT_SETTINGS, **(settings or {})}

    z_label = _resolve_z_key(file, z_label or settings.get("z_data_value"))
    if not z_label:
        return figure, None

    x_data = np.array(file.data["x"])
    y_data = np.array(file.data["y"])

    xy = rotate(np.vstack([x_data, y_data]), settings["mappattern_theta"])
    xy = translate(xy, [settings["mappattern_x"], settings["mappattern_y"]])
    x_data = xy[0, :]
    y_data = xy[1, :]

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
        _, _, gradient_grid = gradient_result
        active_data = gradient_grid[np.isfinite(gradient_grid)]
        colorbar_label = _gradient_label(gradient_mode, z_label)
    else:
        use_gradient_map = False
        active_data = z_data[np.isfinite(z_data)]
        colorbar_label = z_label

    zmin, zmax, manual_scale = _resolve_z_limits(
        active_data,
        settings.get("z_scale_min"),
        settings.get("z_scale_max"),
        force_two_sigma=force_two_sigma,
    )

    colorbar_title = dict(title=dict(text=colorbar_label, side="top"))

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
            colorscale=settings["colormap_value"],
            colorbar=colorbar_title,
            showscale=True,
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
            d_min, d_max = zmin, zmax

            if d_max > d_min:
                norm_zdata = (z_data - d_min) / (d_max - d_min)
                norm_zdata = np.clip(norm_zdata, 0, 1)
            else:
                norm_zdata = np.zeros_like(z_data, dtype=float)
            colors = px.colors.sample_colorscale(
                colorscale=settings["colormap_value"],
                samplepoints=norm_zdata
            )

            shapes.extend(
                [gen_spot(x, y, c, settings["spot_size"], settings["angle_of_incident"]) for x, y, c in zip(x_data, y_data, colors)]
            )

    if settings["sample_outline"]:
        shapes.append(generate_outline(settings))

    if settings["stage_state"]:
        if stage_outline is None:
            cwd = os.getcwd()
            dxf_file = os.path.join(cwd, "src/assets/jaw_stage_outline.dxf")
            stage_outline = dxf_to_path(dxf_file)
        shapes.extend(_clone_shapes(stage_outline))

    if settings["sample_outline"] and settings["ee_state"]:
        if settings["ee_type"] == "radial":
            shapes.append(radial_edge_exclusion_outline(settings))
        elif settings["ee_type"] == "uniform":
            shapes.append(uniform_edge_exclusion_outline(settings))

    xmin, xmax = min(x_data), max(x_data)
    ymin, ymax = min(y_data), max(y_data)
    scale_factor = 0.2
    scale_range = xmax - xmin

    figure.update_layout(
        shapes=shapes,
        xaxis=dict(range=[xmin - scale_factor * scale_range, xmax + scale_factor * scale_range]),
        yaxis=dict(range=[ymin - scale_factor * scale_range, ymax + scale_factor * scale_range]),
    )

    return figure, z_label

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
    
    figure = go.Figure(layout=go.Layout(FIGURE_LAYOUT))
    settings = {**DEFAULT_SETTINGS, **(settings or {})}

    if not selected_file:
        return no_update, no_update

    if not uploaded_files or selected_file not in uploaded_files:
        return figure, no_update

    file = Ellipsometry.from_path_or_stream(uploaded_files[selected_file])
    z_options = sorted(file.get_column_names())

    rendered_figure, _ = _build_main_figure(file, settings, z_label=settings.get("z_data_value"))

    return rendered_figure, z_options


@callback(
    Output(ids.Store.BATCH_MAIN_PLOTS_PAYLOAD, "data"),
    Input(ids.Button.BATCH_MAIN_PLOTS, "n_clicks"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    State(ids.Store.SETTINGS, "data"),
    prevent_initial_call=True,
)
def build_batch_main_plot_payload(n_clicks, selected_file, uploaded_files, settings):
    if not n_clicks:
        return no_update

    if not selected_file or not uploaded_files or selected_file not in uploaded_files:
        logger.warning(
            "Batch main plot payload skipped: selected_file=%s, uploaded_files_present=%s",
            selected_file,
            bool(uploaded_files),
        )
        return no_update

    callback_t0 = time.perf_counter()
    settings = {**DEFAULT_SETTINGS, **(settings or {})}

    logger.info(
        "Batch main plot payload started for file='%s' (n_clicks=%s, gradient_mode=%s, marker_type=%s).",
        selected_file,
        n_clicks,
        settings.get("gradient_mode"),
        settings.get("marker_type"),
    )

    load_t0 = time.perf_counter()
    file = Ellipsometry.from_path_or_stream(uploaded_files[selected_file])
    load_dt = time.perf_counter() - load_t0

    point_count = len(file.data.index)
    z_keys = _resolve_batch_z_keys(file)
    if not z_keys:
        logger.warning("Batch main plot payload skipped: no z-parameter columns found in '%s'.", selected_file)
        return no_update

    logger.info(
        "Batch main plot payload: loaded %d points, %d z-parameters in %.3fs.",
        point_count,
        len(z_keys),
        load_dt,
    )

    stage_outline = None
    if settings.get("stage_state"):
        cwd = os.getcwd()
        dxf_file = os.path.join(cwd, "src/assets/jaw_stage_outline.dxf")
        stage_outline = dxf_to_path(dxf_file)

    root_name = _safe_filename_fragment(os.path.splitext(selected_file)[0])
    payload = []
    build_total = 0.0
    for idx, z_key in enumerate(z_keys, start=1):
        build_t0 = time.perf_counter()
        per_plot_settings = {**settings, "z_data_value": z_key}
        figure, _ = _build_main_figure(
            file,
            per_plot_settings,
            z_label=z_key,
            force_two_sigma=True,
            stage_outline=stage_outline,
        )
        build_dt = time.perf_counter() - build_t0
        build_total += build_dt

        plot_name = _safe_filename_fragment(z_key)
        payload.append(
            {
                "filename": f"{root_name}_{plot_name}.png",
                "figure": json.loads(figure.to_json()),
            }
        )

        logger.info(
            "Batch main plot payload [%d/%d] z='%s': figure prepared in %.3fs.",
            idx,
            len(z_keys),
            z_key,
            build_dt,
        )

    callback_dt = time.perf_counter() - callback_t0
    logger.info(
        "Batch main plot payload completed: plots=%d, build_total=%.3fs, callback_total=%.3fs",
        len(z_keys),
        build_total,
        callback_dt,
    )
    return {
        "request_id": int(time.time() * 1000),
        "plots": payload,
        "width": EXPORT_IMAGE_WIDTH,
        "height": EXPORT_IMAGE_HEIGHT,
        "scale": EXPORT_IMAGE_SCALE,
    }


clientside_callback(
    """
    async function(payload) {
        const noUpdate = window.dash_clientside.no_update;
        if (!payload || !payload.plots || payload.plots.length === 0) {
            return noUpdate;
        }

        if (typeof Plotly === "undefined" || typeof Plotly.downloadImage !== "function") {
            return "Batch export failed: Plotly image download API is unavailable in the browser.";
        }

        const width = payload.width || 1400;
        const height = payload.height || 1000;
        const scale = payload.scale || 1;
        const plots = payload.plots;

        const tempDiv = document.createElement("div");
        tempDiv.style.position = "fixed";
        tempDiv.style.left = "-10000px";
        tempDiv.style.top = "-10000px";
        tempDiv.style.width = `${width}px`;
        tempDiv.style.height = `${height}px`;
        document.body.appendChild(tempDiv);

        const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

        try {
            let plotted = false;
            for (let idx = 0; idx < plots.length; idx++) {
                const plot = plots[idx];
                const figure = plot.figure || {};
                const data = figure.data || [];
                const layout = figure.layout || {};
                const filename = (plot.filename || `batch_plot_${idx + 1}.png`).replace(/\\.png$/i, "");
                console.info(`[Batch main plots] exporting ${idx + 1}/${plots.length}: ${filename}.png`);

                if (!plotted) {
                    await Plotly.newPlot(tempDiv, data, layout, {displayModeBar: false, responsive: false, staticPlot: true});
                    plotted = true;
                } else {
                    await Plotly.react(tempDiv, data, layout, {displayModeBar: false, responsive: false, staticPlot: true});
                }

                await Plotly.downloadImage(tempDiv, {
                    format: "png",
                    filename: filename,
                    width: width,
                    height: height,
                    scale: scale
                });
                console.info(`[Batch main plots] downloaded ${idx + 1}/${plots.length}: ${filename}.png`);

                // Prevent browser download throttling when triggering many files quickly.
                await sleep(120);
            }

            return `Batch main plots downloaded (${plots.length} PNG files).`;
        } catch (error) {
            const message = (error && error.message) ? error.message : String(error);
            console.error("Batch main plot browser export failed:", error);
            return `Batch main plots failed: ${message}`;
        } finally {
            try {
                Plotly.purge(tempDiv);
            } catch (e) {}
            tempDiv.remove();
        }
    }
    """,
    Output(ids.Div.INFO, "children", allow_duplicate=True),
    Input(ids.Store.BATCH_MAIN_PLOTS_PAYLOAD, "data"),
    prevent_initial_call=True,
)
