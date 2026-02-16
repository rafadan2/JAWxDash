from dash import callback, html, no_update, Output, Input, State
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


from src import ids
from src.callbacks.graph_callbacks import (
    _build_gradient_map,
    _gradient_label,
    _resolve_gradient_calc_grid_mode,
    _resolve_z_key,
)
from src.ellipsometry_toolbox.ellipsometry import Ellipsometry
from src.ellipsometry_toolbox.linear_translations import rotate, translate
from src.ellipsometry_toolbox.masking import create_masked_file
from src.templates.settings_template import DEFAULT_SETTINGS


def _empty_figure(title, x_title="", y_title=""):
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor="center", pad=dict(t=10, b=8)),
        margin=dict(l=50, r=30, t=90, b=80),
    )
    if x_title:
        figure.update_xaxes(title=x_title, automargin=True)
    if y_title:
        figure.update_yaxes(title=y_title, automargin=True)
    figure.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text="No data available",
        showarrow=False,
        font=dict(color="gray"),
    )
    return figure


def _safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _linear_fit(x_values, y_values):
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x = np.asarray(x_values)[finite_mask]
    y = np.asarray(y_values)[finite_mask]
    if x.size < 2 or np.isclose(np.nanstd(x), 0.0):
        return None

    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = slope * x + intercept

    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot > 0:
        r_squared = 1.0 - (ss_res / ss_tot)
    else:
        r_squared = np.nan

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r_squared),
        "prediction": prediction,
    }


def _binned_stats(x_values, y_values, n_bins):
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x = np.asarray(x_values)[finite_mask]
    y = np.asarray(y_values)[finite_mask]
    if x.size < 2:
        return None

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if np.isclose(x_min, x_max):
        return None

    edges = np.linspace(x_min, x_max, int(n_bins) + 1)
    indices = np.digitize(x, edges, right=False) - 1
    indices[indices == n_bins] = n_bins - 1

    centers = []
    means = []
    stds = []
    counts = []
    for bin_idx in range(int(n_bins)):
        bin_mask = indices == bin_idx
        if not np.any(bin_mask):
            continue
        y_bin = y[bin_mask]
        centers.append(0.5 * (edges[bin_idx] + edges[bin_idx + 1]))
        means.append(float(np.mean(y_bin)))
        stds.append(float(np.std(y_bin, ddof=1)) if y_bin.size > 1 else 0.0)
        counts.append(int(y_bin.size))

    if not centers:
        return None

    return {
        "x": np.asarray(centers),
        "mean": np.asarray(means),
        "std": np.asarray(stds),
        "count": np.asarray(counts),
    }


def _prepare_active_series(file, settings):
    active_settings = {**DEFAULT_SETTINGS, **(settings or {})}

    if active_settings.get("ee_state"):
        file = create_masked_file(file, active_settings)
    if file.data.empty:
        return None

    z_key = _resolve_z_key(file, active_settings.get("z_data_value"))
    if not z_key:
        return None

    x_data = np.array(file.data["x"])
    y_data = np.array(file.data["y"])
    z_data = file.data[z_key].to_numpy()

    xy = rotate(np.vstack([x_data, y_data]), active_settings["mappattern_theta"])
    xy = translate(xy, [active_settings["mappattern_x"], active_settings["mappattern_y"]])
    x_data = xy[0, :]
    y_data = xy[1, :]

    gradient_mode = active_settings.get("gradient_mode", "none")
    if gradient_mode == "none":
        finite_mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(z_data)
        return x_data[finite_mask], y_data[finite_mask], z_data[finite_mask], z_key, active_settings

    gradient_result = _build_gradient_map(
        x_data,
        y_data,
        z_data,
        gradient_mode,
        grid_mode=_resolve_gradient_calc_grid_mode(active_settings),
        grid_size=active_settings.get("gradient_grid_size"),
        k_nearest=active_settings.get("gradient_k_nearest"),
        polar_center=(active_settings.get("mappattern_x"), active_settings.get("mappattern_y")),
    )
    if not gradient_result:
        finite_mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(z_data)
        return x_data[finite_mask], y_data[finite_mask], z_data[finite_mask], z_key, active_settings

    xi, yi, value_grid = gradient_result
    xx, yy = np.meshgrid(xi, yi)
    flat_x = xx.ravel()
    flat_y = yy.ravel()
    flat_values = value_grid.ravel()
    finite_mask = np.isfinite(flat_x) & np.isfinite(flat_y) & np.isfinite(flat_values)

    return (
        flat_x[finite_mask],
        flat_y[finite_mask],
        flat_values[finite_mask],
        _gradient_label(gradient_mode, z_key),
        active_settings,
    )


def _fit_line_trace(x_values, fit_result):
    x_line = np.asarray([np.min(x_values), np.max(x_values)])
    y_line = fit_result["slope"] * x_line + fit_result["intercept"]
    return x_line, y_line


def _sample_grid_at_points(x_points, y_points, xi, yi, value_grid):
    x = np.asarray(x_points)
    y = np.asarray(y_points)
    xi = np.asarray(xi)
    yi = np.asarray(yi)
    grid = np.asarray(value_grid)

    if x.size == 0:
        return np.asarray([], dtype=float)
    if xi.size < 2 or yi.size < 2 or grid.ndim != 2:
        return np.full(x.shape, np.nan, dtype=float)

    x_idx = np.searchsorted(xi, x, side="left")
    x_idx = np.clip(x_idx, 1, xi.size - 1)
    x_left = xi[x_idx - 1]
    x_right = xi[x_idx]
    use_right_x = np.abs(x - x_right) < np.abs(x - x_left)
    x_idx = np.where(use_right_x, x_idx, x_idx - 1)

    y_idx = np.searchsorted(yi, y, side="left")
    y_idx = np.clip(y_idx, 1, yi.size - 1)
    y_left = yi[y_idx - 1]
    y_right = yi[y_idx]
    use_right_y = np.abs(y - y_right) < np.abs(y - y_left)
    y_idx = np.where(use_right_y, y_idx, y_idx - 1)

    return grid[y_idx, x_idx]


@callback(
    Output(ids.Graph.DISTRIBUTION_XY, "figure"),
    Output(ids.Graph.DISTRIBUTION_RESIDUALS, "figure"),
    Output(ids.Graph.DISTRIBUTION_RADIAL, "figure"),
    Output(ids.Div.DISTRIBUTION_METRICS, "children"),
    Input(ids.Tabs.ANALYSIS, "active_tab"),
    Input(ids.DropDown.UPLOADED_FILES, "value"),
    Input(ids.Store.SETTINGS, "data"),
    State(ids.Store.UPLOADED_FILES, "data"),
)
def update_distribution_tab(active_tab, selected_file, settings, stored_files):
    if active_tab != "distribution":
        return no_update, no_update, no_update, no_update

    empty_xy = _empty_figure("Value vs Axis", "Axis (mm)", "Value")
    empty_res = _empty_figure("Residual Distributions", "Residual", "Count")
    empty_rad = _empty_figure("Radial Profile", "Distance from center (mm)", "Value")

    if not selected_file or not stored_files or selected_file not in stored_files:
        return (
            empty_xy,
            empty_res,
            empty_rad,
            html.Div("Select a file to inspect distribution trends.", className="text-muted"),
        )

    file = Ellipsometry.from_path_or_stream(stored_files[selected_file])
    prepared = _prepare_active_series(file, settings)
    if not prepared:
        return (
            empty_xy,
            empty_res,
            empty_rad,
            html.Div("No valid data for distribution analysis.", className="text-muted"),
        )

    x_coord, y_coord, values, value_label, active_settings = prepared
    if values.size == 0:
        return (
            empty_xy,
            empty_res,
            empty_rad,
            html.Div("No finite values for distribution analysis.", className="text-muted"),
        )

    x_mm = x_coord * 10.0
    y_mm = y_coord * 10.0

    fit_x = _linear_fit(x_mm, values)
    fit_y = _linear_fit(y_mm, values)

    axis_bins = int(np.clip(np.sqrt(values.size), 10, 40))
    binned_x = _binned_stats(x_mm, values, axis_bins)
    binned_y = _binned_stats(y_mm, values, axis_bins)

    fig_xy = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"{value_label} vs X", f"{value_label} vs Y"),
        horizontal_spacing=0.1,
    )
    fig_xy.add_trace(
        go.Scatter(
            x=x_mm,
            y=values,
            mode="markers",
            marker=dict(size=5, opacity=0.35),
            name="Raw points (X)",
        ),
        row=1,
        col=1,
    )
    fig_xy.add_trace(
        go.Scatter(
            x=y_mm,
            y=values,
            mode="markers",
            marker=dict(size=5, opacity=0.35),
            name="Raw points (Y)",
        ),
        row=1,
        col=2,
    )

    if binned_x:
        fig_xy.add_trace(
            go.Scatter(
                x=binned_x["x"],
                y=binned_x["mean"],
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=6),
                name=f"Binned mean (X, {axis_bins})",
            ),
            row=1,
            col=1,
        )
    if binned_y:
        fig_xy.add_trace(
            go.Scatter(
                x=binned_y["x"],
                y=binned_y["mean"],
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=6),
                name=f"Binned mean (Y, {axis_bins})",
            ),
            row=1,
            col=2,
        )

    if fit_x:
        x_line, y_line = _fit_line_trace(x_mm, fit_x)
        fig_xy.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(width=2, dash="dash", color="crimson"),
                name="Linear fit (X)",
            ),
            row=1,
            col=1,
        )
    if fit_y:
        x_line, y_line = _fit_line_trace(y_mm, fit_y)
        fig_xy.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(width=2, dash="dash", color="crimson"),
                name="Linear fit (Y)",
            ),
            row=1,
            col=2,
        )

    fig_xy.update_xaxes(title_text="X (mm)", row=1, col=1)
    fig_xy.update_xaxes(title_text="Y (mm)", row=1, col=2)
    fig_xy.update_yaxes(title_text=value_label, row=1, col=1)
    fig_xy.update_yaxes(title_text=value_label, row=1, col=2)
    fig_xy.update_xaxes(automargin=True, title_standoff=8)
    fig_xy.update_yaxes(automargin=True, title_standoff=8)
    fig_xy.update_layout(
        template="plotly_white",
        title=dict(text="Value Trends vs X/Y", x=0.5, xanchor="center", pad=dict(t=10, b=6)),
        margin=dict(l=55, r=30, t=120, b=110),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )

    fig_residuals = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Residuals after X trend removal", "Residuals after Y trend removal"),
        horizontal_spacing=0.1,
    )

    if fit_x:
        residual_x = values - fit_x["prediction"]
        fig_residuals.add_trace(
            go.Histogram(x=residual_x, name="r_x", marker_color="#1f77b4", opacity=0.75, nbinsx=40),
            row=1,
            col=1,
        )
        fig_residuals.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=1)
    else:
        fig_residuals.add_annotation(
            x=0.5,
            y=0.5,
            text="Insufficient X spread for fit",
            showarrow=False,
            row=1,
            col=1,
        )

    if fit_y:
        residual_y = values - fit_y["prediction"]
        fig_residuals.add_trace(
            go.Histogram(x=residual_y, name="r_y", marker_color="#ff7f0e", opacity=0.75, nbinsx=40),
            row=1,
            col=2,
        )
        fig_residuals.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=2)
    else:
        fig_residuals.add_annotation(
            x=0.5,
            y=0.5,
            text="Insufficient Y spread for fit",
            showarrow=False,
            row=1,
            col=2,
        )

    fig_residuals.update_xaxes(title_text="Residual", row=1, col=1)
    fig_residuals.update_xaxes(title_text="Residual", row=1, col=2)
    fig_residuals.update_yaxes(title_text="Count", row=1, col=1)
    fig_residuals.update_yaxes(title_text="Count", row=1, col=2)
    fig_residuals.update_xaxes(automargin=True, title_standoff=8)
    fig_residuals.update_yaxes(automargin=True, title_standoff=8)
    fig_residuals.update_layout(
        template="plotly_white",
        title=dict(text="Residual Distributions", x=0.5, xanchor="center", pad=dict(t=10, b=6)),
        margin=dict(l=55, r=30, t=105, b=85),
        showlegend=False,
        bargap=0.05,
    )

    x0_mm = _safe_float(active_settings.get("sample_x"), default=np.nan) * 10.0
    y0_mm = _safe_float(active_settings.get("sample_y"), default=np.nan) * 10.0
    if not np.isfinite(x0_mm):
        x0_mm = float(np.median(x_mm))
    if not np.isfinite(y0_mm):
        y0_mm = float(np.median(y_mm))

    radial_distance = np.sqrt((x_mm - x0_mm) ** 2 + (y_mm - y0_mm) ** 2)
    radial_bins = int(np.clip(np.sqrt(values.size), 8, 40))
    radial_stats = _binned_stats(radial_distance, values, radial_bins)

    fig_radial = go.Figure()
    fig_radial.add_trace(
        go.Scatter(
            x=radial_distance,
            y=values,
            mode="markers",
            marker=dict(size=5, opacity=0.3),
            name="Raw points",
        )
    )
    if radial_stats:
        fig_radial.add_trace(
            go.Scatter(
                x=radial_stats["x"],
                y=radial_stats["mean"],
                mode="lines+markers",
                line=dict(width=2, color="black"),
                marker=dict(size=6),
                error_y=dict(type="data", array=radial_stats["std"], visible=True),
                name=f"Radial mean +/- std ({radial_bins})",
            )
        )

    fig_radial.update_layout(
        template="plotly_white",
        title=dict(text="Value vs radial distance from center", x=0.5, xanchor="center", pad=dict(t=10, b=6)),
        xaxis_title="Distance from center (mm)",
        yaxis_title=value_label,
        margin=dict(l=55, r=30, t=100, b=110),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    fig_radial.update_xaxes(automargin=True, title_standoff=8)
    fig_radial.update_yaxes(automargin=True, title_standoff=8)

    if fit_x:
        slope_x_text = f"{fit_x['slope']:.6g}"
        intercept_x_text = f"{fit_x['intercept']:.6g}"
        r2_x_text = f"{fit_x['r2']:.4f}" if np.isfinite(fit_x["r2"]) else "nan"
    else:
        slope_x_text = "n/a"
        intercept_x_text = "n/a"
        r2_x_text = "n/a"

    if fit_y:
        slope_y_text = f"{fit_y['slope']:.6g}"
        intercept_y_text = f"{fit_y['intercept']:.6g}"
        r2_y_text = f"{fit_y['r2']:.4f}" if np.isfinite(fit_y["r2"]) else "nan"
    else:
        slope_y_text = "n/a"
        intercept_y_text = "n/a"
        r2_y_text = "n/a"

    metrics = html.Div(
        [
            html.Div(html.Strong(f"Distribution metrics for {value_label}")),
            html.Div(f"Points analyzed: {values.size}"),
            html.Div(f"X fit: p = ({slope_x_text})x + ({intercept_x_text}), slope unit: {value_label}/mm, R^2: {r2_x_text}"),
            html.Div(f"Y fit: p = ({slope_y_text})y + ({intercept_y_text}), slope unit: {value_label}/mm, R^2: {r2_y_text}"),
            html.Div(f"Radial center used: ({x0_mm:.3f} mm, {y0_mm:.3f} mm)"),
        ]
    )

    return fig_xy, fig_residuals, fig_radial, metrics


@callback(
    Output(ids.Graph.GRADIENT_VIOLIN, "figure"),
    Input(ids.Tabs.ANALYSIS, "active_tab"),
    Input(ids.DropDown.UPLOADED_FILES, "value"),
    Input(ids.Store.SETTINGS, "data"),
    State(ids.Store.UPLOADED_FILES, "data"),
)
def update_gradient_violin_tab(active_tab, selected_file, settings, stored_files):
    if active_tab != "gradient_violin":
        return no_update

    empty_figure = _empty_figure("Gradient Violin Plots", "|grad(Z)|", "Parameter")
    if not selected_file or not stored_files or selected_file not in stored_files:
        return empty_figure

    file = Ellipsometry.from_path_or_stream(stored_files[selected_file])
    active_settings = {**DEFAULT_SETTINGS, **(settings or {})}
    if active_settings.get("ee_state"):
        file = create_masked_file(file, active_settings)
    if file.data.empty:
        return empty_figure

    if "x" not in file.data or "y" not in file.data:
        return empty_figure

    x_data = np.array(file.data["x"])
    y_data = np.array(file.data["y"])
    xy = rotate(np.vstack([x_data, y_data]), active_settings["mappattern_theta"])
    xy = translate(xy, [active_settings["mappattern_x"], active_settings["mappattern_y"]])
    x_data = xy[0, :]
    y_data = xy[1, :]

    parameter_columns = []
    for column in file.data.columns:
        if column in {"x", "y"}:
            continue
        column_values = file.data[column].to_numpy()
        if np.issubdtype(np.asarray(column_values).dtype, np.number):
            parameter_columns.append(column)
    if not parameter_columns:
        return empty_figure

    violin_payload = []
    for parameter in parameter_columns:
        z_data = file.data[parameter].to_numpy()
        gradient_result = _build_gradient_map(
            x_data,
            y_data,
            z_data,
            "magnitude",
            grid_mode=_resolve_gradient_calc_grid_mode(active_settings),
            grid_size=active_settings.get("gradient_grid_size"),
            k_nearest=active_settings.get("gradient_k_nearest"),
            polar_center=(active_settings.get("mappattern_x"), active_settings.get("mappattern_y")),
        )
        if not gradient_result:
            continue

        _, _, gradient_grid = gradient_result
        gradient_values = gradient_grid[np.isfinite(gradient_grid)]
        if gradient_values.size == 0:
            continue

        violin_payload.append((parameter, gradient_values))

    if not violin_payload:
        return empty_figure

    vertical_spacing = min(0.12, max(0.03, 0.26 / len(violin_payload)))
    fig = make_subplots(
        rows=len(violin_payload),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=vertical_spacing,
        subplot_titles=[parameter for parameter, _ in violin_payload],
    )

    for row_idx, (parameter, gradient_values) in enumerate(violin_payload, start=1):
        fig.add_trace(
            go.Violin(
                x=gradient_values,
                y=[parameter] * gradient_values.size,
                orientation="h",
                name=parameter,
                box_visible=True,
                meanline_visible=True,
                points=False,
                spanmode="hard",
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"|grad({parameter})|",
            automargin=True,
            title_standoff=6,
            matches=None,
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(showticklabels=False, title_text="", row=row_idx, col=1)

    fig.update_layout(
        template="plotly_white",
        title=dict(text="|grad(Z)| distribution by parameter", x=0.5, xanchor="center", pad=dict(t=10, b=8)),
        margin=dict(l=55, r=30, t=95, b=80),
        height=max(320, 220 * len(violin_payload) + 60),
        showlegend=False,
    )

    return fig


@callback(
    Output(ids.Graph.SPATIAL_BIN_MAP, "figure"),
    Output(ids.Graph.SPATIAL_BIN_TRENDS, "figure"),
    Output(ids.Div.SPATIAL_BIN_COUNTS, "children"),
    Input(ids.Tabs.ANALYSIS, "active_tab"),
    Input(ids.DropDown.UPLOADED_FILES, "value"),
    Input(ids.Store.SETTINGS, "data"),
    Input(ids.Input.SPATIAL_BIN_RADIAL_COUNT, "value"),
    Input(ids.Input.SPATIAL_BIN_ANGULAR_COUNT, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
)
def update_spatial_binning_tab(
    active_tab,
    selected_file,
    settings,
    radial_bin_count,
    angular_bin_count,
    stored_files,
):
    if active_tab != "spatial_binning":
        return no_update, no_update, no_update

    empty_map = _empty_figure("Spatial Bins", "X (mm)", "Y (mm)")
    empty_trend = _empty_figure("Bin Trends", "Bin number", "Value")

    if not selected_file or not stored_files or selected_file not in stored_files:
        return (
            empty_map,
            empty_trend,
            html.Div("Select a file to analyze spatial bins.", className="text-muted"),
        )

    file = Ellipsometry.from_path_or_stream(stored_files[selected_file])
    if file.data.empty or "x" not in file.data or "y" not in file.data:
        return (
            empty_map,
            empty_trend,
            html.Div("No valid data for spatial binning.", className="text-muted"),
        )

    active_settings = {**DEFAULT_SETTINGS, **(settings or {})}
    z_key = _resolve_z_key(file, active_settings.get("z_data_value"))
    if not z_key:
        return (
            empty_map,
            empty_trend,
            html.Div("No valid Z parameter available for spatial binning.", className="text-muted"),
        )

    x_data = np.array(file.data["x"])
    y_data = np.array(file.data["y"])
    z_data = file.data[z_key].to_numpy()

    xy = rotate(np.vstack([x_data, y_data]), active_settings["mappattern_theta"])
    xy = translate(xy, [active_settings["mappattern_x"], active_settings["mappattern_y"]])
    x_data = xy[0, :]
    y_data = xy[1, :]

    gradient_mode = active_settings.get("gradient_mode", "none")
    if gradient_mode == "none":
        values = z_data
        value_label = z_key
    else:
        gradient_result = _build_gradient_map(
            x_data,
            y_data,
            z_data,
            gradient_mode,
            grid_mode=_resolve_gradient_calc_grid_mode(active_settings),
            grid_size=active_settings.get("gradient_grid_size"),
            k_nearest=active_settings.get("gradient_k_nearest"),
            polar_center=(active_settings.get("mappattern_x"), active_settings.get("mappattern_y")),
        )
        if gradient_result:
            xi, yi, gradient_grid = gradient_result
            values = _sample_grid_at_points(x_data, y_data, xi, yi, gradient_grid)
            value_label = _gradient_label(gradient_mode, z_key)
        else:
            values = z_data
            value_label = z_key

    keep_mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(values)
    x_mm = x_data[keep_mask] * 10.0
    y_mm = y_data[keep_mask] * 10.0
    values = np.asarray(values)[keep_mask]
    if values.size == 0:
        return (
            empty_map,
            empty_trend,
            html.Div("No finite values for spatial binning.", className="text-muted"),
        )

    include_mask = np.ones(values.size, dtype=bool)
    if active_settings.get("ee_state"):
        masked_file = create_masked_file(file, active_settings)
        included_idx = file.data.index.isin(masked_file.data.index)
        include_mask = np.asarray(included_idx)[keep_mask]

    x0_mm = _safe_float(active_settings.get("sample_x"), default=np.nan) * 10.0
    y0_mm = _safe_float(active_settings.get("sample_y"), default=np.nan) * 10.0
    center_source_x = x_mm[include_mask] if np.any(include_mask) else x_mm
    center_source_y = y_mm[include_mask] if np.any(include_mask) else y_mm
    if not np.isfinite(x0_mm):
        x0_mm = float(np.median(center_source_x))
    if not np.isfinite(y0_mm):
        y0_mm = float(np.median(center_source_y))

    dx = x_mm - x0_mm
    dy = y_mm - y0_mm
    radial_distance = np.sqrt(dx * dx + dy * dy)
    angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    try:
        n_radial = int(radial_bin_count)
    except (TypeError, ValueError):
        n_radial = 2
    n_radial = int(np.clip(n_radial, 1, 40))

    try:
        n_angular = int(angular_bin_count)
    except (TypeError, ValueError):
        n_angular = 4
    n_angular = int(np.clip(n_angular, 1, 72))

    interior_bin_count = n_radial * n_angular
    edge_bin_index = interior_bin_count
    bin_index = np.full(values.size, -1, dtype=int)

    radial_edges = np.zeros(n_radial + 1, dtype=float)
    if np.any(include_mask):
        radial_values = radial_distance[include_mask]
        radial_edges = np.quantile(radial_values, np.linspace(0.0, 1.0, n_radial + 1))
        radial_edges = np.maximum.accumulate(np.asarray(radial_edges, dtype=float))

        radial_inner_edges = radial_edges[1:-1]
        radial_bin = np.searchsorted(radial_inner_edges, radial_values, side="right")

        angle_values = angle[include_mask]
        angular_width = 2 * np.pi / n_angular
        angular_bin = np.floor(angle_values / angular_width).astype(int)
        angular_bin = np.clip(angular_bin, 0, n_angular - 1)

        bin_index[include_mask] = radial_bin * n_angular + angular_bin

    if active_settings.get("ee_state"):
        excluded_mask = ~include_mask
        bin_index[excluded_mask] = edge_bin_index

    unassigned = bin_index < 0
    if np.any(unassigned):
        bin_index[unassigned] = edge_bin_index if active_settings.get("ee_state") else 0

    interior_counts = np.asarray([np.sum(bin_index == i) for i in range(interior_bin_count)], dtype=int)
    edge_count = int(np.sum(bin_index == edge_bin_index)) if active_settings.get("ee_state") else 0

    map_fig = go.Figure()
    label_x = []
    label_y = []
    label_text = []
    for b in range(interior_bin_count):
        count = int(interior_counts[b])
        if count == 0:
            continue
        in_bin = bin_index == b
        radial_id = b // n_angular + 1
        angular_id = b % n_angular + 1
        map_fig.add_trace(
            go.Scatter(
                x=x_mm[in_bin],
                y=y_mm[in_bin],
                mode="markers",
                marker=dict(size=6, opacity=0.65),
                name=f"Bin {b + 1} (n={count})",
                hovertemplate=(
                    f"Bin {b + 1} (R{radial_id}, A{angular_id})"
                    + "<br>x: %{x:.3f} mm<br>y: %{y:.3f} mm<br>"
                    + f"{value_label}: %{{customdata:.6g}}<extra></extra>"
                ),
                customdata=values[in_bin],
            )
        )
        label_x.append(float(np.mean(x_mm[in_bin])))
        label_y.append(float(np.mean(y_mm[in_bin])))
        label_text.append(str(b + 1))

    if active_settings.get("ee_state") and edge_count > 0:
        excluded_mask = bin_index == edge_bin_index
        map_fig.add_trace(
            go.Scatter(
                x=x_mm[excluded_mask],
                y=y_mm[excluded_mask],
                mode="markers",
                marker=dict(size=6, opacity=0.6, color="rgba(150,150,150,0.8)"),
                name=f"Edge excluded (n={edge_count})",
                hovertemplate="Edge excluded<br>x: %{x:.3f} mm<br>y: %{y:.3f} mm<br>"
                + f"{value_label}: %{{customdata:.6g}}<extra></extra>",
                customdata=values[excluded_mask],
            )
        )
        label_x.append(float(np.mean(x_mm[excluded_mask])))
        label_y.append(float(np.mean(y_mm[excluded_mask])))
        label_text.append("EE")

    max_radius = float(np.nanmax(radial_distance)) if radial_distance.size else 0.0
    for ring_radius in radial_edges[1:-1]:
        if np.isfinite(ring_radius) and ring_radius > 0:
            map_fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=x0_mm - ring_radius,
                x1=x0_mm + ring_radius,
                y0=y0_mm - ring_radius,
                y1=y0_mm + ring_radius,
                line=dict(color="rgba(30, 30, 30, 0.35)", width=1, dash="dot"),
            )

    for section_idx in range(n_angular):
        theta = 2 * np.pi * section_idx / n_angular
        x_end = x0_mm + max_radius * np.cos(theta)
        y_end = y0_mm + max_radius * np.sin(theta)
        map_fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x0_mm,
            y0=y0_mm,
            x1=x_end,
            y1=y_end,
            line=dict(color="rgba(30, 30, 30, 0.35)", width=1, dash="dot"),
        )

    map_fig.add_trace(
        go.Scatter(
            x=[x0_mm],
            y=[y0_mm],
            mode="markers",
            marker=dict(symbol="x", size=11, color="black"),
            name="Bin center",
            hovertemplate="Center<br>x: %{x:.3f} mm<br>y: %{y:.3f} mm<extra></extra>",
        )
    )
    if label_text:
        map_fig.add_trace(
            go.Scatter(
                x=np.asarray(label_x),
                y=np.asarray(label_y),
                mode="text",
                text=label_text,
                textposition="middle center",
                textfont=dict(size=13, color="black"),
                name="Bin labels",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    map_fig.update_layout(
        template="plotly_white",
        title=dict(text="Spatial bin map (radial x angular)", x=0.5, xanchor="center", pad=dict(t=10, b=6)),
        margin=dict(l=55, r=30, t=95, b=110),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    map_fig.update_xaxes(title_text="X (mm)", automargin=True, title_standoff=8, scaleanchor="y", scaleratio=1)
    map_fig.update_yaxes(title_text="Y (mm)", automargin=True, title_standoff=8)

    trend_fig = go.Figure()
    rng = np.random.default_rng(2026)
    all_bin_indices = list(range(interior_bin_count))
    bin_positions = {b: b + 1 for b in all_bin_indices}
    tick_vals = [b + 1 for b in all_bin_indices]
    tick_text = [str(b + 1) for b in all_bin_indices]
    if active_settings.get("ee_state"):
        all_bin_indices.append(edge_bin_index)
        bin_positions[edge_bin_index] = interior_bin_count + 1
        tick_vals.append(interior_bin_count + 1)
        tick_text.append("EE")

    interior_mean_x = []
    interior_mean_y = []
    interior_mean_std = []
    edge_mean_x = []
    edge_mean_y = []
    edge_mean_std = []
    edge_legend_group = "edge_excluded_group"

    for b in all_bin_indices:
        in_bin = bin_index == b
        y_bin = values[in_bin]
        count = int(y_bin.size)
        x_pos = bin_positions[b]

        mean_value = float(np.mean(y_bin)) if count > 0 else np.nan
        std_value = float(np.std(y_bin, ddof=1)) if count > 1 else 0.0
        if b == edge_bin_index:
            edge_mean_x.append(x_pos)
            edge_mean_y.append(mean_value)
            edge_mean_std.append(std_value)
        else:
            interior_mean_x.append(x_pos)
            interior_mean_y.append(mean_value)
            interior_mean_std.append(std_value)

        if count == 0:
            continue

        jitter_x = np.full(y_bin.size, x_pos, dtype=float) + rng.uniform(-0.18, 0.18, size=y_bin.size)
        series_name = "Edge excluded raw" if b == edge_bin_index else f"Bin {b + 1} raw"
        trend_fig.add_trace(
            go.Scatter(
                x=jitter_x,
                y=y_bin,
                mode="markers",
                marker=dict(size=5, opacity=0.3),
                name=series_name,
                showlegend=bool(active_settings.get("ee_state") and b == edge_bin_index),
                legendgroup=edge_legend_group if b == edge_bin_index else None,
                hovertemplate=(
                    ("Edge excluded" if b == edge_bin_index else f"Bin {b + 1}")
                    + f"<br>{value_label}: "
                    + "%{y:.6g}<extra></extra>"
                ),
            )
        )

    trend_fig.add_trace(
        go.Scatter(
            x=np.asarray(interior_mean_x),
            y=np.asarray(interior_mean_y),
            mode="lines+markers",
            line=dict(width=2, color="black"),
            marker=dict(size=7),
            error_y=dict(type="data", array=np.asarray(interior_mean_std), visible=True),
            name="Bin mean +/- std",
            showlegend=False,
            hovertemplate="Section %{x}<br>Mean: %{y:.6g}<extra></extra>",
        )
    )
    if active_settings.get("ee_state") and edge_count > 0:
        trend_fig.add_trace(
            go.Scatter(
                x=np.asarray(edge_mean_x),
                y=np.asarray(edge_mean_y),
                mode="markers",
                marker=dict(size=7, color="black"),
                error_y=dict(type="data", array=np.asarray(edge_mean_std), visible=True),
                name="Edge excluded mean +/- std",
                showlegend=False,
                legendgroup=edge_legend_group,
                hovertemplate="Section EE<br>Mean: %{y:.6g}<extra></extra>",
            )
        )
    trend_fig.update_layout(
        template="plotly_white",
        title=dict(text="Data and trend by spatial section", x=0.5, xanchor="center", pad=dict(t=10, b=6)),
        margin=dict(l=55, r=30, t=95, b=120),
        showlegend=bool(active_settings.get("ee_state") and edge_count > 0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
            groupclick="togglegroup",
        ),
    )
    trend_fig.update_xaxes(
        title_text="Section number",
        automargin=True,
        title_standoff=8,
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        range=[0.5, tick_vals[-1] + 0.5],
    )
    trend_fig.update_yaxes(title_text=value_label, automargin=True, title_standoff=8)

    radial_lines = []
    for radial_idx in range(n_radial):
        r_min = radial_edges[radial_idx]
        r_max = radial_edges[radial_idx + 1]
        radial_lines.append(
            html.Div(
                f"Radial bin {radial_idx + 1}: r = [{r_min:.4g}, {r_max:.4g}] mm"
            )
        )

    count_lines = []
    for b in range(interior_bin_count):
        radial_id = b // n_angular + 1
        angular_id = b % n_angular + 1
        count_lines.append(
            html.Div(
                f"Bin {b + 1} (R{radial_id}, A{angular_id}): {int(interior_counts[b])} points"
            )
        )
    if active_settings.get("ee_state"):
        count_lines.append(html.Div(f"Edge excluded (EE): {edge_count} points"))

    count_panel = html.Div(
        [
            html.Div(html.Strong("Spatial bin summary")),
            html.Div(f"Scheme: {n_radial} radial bins x {n_angular} angular bins = {interior_bin_count} interior sections"),
            html.Div(f"Center used: ({x0_mm:.3f} mm, {y0_mm:.3f} mm)"),
            html.Div(f"Total points analyzed: {values.size}"),
            html.Div("Radial boundaries are quantile-based on non-edge points for near-equal occupancy."),
            *radial_lines,
            *count_lines,
        ]
    )

    return map_fig, trend_fig, count_panel
