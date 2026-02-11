from dash import callback, html, no_update, Output, Input, State
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


from src import ids
from src.callbacks.graph_callbacks import _build_gradient_map, _gradient_label, _resolve_z_key
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
        grid_mode=active_settings.get("gradient_grid_mode", "auto"),
        grid_size=active_settings.get("gradient_grid_size"),
        k_nearest=active_settings.get("gradient_k_nearest"),
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
