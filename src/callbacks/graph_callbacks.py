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
POLAR_MIN_FILL_RATIO = 0.8
POLAR_ADAPTIVE_R_MIN = 8
POLAR_ADAPTIVE_R_MAX = 256
POLAR_ADAPTIVE_THETA_MIN = 24
POLAR_ADAPTIVE_THETA_MAX = 360
POLAR_ADAPTIVE_MIN_FILL_RATIO = 0.08
POLAR_ADAPTIVE_FILL_K_NEAREST = 12
POLAR_ADAPTIVE_FILL_CHUNK = 128
POLAR_RENDER_GRID_MIN = 120
POLAR_RENDER_GRID_MAX = 360
EXPORT_IMAGE_WIDTH = 1400
EXPORT_IMAGE_HEIGHT = 1000
EXPORT_IMAGE_SCALE = 1


# Adding a placeholder for sample outline
# 8in wafer
r = 1*2.54


def _resolve_gradient_grid_size(point_count, grid_mode="auto", grid_size=None):
    if grid_mode == "manual":
        try:
            resolved_grid_size = int(grid_size)
        except (TypeError, ValueError):
            resolved_grid_size = DEFAULT_SETTINGS["gradient_grid_size"]
        return int(np.clip(resolved_grid_size, GRID_SIZE_MIN, GRID_SIZE_MAX))

    resolved_grid_size = int(np.clip(np.sqrt(point_count) * 4, GRID_SIZE_AUTO_MIN, GRID_SIZE_AUTO_MAX))
    if point_count > 3 * CRITICAL_COUNT:
        resolved_grid_size = min(resolved_grid_size, GRID_SIZE_AUTO_MAX_DENSE)
    return resolved_grid_size


def _resolve_non_spatial_columns(file):
    return [column for column in sorted(file.get_column_names()) if str(column).lower() not in {"x", "y"}]


def _resolve_z_key(file, z_key):
    if z_key and z_key in file.data and str(z_key).lower() not in {"x", "y"}:
        return z_key

    columns = _resolve_non_spatial_columns(file)
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


def _resolve_render_mode(settings):
    mode = (settings or {}).get("render_mode")
    if mode in {"markers", "heatmap"}:
        return mode
    return "markers"


def _resolve_heatmap_grid_mode(settings):
    mode = (settings or {}).get("heatmap_grid_mode")
    if mode in {"auto", "manual"}:
        return mode

    legacy_mode = (settings or {}).get("gradient_grid_mode", "auto")
    return legacy_mode if legacy_mode in {"auto", "manual"} else "auto"


def _resolve_gradient_coordinate_mode(settings):
    mode = (settings or {}).get("gradient_coordinate_mode")
    if mode in {"cartesian", "polar"}:
        return mode

    legacy_mode = (settings or {}).get("gradient_grid_mode")
    return "polar" if legacy_mode == "polar_native" else "cartesian"


def _resolve_gradient_calc_grid_mode(settings):
    if _resolve_gradient_coordinate_mode(settings) == "polar":
        return "polar_native"
    return _resolve_heatmap_grid_mode(settings)


def _cross_2d(o, a, b):
    return ((a[0] - o[0]) * (b[1] - o[1])) - ((a[1] - o[1]) * (b[0] - o[0]))


def _build_convex_hull(points):
    ordered = np.asarray(points, dtype=float)
    if ordered.ndim != 2 or ordered.shape[1] != 2:
        return None

    ordered = np.unique(ordered, axis=0)
    if ordered.shape[0] < 3:
        return None

    sort_idx = np.lexsort((ordered[:, 1], ordered[:, 0]))
    ordered = ordered[sort_idx]

    lower = []
    for point in ordered:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return None

    return np.asarray(hull, dtype=float)


def _mask_points_inside_sample_footprint(sample_x, sample_y, query_x, query_y):
    query_arr = np.asarray(query_x)
    qx = np.asarray(query_x, dtype=float).ravel()
    qy = np.asarray(query_y, dtype=float).ravel()
    inside_mask = np.ones(qx.shape[0], dtype=bool)
    if qx.size == 0:
        return inside_mask.reshape(query_arr.shape)

    sx = np.asarray(sample_x, dtype=float).ravel()
    sy = np.asarray(sample_y, dtype=float).ravel()
    valid_samples = np.isfinite(sx) & np.isfinite(sy)
    if np.count_nonzero(valid_samples) < 3:
        return inside_mask.reshape(query_arr.shape)

    sample_points = np.column_stack([sx[valid_samples], sy[valid_samples]])
    hull = _build_convex_hull(sample_points)
    if hull is None:
        return inside_mask.reshape(query_arr.shape)

    vx = hull[:, 0]
    vy = hull[:, 1]
    ex = np.roll(vx, -1) - vx
    ey = np.roll(vy, -1) - vy

    footprint_span = max(float(np.max(vx) - np.min(vx)), float(np.max(vy) - np.min(vy)), 1.0)
    tolerance = 1e-9 * footprint_span

    chunk_size = 4096
    for start in range(0, qx.size, chunk_size):
        stop = min(start + chunk_size, qx.size)
        local_x = qx[start:stop, None]
        local_y = qy[start:stop, None]
        cross = ex[None, :] * (local_y - vy[None, :]) - ey[None, :] * (local_x - vx[None, :])
        inside_mask[start:stop] = np.all(cross >= -tolerance, axis=1)

    return inside_mask.reshape(query_arr.shape)


def _build_value_map(x_data, y_data, z_data, grid_mode="auto", grid_size=None, k_nearest=None):
    resolved_grid_mode = grid_mode if grid_mode in {"auto", "manual"} else "auto"
    return _interpolate_idw_grid(
        x_data,
        y_data,
        z_data,
        grid_mode=resolved_grid_mode,
        grid_size=grid_size,
        k_nearest=k_nearest,
    )


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

    resolved_grid_size = _resolve_gradient_grid_size(
        point_count,
        grid_mode=grid_mode,
        grid_size=grid_size,
    )

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
    footprint_mask = _mask_points_inside_sample_footprint(x, y, flat_x, flat_y).ravel()
    z_interp[~footprint_mask] = np.nan

    return xi, yi, z_interp.reshape(resolved_grid_size, resolved_grid_size)


def _resolve_adaptive_polar_bin_counts(r_values, point_count):
    r_estimates = [
        np.unique(np.round(r_values, decimals=2)).size,
        np.unique(np.round(r_values, decimals=3)).size,
        int(np.round(np.sqrt(point_count))),
    ]
    r_estimates = [value for value in r_estimates if value >= 3]
    radial_guess = int(np.median(r_estimates)) if r_estimates else int(np.round(np.sqrt(point_count)))

    density_cap = max(POLAR_ADAPTIVE_R_MIN, int(np.sqrt(point_count) * 1.5))
    density_cap = min(density_cap, POLAR_ADAPTIVE_R_MAX)
    radial_bins = int(np.clip(radial_guess, POLAR_ADAPTIVE_R_MIN, density_cap))
    angular_bins = int(
        np.clip(
            np.rint(point_count / max(radial_bins, 1)),
            POLAR_ADAPTIVE_THETA_MIN,
            POLAR_ADAPTIVE_THETA_MAX,
        )
    )
    return radial_bins, angular_bins


def _fill_sparse_polar_grid(r_levels, theta_levels, z_grid, k_nearest=POLAR_ADAPTIVE_FILL_K_NEAREST):
    filled = np.asarray(z_grid, dtype=float).copy()
    valid_mask = np.isfinite(filled)
    if np.all(valid_mask):
        return filled, None

    known_count = int(np.count_nonzero(valid_mask))
    if known_count < 3:
        return None, f"too few occupied polar bins to fill holes: {known_count}"

    r_mesh = np.broadcast_to(r_levels[:, np.newaxis], filled.shape)
    theta_mesh = np.broadcast_to(theta_levels[np.newaxis, :], filled.shape)

    known_x = r_mesh[valid_mask] * np.cos(theta_mesh[valid_mask])
    known_y = r_mesh[valid_mask] * np.sin(theta_mesh[valid_mask])
    known_z = filled[valid_mask]

    query_mask = ~valid_mask
    query_x = r_mesh[query_mask] * np.cos(theta_mesh[query_mask])
    query_y = r_mesh[query_mask] * np.sin(theta_mesh[query_mask])
    if query_x.size == 0:
        return filled, None

    k_eff = min(int(k_nearest), known_count)
    if k_eff < 1:
        return None, "invalid nearest-neighbor count for polar hole fill"

    eps = 1e-12
    query_values = np.empty(query_x.shape[0], dtype=float)
    for start in range(0, query_x.shape[0], POLAR_ADAPTIVE_FILL_CHUNK):
        stop = min(start + POLAR_ADAPTIVE_FILL_CHUNK, query_x.shape[0])

        dx = query_x[start:stop, None] - known_x[None, :]
        dy = query_y[start:stop, None] - known_y[None, :]
        dist2 = dx * dx + dy * dy

        if k_eff < known_count:
            neighbor_idx = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
            selected_dist2 = np.take_along_axis(dist2, neighbor_idx, axis=1)
            selected_z = known_z[neighbor_idx]
        else:
            selected_dist2 = dist2
            selected_z = known_z[None, :]

        nearest_idx = np.argmin(selected_dist2, axis=1)
        row_idx = np.arange(selected_dist2.shape[0])
        closest_dist2 = selected_dist2[row_idx, nearest_idx]

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

        query_values[start:stop] = chunk_values

    filled[query_mask] = query_values
    return filled, None


def _build_adaptive_polar_lattice(r_values, theta_values, z_values):
    point_count = z_values.size
    radial_bins, angular_bins = _resolve_adaptive_polar_bin_counts(r_values, point_count)

    r_min = float(np.min(r_values))
    r_max = float(np.max(r_values))
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max <= r_min:
        return None, "invalid radial extent for adaptive lattice"

    r_edges = np.linspace(r_min, r_max, radial_bins + 1)
    theta_edges = np.linspace(0.0, 2 * np.pi, angular_bins + 1)
    r_levels = 0.5 * (r_edges[:-1] + r_edges[1:])
    theta_levels = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    if r_levels.size < 3 or theta_levels.size < 8:
        return None, f"adaptive lattice too small: r_levels={r_levels.size}, theta_levels={theta_levels.size}"

    r_idx = np.searchsorted(r_edges, r_values, side="right") - 1
    r_idx = np.clip(r_idx, 0, radial_bins - 1)

    theta_wrapped = np.mod(theta_values, 2 * np.pi)
    theta_idx = np.searchsorted(theta_edges, theta_wrapped, side="right") - 1
    theta_idx = np.clip(theta_idx, 0, angular_bins - 1)

    z_sum = np.zeros((radial_bins, angular_bins), dtype=float)
    z_count = np.zeros((radial_bins, angular_bins), dtype=float)
    np.add.at(z_sum, (r_idx, theta_idx), z_values)
    np.add.at(z_count, (r_idx, theta_idx), 1.0)

    total_cells = radial_bins * angular_bins
    occupied_cells = int(np.count_nonzero(z_count))
    fill_ratio = float(occupied_cells / total_cells) if total_cells > 0 else 0.0
    if fill_ratio < POLAR_ADAPTIVE_MIN_FILL_RATIO:
        return (
            None,
            "adaptive lattice fill ratio too low "
            f"(fill={fill_ratio:.3f}, required>={POLAR_ADAPTIVE_MIN_FILL_RATIO:.2f}, "
            f"occupied={occupied_cells}, total={total_cells})",
        )

    min_occupied = min(total_cells, max(12, int(np.ceil(point_count * 0.1))))
    if occupied_cells < min_occupied:
        return (
            None,
            f"adaptive lattice has too few occupied cells ({occupied_cells}, need >= {min_occupied})",
        )

    z_grid = np.divide(
        z_sum,
        z_count,
        out=np.full_like(z_sum, np.nan),
        where=z_count > 0,
    )

    filled_grid, fill_reason = _fill_sparse_polar_grid(
        r_levels,
        theta_levels,
        z_grid,
        k_nearest=POLAR_ADAPTIVE_FILL_K_NEAREST,
    )
    if filled_grid is None:
        return None, f"adaptive polar hole fill failed: {fill_reason}"

    details = (
        "adaptive bins "
        f"(r_bins={radial_bins}, theta_bins={angular_bins}, "
        f"occupied={occupied_cells}/{total_cells}, fill={fill_ratio:.3f})"
    )
    return (r_levels, theta_levels, filled_grid), details


def _build_polar_lattice(x_data, y_data, z_data, center_x=None, center_y=None):
    finite_mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(z_data)
    x = np.asarray(x_data)[finite_mask]
    y = np.asarray(y_data)[finite_mask]
    z = np.asarray(z_data)[finite_mask]
    if z.size < 24:
        return None, f"insufficient finite samples for polar lattice: {z.size} (need >= 24)"

    if center_x is None or not np.isfinite(center_x):
        center_x = float(np.median(x))
    if center_y is None or not np.isfinite(center_y):
        center_y = float(np.median(y))

    dx = x - center_x
    dy = y - center_y
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    def _fallback_to_adaptive(strict_reason):
        adaptive_lattice, adaptive_details = _build_adaptive_polar_lattice(r, theta, z)
        if adaptive_lattice is None:
            return None, f"{strict_reason}; adaptive binning failed: {adaptive_details}"

        r_levels_adaptive, theta_levels_adaptive, z_grid_adaptive = adaptive_lattice
        logger.debug(
            "Polar lattice fallback to adaptive bins (%s): %s",
            strict_reason,
            adaptive_details,
        )
        return (
            center_x,
            center_y,
            r_levels_adaptive,
            theta_levels_adaptive,
            z_grid_adaptive,
        ), None

    best = None
    candidate_summaries = []
    for decimals in (6, 5, 4, 3):
        r_levels, r_idx = np.unique(np.round(r, decimals=decimals), return_inverse=True)
        theta_levels, theta_idx = np.unique(np.round(theta, decimals=decimals), return_inverse=True)
        if r_levels.size < 3 or theta_levels.size < 8:
            candidate_summaries.append(
                f"d={decimals}:r_levels={r_levels.size},theta_levels={theta_levels.size} (below minimum)"
            )
            continue

        cell_count = r_levels.size * theta_levels.size
        if cell_count <= 0:
            candidate_summaries.append(f"d={decimals}:invalid cell_count={cell_count}")
            continue

        z_count = np.zeros((r_levels.size, theta_levels.size), dtype=float)
        np.add.at(z_count, (r_idx, theta_idx), 1.0)
        fill_ratio = float(np.count_nonzero(z_count) / cell_count)
        candidate_summaries.append(
            f"d={decimals}:r_levels={r_levels.size},theta_levels={theta_levels.size},fill={fill_ratio:.3f}"
        )

        if fill_ratio < POLAR_MIN_FILL_RATIO:
            continue

        if (
            best is None
            or cell_count > best["cell_count"]
            or (cell_count == best["cell_count"] and fill_ratio > best["fill_ratio"])
        ):
            best = dict(
                r_levels=r_levels,
                theta_levels=theta_levels,
                r_idx=r_idx,
                theta_idx=theta_idx,
                fill_ratio=fill_ratio,
                cell_count=cell_count,
            )

    if not best:
        candidate_msg = "; ".join(candidate_summaries) if candidate_summaries else "no viable rounding candidates"
        return _fallback_to_adaptive(
            f"no polar lattice met strict fill threshold {POLAR_MIN_FILL_RATIO:.2f}; candidates: {candidate_msg}"
        )

    r_levels = best["r_levels"]
    theta_levels = best["theta_levels"]
    r_idx = best["r_idx"]
    theta_idx = best["theta_idx"]

    theta_gaps = np.diff(np.concatenate([theta_levels, [theta_levels[0] + (2 * np.pi)]]))
    finite_theta_gaps = theta_gaps[np.isfinite(theta_gaps) & (theta_gaps > 0)]
    if finite_theta_gaps.size < 2:
        return _fallback_to_adaptive("insufficient positive theta gaps after strict lattice construction")
    median_theta_gap = float(np.median(finite_theta_gaps))
    if not np.isfinite(median_theta_gap) or median_theta_gap <= 0:
        return _fallback_to_adaptive(f"invalid theta gap median from strict lattice: {median_theta_gap}")

    max_theta_gap = float(np.max(finite_theta_gaps))
    if max_theta_gap > (3.5 * median_theta_gap):
        gap_ratio = max_theta_gap / median_theta_gap if median_theta_gap > 0 else np.inf
        return _fallback_to_adaptive(
            "strict theta spacing too non-uniform "
            f"(max_gap={max_theta_gap:.6g}, median_gap={median_theta_gap:.6g}, ratio={gap_ratio:.2f})"
        )

    z_sum = np.zeros((r_levels.size, theta_levels.size), dtype=float)
    z_count = np.zeros((r_levels.size, theta_levels.size), dtype=float)
    np.add.at(z_sum, (r_idx, theta_idx), z)
    np.add.at(z_count, (r_idx, theta_idx), 1.0)
    z_grid = np.divide(
        z_sum,
        z_count,
        out=np.full_like(z_sum, np.nan),
        where=z_count > 0,
    )

    logger.debug(
        "Polar lattice strict reconstruction succeeded: r_levels=%d, theta_levels=%d, fill=%.3f",
        r_levels.size,
        theta_levels.size,
        best["fill_ratio"],
    )
    return (center_x, center_y, r_levels, theta_levels, z_grid), None


def _build_polar_gradient_grid(r_levels, theta_levels, z_grid, gradient_mode):
    if z_grid.ndim != 2 or z_grid.shape[0] < 3 or z_grid.shape[1] < 8:
        return (
            None,
            f"polar lattice shape unsupported for derivatives: shape={z_grid.shape} (need >=3 radial and >=8 angular)",
        )
    if r_levels.ndim != 1 or theta_levels.ndim != 1:
        return None, "polar axis arrays are not 1D"
    if np.any(~np.isfinite(r_levels)) or np.any(~np.isfinite(theta_levels)):
        return None, "polar axis arrays contain non-finite values"
    if np.any(np.diff(r_levels) <= 0):
        return None, "radial levels are not strictly increasing"

    try:
        dz_dr = np.gradient(z_grid, r_levels, axis=0, edge_order=1)

        theta_ext = np.empty(theta_levels.size + 2, dtype=float)
        theta_ext[1:-1] = theta_levels
        theta_ext[0] = theta_levels[-1] - (2 * np.pi)
        theta_ext[-1] = theta_levels[0] + (2 * np.pi)
        z_ext = np.concatenate([z_grid[:, -1:], z_grid, z_grid[:, :1]], axis=1)
        dz_dtheta_ext = np.gradient(z_ext, theta_ext, axis=1, edge_order=1)
        dz_dtheta = dz_dtheta_ext[:, 1:-1]

        theta_mesh = np.broadcast_to(theta_levels[np.newaxis, :], z_grid.shape)
        r_mesh = np.broadcast_to(r_levels[:, np.newaxis], z_grid.shape)

        radial_eps = max(float(np.nanmin(np.diff(r_levels))) * 0.5 if r_levels.size > 1 else 0.0, 1e-9)
        valid_r = r_mesh > radial_eps
        if not np.any(valid_r):
            return None, f"all radial locations are near center (r <= {radial_eps:.6g})"

        inv_r = np.divide(1.0, r_mesh, out=np.zeros_like(r_mesh), where=valid_r)
        inv_r2 = np.divide(1.0, r_mesh * r_mesh, out=np.zeros_like(r_mesh), where=valid_r)
        angular_component = dz_dtheta * inv_r

        if gradient_mode == "dx":
            gradient_polar = np.cos(theta_mesh) * dz_dr - np.sin(theta_mesh) * angular_component
        elif gradient_mode == "dy":
            gradient_polar = np.sin(theta_mesh) * dz_dr + np.cos(theta_mesh) * angular_component
        elif gradient_mode == "laplacian":
            radial_flux = r_mesh * dz_dr
            dr_radial_flux = np.gradient(radial_flux, r_levels, axis=0, edge_order=1)
            d2z_dtheta2_ext = np.gradient(dz_dtheta_ext, theta_ext, axis=1, edge_order=1)
            d2z_dtheta2 = d2z_dtheta2_ext[:, 1:-1]
            gradient_polar = (dr_radial_flux * inv_r) + (d2z_dtheta2 * inv_r2)
            gradient_polar = np.where(valid_r, gradient_polar, np.nan)
        else:
            gradient_polar = np.hypot(dz_dr, angular_component)
    except Exception as exc:
        return None, f"exception during polar derivative calculation: {exc}"

    return gradient_polar, None


def _resample_polar_to_cartesian_grid(
    x_data,
    y_data,
    center_x,
    center_y,
    r_levels,
    theta_levels,
    value_grid,
    render_grid_size=None,
):
    if render_grid_size is None:
        point_count = value_grid.size
        resolved_grid_size = _resolve_gradient_grid_size(point_count, grid_mode="auto")
    else:
        resolved_grid_size = int(np.clip(render_grid_size, GRID_SIZE_MIN, POLAR_RENDER_GRID_MAX))

    x_min, x_max = float(np.nanmin(x_data)), float(np.nanmax(x_data))
    y_min, y_max = float(np.nanmin(y_data)), float(np.nanmax(y_data))
    xi = np.linspace(x_min, x_max, resolved_grid_size)
    yi = np.linspace(y_min, y_max, resolved_grid_size)
    xx, yy = np.meshgrid(xi, yi)

    query_r = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2).ravel()
    query_theta = np.mod(np.arctan2(yy - center_y, xx - center_x), 2 * np.pi).ravel()
    flat_values = np.full(query_r.shape, np.nan, dtype=float)
    footprint_mask = _mask_points_inside_sample_footprint(x_data, y_data, xx.ravel(), yy.ravel()).ravel()

    valid_r = (query_r >= r_levels[0]) & (query_r <= r_levels[-1])
    valid_r = valid_r & footprint_mask
    if not np.any(valid_r):
        return xi, yi, flat_values.reshape(resolved_grid_size, resolved_grid_size)

    local_r = query_r[valid_r]
    local_theta = query_theta[valid_r]

    r_high = np.searchsorted(r_levels, local_r, side="left")
    r_high = np.clip(r_high, 1, r_levels.size - 1)
    r_low = r_high - 1
    r0 = r_levels[r_low]
    r1 = r_levels[r_high]
    radial_fraction = np.divide(
        local_r - r0,
        r1 - r0,
        out=np.zeros_like(local_r),
        where=(r1 - r0) > 0,
    )

    theta_ext = np.concatenate([theta_levels, [theta_levels[0] + (2 * np.pi)]])
    theta_wrapped = np.where(local_theta < theta_levels[0], local_theta + (2 * np.pi), local_theta)
    theta_high_ext = np.searchsorted(theta_ext, theta_wrapped, side="right")
    theta_high_ext = np.clip(theta_high_ext, 1, theta_ext.size - 1)
    theta_low_ext = theta_high_ext - 1

    theta0 = theta_ext[theta_low_ext]
    theta1 = theta_ext[theta_high_ext]
    angular_fraction = np.divide(
        theta_wrapped - theta0,
        theta1 - theta0,
        out=np.zeros_like(theta_wrapped),
        where=(theta1 - theta0) > 0,
    )

    theta_low = np.mod(theta_low_ext, theta_levels.size)
    theta_high = np.mod(theta_high_ext, theta_levels.size)

    v00 = value_grid[r_low, theta_low]
    v01 = value_grid[r_low, theta_high]
    v10 = value_grid[r_high, theta_low]
    v11 = value_grid[r_high, theta_high]

    w00 = (1.0 - radial_fraction) * (1.0 - angular_fraction)
    w01 = (1.0 - radial_fraction) * angular_fraction
    w10 = radial_fraction * (1.0 - angular_fraction)
    w11 = radial_fraction * angular_fraction

    stacked_values = np.vstack([v00, v01, v10, v11])
    stacked_weights = np.vstack([w00, w01, w10, w11])
    valid_values = np.isfinite(stacked_values)
    weighted_sum = np.sum(np.where(valid_values, stacked_weights * stacked_values, 0.0), axis=0)
    weight_sum = np.sum(np.where(valid_values, stacked_weights, 0.0), axis=0)
    interpolated = np.divide(
        weighted_sum,
        weight_sum,
        out=np.full_like(weighted_sum, np.nan),
        where=weight_sum > 0,
    )

    flat_values[valid_r] = interpolated
    return xi, yi, flat_values.reshape(resolved_grid_size, resolved_grid_size)


def _build_polar_native_gradient_map(x_data, y_data, z_data, gradient_mode, polar_center=None):
    finite_mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(z_data)
    x = np.asarray(x_data)[finite_mask]
    y = np.asarray(y_data)[finite_mask]
    if x.size < 24:
        return None, f"insufficient finite samples: {x.size} (need >= 24)"

    center_x = None
    center_y = None
    if polar_center:
        try:
            center_x = float(polar_center[0])
            center_y = float(polar_center[1])
        except (TypeError, ValueError, IndexError):
            center_x = None
            center_y = None

    lattice, lattice_reason = _build_polar_lattice(
        x,
        y,
        np.asarray(z_data)[finite_mask],
        center_x=center_x,
        center_y=center_y,
    )
    if lattice is None:
        return None, lattice_reason

    center_x, center_y, r_levels, theta_levels, z_polar = lattice
    gradient_polar, gradient_reason = _build_polar_gradient_grid(r_levels, theta_levels, z_polar, gradient_mode)
    if gradient_polar is None:
        return None, gradient_reason

    polar_resolution_hint = max(r_levels.size, theta_levels.size) * 2
    render_grid_size = int(np.clip(polar_resolution_hint, POLAR_RENDER_GRID_MIN, POLAR_RENDER_GRID_MAX))
    resampled = _resample_polar_to_cartesian_grid(
        x,
        y,
        center_x,
        center_y,
        r_levels,
        theta_levels,
        gradient_polar,
        render_grid_size=render_grid_size,
    )
    _, _, resampled_grid = resampled
    finite_count = int(np.count_nonzero(np.isfinite(resampled_grid)))
    if finite_count == 0:
        return None, "resampled polar gradient contains no finite values on Cartesian grid"

    logger.debug(
        "Polar-native reconstruction succeeded: points=%d, r_levels=%d, theta_levels=%d, render_grid=%d, finite_cells=%d",
        x.size,
        r_levels.size,
        theta_levels.size,
        render_grid_size,
        finite_count,
    )
    return resampled, None


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
    if grid_mode == "polar_native":
        polar_result, polar_reason = _build_polar_native_gradient_map(
            x_data,
            y_data,
            z_data,
            gradient_mode,
            polar_center=polar_center,
        )
        if polar_result is not None:
            return polar_result
        logger.warning(
            "Polar-native gradient reconstruction failed (%s); falling back to IDW interpolation.",
            polar_reason or "unknown reason",
        )

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


def _apply_map_layout_and_shapes(figure, settings, x_data, y_data, stage_outline=None, base_shapes=None):
    shapes = _clone_shapes(base_shapes or [])

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

    x_values = np.asarray(x_data, dtype=float)
    y_values = np.asarray(y_data, dtype=float)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(finite_mask):
        figure.update_layout(shapes=shapes)
        return figure

    x_plot = x_values[finite_mask]
    y_plot = y_values[finite_mask]
    xmin, xmax = float(np.min(x_plot)), float(np.max(x_plot))
    ymin, ymax = float(np.min(y_plot)), float(np.max(y_plot))

    span = max(xmax - xmin, ymax - ymin, 1e-6)
    scale_factor = 0.2

    figure.update_layout(
        shapes=shapes,
        xaxis=dict(range=[xmin - scale_factor * span, xmax + scale_factor * span]),
        yaxis=dict(range=[ymin - scale_factor * span, ymax + scale_factor * span]),
    )
    return figure


def _resolve_batch_z_keys(file):
    return _resolve_non_spatial_columns(file)


def _safe_filename_fragment(value):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "plot"


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

    render_mode = _resolve_render_mode(settings)
    gradient_mode = settings.get("gradient_mode", "none")
    heatmap_grid_mode = _resolve_heatmap_grid_mode(settings)
    gradient_coordinate_mode = _resolve_gradient_coordinate_mode(settings)
    gradient_calc_grid_mode = _resolve_gradient_calc_grid_mode(settings)
    gradient_grid_size = settings.get("gradient_grid_size")
    gradient_k_nearest = settings.get("gradient_k_nearest")
    use_gradient_map = gradient_mode != "none"
    gradient_result = (
        _build_gradient_map(
            x_data,
            y_data,
            z_data,
            gradient_mode,
            grid_mode=gradient_calc_grid_mode,
            grid_size=gradient_grid_size,
            k_nearest=gradient_k_nearest,
            polar_center=(settings.get("mappattern_x"), settings.get("mappattern_y")),
        )
        if use_gradient_map
        else None
    )

    heatmap_result = None
    plot_values = np.asarray(z_data, dtype=float)
    if render_mode == "heatmap":
        if use_gradient_map and gradient_result:
            heatmap_result = gradient_result
            _, _, heatmap_values = heatmap_result
            active_data = heatmap_values[np.isfinite(heatmap_values)]
            colorbar_label = _gradient_label(gradient_mode, z_label)
        else:
            value_map = _build_value_map(
                x_data,
                y_data,
                z_data,
                grid_mode=heatmap_grid_mode,
                grid_size=gradient_grid_size,
                k_nearest=gradient_k_nearest,
            )
            if value_map:
                heatmap_result = value_map
                _, _, heatmap_values = heatmap_result
                active_data = heatmap_values[np.isfinite(heatmap_values)]
                colorbar_label = z_label
                use_gradient_map = False
            else:
                render_mode = "markers"
                use_gradient_map = False
                active_data = z_data[np.isfinite(z_data)]
                colorbar_label = z_label
    elif use_gradient_map and gradient_result:
        xi, yi, gradient_grid = gradient_result
        sampled_gradient = _sample_grid_at_points(x_data, y_data, xi, yi, gradient_grid)
        sampled_mask = np.isfinite(sampled_gradient)
        if np.any(sampled_mask):
            plot_values = sampled_gradient
            active_data = sampled_gradient[sampled_mask]
            colorbar_label = _gradient_label(gradient_mode, z_label)
        else:
            use_gradient_map = False
            active_data = z_data[np.isfinite(z_data)]
            colorbar_label = z_label
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

    if render_mode == "heatmap" and heatmap_result:
        xi, yi, heatmap_values = heatmap_result
        heatmap_trace = dict(
            x=xi,
            y=yi,
            z=heatmap_values,
            colorscale=settings["colormap_value"],
            colorbar=colorbar_title,
            hovertemplate=f"x: %{{x}}<br>y: %{{y}}<br>{colorbar_label}: %{{z}}<extra></extra>",
            showscale=True,
            connectgaps=False,
        )
        if use_gradient_map and gradient_coordinate_mode == "polar":
            heatmap_trace["zsmooth"] = "best"
        if manual_scale:
            heatmap_trace.update(zmin=zmin, zmax=zmax, zauto=False)

        figure.add_trace(go.Heatmap(**heatmap_trace))
    else:
        finite_mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(plot_values)
        x_plot = x_data[finite_mask]
        y_plot = y_data[finite_mask]
        values_plot = plot_values[finite_mask]
        is_ellipse = settings["marker_type"] == "ellipse"

        marker = dict(
            size=settings["marker_size"],
            opacity=0 if is_ellipse else 1,
            color=values_plot,
            colorscale=settings["colormap_value"],
            colorbar=colorbar_title,
            showscale=True,
        )
        if manual_scale:
            marker.update(cmin=zmin, cmax=zmax, cauto=False)

        primary_trace = dict(
            x=x_plot,
            y=y_plot,
            mode='markers',
            marker=marker,
            hovertemplate=f"x: %{{x}}<br>y: %{{y}}<br>{colorbar_label}: %{{marker.color}}<extra></extra>",
            showlegend=False,
        )

        figure.add_trace(go.Scatter(**primary_trace))

        if is_ellipse and values_plot.size:
            d_min, d_max = zmin, zmax

            if d_max > d_min:
                norm_values = (values_plot - d_min) / (d_max - d_min)
                norm_values = np.clip(norm_values, 0, 1)
            else:
                norm_values = np.zeros_like(values_plot, dtype=float)
            colors = px.colors.sample_colorscale(
                colorscale=settings["colormap_value"],
                samplepoints=norm_values
            )

            shapes.extend(
                [gen_spot(x, y, c, settings["spot_size"], settings["angle_of_incident"]) for x, y, c in zip(x_plot, y_plot, colors)]
            )

    _apply_map_layout_and_shapes(
        figure,
        settings,
        x_data,
        y_data,
        stage_outline=stage_outline,
        base_shapes=shapes,
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
    z_options = _resolve_non_spatial_columns(file)

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

    settings = {**DEFAULT_SETTINGS, **(settings or {})}

    file = Ellipsometry.from_path_or_stream(uploaded_files[selected_file])

    z_keys = _resolve_batch_z_keys(file)
    if not z_keys:
        logger.warning("Batch main plot payload skipped: no z-parameter columns found in '%s'.", selected_file)
        return no_update

    stage_outline = None
    if settings.get("stage_state"):
        cwd = os.getcwd()
        dxf_file = os.path.join(cwd, "src/assets/jaw_stage_outline.dxf")
        stage_outline = dxf_to_path(dxf_file)

    root_name = _safe_filename_fragment(os.path.splitext(selected_file)[0])
    payload = []
    for z_key in z_keys:
        per_plot_settings = {**settings, "z_data_value": z_key}
        figure, _ = _build_main_figure(
            file,
            per_plot_settings,
            z_label=z_key,
            force_two_sigma=True,
            stage_outline=stage_outline,
        )

        plot_name = _safe_filename_fragment(z_key)
        payload.append(
            {
                "filename": f"{root_name}_{plot_name}.png",
                "figure": json.loads(figure.to_json()),
            }
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

        if (typeof Plotly === "undefined" || typeof Plotly.toImage !== "function") {
            return "Batch export failed: Plotly image export API is unavailable in the browser.";
        }

        if (typeof window.JSZip === "undefined") {
            return "Batch export failed: JSZip is unavailable in the browser.";
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

        try {
            const zip = new window.JSZip();
            let plotted = false;
            for (let idx = 0; idx < plots.length; idx++) {
                const plot = plots[idx];
                const figure = plot.figure || {};
                const data = figure.data || [];
                const layout = figure.layout || {};
                const filename = (plot.filename || `batch_plot_${idx + 1}.png`).replace(/\\.png$/i, "");

                if (!plotted) {
                    await Plotly.newPlot(tempDiv, data, layout, {displayModeBar: false, responsive: false, staticPlot: true});
                    plotted = true;
                } else {
                    await Plotly.react(tempDiv, data, layout, {displayModeBar: false, responsive: false, staticPlot: true});
                }

                const pngDataUrl = await Plotly.toImage(tempDiv, {
                    format: "png",
                    width: width,
                    height: height,
                    scale: scale
                });
                const base64Data = (pngDataUrl || "").split(",")[1];
                if (!base64Data) {
                    throw new Error(`PNG export failed for ${filename}.png.`);
                }
                zip.file(`${filename}.png`, base64Data, {base64: true});
            }

            const zipBlob = await zip.generateAsync({type: "blob"});
            const zipUrl = URL.createObjectURL(zipBlob);
            const anchor = document.createElement("a");
            anchor.href = zipUrl;
            anchor.download = "batch_plots.zip";
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            URL.revokeObjectURL(zipUrl);

            return `Batch main plots downloaded (batch_plots.zip with ${plots.length} PNG files).`;
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
