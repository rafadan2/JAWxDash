from dash import callback, clientside_callback, dcc, html, no_update, Output, Input, State, ALL, MATCH
import json
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from itertools import zip_longest
import re
import time


from src import ids
from src.callbacks.graph_callbacks import (
    _build_gradient_map,
    _gradient_label,
    _resolve_gradient_calc_grid_mode,
    _resolve_z_key,
    _resolve_z_limits,
)
from src.ellipsometry_toolbox.ellipsometry import Ellipsometry
from src.ellipsometry_toolbox.linear_translations import rotate, translate
from src.ellipsometry_toolbox.masking import (
    create_masked_file,
    radial_edge_exclusion_outline,
    uniform_edge_exclusion_outline,
)
from src.utils.sample_outlines import generate_outline
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


def _safe_filename_fragment(value):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "plot"


def _resolve_spatial_batch_z_keys(file):
    return [column for column in sorted(file.get_column_names()) if column.lower() not in {"x", "y"}]


def _normalize_spatial_batch_mode(mode_raw):
    mode_text = str(mode_raw or "").strip().lower()
    if mode_text in {"percent", "percentage", "pct", "%", "p"}:
        return "percent"
    if mode_text in {"sigma", "sig", "s", "\u03c3"}:
        return "sigma"
    return None


def _default_spatial_batch_variation_value(mode):
    return 2.0 if mode == "sigma" else 5.0


def _normalize_spatial_batch_z_name(z_name):
    collapsed = re.sub(r"[^a-z0-9]+", " ", str(z_name or "").lower())
    return " ".join(collapsed.split())


def _spatial_batch_default_preset(z_name, fallback_mode):
    mode = fallback_mode if fallback_mode in {"percent", "sigma"} else "percent"
    value = _default_spatial_batch_variation_value(mode)
    excluded = False

    normalized = _normalize_spatial_batch_z_name(z_name)

    if normalized in {"a", "b", "mse"}:
        return dict(mode="sigma", value=2.0, excluded=False)

    if normalized in {"fit ok", "hardware ok", "sigint", "tilt x", "tilt y", "z align"}:
        return dict(mode=mode, value=value, excluded=True)

    if re.fullmatch(r"point(?:\s+(?:number|no|num))?", normalized):
        return dict(mode=mode, value=value, excluded=True)

    if "n of cauchy" in normalized:
        return dict(mode="sigma", value=2.0, excluded=False)

    return dict(mode=mode, value=value, excluded=excluded)


def _parse_spatial_batch_component_rules(
    mode_values,
    mode_ids,
    value_values,
    value_ids,
    exclude_values,
    exclude_ids,
):
    rules = {}
    excluded_z = set()
    errors = []
    mode_values = mode_values or []
    mode_ids = mode_ids or []
    value_values = value_values or []
    value_ids = value_ids or []
    exclude_values = exclude_values or []
    exclude_ids = exclude_ids or []
    if not mode_ids and not value_ids and not exclude_ids:
        return rules, excluded_z, errors

    modes_by_z = {}
    values_by_z = {}

    for idx, component_id in enumerate(mode_ids, start=1):
        if not isinstance(component_id, dict):
            errors.append(f"row {idx}: invalid mode selector id")
            continue

        z_name = str(component_id.get("z", "")).strip()
        if not z_name:
            continue

        mode_raw = mode_values[idx - 1] if idx - 1 < len(mode_values) else None
        mode = _normalize_spatial_batch_mode(mode_raw)
        if not mode:
            errors.append(f"row {idx} ({z_name}): unknown mode '{mode_raw}'")
            continue
        modes_by_z[z_name.lower()] = mode

    for idx, component_id in enumerate(value_ids, start=1):
        if not isinstance(component_id, dict):
            errors.append(f"row {idx}: invalid value input id")
            continue

        z_name = str(component_id.get("z", "")).strip()
        if not z_name:
            continue

        value_raw = value_values[idx - 1] if idx - 1 < len(value_values) else None
        value = _safe_float(value_raw, default=np.nan)
        if not np.isfinite(value) or value < 0:
            errors.append(f"row {idx} ({z_name}): invalid value '{value_raw}'")
            continue
        values_by_z[z_name.lower()] = float(value)

    for idx, component_id in enumerate(exclude_ids, start=1):
        if not isinstance(component_id, dict):
            errors.append(f"row {idx}: invalid exclude checkbox id")
            continue

        z_name = str(component_id.get("z", "")).strip()
        if not z_name:
            continue

        exclude_raw = exclude_values[idx - 1] if idx - 1 < len(exclude_values) else []
        is_excluded = False
        if isinstance(exclude_raw, (list, tuple, set)):
            is_excluded = "exclude" in exclude_raw
        elif isinstance(exclude_raw, bool):
            is_excluded = exclude_raw

        if is_excluded:
            excluded_z.add(z_name.lower())

    for z_name in set(modes_by_z).intersection(values_by_z):
        rules[z_name] = dict(mode=modes_by_z[z_name], value=values_by_z[z_name])

    for z_name in set(modes_by_z).difference(values_by_z):
        errors.append(f"{z_name}: missing value")
    for z_name in set(values_by_z).difference(modes_by_z):
        errors.append(f"{z_name}: missing mode")

    return rules, excluded_z, errors


def _children_as_list(component):
    if component is None:
        return []
    children = getattr(component, "children", None)
    if children is None:
        return []
    if isinstance(children, (list, tuple)):
        return list(children)
    return [children]


def _component_text(component):
    if component is None:
        return ""
    if isinstance(component, (str, int, float, np.number)):
        return str(component).strip()

    children = getattr(component, "children", None)
    if children is None:
        return ""

    parts = []
    if isinstance(children, (list, tuple)):
        for child in children:
            text = _component_text(child)
            if text:
                parts.append(text)
    else:
        text = _component_text(children)
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _component_type_name(component):
    return getattr(getattr(component, "__class__", None), "__name__", "").lower()


def _find_first_component_by_type(component, type_name):
    if component is None:
        return None
    if _component_type_name(component) == str(type_name or "").lower():
        return component

    for child in _children_as_list(component):
        found = _find_first_component_by_type(child, type_name)
        if found is not None:
            return found
    return None


def _extract_html_table_data(table_component):
    headers = []
    rows = []
    if table_component is None:
        return headers, rows

    table_children = _children_as_list(table_component)
    thead = next((child for child in table_children if _component_type_name(child) == "thead"), None)
    tbody = next((child for child in table_children if _component_type_name(child) == "tbody"), None)

    if thead is not None:
        header_rows = _children_as_list(thead)
        if header_rows:
            first_header_row = header_rows[0]
            headers = [_component_text(cell) for cell in _children_as_list(first_header_row)]

    if tbody is not None:
        for row_component in _children_as_list(tbody):
            row_values = [_component_text(cell) for cell in _children_as_list(row_component)]
            if any(value.strip() for value in row_values):
                rows.append(row_values)

    return headers, rows


def _rows_to_columns(rows, column_count):
    row_list = [list(row) for row in (rows or [])]
    if column_count < 1:
        column_count = 1

    if not row_list:
        row_list = [[""] * column_count]

    normalized_rows = []
    for row in row_list:
        padded = list(row[:column_count])
        if len(padded) < column_count:
            padded.extend([""] * (column_count - len(padded)))
        normalized_rows.append(padded)

    return [list(column) for column in zip_longest(*normalized_rows, fillvalue="")]


def _table_fill_colors(column_count, row_count):
    row_colors = [
        "rgb(255,255,255)" if idx % 2 == 0 else "rgb(248,249,250)"
        for idx in range(max(row_count, 1))
    ]
    return [row_colors for _ in range(max(column_count, 1))]


def _parse_percentage_from_text(value):
    match = re.search(r"[-+]?\d*\.?\d+", str(value or ""))
    if not match:
        return np.nan
    return _safe_float(match.group(0), default=np.nan)


def _conformance_pct_color(percent_value):
    if not np.isfinite(percent_value):
        return "rgb(33,37,41)"
    if percent_value >= 95.0:
        return "rgb(25,135,84)"
    if percent_value >= 80.0:
        return "rgb(255,193,7)"
    return "rgb(220,53,69)"


def _build_summary_card_overlays(summary_rows):
    context_text = ""
    cards = []
    for label, value in summary_rows or []:
        label_text = str(label or "").strip()
        value_text = str(value or "").strip()
        if not label_text and not value_text:
            continue
        if label_text.lower() == "context":
            context_text = value_text
            continue
        cards.append((label_text or "Summary", value_text or "-"))

    if not cards:
        cards = [("Summary", "No summary values available.")]

    card_count = len(cards)
    card_columns = min(4, max(card_count, 1))
    card_rows = int(np.ceil(card_count / card_columns))

    left_margin = 0.02
    right_margin = 0.98
    horizontal_gap = 0.018 if card_columns > 1 else 0.0
    top_bound = 0.80 if context_text else 0.94
    bottom_bound = 0.08
    vertical_gap = 0.08 if card_rows > 1 else 0.0

    card_width = (
        (right_margin - left_margin - horizontal_gap * (card_columns - 1))
        / max(card_columns, 1)
    )
    card_height = (
        (top_bound - bottom_bound - vertical_gap * (card_rows - 1))
        / max(card_rows, 1)
    )

    shapes = []
    annotations = []

    if context_text:
        annotations.append(
            dict(
                x=0.5,
                y=0.98,
                xref="x3",
                yref="y3",
                text=context_text,
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=11, color="rgb(108,117,125)"),
            )
        )

    for card_idx, (label, value) in enumerate(cards):
        row_idx = card_idx // card_columns
        col_idx = card_idx % card_columns

        x0 = left_margin + col_idx * (card_width + horizontal_gap)
        x1 = x0 + card_width
        y1 = top_bound - row_idx * (card_height + vertical_gap)
        y0 = y1 - card_height

        shapes.append(
            dict(
                type="rect",
                xref="x3",
                yref="y3",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(color="rgb(222,226,230)", width=1),
                fillcolor="rgb(248,249,250)",
                layer="below",
            )
        )

        pad_x = card_width * 0.04
        annotations.append(
            dict(
                x=x0 + pad_x,
                y=y1 - (card_height * 0.15),
                xref="x3",
                yref="y3",
                text=str(label).upper(),
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                font=dict(size=10, color="rgb(108,117,125)"),
            )
        )
        annotations.append(
            dict(
                x=x0 + pad_x,
                y=y0 + (card_height * 0.34),
                xref="x3",
                yref="y3",
                text=f"<b>{value}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                align="left",
                font=dict(size=12, color="rgb(33,37,41)"),
            )
        )

    return shapes, annotations


def _extract_spatial_summary_snapshot(count_panel):
    summary = {
        "title": "Spatial bin summary",
        "subtitle": "",
        "summary_rows": [],
        "radial_title": "Radial boundaries (mm)",
        "radial_subtitle": "",
        "radial_headers": [],
        "radial_rows": [],
        "conformance_title": "Conformance by section",
        "conformance_headers": [],
        "conformance_rows": [],
    }

    if isinstance(count_panel, str):
        summary["summary_rows"] = [("Summary", count_panel)]
        return summary

    root_children = _children_as_list(count_panel)
    if not root_children:
        return summary

    title_text = _component_text(root_children[0]) if len(root_children) > 0 else ""
    subtitle_text = _component_text(root_children[1]) if len(root_children) > 1 else ""
    if title_text:
        summary["title"] = title_text
    summary["subtitle"] = subtitle_text

    if len(root_children) > 2:
        cards_container = root_children[2]
        for card in _children_as_list(cards_container):
            card_children = _children_as_list(card)
            if len(card_children) < 2:
                continue
            label = _component_text(card_children[0])
            value = _component_text(card_children[1])
            if label or value:
                summary["summary_rows"].append((label, value))

    if summary["subtitle"]:
        summary["summary_rows"].insert(0, ("Context", summary["subtitle"]))

    if len(root_children) > 3:
        sections_row = root_children[3]
        section_columns = _children_as_list(sections_row)

        if len(section_columns) > 0:
            radial_column = section_columns[0]
            radial_children = _children_as_list(radial_column)
            if radial_children:
                radial_title = _component_text(radial_children[0])
                if radial_title:
                    summary["radial_title"] = radial_title
            if len(radial_children) > 1:
                summary["radial_subtitle"] = _component_text(radial_children[1])

            radial_table = _find_first_component_by_type(radial_column, "table")
            radial_headers, radial_rows = _extract_html_table_data(radial_table)
            summary["radial_headers"] = radial_headers
            summary["radial_rows"] = radial_rows

        if len(section_columns) > 1:
            conformance_column = section_columns[1]
            conformance_children = _children_as_list(conformance_column)
            if conformance_children:
                conformance_title = _component_text(conformance_children[0])
                if conformance_title:
                    summary["conformance_title"] = conformance_title

            conformance_table = _find_first_component_by_type(conformance_column, "table")
            conformance_headers, conformance_rows = _extract_html_table_data(conformance_table)
            summary["conformance_headers"] = conformance_headers
            summary["conformance_rows"] = conformance_rows

    if not summary["summary_rows"]:
        summary["summary_rows"] = [("Summary", "No summary values available.")]

    if not summary["radial_rows"]:
        summary["radial_rows"] = [["No radial boundary data available.", "", ""]]
    if not summary["conformance_rows"]:
        summary["conformance_rows"] = [["No conformance data available.", "", "", "", ""]]

    return summary


def _copy_axis_layout(target_axis, source_axis):
    if source_axis is None:
        return
    source_dict = source_axis.to_plotly_json()
    blocked_keys = {"domain", "anchor", "overlaying", "position", "matches"}
    for key, value in source_dict.items():
        if key in blocked_keys:
            continue
        target_axis[key] = value


def _remap_axis_reference(reference, axis_letter):
    if not isinstance(reference, str):
        return reference
    if reference == axis_letter:
        return f"{axis_letter}2"
    if reference == f"{axis_letter} domain":
        return f"{axis_letter}2 domain"
    return reference


def _build_spatial_batch_snapshot_figure(z_key, map_fig, trend_fig, count_panel):
    summary = _extract_spatial_summary_snapshot(count_panel)
    summary_rows = summary["summary_rows"]
    summary_shapes, summary_annotations = _build_summary_card_overlays(summary_rows)

    radial_headers = summary["radial_headers"] or ["Bin", "r min", "r max"]
    radial_rows = summary["radial_rows"]
    radial_display_rows = []
    for row in radial_rows:
        normalized_row = list(row[: len(radial_headers)])
        if len(normalized_row) < len(radial_headers):
            normalized_row.extend([""] * (len(radial_headers) - len(normalized_row)))
        if normalized_row and str(normalized_row[0]).strip():
            normalized_row[0] = f"<b>{normalized_row[0]}</b>"
        radial_display_rows.append(normalized_row)
    radial_columns = _rows_to_columns(radial_display_rows, len(radial_headers))

    conformance_headers = summary["conformance_headers"] or [
        "Section",
        "Region",
        "Points",
        "Conformal",
        "Conformal %",
    ]
    conformance_rows = summary["conformance_rows"]
    conformance_pct_index = next(
        (idx for idx, header in enumerate(conformance_headers) if "%" in str(header)),
        None,
    )
    if conformance_pct_index is None and conformance_headers:
        conformance_pct_index = len(conformance_headers) - 1

    conformance_display_rows = []
    conformance_row_fill = []
    conformance_pct_colors = []
    for row_idx, row in enumerate(conformance_rows):
        normalized_row = list(row[: len(conformance_headers)])
        if len(normalized_row) < len(conformance_headers):
            normalized_row.extend([""] * (len(conformance_headers) - len(normalized_row)))

        section_text = str(normalized_row[0]).strip().lower() if normalized_row else ""
        region_text = str(normalized_row[1]).strip().lower() if len(normalized_row) > 1 else ""
        is_edge_row = section_text == "ee" or "edge excluded" in region_text
        if is_edge_row:
            conformance_row_fill.append("rgb(233,236,239)")
        else:
            conformance_row_fill.append("rgb(255,255,255)" if row_idx % 2 == 0 else "rgb(248,249,250)")

        if normalized_row and str(normalized_row[0]).strip():
            normalized_row[0] = f"<b>{normalized_row[0]}</b>"

        pct_color = "rgb(33,37,41)"
        if conformance_pct_index is not None and conformance_pct_index < len(normalized_row):
            pct_text = str(normalized_row[conformance_pct_index]).strip()
            pct_value = _parse_percentage_from_text(pct_text)
            pct_color = _conformance_pct_color(pct_value)
            if pct_text:
                normalized_row[conformance_pct_index] = f"<b>{pct_text}</b>"
        conformance_pct_colors.append(pct_color)
        conformance_display_rows.append(normalized_row)

    conformance_columns = _rows_to_columns(conformance_display_rows, len(conformance_headers))
    conformance_fill_colors = [list(conformance_row_fill) for _ in range(max(len(conformance_headers), 1))]
    conformance_font_colors = [
        ["rgb(33,37,41)" for _ in range(max(len(conformance_display_rows), 1))]
        for _ in range(max(len(conformance_headers), 1))
    ]
    if (
        conformance_pct_index is not None
        and conformance_pct_index < len(conformance_font_colors)
        and conformance_pct_colors
    ):
        for row_idx, pct_color in enumerate(conformance_pct_colors):
            if row_idx < len(conformance_font_colors[conformance_pct_index]):
                conformance_font_colors[conformance_pct_index][row_idx] = pct_color

    snapshot_fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "table"}, {"type": "table"}],
        ],
        row_heights=[0.48, 0.24, 0.28],
        column_widths=[0.66, 0.34],
        horizontal_spacing=0.14,
        vertical_spacing=0.08,
        subplot_titles=(
            "Spatial bin map",
            "Data and trend by spatial section",
            summary["title"],
            summary["radial_title"],
            summary["conformance_title"],
        ),
    )
    base_subplot_annotations = [
        annotation.to_plotly_json() for annotation in (snapshot_fig.layout.annotations or [])
    ]

    for trace in map_fig.data:
        snapshot_fig.add_trace(trace, row=1, col=1)
    for trace in trend_fig.data:
        snapshot_fig.add_trace(trace, row=1, col=2)

    snapshot_fig.update_xaxes(
        row=2,
        col=1,
        visible=False,
        range=[0, 1],
        fixedrange=True,
        showgrid=False,
        zeroline=False,
    )
    snapshot_fig.update_yaxes(
        row=2,
        col=1,
        visible=False,
        range=[0, 1],
        fixedrange=True,
        showgrid=False,
        zeroline=False,
    )

    snapshot_fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{header}</b>" for header in radial_headers],
                fill_color="rgb(240,240,240)",
                align="left",
                font=dict(size=12),
            ),
            cells=dict(
                values=radial_columns,
                align="left",
                font=dict(size=11),
                height=22,
                fill_color=_table_fill_colors(len(radial_headers), len(radial_display_rows)),
            ),
        ),
        row=3,
        col=1,
    )

    snapshot_fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{header}</b>" for header in conformance_headers],
                fill_color="rgb(240,240,240)",
                align="left",
                font=dict(size=12),
            ),
            cells=dict(
                values=conformance_columns,
                align="left",
                font=dict(size=11, color=conformance_font_colors),
                height=22,
                fill_color=conformance_fill_colors,
            ),
        ),
        row=3,
        col=2,
    )

    _copy_axis_layout(snapshot_fig.layout.xaxis, map_fig.layout.xaxis)
    _copy_axis_layout(snapshot_fig.layout.yaxis, map_fig.layout.yaxis)
    _copy_axis_layout(snapshot_fig.layout.xaxis2, trend_fig.layout.xaxis)
    _copy_axis_layout(snapshot_fig.layout.yaxis2, trend_fig.layout.yaxis)

    map_shapes = []
    for shape in (map_fig.layout.shapes or []):
        map_shapes.append(shape.to_plotly_json())

    trend_shapes = []
    for shape in (trend_fig.layout.shapes or []):
        shape_json = shape.to_plotly_json()
        shape_json["xref"] = _remap_axis_reference(shape_json.get("xref"), "x")
        shape_json["yref"] = _remap_axis_reference(shape_json.get("yref"), "y")
        trend_shapes.append(shape_json)

    map_annotations = []
    for annotation in (map_fig.layout.annotations or []):
        map_annotations.append(annotation.to_plotly_json())

    trend_annotations = []
    for annotation in (trend_fig.layout.annotations or []):
        annotation_json = annotation.to_plotly_json()
        annotation_json["xref"] = _remap_axis_reference(annotation_json.get("xref"), "x")
        annotation_json["yref"] = _remap_axis_reference(annotation_json.get("yref"), "y")
        trend_annotations.append(annotation_json)

    layout_updates = dict(
        template="plotly_white",
        title=dict(text=f"Spatial binning snapshot: {z_key}", x=0.5, xanchor="center", pad=dict(t=10, b=6)),
        margin=dict(l=55, r=40, t=100, b=40),
        showlegend=False,
    )

    map_coloraxis = getattr(map_fig.layout, "coloraxis", None)
    if map_coloraxis is not None:
        coloraxis_json = map_coloraxis.to_plotly_json()
        if coloraxis_json:
            x_domain = getattr(snapshot_fig.layout.xaxis, "domain", None)
            y_domain = getattr(snapshot_fig.layout.yaxis, "domain", None)
            colorbar = coloraxis_json.get("colorbar", {})
            if (
                isinstance(x_domain, (list, tuple))
                and len(x_domain) == 2
                and isinstance(y_domain, (list, tuple))
                and len(y_domain) == 2
            ):
                colorbar.update(
                    x=min(float(x_domain[1]) + 0.012, 0.99),
                    xanchor="left",
                    y=float((y_domain[0] + y_domain[1]) / 2.0),
                    yanchor="middle",
                    len=float(y_domain[1] - y_domain[0]),
                    lenmode="fraction",
                    thickness=16,
                )
            coloraxis_json["colorbar"] = colorbar
            layout_updates["coloraxis"] = coloraxis_json

    snapshot_fig.update_layout(
        **layout_updates,
        shapes=map_shapes + trend_shapes + summary_shapes,
        annotations=base_subplot_annotations + map_annotations + trend_annotations + summary_annotations,
    )

    return snapshot_fig


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
    Output(ids.Div.SPATIAL_BIN_ACCEPTED_VARIATION_UNIT, "children"),
    Input(ids.RadioItems.SPATIAL_BIN_ACCEPTED_VARIATION_MODE, "value"),
)
def update_spatial_bin_variation_unit(variation_mode):
    return "\u03c3" if variation_mode == "sigma" else "%"


@callback(
    Output(ids.Input.SPATIAL_BIN_ACCEPTED_VARIATION, "value"),
    Input(ids.RadioItems.SPATIAL_BIN_ACCEPTED_VARIATION_MODE, "value"),
)
def update_spatial_bin_variation_default(variation_mode):
    return 2.0 if variation_mode == "sigma" else 5.0


@callback(
    Output(ids.Input.SPATIAL_BIN_BATCH_RULES, "children"),
    Input(ids.DropDown.UPLOADED_FILES, "value"),
    Input(ids.Store.UPLOADED_FILES, "data"),
    State(ids.RadioItems.SPATIAL_BIN_ACCEPTED_VARIATION_MODE, "value"),
)
def populate_spatial_batch_rules_table(
    selected_file,
    stored_files,
    accepted_variation_mode,
):
    if not selected_file or not stored_files or selected_file not in stored_files:
        return html.Div("Select a file to configure batch rules.", className="text-muted small")

    file = Ellipsometry.from_path_or_stream(stored_files[selected_file])
    z_keys = _resolve_spatial_batch_z_keys(file)
    if not z_keys:
        return html.Div("No Z parameters available for batch export.", className="text-muted small")

    default_mode = accepted_variation_mode if accepted_variation_mode in {"percent", "sigma"} else "percent"

    rows = []
    for z_key in z_keys:
        preset = _spatial_batch_default_preset(z_key, default_mode)
        rows.append(
            html.Tr(
                [
                    html.Td(z_key, className="fw-semibold"),
                    html.Td(
                        dcc.RadioItems(
                            id={"type": "spatial_batch_mode", "z": z_key},
                            options=[
                                {"label": "Percent", "value": "percent"},
                                {"label": "\u03c3", "value": "sigma"},
                            ],
                            value=preset["mode"],
                            inline=True,
                            className="mb-0",
                            labelStyle={"margin-right": "14px", "margin-bottom": "0"},
                        )
                    ),
                    html.Td(
                        dcc.Input(
                            id={"type": "spatial_batch_value", "z": z_key},
                            type="number",
                            value=float(preset["value"]),
                            min=0,
                            step=0.1,
                            debounce=True,
                            className="form-control form-control-sm",
                        ),
                        style={"minWidth": "120px", "maxWidth": "180px"},
                    ),
                    html.Td(
                        dcc.Checklist(
                            id={"type": "spatial_batch_exclude", "z": z_key},
                            options=[{"label": "", "value": "exclude"}],
                            value=["exclude"] if preset["excluded"] else [],
                            className="mb-0",
                            style={"display": "flex", "justifyContent": "center"},
                        ),
                    ),
                ]
            )
        )

    return html.Div(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Z parameter"),
                            html.Th("Mode"),
                            html.Th("Value"),
                            html.Th("Exclude"),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            className="table table-sm table-striped table-bordered align-middle mb-0",
        ),
        className="table-responsive",
    )


@callback(
    Output({"type": "spatial_batch_value", "z": MATCH}, "value"),
    Input({"type": "spatial_batch_mode", "z": MATCH}, "value"),
    prevent_initial_call=True,
)
def update_spatial_batch_row_default_value(mode):
    return _default_spatial_batch_variation_value(mode)


@callback(
    Output(ids.Graph.SPATIAL_BIN_MAP, "figure"),
    Output(ids.Graph.SPATIAL_BIN_TRENDS, "figure"),
    Output(ids.Div.SPATIAL_BIN_COUNTS, "children"),
    Input(ids.Tabs.ANALYSIS, "active_tab"),
    Input(ids.DropDown.UPLOADED_FILES, "value"),
    Input(ids.Store.SETTINGS, "data"),
    Input(ids.Input.SPATIAL_BIN_RADIAL_COUNT, "value"),
    Input(ids.Input.SPATIAL_BIN_ANGULAR_COUNT, "value"),
    Input(ids.Input.SPATIAL_BIN_ACCEPTED_VARIATION, "value"),
    Input(ids.RadioItems.SPATIAL_BIN_ACCEPTED_VARIATION_MODE, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
)
def update_spatial_binning_tab(
    active_tab,
    selected_file,
    settings,
    radial_bin_count,
    angular_bin_count,
    accepted_variation_value,
    accepted_variation_mode,
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
        n_radial = 1
    n_radial = int(np.clip(n_radial, 1, 40))

    try:
        n_angular = int(angular_bin_count)
    except (TypeError, ValueError):
        n_angular = 4
    n_angular = int(np.clip(n_angular, 1, 72))

    variation_mode = accepted_variation_mode if accepted_variation_mode in {"percent", "sigma"} else "percent"
    default_variation = 2.0 if variation_mode == "sigma" else 5.0
    try:
        accepted_variation_value = float(accepted_variation_value)
    except (TypeError, ValueError):
        accepted_variation_value = default_variation
    accepted_variation_value = float(np.clip(accepted_variation_value, 0.0, 1000.0))

    reference_mask = include_mask if active_settings.get("ee_state") else np.ones(values.size, dtype=bool)
    if not np.any(reference_mask):
        reference_mask = np.ones(values.size, dtype=bool)

    reference_values = values[reference_mask]
    reference_median = float(np.median(reference_values)) if reference_values.size else float(np.median(values))
    reference_std = float(np.std(reference_values, ddof=1)) if reference_values.size > 1 else 0.0
    if variation_mode == "sigma":
        tolerance_abs = accepted_variation_value * reference_std
    else:
        tolerance_abs = abs(reference_median) * accepted_variation_value / 100.0
    lower_limit = reference_median - tolerance_abs
    upper_limit = reference_median + tolerance_abs
    if np.isclose(tolerance_abs, 0.0):
        conformal_mask = np.isclose(values, reference_median, atol=np.finfo(float).eps, rtol=0.0)
    else:
        conformal_mask = np.abs(values - reference_median) <= tolerance_abs

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
    interior_conformal_counts = np.asarray(
        [np.sum(conformal_mask & (bin_index == i)) for i in range(interior_bin_count)],
        dtype=int,
    )
    edge_conformal_count = int(np.sum(conformal_mask & (bin_index == edge_bin_index))) if active_settings.get("ee_state") else 0
    reference_conformal_total = int(np.sum(conformal_mask[reference_mask]))
    reference_total = int(np.sum(reference_mask))
    reference_conformal_pct = 100.0 * reference_conformal_total / reference_total if reference_total > 0 else 0.0

    map_fig = go.Figure()
    label_x = []
    label_y = []
    label_text = []
    map_active_values = values[np.isfinite(values)]
    force_two_sigma_map_scale = bool(active_settings.get("_spatial_force_two_sigma_z_scale"))
    map_zmin, map_zmax, map_manual_scale = _resolve_z_limits(
        map_active_values,
        active_settings.get("z_scale_min"),
        active_settings.get("z_scale_max"),
        force_two_sigma=force_two_sigma_map_scale,
    )
    map_colorbar = dict(title=dict(text=value_label, side="top"))
    map_coloraxis = dict(
        colorscale=active_settings["colormap_value"],
        colorbar=map_colorbar,
    )
    if map_manual_scale:
        map_coloraxis.update(cmin=map_zmin, cmax=map_zmax, cauto=False)

    interior_mask = bin_index < interior_bin_count
    if np.any(interior_mask):
        interior_bins = bin_index[interior_mask].astype(int)
        interior_radial_ids = interior_bins // n_angular + 1
        interior_angular_ids = interior_bins % n_angular + 1
        interior_customdata = np.column_stack(
            [
                interior_bins + 1,
                interior_radial_ids,
                interior_angular_ids,
                values[interior_mask],
            ]
        )
        interior_marker = dict(
            size=6,
            opacity=0.7,
            color=values[interior_mask],
            coloraxis="coloraxis",
        )
        map_fig.add_trace(
            go.Scatter(
                x=x_mm[interior_mask],
                y=y_mm[interior_mask],
                mode="markers",
                marker=interior_marker,
                name="Data points",
                showlegend=False,
                hovertemplate=(
                    "Section %{customdata[0]:.0f} (R%{customdata[1]:.0f}, A%{customdata[2]:.0f})"
                    + "<br>x: %{x:.3f} mm<br>y: %{y:.3f} mm<br>"
                    + f"{value_label}: %{{customdata[3]:.6g}}<extra></extra>"
                ),
                customdata=interior_customdata,
            )
        )

    for b in range(interior_bin_count):
        count = int(interior_counts[b])
        if count == 0:
            continue
        in_bin = bin_index == b
        label_x.append(float(np.mean(x_mm[in_bin])))
        label_y.append(float(np.mean(y_mm[in_bin])))
        label_text.append(str(b + 1))

    if active_settings.get("ee_state") and edge_count > 0:
        excluded_mask = bin_index == edge_bin_index
        edge_marker = dict(
            size=6,
            opacity=0.65,
            color=values[excluded_mask],
            coloraxis="coloraxis",
        )
        map_fig.add_trace(
            go.Scatter(
                x=x_mm[excluded_mask],
                y=y_mm[excluded_mask],
                mode="markers",
                marker=edge_marker,
                name="Edge excluded",
                showlegend=False,
                hovertemplate="Edge excluded<br>x: %{x:.3f} mm<br>y: %{y:.3f} mm<br>"
                + f"{value_label}: %{{customdata:.6g}}<extra></extra>",
                customdata=values[excluded_mask],
            )
        )

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

    outline_settings_mm = {**active_settings}
    for key in (
        "sample_x",
        "sample_y",
        "sample_radius",
        "sample_width",
        "sample_height",
        "ee_distance",
    ):
        outline_settings_mm[key] = _safe_float(active_settings.get(key), default=0.0) * 10.0

    if active_settings.get("sample_outline"):
        sample_outline_shape = generate_outline(outline_settings_mm)
        if isinstance(sample_outline_shape, dict) and sample_outline_shape.get("type"):
            map_fig.add_shape(sample_outline_shape)

    if active_settings.get("sample_outline") and active_settings.get("ee_state"):
        ee_shape = {}
        if active_settings.get("ee_type") == "radial":
            ee_shape = radial_edge_exclusion_outline(outline_settings_mm)
        elif active_settings.get("ee_type") == "uniform":
            ee_shape = uniform_edge_exclusion_outline(outline_settings_mm)
        if isinstance(ee_shape, dict) and ee_shape.get("type"):
            map_fig.add_shape(ee_shape)

    map_fig.add_trace(
        go.Scatter(
            x=[x0_mm],
            y=[y0_mm],
            mode="markers",
            marker=dict(symbol="x", size=11, color="black"),
            name="Bin center",
            showlegend=False,
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
                textfont=dict(size=13, color="rgba(193, 0, 1, 1)"),
                name="Bin labels",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    map_fig.update_layout(
        template="plotly_white",
        title=dict(text="Spatial bin map (radial x angular)", x=0.5, xanchor="center", pad=dict(t=10, b=6)),
        margin=dict(l=55, r=30, t=95, b=85),
        coloraxis=map_coloraxis,
        showlegend=False,
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

    finite_values = values[np.isfinite(values)]
    if finite_values.size > 0:
        raw_y_min = float(np.min(finite_values))
        raw_y_max = float(np.max(finite_values))
    else:
        raw_y_min = lower_limit
        raw_y_max = upper_limit
    span = raw_y_max - raw_y_min
    y_padding = 0.08 * span if span > 0 else max(abs(reference_median) * 0.1, 1.0)
    plot_y_min = min(raw_y_min, lower_limit) - y_padding
    plot_y_max = max(raw_y_max, upper_limit) + y_padding

    if lower_limit > plot_y_min:
        trend_fig.add_hrect(
            y0=plot_y_min,
            y1=lower_limit,
            fillcolor="rgba(220, 30, 30, 0.12)",
            line_width=0,
            layer="below",
        )
    if upper_limit < plot_y_max:
        trend_fig.add_hrect(
            y0=upper_limit,
            y1=plot_y_max,
            fillcolor="rgba(220, 30, 30, 0.12)",
            line_width=0,
            layer="below",
        )

    trend_fig.add_hline(
        y=reference_median,
        line=dict(color="rgba(25,25,25,0.75)", width=1.5, dash="dash"),
    )
    if np.isclose(lower_limit, upper_limit):
        trend_fig.add_hline(
            y=upper_limit,
            line=dict(color="rgba(175,20,20,0.7)", width=1.2, dash="dot"),
        )
    else:
        trend_fig.add_hline(
            y=lower_limit,
            line=dict(color="rgba(175,20,20,0.7)", width=1.2, dash="dot"),
        )
        trend_fig.add_hline(
            y=upper_limit,
            line=dict(color="rgba(175,20,20,0.7)", width=1.2, dash="dot"),
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
    trend_fig.update_yaxes(
        title_text=value_label,
        automargin=True,
        title_standoff=8,
        range=[plot_y_min, plot_y_max],
    )

    reference_scope = "non-EE points" if active_settings.get("ee_state") else "all points"
    if variation_mode == "sigma":
        variation_window = f"+/-{accepted_variation_value:.3g}\u03c3"
        variation_detail = f"Reference \u03c3: {reference_std:.6g} {value_label}"
    else:
        variation_window = f"+/-{accepted_variation_value:.3g}%"
        variation_detail = "Reference \u03c3 available when mode is \u03c3"

    def _summary_card(label, value):
        return html.Div(
            [
                html.Div(label, className="text-uppercase text-muted small"),
                html.Div(value, className="fw-semibold"),
            ],
            className="col-12 col-sm-6 col-xl-3 border rounded p-2 bg-light",
        )

    radial_rows = []
    for radial_idx in range(n_radial):
        r_min = radial_edges[radial_idx]
        r_max = radial_edges[radial_idx + 1]
        radial_rows.append(
            html.Tr(
                [
                    html.Td(f"R{radial_idx + 1}"),
                    html.Td(f"{r_min:.4g}"),
                    html.Td(f"{r_max:.4g}"),
                ]
            )
        )

    bin_rows = []
    for b in range(interior_bin_count):
        radial_id = b // n_angular + 1
        angular_id = b % n_angular + 1
        bin_total = int(interior_counts[b])
        bin_conformal = int(interior_conformal_counts[b])
        bin_conformal_pct = 100.0 * bin_conformal / bin_total if bin_total > 0 else 0.0
        pct_class = (
            "text-success fw-semibold"
            if bin_conformal_pct >= 95.0
            else "text-warning fw-semibold"
            if bin_conformal_pct >= 80.0
            else "text-danger fw-semibold"
        )
        bin_rows.append(
            html.Tr(
                [
                    html.Td(str(b + 1)),
                    html.Td(f"R{radial_id} / A{angular_id}"),
                    html.Td(f"{bin_total}"),
                    html.Td(f"{bin_conformal}/{bin_total}" if bin_total > 0 else "0/0"),
                    html.Td(f"{bin_conformal_pct:.1f}%", className=pct_class),
                ]
            )
        )
    if active_settings.get("ee_state"):
        edge_conformal_pct = 100.0 * edge_conformal_count / edge_count if edge_count > 0 else 0.0
        edge_pct_class = (
            "text-success fw-semibold"
            if edge_conformal_pct >= 95.0
            else "text-warning fw-semibold"
            if edge_conformal_pct >= 80.0
            else "text-danger fw-semibold"
        )
        bin_rows.append(
            html.Tr(
                [
                    html.Td("EE"),
                    html.Td("Edge excluded"),
                    html.Td(f"{edge_count}"),
                    html.Td(f"{edge_conformal_count}/{edge_count}" if edge_count > 0 else "0/0"),
                    html.Td(f"{edge_conformal_pct:.1f}%", className=edge_pct_class),
                ],
                className="table-secondary",
            )
        )

    count_panel = html.Div(
        [
            html.Div("Spatial bin summary", className="fw-semibold fs-5"),
            html.Div(
                f"Center: ({x0_mm:.3f} mm, {y0_mm:.3f} mm) | Reference set: {reference_scope}",
                className="text-muted small mb-3",
            ),
            html.Div(
                [
                    _summary_card(
                        "Scheme",
                        f"{n_radial} radial x {n_angular} angular ({interior_bin_count} interior sections)",
                    ),
                    _summary_card("Points analyzed", f"{int(values.size):,}"),
                    _summary_card(
                        "Conformal (reference set)",
                        f"{reference_conformal_pct:.1f}% ({reference_conformal_total}/{reference_total})",
                    ),
                    _summary_card("Accepted window", variation_window),
                    _summary_card("Reference median", f"{reference_median:.6g} {value_label}"),
                    _summary_card("Conformal range", f"[{lower_limit:.6g}, {upper_limit:.6g}]"),
                    _summary_card("Reference spread", variation_detail),
                ],
                className="row g-2 mb-3",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Radial boundaries (mm)", className="fw-semibold mb-2"),
                            html.Div(
                                "Quantile-based on non-edge points for near-equal occupancy.",
                                className="text-muted small mb-2",
                            ),
                            html.Div(
                                html.Table(
                                    [
                                        html.Thead(
                                            html.Tr(
                                                [
                                                    html.Th("Bin"),
                                                    html.Th("r min"),
                                                    html.Th("r max"),
                                                ]
                                            )
                                        ),
                                        html.Tbody(radial_rows),
                                    ],
                                    className="table table-sm table-striped table-bordered align-middle mb-0",
                                ),
                                className="table-responsive",
                            ),
                        ],
                        className="col-12 col-lg-4",
                    ),
                    html.Div(
                        [
                            html.Div("Conformance by section", className="fw-semibold mb-2"),
                            html.Div(
                                html.Table(
                                    [
                                        html.Thead(
                                            html.Tr(
                                                [
                                                    html.Th("Section"),
                                                    html.Th("Region"),
                                                    html.Th("Points"),
                                                    html.Th("Conformal"),
                                                    html.Th("Conformal %"),
                                                ]
                                            )
                                        ),
                                        html.Tbody(bin_rows),
                                    ],
                                    className="table table-sm table-striped table-bordered align-middle mb-0",
                                ),
                                className="table-responsive",
                            ),
                        ],
                        className="col-12 col-lg-8",
                    ),
                ],
                className="row g-3",
            ),
        ],
        className="border rounded bg-white p-3",
    )
    return map_fig, trend_fig, count_panel


@callback(
    Output(ids.Store.BATCH_SPATIAL_BINNING_PAYLOAD, "data"),
    Input(ids.Button.BATCH_SPATIAL_BINNING, "n_clicks"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    State(ids.Store.SETTINGS, "data"),
    State(ids.Input.SPATIAL_BIN_RADIAL_COUNT, "value"),
    State(ids.Input.SPATIAL_BIN_ANGULAR_COUNT, "value"),
    State(ids.Input.SPATIAL_BIN_ACCEPTED_VARIATION, "value"),
    State(ids.RadioItems.SPATIAL_BIN_ACCEPTED_VARIATION_MODE, "value"),
    State({"type": "spatial_batch_mode", "z": ALL}, "value"),
    State({"type": "spatial_batch_mode", "z": ALL}, "id"),
    State({"type": "spatial_batch_value", "z": ALL}, "value"),
    State({"type": "spatial_batch_value", "z": ALL}, "id"),
    State({"type": "spatial_batch_exclude", "z": ALL}, "value"),
    State({"type": "spatial_batch_exclude", "z": ALL}, "id"),
    prevent_initial_call=True,
)
def build_batch_spatial_binning_payload(
    n_clicks,
    selected_file,
    stored_files,
    settings,
    radial_bin_count,
    angular_bin_count,
    accepted_variation_value,
    accepted_variation_mode,
    batch_mode_values,
    batch_mode_ids,
    batch_value_values,
    batch_value_ids,
    batch_exclude_values,
    batch_exclude_ids,
):
    if not n_clicks:
        return no_update

    if not selected_file or not stored_files or selected_file not in stored_files:
        return no_update

    file = Ellipsometry.from_path_or_stream(stored_files[selected_file])
    z_keys = _resolve_spatial_batch_z_keys(file)
    if not z_keys:
        return no_update

    default_mode = accepted_variation_mode if accepted_variation_mode in {"percent", "sigma"} else "percent"
    default_value = _safe_float(
        accepted_variation_value,
        default=_default_spatial_batch_variation_value(default_mode),
    )
    if not np.isfinite(default_value) or default_value < 0:
        default_value = _default_spatial_batch_variation_value(default_mode)

    rules, excluded_z, parse_errors = _parse_spatial_batch_component_rules(
        batch_mode_values,
        batch_mode_ids,
        batch_value_values,
        batch_value_ids,
        batch_exclude_values,
        batch_exclude_ids,
    )
    root_name = _safe_filename_fragment(selected_file.rsplit(".", 1)[0])
    base_settings = {**(settings or {})}

    plots = []
    for z_key in z_keys:
        if z_key.lower() in excluded_z:
            continue

        override = rules.get(z_key.lower())
        mode = override["mode"] if override else default_mode
        value = float(override["value"]) if override else float(default_value)

        batch_settings = {
            **base_settings,
            "z_data_value": z_key,
            "_spatial_force_two_sigma_z_scale": True,
        }
        map_fig, trend_fig, count_panel = update_spatial_binning_tab(
            "spatial_binning",
            selected_file,
            batch_settings,
            radial_bin_count,
            angular_bin_count,
            value,
            mode,
            stored_files,
        )

        if not isinstance(map_fig, go.Figure) or not isinstance(trend_fig, go.Figure):
            continue

        snapshot_fig = _build_spatial_batch_snapshot_figure(z_key, map_fig, trend_fig, count_panel)

        z_name = _safe_filename_fragment(z_key)
        plots.append(
            {
                "filename": f"{root_name}_{z_name}_spatial_snapshot.png",
                "figure": json.loads(snapshot_fig.to_json()),
            }
        )

    if not plots:
        return no_update

    processed_z_count = sum(1 for z_key in z_keys if z_key.lower() not in excluded_z)
    overridden_count = sum(1 for z_key in z_keys if z_key.lower() in rules and z_key.lower() not in excluded_z)
    message = (
        f"Prepared {len(plots)} spatial-binning snapshot(s) "
        f"({processed_z_count}/{len(z_keys)} Z parameters processed, overrides used for {overridden_count})."
    )
    if excluded_z:
        message += f" Excluded {len(excluded_z)} Z parameter(s)."
    if parse_errors:
        message += f" Ignored {len(parse_errors)} invalid batch row(s)."

    return {
        "request_id": int(time.time() * 1000),
        "zip_name": f"{root_name}_spatial_binning.zip",
        "plots": plots,
        "warnings": parse_errors,
        "message": message,
        "width": 1600,
        "height": 1500,
        "scale": 1,
    }


clientside_callback(
    """
    async function(payload) {
        const noUpdate = window.dash_clientside.no_update;
        if (!payload || !payload.plots || payload.plots.length === 0) {
            return noUpdate;
        }

        if (typeof Plotly === "undefined" || typeof Plotly.toImage !== "function") {
            return "Spatial binning batch export failed: Plotly image export API is unavailable in the browser.";
        }
        if (typeof window.JSZip === "undefined") {
            return "Spatial binning batch export failed: JSZip is unavailable in the browser.";
        }

        const width = payload.width || 1400;
        const height = payload.height || 1000;
        const scale = payload.scale || 1;
        const plots = payload.plots;
        const warnings = payload.warnings || [];
        const zipName = payload.zip_name || "spatial_binning_batch.zip";

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
                const filename = (plot.filename || `spatial_binning_plot_${idx + 1}.png`).replace(/\\.png$/i, "");

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

            if (warnings.length > 0) {
                zip.file("batch_rule_warnings.txt", warnings.join("\\n"));
            }

            const zipBlob = await zip.generateAsync({type: "blob"});
            const zipUrl = URL.createObjectURL(zipBlob);
            const anchor = document.createElement("a");
            anchor.href = zipUrl;
            anchor.download = zipName;
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            URL.revokeObjectURL(zipUrl);

            if (payload.message) {
                return payload.message;
            }
            return `Spatial binning batch download complete (${plots.length} PNG files).`;
        } catch (error) {
            const message = (error && error.message) ? error.message : String(error);
            console.error("Spatial binning batch browser export failed:", error);
            return `Spatial binning batch export failed: ${message}`;
        } finally {
            try {
                Plotly.purge(tempDiv);
            } catch (e) {}
            tempDiv.remove();
        }
    }
    """,
    Output(ids.Div.INFO, "children", allow_duplicate=True),
    Input(ids.Store.BATCH_SPATIAL_BINNING_PAYLOAD, "data"),
    prevent_initial_call=True,
)

