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
from src.utils.file_manager import get_file_path, get_file_title


from src.templates.graph_template import FIGURE_LAYOUT

logger = logging.getLogger(__name__)


CRITICAL_COUNT = 500


# Adding a placeholder for sample outline
# 8in wafer
r = 1*2.54

@callback(
        Output(ids.Graph.MAIN, "figure"),
        Output(ids.DropDown.Z_DATA, "options"),
        Input(ids.DropDown.UPLOADED_FILES, "value"),
        Input(ids.Store.UPLOADED_FILES, "data"),
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
    

    # If no file or z-data-value selected, return an empty figure
    if not selected_file:
        return no_update, no_update
    
    
    # A sample has been selected, now let's unpack
    file_path = get_file_path(uploaded_files, selected_file)
    if not file_path:
        return no_update, no_update

    file = Ellipsometry.from_path_or_stream(file_path)


    # Setting z-data-value default if non selected
    if not settings["z_data_value"]:
        settings["z_data_value"] = sorted(file.get_column_names())[1]

    
    # exposing x,y,z data directly
    x_data = np.array(file.data["x"])
    y_data = np.array(file.data["y"])

    xy = rotate(np.vstack([x_data, y_data]), settings["mappattern_theta"])
    xy = translate(xy, [settings["mappattern_x"], settings["mappattern_y"]])
    x_data = xy[0,:]
    y_data = xy[1,:]
    
    z_data = file.data[settings["z_data_value"]].to_numpy()
    
    z_scale_min = settings.get("z_scale_min")
    z_scale_max = settings.get("z_scale_max")
    auto_min = np.nanmin(z_data)
    auto_max = np.nanmax(z_data)
    zmin = auto_min if z_scale_min is None else z_scale_min
    zmax = auto_max if z_scale_max is None else z_scale_max
    manual_scale = (z_scale_min is not None) or (z_scale_max is not None)

    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin >= zmax:
        zmin, zmax = auto_min, auto_max
        manual_scale = False


    # List for holding 'shapes'
    shapes = []

    
    marker = dict(
        size=15 if settings["marker_type"]=="point" else 1,
        color=z_data,  # numeric value
        colorscale=settings["colormap_value"],  # set the colormap
        colorbar=dict(title=settings["z_data_value"] or "value"),  # optional colorbar
        showscale=True  # show the color scale
    )

    if manual_scale:
        marker.update(cmin=zmin, cmax=zmax, cauto=False)

    # Plotting trace
    figure.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=marker,
        hovertemplate="x: %{x}<br>y: %{y}<br>z: %{marker.color}<extra></extra>",
    ))


    if settings["marker_type"] == "ellipse":
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

    title = get_file_title(uploaded_files, selected_file)
    if title:
        figure.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=18),
            )
        )


    return figure, sorted(file.get_column_names())
