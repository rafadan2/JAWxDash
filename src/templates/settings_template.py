import plotly.express as px
import os



#UPLOAD_DIRECTORY = os.path.join(os.getcwd(), "tmp", "uploads")
UPLOAD_DIRECTORY = "/app/uploads"
ASSETS_DIRECTORY = "/assets"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


DEFAULT_SETTINGS = dict(
    # Spot settings
    marker_type="point",
    marker_size=15,
    angle_of_incident=65,
    spot_size=0.3,

    # Sample settings
    colormap_value="viridis",
    colormap_options=sorted([colorscale for colorscale in px.colors.named_colorscales()]),
    sample_outline="sector",
    z_data_value="",
    render_mode="markers",
    gradient_mode="none",
    gradient_coordinate_mode="cartesian",
    heatmap_grid_mode="auto",
    gradient_grid_mode="auto",
    gradient_grid_size=96,
    gradient_k_nearest=24,
    z_scale_min=None,
    z_scale_max=None,

    # Mappattern offset
    mappattern_x=0.0,
    mappattern_y=2.5,
    mappattern_theta=4.1,

    # Sample offset
    sample_x=-0.067,
    sample_y=2.948,
    sample_theta=4.1+225,
    sample_radius=2*2.54,
    sample_width=2*2.54,
    sample_height=2*2.54,

    # Edge exclusion
    ee_state=False,
    ee_type="radial",
    ee_distance=1.0,
    ee_batch_processing=False,

    # Stage outline
    stage_state=False,
)
