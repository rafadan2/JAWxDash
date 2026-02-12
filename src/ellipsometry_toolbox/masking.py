import numpy as np
from copy import copy
import logging

from src.ellipsometry_toolbox.ellipsometry import Ellipsometry
from src.ellipsometry_toolbox.linear_translations import rotate, translate


logger = logging.getLogger(__name__)


def create_masked_file(file:Ellipsometry, settings:dict) -> Ellipsometry:
    out_file = copy(file)

    if file.data.empty:
        out_file.data = file.data.copy()
        return out_file

    if "x" not in file.data or "y" not in file.data:
        logger.warning("Edge exclusion skipped: file is missing x/y columns.")
        out_file.data = file.data.copy()
        return out_file

    # Getting instrumented coordinate of the measurement
    xy_off = file.offset(
        x=settings.get("mappattern_x", 0.0),
        y=settings.get("mappattern_y", 0.0),
        theta=settings.get("mappattern_theta", 0.0),
    )

    x_data = xy_off[:, 0]
    y_data = xy_off[:, 1]
    point_count = len(file.data.index)

    sample_outline = settings.get("sample_outline")
    ee_type = settings.get("ee_type")
    ee_distance = settings.get("ee_distance", 0.0)
    try:
        edge_distance = float(ee_distance)
    except (TypeError, ValueError):
        edge_distance = 0.0
    if not np.isfinite(edge_distance):
        edge_distance = 0.0
    edge_distance = max(0.0, edge_distance)

    full_mask = np.ones(point_count, dtype=bool)
    empty_mask = np.zeros(point_count, dtype=bool)
    eps = 1e-12

    if sample_outline == "circle":
        cx = float(settings.get("sample_x", 0.0))
        cy = float(settings.get("sample_y", 0.0))
        inner_radius = float(settings.get("sample_radius", 0.0)) - edge_distance
        if inner_radius <= 0:
            mask = empty_mask
        else:
            dx = x_data - cx
            dy = y_data - cy
            mask = (dx * dx + dy * dy) <= (inner_radius * inner_radius + eps)

    elif sample_outline == "rectangle_corner":
        if ee_type != "uniform":
            logger.warning("Edge exclusion type '%s' for outline '%s' is not implemented for data masking. Using unmasked data.", ee_type, sample_outline)
            mask = full_mask
        else:
            x0 = float(settings.get("sample_x", 0.0))
            y0 = float(settings.get("sample_y", 0.0))
            theta = float(settings.get("sample_theta", 0.0))
            width = float(settings.get("sample_width", 0.0))
            height = float(settings.get("sample_height", 0.0))

            x_min = edge_distance
            y_min = edge_distance
            x_max = width - edge_distance
            y_max = height - edge_distance
            if x_max <= x_min or y_max <= y_min:
                mask = empty_mask
            else:
                theta_rad = np.deg2rad(theta)
                cos_t = np.cos(theta_rad)
                sin_t = np.sin(theta_rad)
                dx = x_data - x0
                dy = y_data - y0

                # Transform world coordinates back to local rectangle coordinates.
                x_local = cos_t * dx + sin_t * dy
                y_local = -sin_t * dx + cos_t * dy

                mask = (
                    (x_local >= x_min - eps)
                    & (x_local <= x_max + eps)
                    & (y_local >= y_min - eps)
                    & (y_local <= y_max + eps)
                )

    elif sample_outline == "sector":
        cx = float(settings.get("sample_x", 0.0))
        cy = float(settings.get("sample_y", 0.0))
        outer_radius = float(settings.get("sample_radius", 0.0))
        inner_radius = outer_radius - edge_distance
        theta_start = np.deg2rad(float(settings.get("sample_theta", 0.0)))
        theta_end = theta_start + 0.5 * np.pi

        dx = x_data - cx
        dy = y_data - cy
        r = np.sqrt(dx * dx + dy * dy)

        e1x, e1y = np.cos(theta_start), np.sin(theta_start)
        e2x, e2y = np.cos(theta_end), np.sin(theta_end)
        cross_start = e1x * dy - e1y * dx
        cross_end = e2x * dy - e2y * dx

        if inner_radius <= 0:
            mask = empty_mask
        elif ee_type == "radial":
            mask = (r <= inner_radius + eps) & (cross_start >= -eps) & (cross_end <= eps)
        elif ee_type == "uniform":
            mask = (
                (r <= inner_radius + eps)
                & (cross_start >= edge_distance - eps)
                & (cross_end <= -(edge_distance - eps))
            )
        else:
            logger.warning("Edge exclusion type '%s' for outline '%s' is not implemented for data masking. Using unmasked data.", ee_type, sample_outline)
            mask = full_mask

    else:
        logger.warning("Sample outline '%s' is not implemented for data masking. Using unmasked data.", sample_outline)
        mask = full_mask

    if mask.shape[0] != point_count:
        logger.error("Invalid edge-exclusion mask shape %s for %s points. Using unmasked data.", mask.shape, point_count)
        mask = full_mask

    out_file.data = file.data.loc[mask].copy()

    return out_file



def _circle_edge_exclusion_(x, y, radius, ee_distance) -> dict:
    
    return dict(
            type="circle",
            x0=x - radius + ee_distance,
            y0=y - radius + ee_distance,
            x1=x + radius - ee_distance,
            y1=y + radius - ee_distance,
            line=dict(color="rgba(193, 0, 1, 255)", width=2),
        )



def radial_edge_exclusion_outline(settings):

    x, y = settings["sample_x"], settings["sample_y"]
    d = settings["ee_distance"]
    r =  settings["sample_radius"]


    # Circle outline
    if settings["sample_outline"] == "circle":

        return _circle_edge_exclusion_(x, y, r, d)
    

    # Rectangle (corner) outline
    elif settings["sample_outline"] == "rectangle_corner":
        logger.warning("Rectangle (corner) radial edge exclusion method not implemented")

        return dict()
    

    # Sector outline
    elif settings["sample_outline"] == "sector":
        
        t = settings["sample_theta"]

        t1 = np.deg2rad(t)
        t2 = t1 + 0.5*np.pi


        # Creating arc
        t = np.linspace(t1, t2, 50)
        x_arc = x + (r-d) * np.cos(t)
        y_arc = y + (r-d) * np.sin(t)


        path = f"M {x},{y}"
        for xc, yc in zip(x_arc, y_arc):
            path += f" L{xc},{yc}"

        path += f" L{x},{y} Z"

        
        return dict(
            type="path",
            path=path, #sector
            line=dict(color="rgba(193, 0, 1, 255)", width=2),
        )
    

    else:
        return dict()



def uniform_edge_exclusion_outline(settings):

    x, y = settings["sample_x"],  settings["sample_y"]
    d = settings["ee_distance"]

    # Circle outline
    if settings["sample_outline"] == "circle":
        r =  settings["sample_radius"]

        return _circle_edge_exclusion_(x, y, r, d)
    

    # Rectangle (corner) outline
    elif settings["sample_outline"] == "rectangle_corner":
        x, y = settings["sample_x"], settings["sample_y"]
        w, h = settings["sample_width"], settings["sample_height"]
        d = settings["ee_distance"]

        xc = (0+d, w-d, w-d, 0+d, 0+d)
        yc = (0+d, 0+d, h-d, h-d, 0+d)
        xy = np.asarray([xc, yc])

        # Rotate and translate
        xy_rot = rotate(xy, settings["sample_theta"])
        xy_rot_tran = translate(xy_rot, np.asarray([settings["sample_x"], settings["sample_y"]]))

        x = xy_rot_tran[0,:]
        y = xy_rot_tran[1,:]

        path = f"M {x[0]},{y[0]}"

        for xc, yc in zip(x[1:-1], y[1:-1]):
            path += f" L{xc},{yc}"

        path += f" L{x[-1]},{y[-1]} Z"

        return dict(
            type="path",
            path=path,
            line=dict(color="rgba(193, 0, 1, 255)", width=2),
        )
    
    
    # Sector outline
    elif settings["sample_outline"] == "sector":
        theta = settings["sample_theta"]
        r = settings["sample_radius"]
        
        t1 = np.deg2rad(theta)
        t2 = t1 + 0.5*np.pi

        # Calculate 'intermitten' angle
        t_ee = (t2-t1)/2 + t1

        # Calculate 'intermitten' length
        c = d/np.cos(np.pi/4)

        # Calculate 90deg corner of shrunken sector
        x_ee = np.cos(t_ee) * c + x
        y_ee = np.sin(t_ee) * c + y

        # Calc offset angle 
        t_o = np.arcsin(d/(r-d))


        # Creating arc
        t = np.linspace(t1+t_o, t2-t_o, 50)
        x_arc = x + (r-d) * np.cos(t)
        y_arc = y + (r-d) * np.sin(t)

        path = f"M {x_ee},{y_ee}"
        for xc, yc in zip(x_arc, y_arc):
            path += f" L{xc},{yc}"

        path += f" L{x_ee},{y_ee} Z"


        return dict(
            type="path",
            path=path, #sector
            line=dict(color="rgba(193, 0, 1, 255)", width=2),
        )
    
    else:
        return dict()

