import math
import numpy as np
from src.math_utils import (
    dir_from_equirect,
    focal_length_pixels,
    project_fisheye_equidistant_rect,
    project_rectilinear_pinhole,
    pack_u16,
)

def generate_lookup_rgba8(
    single_lens_w: int,
    single_lens_h: int,
    out_w: int,
    out_h: int,
    hfov_deg: float,
    texture_fov_deg: float = 180.0,
    distortion_type: str = "fisheye",
) -> np.ndarray:
    # Generate lookup for a SINGLE canonical camera at Yaw=0
    # Map from Sphere Pixels (out_w, out_h) -> Camera Pixels (single_lens_w, single_lens_h)
    
    model = str(distortion_type or "fisheye").strip().lower()
    if model not in ("fisheye", "corrected"):
        model = "fisheye"

    # HFOV is across width of the single lens image; pick focal length for the chosen projection model.
    f_pix = focal_length_pixels(model, single_lens_w, hfov_deg)

    cx = single_lens_w * 0.5
    cy = single_lens_h * 0.5
    projector = project_fisheye_equidistant_rect if model == "fisheye" else project_rectilinear_pinhole
    
    hfov_rad = math.radians(texture_fov_deg)

    lookup = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    for y in range(out_h):
        v = (y + 0.5) / out_h
        for x in range(out_w):
            u = (x + 0.5) / out_w
            
            # Map U to the requested Texture FOV
            d_world = dir_from_equirect(u, v, hfov_rad)

            # Project to camera (Identity rotation)
            d_cam = d_world
            p = projector(d_cam, f_pix, cx, cy)
            
            if p is None:
                lookup[y, x] = (0, 0, 0, 0)
                continue
            
            px, py, theta = p

            # Rectangular crop check
            if not (0.0 <= py < single_lens_h):
                lookup[y, x] = (0, 0, 0, 0)
                continue
            if not (0.0 <= px < single_lens_w):
                lookup[y, x] = (0, 0, 0, 0)
                continue

            u_src = (px + 0.5) / single_lens_w
            v_src = (py + 0.5) / single_lens_h

            u16 = int(round(u_src * 65535.0))
            v16 = int(round(v_src * 65535.0))
            uh, ul = pack_u16(u16)
            vh, vl = pack_u16(v16)
            lookup[y, x] = (uh, ul, vh, vl)

    return lookup


def generate_lookup_float(
    single_lens_w: int,
    single_lens_h: int,
    out_w: int,
    out_h: int,
    hfov_deg: float,
    texture_fov_deg: float = 180.0,
    distortion_type: str = "fisheye",
) -> np.ndarray:
    # Generate lookup for a SINGLE canonical camera at Yaw=0
    # Map from Sphere Pixels (out_w, out_h) -> Camera Pixels (single_lens_w, single_lens_h)
    # Output: (out_h, out_w, 2) float32 array.
    # Invalid pixels are set to (-1, -1).

    model = str(distortion_type or "fisheye").strip().lower()
    if model not in ("fisheye", "corrected"):
        model = "fisheye"

    f_pix = focal_length_pixels(model, single_lens_w, hfov_deg)

    cx = single_lens_w * 0.5
    cy = single_lens_h * 0.5
    projector = project_fisheye_equidistant_rect if model == "fisheye" else project_rectilinear_pinhole
    
    # Init with sentinel -1
    lookup = np.full((out_h, out_w, 2), -1.0, dtype=np.float32)
    
    hfov_rad = math.radians(texture_fov_deg)

    for y in range(out_h):
        v = (y + 0.5) / out_h
        for x in range(out_w):
            u = (x + 0.5) / out_w
            
            d_world = dir_from_equirect(u, v, hfov_rad)

            # Project to camera (Identity rotation)
            d_cam = d_world
            p = projector(d_cam, f_pix, cx, cy)
            
            if p is None:
                continue
            
            px, py, theta = p

            # Rectangular crop check
            if not (0.0 <= py < single_lens_h):
                continue
            if not (0.0 <= px < single_lens_w):
                continue

            u_src = (px + 0.5) / single_lens_w
            v_src = (py + 0.5) / single_lens_h

            lookup[y, x] = (u_src, v_src)

    return lookup
