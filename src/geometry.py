import math
import numpy as np

def make_inside_sphere(lat_steps=64, lon_steps=128, radius=10.0, fov_deg=360.0):
    # Inward-facing sphere (or partial sphere) with equirect UVs.
    verts, uvs, idx = [], [], []

    fov_rad = math.radians(fov_deg)

    for i in range(lat_steps + 1):
        v = i / lat_steps
        lat = (0.5 - v) * math.pi
        cl = math.cos(lat)
        sl = math.sin(lat)
        for j in range(lon_steps + 1):
            u = j / lon_steps
            # Map U (0..1) to longitude range [-FOV/2, +FOV/2]
            lon = (u - 0.5) * fov_rad
            
            cx = math.cos(lon)
            sx = math.sin(lon)

            x = radius * cl * sx
            y = radius * sl
            z = radius * cl * cx
            verts.append((x, y, z))
            # Flip U and V to match "inside-looking-out" convention
            uvs.append((1.0 - u, 1.0 - v))

    def vid(i, j):
        return i * (lon_steps + 1) + j

    for i in range(lat_steps):
        for j in range(lon_steps):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i + 1, j + 1)
            d = vid(i, j + 1)
            # Reverse winding so inside is visible.
            idx.extend([a, c, b])
            idx.extend([a, d, c])

    return (
        np.array(verts, dtype=np.float32),
        np.array(uvs, dtype=np.float32),
        np.array(idx, dtype=np.uint32),
    )

def make_quad():
    # Full screen quad with flipped U to match sphere's inside-out winding
    # Sphere uses (1-u), so we must also run 1..0 left-to-right to match.
    # V is also flipped relative to Sphere (0 at top here vs 1 at top on sphere), 
    # but let's stick to fixing horizontal mirror first unless upside-down.
    # Positions x,y in [-1, 1]
    verts = np.array([
        -1.0, -1.0, 0.0,
         1.0, -1.0, 0.0,
         1.0,  1.0, 0.0,
        -1.0,  1.0, 0.0
    ], dtype=np.float32)
    
    # UVs: BL, BR, TR, TL
    # Flip U: 1 at Left, 0 at Right.
    uvs = np.array([
        0.0, 0.0, 
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    ], dtype=np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return verts, uvs, idx
