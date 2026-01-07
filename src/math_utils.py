import math
import numpy as np

def mat4_perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    fovy = math.radians(fovy_deg)
    f = 1.0 / math.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m

def mat4_from_yaw_pitch_roll(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    Ry = np.array([
        [ cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    Rx = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  cp, -sp, 0.0],
        [0.0,  sp,  cp, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    Rz = np.array([
        [ cr, -sr, 0.0, 0.0],
        [ sr,  cr, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    # Return View Matrix V = Rz(roll) * Rx(pitch) * Ry(yaw)
    return Rz @ Rx @ Ry

def rotation_yaw_deg(yaw_deg: float) -> np.ndarray:
    y = math.radians(yaw_deg)
    c, s = math.cos(y), math.sin(y)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)

def dir_from_equirect(u: float, v: float, hfov_rad: float = 2.0 * math.pi) -> np.ndarray:
    # World frame: +X right, +Y up, +Z forward
    # u=0.5 -> lon=0 -> +Z
    lon = (u - 0.5) * hfov_rad
    lat = (0.5 - v) * math.pi

    cl = math.cos(lat)
    sl = math.sin(lat)

    x = cl * math.sin(lon)
    y = sl
    z = cl * math.cos(lon)

    d = np.array([x, y, z], dtype=np.float64)
    return d / np.linalg.norm(d)

def project_fisheye_equidistant_rect(d_cam: np.ndarray, f_pix: float, cx: float, cy: float):
    # Equidistant: r = f * theta; no explicit theta cutoff.
    z = d_cam[2]
    if z <= 0.0:
        return None  # behind camera (front hemisphere only)

    theta = math.acos(max(-1.0, min(1.0, z)))
    r_xy = math.hypot(d_cam[0], d_cam[1])
    if r_xy < 1e-12:
        return (cx, cy, theta)

    r = f_pix * theta
    px = cx + r * (d_cam[0] / r_xy)
    py = cy + r * (d_cam[1] / r_xy)
    return (px, py, theta)

def pack_u16(v: int):
    v = max(0, min(65535, int(v)))
    return (v >> 8) & 0xFF, v & 0xFF
