#!/usr/bin/env python3
import sys
import os
import math
import time
import ctypes
import argparse
import subprocess
import re

import numpy as np
import cv2
import glfw
from OpenGL import GL
from PIL import Image


# -----------------------------
# Shaders (OpenGL 2.1 baseline)
# -----------------------------
VERT_SRC = r"""
#version 120
attribute vec3 a_pos;
attribute vec2 a_uv;

uniform mat4 u_mvp;

varying vec2 v_uv;

void main() {
    v_uv = a_uv;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"""

FRAG_SRC = r"""
#version 120
uniform sampler2D u_src;     // combined source frame (left|right)
uniform sampler2D u_lookup;  // RGBA8 packed UV lookup (u16 in RG, v16 in BA)
uniform float u_uv_offset_x; // 0.0 for Left, 0.5 for Right

varying vec2 v_uv;

vec2 unpack_uv_rgba8(vec4 t) {
    // t is 0..1; reconstruct u16/v16 from RG/BA bytes
    float r = floor(t.r * 255.0 + 0.5);
    float g = floor(t.g * 255.0 + 0.5);
    float b = floor(t.b * 255.0 + 0.5);
    float a = floor(t.a * 255.0 + 0.5);

    float u16 = r * 256.0 + g;
    float v16 = b * 256.0 + a;

    return vec2(u16 / 65535.0, v16 / 65535.0);
}

void main() {
    vec4 lu = texture2D(u_lookup, v_uv);

    // Invalid pixels: check if the lookup value is black (0,0,0,0)
    // We cannot use alpha alone because alpha contains part of the V coordinate.
    // 1/255 ~= 0.0039. (1/255)^2 ~= 0.000015.
    // Previous threshold 0.0001 (10e-5) was too high, discarding valid low values (1/255, 2/255).
    // Use a threshold smaller than (1/255)^2.
    if (dot(lu, lu) < 0.000001) {
        discard; // Discard invalid pixels to allow multiple spheres to overlap/composite
    }

    vec2 src_uv = unpack_uv_rgba8(lu);
    
    // Scale and offset for side-by-side layout
    // The lookup returns 0..1 for the single lens.
    // We compress it to 0..0.5 and shift by offset.
    src_uv.x = src_uv.x * 0.5 + u_uv_offset_x;
    
    gl_FragColor = texture2D(u_src, src_uv);
}
"""

FRAG_SRC_FLOAT = r"""
#version 120
uniform sampler2D u_src;     // combined source frame (left|right)
uniform sampler2D u_lookup;  // RG16F float lookup (u in R, v in G)
uniform float u_uv_offset_x; // 0.0 for Left, 0.5 for Right

varying vec2 v_uv;

void main() {
    // Sample high-precision float texture directly
    vec2 lu = texture2D(u_lookup, v_uv).rg;

    // Check for sentinel value (e.g. negative) to indicate invalid mapping
    if (lu.x < -0.0001) {
        discard; 
    }

    vec2 src_uv = lu;
    
    // Scale and offset for side-by-side layout
    src_uv.x = src_uv.x * 0.5 + u_uv_offset_x;
    
    gl_FragColor = texture2D(u_src, src_uv);
}
"""


# -----------------------------
# OpenGL helpers
# -----------------------------
def compile_shader(src: str, shader_type):
    sh = GL.glCreateShader(shader_type)
    GL.glShaderSource(sh, src)
    GL.glCompileShader(sh)
    ok = GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS)
    if not ok:
        log = GL.glGetShaderInfoLog(sh).decode("utf-8", "replace")
        raise RuntimeError(f"Shader compile failed:\n{log}")
    return sh


def link_program(vs, fs):
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)
    ok = GL.glGetProgramiv(prog, GL.GL_LINK_STATUS)
    if not ok:
        log = GL.glGetProgramInfoLog(prog).decode("utf-8", "replace")
        raise RuntimeError(f"Program link failed:\n{log}")
    return prog


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
    # Full screen quad 0..1 UV
    # Positions x,y in [-1, 1]
    verts = np.array([
        -1.0, -1.0, 0.0,
         1.0, -1.0, 0.0,
         1.0,  1.0, 0.0,
        -1.0,  1.0, 0.0
    ], dtype=np.float32)
    # Standard UVs
    uvs = np.array([
        0.0, 1.0, 
        1.0, 1.0,
        1.0, 0.0,
        0.0, 0.0
    ], dtype=np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return verts, uvs, idx


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
    # Applied to world vector v: v' = Rz @ Rx @ Ry @ v
    # This corresponds to: Yaw world, then Pitch pitched-world, then Rolled.
    return Rz @ Rx @ Ry


def query_dshow_options_ffmpeg(device_index: int):
    """
    Uses ffmpeg to list DirectShow device options for the given index.
    Outputs to console.
    """
    print(f"[video] Querying capabilities for dshow index {device_index} via ffmpeg...")
    try:
        # HARDCODED override for debugging as per user request
        # The parsing of "ffmpeg -list_devices" was failing due to multiline output or formatting.
        dev_name = "USB Camera"
        print(f"[video] Using hardcoded device name: '{dev_name}'")
        
        # CMD: ffmpeg -list_options true -f dshow -i video="Device Name"
        cmd_opt_str = f'ffmpeg -list_options true -f dshow -i video="{dev_name}"'
        res_opt = subprocess.run(cmd_opt_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', shell=True)
        
        print(f"[video] Supported formats for '{dev_name}':")
        for line in res_opt.stderr.splitlines():
            # Filter for relevant lines
            if "pixel_format" in line or "vcodec" in line or "fps" in line:
                 # Clean up log prefix if present
                 txt = line.split("]", 1)[-1].strip() if "]" in line else line.strip()
                 print(f"  {txt}")

    except FileNotFoundError:
        print("[video] ffmpeg not found in PATH. Cannot listing capabilities.")
    except Exception as e:
        print(f"[video] Error querying dshow options: {e}")


# -----------------------------
# Video capture helper
# -----------------------------
def open_video_source(src: str, width: int = None, height: int = None):
    # src:
    #   "0"            default camera
    #   "dshow:0"      DirectShow
    #   "path\to\file"
    if src.startswith("dshow:"):
        idx = int(src.split(":", 1)[1])
        
        # Query options BEFORE opening, to avoid busy device issues
        query_dshow_options_ffmpeg(idx)
        
        if width is not None and height is not None:
            print(f"[video] Opening DSHOW device {idx} with explicit params in constructor...")
            # Pass properties directly to constructor to avoid bandwidth locking on default format
            # NOTE: params must be integers! Passing floats causes type error in cv2 binding
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW, [
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
                cv2.CAP_PROP_FRAME_WIDTH, width,
                cv2.CAP_PROP_FRAME_HEIGHT, height,
                cv2.CAP_PROP_FPS, 30
            ])
        else:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            
        is_device = True
    else:
        try:
            idx = int(src)
            # Try to infer device backend or use default
            if width is not None and height is not None:
                 print(f"[video] Opening device {idx} with explicit params...")
                 cap = cv2.VideoCapture(idx, cv2.CAP_ANY, [
                    cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
                    cv2.CAP_PROP_FRAME_WIDTH, width,
                    cv2.CAP_PROP_FRAME_HEIGHT, height,
                    cv2.CAP_PROP_FPS, 30
                ]) 
            else:
                cap = cv2.VideoCapture(idx)
            is_device = True
        except ValueError:
            cap = cv2.VideoCapture(src)
            is_device = False

    if is_device:
        # Fallback setting if constructor params didn't stick or weren't used
        fourcc_now = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_now = "".join([chr((fourcc_now >> 8 * i) & 0xFF) for i in range(4)])
        
        if codec_now != 'MJPG':
             print(f"[video] Constructor params didn't force MJPG (got {codec_now}). Trying legacy set()...")
             # Try setting again
             cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
             cap.set(cv2.CAP_PROP_FPS, 30.0)
             if width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
             if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if width is not None and height is not None:
        # Verify
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[debug] Requested {width}x{height}, got {actual_w}x{actual_h}")
        
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {src}")

    # Report actual codec
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # Decode 4-character code (little-endian usually)
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"[video] Active FOURCC: '{codec}' (Int: {fourcc})")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[video] Active FPS: {fps}")

    return cap


# -----------------------------
# Lookup generation (HFOV across width)
# -----------------------------
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


def generate_lookup_rgba8(
    single_lens_w: int,
    single_lens_h: int,
    out_w: int,
    out_h: int,
    hfov_deg: float,
    texture_fov_deg: float = 180.0,
) -> np.ndarray:
    # Generate lookup for a SINGLE canonical camera at Yaw=0
    # Map from Sphere Pixels (out_w, out_h) -> Camera Pixels (single_lens_w, single_lens_h)
    
    # HFOV is across width of the single lens image
    theta_x = math.radians(hfov_deg * 0.5)
    if theta_x <= 1e-9:
        raise ValueError("hfov_deg must be > 0")
    r_x = single_lens_w * 0.5
    f_pix = r_x / theta_x

    cx = single_lens_w * 0.5
    cy = single_lens_h * 0.5
    
    # We always simulate a camera pointing at -Z (or +Z depending on convention).
    # dir_from_equirect returns: +Z forward.
    # So we use Identity rotation.
    
    # Use hfov_rad based on the geometry FOV (e.g., 180 deg)
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
            p = project_fisheye_equidistant_rect(d_cam, f_pix, cx, cy)
            
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

    # For informational logging
    theta_y = (single_lens_h * 0.5) / f_pix
    vfov_deg = math.degrees(2.0 * theta_y)
    print(f"[lookup] Canonical Lens: f_pix={f_pix:.3f} from HFOV={hfov_deg}° across w={single_lens_w}px; implied VFOV≈{vfov_deg:.1f}°")
    return lookup


def generate_lookup_float(
    single_lens_w: int,
    single_lens_h: int,
    out_w: int,
    out_h: int,
    hfov_deg: float,
    texture_fov_deg: float = 180.0,
) -> np.ndarray:
    # Generate lookup for a SINGLE canonical camera at Yaw=0
    # Map from Sphere Pixels (out_w, out_h) -> Camera Pixels (single_lens_w, single_lens_h)
    # Output: (out_h, out_w, 2) float32 array.
    # Invalid pixels are set to (-1, -1).

    theta_x = math.radians(hfov_deg * 0.5)
    if theta_x <= 1e-9:
        raise ValueError("hfov_deg must be > 0")
    r_x = single_lens_w * 0.5
    f_pix = r_x / theta_x

    cx = single_lens_w * 0.5
    cy = single_lens_h * 0.5
    
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
            p = project_fisheye_equidistant_rect(d_cam, f_pix, cx, cy)
            
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

    theta_y = (single_lens_h * 0.5) / f_pix
    vfov_deg = math.degrees(2.0 * theta_y)
    print(f"[lookup-float] Canonical Lens: f_pix={f_pix:.3f} from HFOV={hfov_deg}° across w={single_lens_w}px; implied VFOV≈{vfov_deg:.1f}°")
    return lookup


def generate_remap_maps(lookup_data, src_w: int, src_h: int):
    # Retrieve maps for cv2.remap.
    # lookup_data can be:
    #   - RGBA8 (H,W,4) uint8
    #   - RG Float (H,W,2) float32
    
    h, w = lookup_data.shape[:2]
    
    if lookup_data.dtype == np.uint8 and lookup_data.shape[2] == 4:
        # RGBA8 packed mode
        R = lookup_data[:,:,0].astype(np.float32)
        G = lookup_data[:,:,1].astype(np.float32)
        B = lookup_data[:,:,2].astype(np.float32)
        A = lookup_data[:,:,3].astype(np.float32)
        
        u_norm = (R * 256.0 + G) / 65535.0
        v_norm = (B * 256.0 + A) / 65535.0
        
    elif (lookup_data.dtype == np.float32 or lookup_data.dtype == np.float16) and lookup_data.shape[2] == 2:
        # Float mode
        u_norm = lookup_data[:,:,0]
        v_norm = lookup_data[:,:,1]
        
        # In float mode, we used -1 for invalid.
        # cv2.remap treats coordinates. If they are out of image, it handles border.
        # Our src is 0..src_w. 
        # If u_norm is -1, map_x will be -src_w. This is "valid" in the sense that it's outside.
        # So we can just pass it through.
        
    else:
        raise ValueError(f"Unknown lookup format: {lookup_data.shape} {lookup_data.dtype}")
    
    map_x = u_norm * src_w
    map_y = v_norm * src_h
    
    return map_x, map_y


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help='Video source: "0", "dshow:0", or file path')
    ap.add_argument("--hfov", type=float, default=160.0, help="Horizontal FOV across width of each half-image")
    ap.add_argument("--out_w", type=int, default=1024, help="Lookup/output pano width (2:1 recommended)")
    ap.add_argument("--out_h", type=int, default=512, help="Lookup/output pano height (2:1 recommended)")
    ap.add_argument("--lookup_path", type=str, default="lookup_autogen.png", help="Where to write the lookup PNG")
    ap.add_argument("--left_yaw", type=float, default=0.0)
    ap.add_argument("--right_yaw", type=float, default=180.0)
    ap.add_argument("--width", type=int, help="Requested capture width")
    ap.add_argument("--height", type=int, help="Requested capture height")
    ap.add_argument("--use_packed", action="store_true", help="Use legacy packed RGBA8 lookup instead of Float")
    args = ap.parse_args()

    cap = open_video_source(args.source, args.width, args.height)

    # Read one frame to lock resolution (critical)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read initial frame from video source")
    
    src_h, src_w = frame.shape[:2]
    print(f"Captured video frame size: {src_w}x{src_h}")
    src_h, src_w = frame.shape[:2]
    print(f"[capture] source frame = {src_w}x{src_h}")
    
    if src_w % 2 != 0:
        raise ValueError("Source width must be even (side-by-side mode)")
    half_w = src_w // 2

    # Calculate texture/mesh FOV
    # The user suggested 180 degrees.
    # We will use 180.0 as the span for the lookup and mesh.
    # This means the texture covers -90 to +90 instead of -180 to +180.
    PROJ_FOV = 180.0

    # --------------------------------------------------------------------------
    # Lookup Generation / Loading
    # --------------------------------------------------------------------------
    if args.use_packed:
        # Legacy RGBA8 Packed Path
        cache_filename = f"lookup_SingleLens_Packed_{half_w}x{src_h}_O{args.out_w}x{args.out_h}_fov{args.hfov}_pfov{PROJ_FOV}.png"
        
        if os.path.exists(cache_filename):
            print(f"[lookup] Loading cached PACKED lookup from {cache_filename}")
            img = Image.open(cache_filename).convert("RGBA")
            if img.size != (args.out_w, args.out_h):
                 print("[lookup] Cached image size mismatch, regenerating...")
                 lookup = generate_lookup_rgba8(half_w, src_h, args.out_w, args.out_h, args.hfov, PROJ_FOV)
                 Image.fromarray(lookup, "RGBA").save(cache_filename)
            else:
                 lookup = np.array(img, dtype=np.uint8)
        else:
            print(f"[lookup] Generating new PACKED single-lens lookup: {cache_filename}")
            lookup = generate_lookup_rgba8(half_w, src_h, args.out_w, args.out_h, args.hfov, PROJ_FOV)
            Image.fromarray(lookup, "RGBA").save(cache_filename)
            print(f"[lookup] Saved cache to {cache_filename}")
            
    else:
        # New Default Float Path
        cache_filename = f"lookup_SingleLens_Float_{half_w}x{src_h}_O{args.out_w}x{args.out_h}_fov{args.hfov}_pfov{PROJ_FOV}.npy"
        
        if os.path.exists(cache_filename):
            print(f"[lookup] Loading cached FLOAT lookup from {cache_filename}")
            try:
                lookup = np.load(cache_filename)
                if lookup.shape != (args.out_h, args.out_w, 2):
                    print(f"[lookup] Cached shape mismatch {lookup.shape}, regenerating...")
                    lookup = generate_lookup_float(half_w, src_h, args.out_w, args.out_h, args.hfov, PROJ_FOV)
                    np.save(cache_filename, lookup)
            except Exception as e:
                print(f"[lookup] Failed to load cache: {e}. Regenerating...")
                lookup = generate_lookup_float(half_w, src_h, args.out_w, args.out_h, args.hfov, PROJ_FOV)
                np.save(cache_filename, lookup)
        else:
            print(f"[lookup] Generating new FLOAT single-lens lookup: {cache_filename}")
            lookup = generate_lookup_float(half_w, src_h, args.out_w, args.out_h, args.hfov, PROJ_FOV)
            np.save(cache_filename, lookup)
            print(f"[lookup] Saved cache to {cache_filename}")

    # Create CV2 remap maps once.
    # We combine them to show both unwarped views side-by-side.
    print("Generating CV2 remap maps (Dual View - Limited FOV)...")
    map_x_single, map_y_single = generate_remap_maps(lookup, half_w, src_h)
    
    # Left side: 0..half_w. Right side: half_w..src_w
    map_x = np.hstack([map_x_single, map_x_single + half_w]).astype(np.float32)
    map_y = np.hstack([map_y_single, map_y_single]).astype(np.float32)

    # Flip the maps vertically so the 2D window is not upside down
    map_x = np.flipud(map_x)
    map_y = np.flipud(map_y)

    # Initialize GLFW / OpenGL
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    win_w, win_h = 1280, 720
    window = glfw.create_window(win_w, win_h, "Sphere Viewer (lookup autogen)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Build sphere (Partial)
    verts, uvs, indices = make_inside_sphere(
        lat_steps=64, 
        lon_steps=64, # reduced since half sphere
        radius=10.0, 
        fov_deg=PROJ_FOV
    )
    
    # Build Quad for flat view
    q_verts, q_uvs, q_indices = make_quad()

    # Compile shaders
    vs = compile_shader(VERT_SRC, GL.GL_VERTEX_SHADER)
    
    # Choose Fragment Shader based on mode
    if args.use_packed:
        fs = compile_shader(FRAG_SRC, GL.GL_FRAGMENT_SHADER)
    else:
        fs = compile_shader(FRAG_SRC_FLOAT, GL.GL_FRAGMENT_SHADER)
        
    prog = link_program(vs, fs)

    a_pos = GL.glGetAttribLocation(prog, "a_pos")
    a_uv = GL.glGetAttribLocation(prog, "a_uv")
    u_mvp = GL.glGetUniformLocation(prog, "u_mvp")
    u_src = GL.glGetUniformLocation(prog, "u_src")
    u_lookup = GL.glGetUniformLocation(prog, "u_lookup")
    u_uv_offset_x = GL.glGetUniformLocation(prog, "u_uv_offset_x")

    # Buffers
    vbo_pos = GL.glGenBuffers(1)
    vbo_uv = GL.glGenBuffers(1)
    ebo = GL.glGenBuffers(1)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_pos)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_uv)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL.GL_STATIC_DRAW)

    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)

    # Quad Buffers
    q_vbo_pos = GL.glGenBuffers(1)
    q_vbo_uv = GL.glGenBuffers(1)
    q_ebo = GL.glGenBuffers(1)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, q_vbo_pos)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, q_verts.nbytes, q_verts, GL.GL_STATIC_DRAW)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, q_vbo_uv)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, q_uvs.nbytes, q_uvs, GL.GL_STATIC_DRAW)

    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, q_ebo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, q_indices.nbytes, q_indices, GL.GL_STATIC_DRAW)

    # Source texture (allocate once)
    tex_src = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex_src)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, src_w, src_h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)

    # Lookup texture (packed or float)
    tex_lu = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex_lu)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    if args.use_packed:
        # Packed RGBA8 -> NEAREST filtering
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, args.out_w, args.out_h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, lookup)
    else:
        # Float RG16F -> LINEAR filtering (Smooth interpolation!)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        
        # Upload using GL_RG32F equivalent for simplicity in Python (we have float32 array)
        # Internal format: GL_RG16F (0x822F) for storage efficiency
        # Format: GL_RG (0x8227)
        # Type: GL_FLOAT
        
        # If constants aren't available in older PyOpenGL context, we can hardcode integers or standard ones
        # GL_RG16F = 0x822F
        # GL_RG = 0x8227
        # PyOpenGL should expose them.
        try:
             internal_fmt = GL.GL_RG16F
             fmt = GL.GL_RG
        except AttributeError:
             # Fallback if specific enums missing
             print("[warn] GL_RG16F/GL_RG not found in GL module, trying hardcoded values...")
             internal_fmt = 0x822F
             fmt = 0x8227
             
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_fmt, args.out_w, args.out_h, 0, fmt, GL.GL_FLOAT, lookup)

    # Navigation state
    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    fov = 70.0
    view_sphere = True

    dragging = False
    last_x = 0.0
    last_y = 0.0
    mouse_sens = 0.12  # degrees per pixel

    def clamp_pitch(p):
        return max(-89.0, min(89.0, p))

    def on_key(win, key, scancode, action, mods):
        nonlocal yaw, pitch, roll, fov, view_sphere
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_R:
            yaw = pitch = roll = 0.0
            fov = 70.0
        elif key == glfw.KEY_Q:
            roll -= 2.0
        elif key == glfw.KEY_E:
            view_sphere = not view_sphere
            print(f"[view] Mode: {'Sphere' if view_sphere else 'Equirect'}")

    def on_mouse_button(win, button, action, mods):
        nonlocal dragging, last_x, last_y
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                dragging = True
                last_x, last_y = glfw.get_cursor_pos(win)
            elif action == glfw.RELEASE:
                dragging = False

    def on_cursor_pos(win, x, y):
        nonlocal yaw, pitch, last_x, last_y
        if not dragging:
            return
        dx = x - last_x
        dy = y - last_y
        last_x, last_y = x, y
        yaw += dx * mouse_sens
        pitch += dy * mouse_sens
        pitch = clamp_pitch(pitch)

    def on_scroll(win, xoff, yoff):
        nonlocal fov
        fov -= yoff * 2.0
        fov = max(20.0, min(180.0, fov))
        print(f"[nav] FOV={fov:.1f}")

    glfw.set_key_callback(window, on_key)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor_pos)
    glfw.set_scroll_callback(window, on_scroll)

    GL.glDisable(GL.GL_CULL_FACE)
    GL.glEnable(GL.GL_DEPTH_TEST)

    # FPS log
    last_t = time.time()
    frames = 0

    # Main loop
    while not glfw.window_should_close(window):
        glfw.poll_events()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Ensure same size as initial (otherwise lookup is wrong)
        if frame.shape[1] != src_w or frame.shape[0] != src_h:
            raise RuntimeError(f"Source resolution changed at runtime: got {frame.shape[1]}x{frame.shape[0]}, expected {src_w}x{src_h}")
        
        # Show Raw Input
        cv2.imshow("Raw Input", frame)

        # Show 2D Remapped (Equirectangular) in separate window
        remapped_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Remapped 2D", remapped_frame)
        
        if cv2.waitKey(1) == 27: # ESC
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Upload video
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_src)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, src_w, src_h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)

        fb_w, fb_h = glfw.get_framebuffer_size(window)
        GL.glViewport(0, 0, fb_w, fb_h)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        aspect = fb_w / float(fb_h if fb_h else 1)
        
        if view_sphere:
            proj = mat4_perspective(fov, aspect, 0.1, 100.0)
            view = mat4_from_yaw_pitch_roll(yaw, pitch, roll)

            # Use Sphere Buffers
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_pos)
            GL.glEnableVertexAttribArray(a_pos)
            GL.glVertexAttribPointer(a_pos, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_uv)
            GL.glEnableVertexAttribArray(a_uv)
            GL.glVertexAttribPointer(a_uv, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))

            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)

            for cam_idx, (cam_yaw, uv_offset) in enumerate([
                (args.left_yaw, 0.0), 
                (args.right_yaw, 0.5)
            ]):
                cy = math.cos(math.radians(cam_yaw))
                sy = math.sin(math.radians(cam_yaw))
                model = np.array([
                    [ cy, 0.0, sy, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-sy, 0.0, cy, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ], dtype=np.float32)
                
                mvp = proj @ view @ model

                GL.glUseProgram(prog)
                GL.glUniformMatrix4fv(u_mvp, 1, GL.GL_FALSE, mvp.T)
                GL.glUniform1f(u_uv_offset_x, uv_offset)

                GL.glUniform1i(u_src, 0)
                GL.glActiveTexture(GL.GL_TEXTURE1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex_lu)
                GL.glUniform1i(u_lookup, 1)

                GL.glDrawElements(GL.GL_TRIANGLES, indices.size, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))

        else:
            # Flat Equirect View using Quads
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, q_vbo_pos)
            GL.glEnableVertexAttribArray(a_pos)
            GL.glVertexAttribPointer(a_pos, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, q_vbo_uv)
            GL.glEnableVertexAttribArray(a_uv)
            GL.glVertexAttribPointer(a_uv, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))

            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, q_ebo)

            # Draw Left and Right side-by-side
            for offset_screen, uv_offset in [(-0.5, 0.0), (0.5, 0.5)]:
                 mvp = np.array([
                     [0.5, 0.0, 0.0, offset_screen],
                     [0.0, -1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]
                 ], dtype=np.float32)
                 
                 GL.glUseProgram(prog)
                 GL.glUniformMatrix4fv(u_mvp, 1, GL.GL_TRUE, mvp)
                 GL.glUniform1f(u_uv_offset_x, uv_offset)
                 
                 GL.glUniform1i(u_src, 0)
                 GL.glActiveTexture(GL.GL_TEXTURE1)
                 GL.glBindTexture(GL.GL_TEXTURE_2D, tex_lu)
                 GL.glUniform1i(u_lookup, 1)

                 GL.glDrawElements(GL.GL_TRIANGLES, q_indices.size, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))

        glfw.swap_buffers(window)

        frames += 1
        now = time.time()
        if now - last_t >= 2.0:
            fps = frames / (now - last_t)
            print(f"[perf] FPS={fps:.1f} src={src_w}x{src_h} lookup={args.out_w}x{args.out_h} yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f} fov={fov:.1f}")
            frames = 0
            last_t = now

    cap.release()
    glfw.terminate()


if __name__ == "__main__":
    # Example usage:
    #   python viewer_with_lookupgen_win.py --source dshow:0 --hfov 160 --out_w 1024 --out_h 512
    main()
