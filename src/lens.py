import os
import numpy as np
from OpenGL import GL
from src.capture import CameraDevice
from src.lookup import generate_lookup_float
from src.constants import LOOKUP_WIDTH, LOOKUP_HEIGHT, PROJECTION_FOV_DEG, DEFAULT_FOV


def _clamp01(x: float) -> float:
     if x < 0.0:
          return 0.0
     if x > 1.0:
          return 1.0
     return x


def _ensure_edge_distance_mask_png(
    mask_filename: str,
    lookup_data: np.ndarray,
    mask_mindistance: float,
) -> None:
    if os.path.exists(mask_filename):
        return

    try:
        import cv2
        from PIL import Image
    except Exception as e:
        print(f"[warn] Cannot generate mask PNG (missing deps): {e}")
        return

    if lookup_data.ndim != 3 or lookup_data.shape[2] < 2:
        print(f"[warn] Cannot generate mask PNG (unexpected lookup shape): {lookup_data.shape}")
        return

    u = lookup_data[:, :, 0]
    v = lookup_data[:, :, 1]
    valid = np.isfinite(u) & np.isfinite(v) & (u >= 0.0) & (v >= 0.0)

    # OpenCV distanceTransform returns distance (in pixels) to the nearest zero pixel.
    # We treat invalid as 0 and valid as 255.
    valid_u8 = valid.astype(np.uint8) * 255
    dist = cv2.distanceTransform(valid_u8, cv2.DIST_L2, 5).astype(np.float32)

    # Approximate "distance from the usable edge" by making boundary pixels 0.
    dist = np.maximum(dist - 1.0, 0.0)

    dmax = float(dist.max())
    if dmax <= 1e-6:
        # Degenerate case: no valid pixels or region is too thin for a meaningful ramp.
        mask_u8 = valid_u8
    else:
        mask_mindistance = _clamp01(float(mask_mindistance))
        denom = max((1.0 - mask_mindistance) * dmax, 1e-6)
        mask = np.clip(dist / denom, 0.0, 1.0)
        mask_u8 = (mask * 255.0 + 0.5).astype(np.uint8)

    try:
        Image.fromarray(mask_u8, mode="L").save(mask_filename)
        print(f"[lens] Wrote edge mask PNG {mask_filename}")
    except Exception as e:
        print(f"[warn] Failed to write mask PNG {mask_filename}: {e}")

class LensView:
    """Represents a single optical lens projecting onto the sphere"""
    def __init__(self, camera: CameraDevice, cam_config: dict):
        self.camera = camera
        
        # Config extraction
        self.fov = float(cam_config.get("fov", DEFAULT_FOV))
        self.mask_mindistance = _clamp01(float(cam_config.get("mask_mindistance", 0.0)))
        cam_type = cam_config.get("type", "single")
        
        # Determine UV mapping settings based on type
        if cam_type == "dual_left":
             self.uv_scale_x = 0.5
             self.uv_offset_x = 0.0
             pixel_w_divisor = 2
        elif cam_type == "dual_right":
             self.uv_scale_x = 0.5
             self.uv_offset_x = 0.5
             pixel_w_divisor = 2
        else: # single
             self.uv_scale_x = 1.0
             self.uv_offset_x = 0.0
             pixel_w_divisor = 1

        # Transform (Use explicit values from config)
        self.world_yaw = float(cam_config.get("yaw", 0.0))
        self.world_pitch = float(cam_config.get("pitch", 0.0))
        self.world_roll = float(cam_config.get("roll", 0.0))
        self.orientation = float(cam_config.get("orientation", 0.0))
        
        # Generate Lookup
        self.out_w = LOOKUP_WIDTH
        self.out_h = LOOKUP_HEIGHT
        PROJ_FOV = PROJECTION_FOV_DEG
        
        # Calculate source slice dims for lookup gen
        lens_pixel_w = camera.actual_w // pixel_w_divisor
        lens_pixel_h = camera.actual_h
        
        # Using simple naming scheme based on properties to reuse cache
        cache_filename = f"lookup_{lens_pixel_w}x{lens_pixel_h}_to_{self.out_w}x{self.out_h}_fov{self.fov}_pfov{PROJ_FOV}.npy"
        
        if os.path.exists(cache_filename):
            print(f"[lens] Loading cache {cache_filename}")
            self.lookup_data = np.load(cache_filename)
        else:
            print(f"[lens] Generating lookup {cache_filename}")
            self.lookup_data = generate_lookup_float(lens_pixel_w, lens_pixel_h, self.out_w, self.out_h, self.fov, PROJ_FOV)
            np.save(cache_filename, self.lookup_data)

            # Generate an 8-bit grayscale mask PNG (if missing) that encodes normalized
            # distance-from-edge for the usable region of the lookup.
            mask_base = os.path.splitext(cache_filename)[0]
            mask_filename = f"{mask_base}_edgemask_mindist{self.mask_mindistance:.3f}.png"
            _ensure_edge_distance_mask_png(mask_filename, self.lookup_data, self.mask_mindistance)

        # Create GL Texture for Lookup
        self.tex_lookup = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_lookup)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        
        # Format (Float16/32 -> RG16F)
        try:
             # GL_RG16F = 0x822F, GL_RG = 0x8227
             internal_fmt = 0x822F
             fmt = 0x8227
             # Try standard names if available
             if hasattr(GL, 'GL_RG16F'): internal_fmt = GL.GL_RG16F
             if hasattr(GL, 'GL_RG'): fmt = GL.GL_RG
             
             GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_fmt, self.out_w, self.out_h, 0, fmt, GL.GL_FLOAT, self.lookup_data)
        except Exception as e:
             # Fallback
             print(f"[warn] Failed to upload float texture: {e}")
