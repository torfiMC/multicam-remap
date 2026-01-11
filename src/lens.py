import os
import numpy as np
from OpenGL import GL
from src.capture import CameraDevice
from src.lookup import generate_lookup_float
from src.constants import LOOKUP_WIDTH, LOOKUP_HEIGHT, PROJECTION_FOV_DEG, DEFAULT_FOV


def _sanitize_filename_component(s: str) -> str:
    s = str(s or "")
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_-")
    return cleaned or "camera"


def _clamp01(x: float) -> float:
     if x < 0.0:
          return 0.0
     if x > 1.0:
          return 1.0
     return x


def _ensure_edge_distance_mask_png(
    mask_filename: str,
    single_lens_w: int,
    single_lens_h: int,
    out_w: int,
    out_h: int,
    hfov_deg: float,
    texture_fov_deg: float,
    mask_mindistance: float,
    maskblur: int = 0,
    force_regen: bool = False,
    supersample: int = 4,
) -> None:
    if (not force_regen) and os.path.exists(mask_filename):
        return

    try:
        import cv2
        from PIL import Image
    except Exception as e:
        print(f"[warn] Cannot generate mask PNG (missing deps): {e}")
        return

    import math

    try:
        ss = int(supersample)
    except Exception:
        ss = 4
    if ss < 1:
        ss = 1

    high_w = int(out_w) * ss
    high_h = int(out_h) * ss

    try:
        theta_x = math.radians(float(hfov_deg) * 0.5)
        if theta_x <= 1e-9:
            raise ValueError("hfov_deg must be > 0")
        r_x = float(single_lens_w) * 0.5
        f_pix = r_x / theta_x
        cx = float(single_lens_w) * 0.5
        cy = float(single_lens_h) * 0.5
        hfov_rad = math.radians(float(texture_fov_deg))
    except Exception as e:
        print(f"[warn] Cannot generate mask PNG (bad parameters): {e}")
        return

    # Build a high-resolution binary validity mask directly from the projection math.
    # This avoids inheriting aliasing from the sampled lookup .npy.
    x = (np.arange(high_w, dtype=np.float32) + 0.5) / float(high_w)
    lon = (x - 0.5) * np.float32(hfov_rad)
    sin_lon = np.sin(lon, dtype=np.float32)
    cos_lon = np.cos(lon, dtype=np.float32)

    y = (np.arange(high_h, dtype=np.float32) + 0.5) / float(high_h)
    lat = (0.5 - y) * np.float32(math.pi)
    sin_lat = np.sin(lat, dtype=np.float32)
    cos_lat = np.cos(lat, dtype=np.float32)

    valid_u8 = np.zeros((high_h, high_w), dtype=np.uint8)
    eps = np.float32(1e-12)
    w_f = np.float32(single_lens_w)
    h_f = np.float32(single_lens_h)
    cx_f = np.float32(cx)
    cy_f = np.float32(cy)
    f_pix_f = np.float32(f_pix)

    for yi in range(high_h):
        cl = cos_lat[yi]
        sl = sin_lat[yi]

        dx = cl * sin_lon
        dy = sl
        dz = cl * cos_lon

        zpos = dz > 0.0
        zc = np.clip(dz, -1.0, 1.0)
        theta = np.arccos(zc).astype(np.float32)

        r_xy = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
        inv_rxy = np.where(r_xy > eps, 1.0 / r_xy, 0.0).astype(np.float32)

        r = f_pix_f * theta
        px = cx_f + r * (dx * inv_rxy)
        py = cy_f + r * (dy * inv_rxy)

        valid = (
            zpos
            & (px >= 0.0)
            & (px < w_f)
            & (py >= 0.0)
            & (py < h_f)
        )
        valid_u8[yi, :] = valid.astype(np.uint8) * 255

    dist = cv2.distanceTransform(valid_u8, cv2.DIST_L2, 5).astype(np.float32)

    # Make boundary pixels 0, then convert distance units back to output-pixel units.
    dist = np.maximum(dist - 1.0, 0.0) / float(ss)

    dmax = float(dist.max())
    if dmax <= 1e-6:
        mask_u8 = valid_u8
        if ss != 1:
            mask_u8 = cv2.resize(mask_u8, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)
    else:
        mask_mindistance = _clamp01(float(mask_mindistance))
        denom = max((1.0 - mask_mindistance) * dmax, 1e-6)
        mask_f = np.clip(dist / denom, 0.0, 1.0).astype(np.float32)
        if ss != 1:
            mask_f = cv2.resize(mask_f, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)
        mask_u8 = (mask_f * 255.0 + 0.5).astype(np.uint8)

    # Optional Gaussian blur for a softer transition.
    try:
        k = int(maskblur)
    except Exception:
        k = 0
    if k >= 3:
        if (k % 2) == 0:
            k += 1
        mask_u8 = cv2.GaussianBlur(mask_u8, (k, k), 0)

    try:
        Image.fromarray(mask_u8, mode="L").save(mask_filename)
        print(f"[lens] Wrote edge mask PNG {mask_filename}")
    except Exception as e:
        print(f"[warn] Failed to write mask PNG {mask_filename}: {e}")

class LensView:
    """Represents a single optical lens projecting onto the sphere"""
    def __init__(
        self,
        camera: CameraDevice,
        cam_config: dict,
        softborder: bool = False,
        cache_lookup: bool = True,
        maskblur: int = 0,
    ):
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

        cam_name_prefix = _sanitize_filename_component(cam_config.get("name", "camera"))
        
        # Using simple naming scheme based on properties to reuse cache
        cache_filename = f"{cam_name_prefix}_lookup_{lens_pixel_w}x{lens_pixel_h}_to_{self.out_w}x{self.out_h}_fov{self.fov}_pfov{PROJ_FOV}.npy"

        should_use_cache = bool(cache_lookup)
        if should_use_cache and os.path.exists(cache_filename):
            try:
                print(f"[lens] Loading cache {cache_filename}")
                self.lookup_data = np.load(cache_filename)
            except Exception as e:
                print(f"[warn] Failed to load lookup cache {cache_filename}: {e}")
                should_use_cache = False

        if (not should_use_cache) or (not os.path.exists(cache_filename)):
            print(f"[lens] Generating lookup {cache_filename}")
            self.lookup_data = generate_lookup_float(
                lens_pixel_w,
                lens_pixel_h,
                self.out_w,
                self.out_h,
                self.fov,
                PROJ_FOV,
            )
            try:
                np.save(cache_filename, self.lookup_data)
            except Exception as e:
                print(f"[warn] Failed to write lookup cache {cache_filename}: {e}")

        # Generate an 8-bit grayscale mask PNG (if missing) that encodes normalized
        # distance-from-edge for the usable region of the lookup.
        mask_base = os.path.splitext(cache_filename)[0]
        try:
            maskblur_i = max(0, int(maskblur))
        except Exception:
            maskblur_i = 0
        self.mask_filename = f"{mask_base}_edgemask_mindist{self.mask_mindistance:.3f}_blur{maskblur_i}.png"
        _ensure_edge_distance_mask_png(
            self.mask_filename,
            single_lens_w=lens_pixel_w,
            single_lens_h=lens_pixel_h,
            out_w=self.out_w,
            out_h=self.out_h,
            hfov_deg=self.fov,
            texture_fov_deg=PROJ_FOV,
            mask_mindistance=self.mask_mindistance,
            maskblur=maskblur_i,
            force_regen=(not bool(cache_lookup)),
        )

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

        # Optional mask texture for soft border blending
        self.tex_mask = 0
        if softborder:
            try:
                from PIL import Image

                img = Image.open(self.mask_filename).convert('L')
                mask_u8 = np.array(img, dtype=np.uint8)

                # Upload as a single-channel texture (GL_LUMINANCE for GL2.1 compatibility)
                self.tex_mask = GL.glGenTextures(1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_mask)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_LUMINANCE,
                    self.out_w,
                    self.out_h,
                    0,
                    GL.GL_LUMINANCE,
                    GL.GL_UNSIGNED_BYTE,
                    mask_u8,
                )
            except Exception as e:
                print(f"[warn] Failed to load/upload mask texture '{getattr(self, 'mask_filename', '')}': {e}")
                self.tex_mask = 0

            # Ensure we always have a valid texture bound when softborder is enabled.
            if not self.tex_mask:
                try:
                    self.tex_mask = GL.glGenTextures(1)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_mask)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                    white = np.array([[255]], dtype=np.uint8)
                    GL.glTexImage2D(
                        GL.GL_TEXTURE_2D,
                        0,
                        GL.GL_LUMINANCE,
                        1,
                        1,
                        0,
                        GL.GL_LUMINANCE,
                        GL.GL_UNSIGNED_BYTE,
                        white,
                    )
                except Exception as e:
                    print(f"[warn] Failed to create fallback mask texture: {e}")
                    self.tex_mask = 0
