import os
import numpy as np
from OpenGL import GL
from src.capture import CameraDevice
from src.lookup import generate_lookup_float

class LensView:
    """Represents a single optical lens projecting onto the sphere"""
    def __init__(self, camera: CameraDevice, cam_config: dict):
        self.camera = camera
        
        # Config extraction
        self.fov = float(cam_config.get("fov", 160.0))
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
        self.out_w = 1024
        self.out_h = 512
        PROJ_FOV = 180.0
        
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
