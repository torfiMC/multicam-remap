import os
import sys
import time
import ctypes
import logging
import yaml
import numpy as np
import glfw
from OpenGL import GL

from src.capture import CameraDevice
from src.lens import LensView
from src.geometry import make_inside_sphere, make_quad
from src.shaders import compile_shader, link_program, VERT_SRC, FRAG_SRC_FLOAT, FRAG_SRC_FLOAT_SOFTBORDER
from src.math_utils import mat4_perspective, mat4_from_yaw_pitch_roll
from src.constants import WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE
from src.input_handler import InputHandler

log = logging.getLogger(__name__)

class App:
    def __init__(self, config_path: str, fullscreen: bool = False):
        self.config_path = config_path
        self.fullscreen = fullscreen
        self._load_config()
        self._init_window()
        self._init_devices()
        self._init_gl()
        self._init_state()
        
        # Initialize Input Handler
        self.input_handler = InputHandler(self)
        
        # Callbacks routed to Input Handler
        glfw.set_key_callback(self.window, self.input_handler.on_key)
        glfw.set_mouse_button_callback(self.window, self.input_handler.on_mouse)
        glfw.set_cursor_pos_callback(self.window, self.input_handler.on_cursor)
        glfw.set_scroll_callback(self.window, self.input_handler.on_scroll)

    def _load_config(self):
        if not os.path.exists(self.config_path):
             raise RuntimeError(f"Config file {self.config_path} not found.")

        with open(self.config_path, 'r') as f:
            self.config_data = yaml.safe_load(f)
        
        raw_cams = self.config_data.get('cameras', [])
        if not raw_cams:
             raise RuntimeError("No cameras defined in config file.")

        self.cam_configs = []
        for cc in raw_cams:
            enabled = cc.get("enabled", True)
            if not enabled:
                log.info(f"[config] Skipping disabled camera id={cc.get('id')} name={cc.get('name')}")
                continue
            self.cam_configs.append(cc)

        if not self.cam_configs:
            raise RuntimeError("All cameras are disabled in config file.")

        log.info(f"[config] Active cameras: {len(self.cam_configs)}")
        for cc in self.cam_configs:
            log.info(f"[config] cam id={cc.get('id')} name={cc.get('name')} type={cc.get('type')} res={cc.get('resolution')} fmt={cc.get('format')} fps={cc.get('fps', 0)}")

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        monitor = glfw.get_primary_monitor() if self.fullscreen else None
        mode = glfw.get_video_mode(monitor) if monitor else None
        
        width = mode.size.width if mode else WINDOW_WIDTH
        height = mode.size.height if mode else WINDOW_HEIGHT

        self.window = glfw.create_window(width, height, WINDOW_TITLE, monitor, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

    def _init_devices(self):
        self.device_registry = {} # id_str -> CameraDevice
        self.devices = [] # Unique list of devices to update
        self.lenses = []
        self.lens_configs = []  # aligns with self.lenses
        self.lens_config_indices = []  # index into self.cam_configs for each lens
        failed_dev_ids = set()
        
        try:
            for cc in self.cam_configs:
                dev_id = str(cc.get("id", "0"))
                
                # Retrieve or Create Device
                if dev_id in self.device_registry:
                    dev = self.device_registry[dev_id]
                    log.info(f"Reusing existing device {dev_id} for '{cc.get('name')}'")
                else:
                    log.info(f"Initializing new device {dev_id} for '{cc.get('name')}'")
                    dev = CameraDevice(cc)
                except Exception as e:
                    print(f"[warn] Failed to open device for '{cam_name}' ({dev_id}): {type(e).__name__}: {e}")
                    failed_dev_ids.add(dev_id)
                    continue
                self.device_registry[dev_id] = dev
                self.devices.append(dev)

            # Create lens mapping for this camera config
            try:
                lens = LensView(
                    dev,
                    cc,
                    softborder=self.softborder,
                    cache_lookup=self.cache_lookup,
                    maskblur=self.maskblur,
                )
            except Exception as e:
                print(f"[warn] Failed to initialize lens for '{cam_name}' ({dev_id}): {type(e).__name__}: {e}")
                continue

            self.lenses.append(lens)
            self.lens_configs.append(cc)
            self.lens_config_indices.append(i)

        if not self.lenses:
            print("[warn] No cameras could be initialized; running with an empty scene.")

    def _init_gl(self):
        PROJ_FOV = 180.0
        # Geometry
        self.verts, self.uvs, self.indices = make_inside_sphere(
            lat_steps=64, lon_steps=64, radius=10.0, fov_deg=PROJ_FOV
        )
        
        # GL Objects
        self.vbo_pos = GL.glGenBuffers(1)
        self.vbo_uv = GL.glGenBuffers(1)
        self.ebo = GL.glGenBuffers(1)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_pos)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.verts.nbytes, self.verts, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_uv)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.uvs.nbytes, self.uvs, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL.GL_STATIC_DRAW)

        # Quad Geometry (for 2D View)
        self.q_verts, self.q_uvs, self.q_indices = make_quad()
        
        self.q_vbo_pos = GL.glGenBuffers(1)
        self.q_vbo_uv = GL.glGenBuffers(1)
        self.q_ebo = GL.glGenBuffers(1)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.q_vbo_pos)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.q_verts.nbytes, self.q_verts, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.q_vbo_uv)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.q_uvs.nbytes, self.q_uvs, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.q_ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.q_indices.nbytes, self.q_indices, GL.GL_STATIC_DRAW)

        # Shader
        vs = compile_shader(VERT_SRC, GL.GL_VERTEX_SHADER)
        fs_src = FRAG_SRC_FLOAT_SOFTBORDER if getattr(self, 'softborder', False) else FRAG_SRC_FLOAT
        fs = compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)
        self.prog = link_program(vs, fs)

        # Uniform locs
        self.u_locs = {
            'a_pos': GL.glGetAttribLocation(self.prog, "a_pos"),
            'a_uv': GL.glGetAttribLocation(self.prog, "a_uv"),
            'u_mvp': GL.glGetUniformLocation(self.prog, "u_mvp"),
            'u_src': GL.glGetUniformLocation(self.prog, "u_src"),
            'u_lookup': GL.glGetUniformLocation(self.prog, "u_lookup"),
            'u_uv_offset_x': GL.glGetUniformLocation(self.prog, "u_uv_offset_x"),
            'u_uv_scale_x': GL.glGetUniformLocation(self.prog, "u_uv_scale_x"),
        }

        if getattr(self, 'softborder', False):
            self.u_locs['u_mask'] = GL.glGetUniformLocation(self.prog, "u_mask")

        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_DEPTH_TEST)

        if getattr(self, 'softborder', False):
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        else:
            GL.glDisable(GL.GL_BLEND)

    def _init_state(self):
        vc = getattr(self, 'viewer_config', None) or {}
        self.yaw = float(vc.get('yaw', 0.0))
        self.pitch = float(vc.get('pitch', 0.0))
        self.roll = float(vc.get('roll', 0.0))
        self.fov = float(vc.get('fov', 70.0))
        self.view_sphere = True
        
        self.edit_mode = False
        self.sel_lens_idx = 0
        self.sel_attr_idx = 0

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self._update()
            self._render()
        
        # Cleanup
        for dev in self.devices:
            dev.stop()
        glfw.terminate()

    def _update(self):
         for dev in self.devices:
            dev.update()
            dev.upload_texture()

    def _render(self):
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        GL.glViewport(0, 0, fb_w, fb_h)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        aspect = fb_w / float(fb_h if fb_h else 1)

        if self.view_sphere:
            proj = mat4_perspective(self.fov, aspect, 0.1, 100.0)
            view = mat4_from_yaw_pitch_roll(self.yaw, self.pitch, self.roll)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_pos)
            GL.glEnableVertexAttribArray(self.u_locs['a_pos'])
            GL.glVertexAttribPointer(self.u_locs['a_pos'], 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
            
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_uv)
            GL.glEnableVertexAttribArray(self.u_locs['a_uv'])
            GL.glVertexAttribPointer(self.u_locs['a_uv'], 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
            
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)

            GL.glUseProgram(self.prog)
            GL.glUniform1i(self.u_locs['u_src'], 0)
            GL.glUniform1i(self.u_locs['u_lookup'], 1)
            if getattr(self, 'softborder', False):
                GL.glUniform1i(self.u_locs['u_mask'], 2)

            # Render in reverse order (Painter's Algorithm)
            for lens in reversed(self.lenses): 
                 m_world = mat4_from_yaw_pitch_roll(lens.world_yaw, lens.world_pitch, lens.world_roll)
                 m_orient = mat4_from_yaw_pitch_roll(0.0, 0.0, lens.orientation)
                 model = m_world @ m_orient
                 mvp = proj @ view @ model
                 
                 GL.glUniformMatrix4fv(self.u_locs['u_mvp'], 1, GL.GL_TRUE, mvp)
                 GL.glUniform1f(self.u_locs['u_uv_offset_x'], lens.uv_offset_x)
                 GL.glUniform1f(self.u_locs['u_uv_scale_x'], lens.uv_scale_x)

                 GL.glActiveTexture(GL.GL_TEXTURE0)
                 GL.glBindTexture(GL.GL_TEXTURE_2D, lens.camera.tex_id)
                 
                 GL.glActiveTexture(GL.GL_TEXTURE1)
                 GL.glBindTexture(GL.GL_TEXTURE_2D, lens.tex_lookup)

                 if getattr(self, 'softborder', False):
                     GL.glActiveTexture(GL.GL_TEXTURE2)
                     GL.glBindTexture(GL.GL_TEXTURE_2D, getattr(lens, 'tex_mask', 0))

                 GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)

        else:
            # Equirectangular / 2D View
            # Use an identity matrix for MVP to render flat to screen
            mvp = np.eye(4, dtype=np.float32)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.q_vbo_pos)
            GL.glEnableVertexAttribArray(self.u_locs['a_pos'])
            GL.glVertexAttribPointer(self.u_locs['a_pos'], 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
            
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.q_vbo_uv)
            GL.glEnableVertexAttribArray(self.u_locs['a_uv'])
            GL.glVertexAttribPointer(self.u_locs['a_uv'], 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
            
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.q_ebo)

            GL.glUseProgram(self.prog)
            GL.glUniform1i(self.u_locs['u_src'], 0)
            GL.glUniform1i(self.u_locs['u_lookup'], 1)
            if getattr(self, 'softborder', False):
                GL.glUniform1i(self.u_locs['u_mask'], 2)
            GL.glUniformMatrix4fv(self.u_locs['u_mvp'], 1, GL.GL_TRUE, mvp)

            # Render in reverse order. Note: This overlays multiple cameras on 0..1 UV.
            # Since the lenses map to the SAME Equirectangular space (implied by the lookup),
            # this will composite them.
            for lens in reversed(self.lenses): 
                 GL.glUniform1f(self.u_locs['u_uv_offset_x'], lens.uv_offset_x)
                 GL.glUniform1f(self.u_locs['u_uv_scale_x'], lens.uv_scale_x)

                 GL.glActiveTexture(GL.GL_TEXTURE0)
                 GL.glBindTexture(GL.GL_TEXTURE_2D, lens.camera.tex_id)
                 
                 GL.glActiveTexture(GL.GL_TEXTURE1)
                 GL.glBindTexture(GL.GL_TEXTURE_2D, lens.tex_lookup)

                 if getattr(self, 'softborder', False):
                     GL.glActiveTexture(GL.GL_TEXTURE2)
                     GL.glBindTexture(GL.GL_TEXTURE_2D, getattr(lens, 'tex_mask', 0))

                 GL.glDrawElements(GL.GL_TRIANGLES, len(self.q_indices), GL.GL_UNSIGNED_INT, None)


        glfw.swap_buffers(self.window)

    def save_config(self):
        print(f"[edit] Saving configuration to {self.config_path}...")
        try:
            indices = getattr(self, 'lens_config_indices', None)
            if not indices:
                indices = list(range(len(self.lenses)))

            for lens_idx, lens in enumerate(self.lenses):
                cfg_idx = indices[lens_idx]
                cfg = self.cam_configs[cfg_idx]
                cfg['yaw'] = float(lens.world_yaw)
                cfg['pitch'] = float(lens.world_pitch)
                cfg['roll'] = float(lens.world_roll)
                cfg['orientation'] = float(lens.orientation)
            
            with open(self.config_path, 'w') as f:
                yaml.dump({'cameras': self.cam_configs}, f, sort_keys=False)
            print("[edit] Save complete.")
        except Exception as e:
            print(f"[edit] Error saving config: {e}")
