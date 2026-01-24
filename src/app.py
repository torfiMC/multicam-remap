import os
import yaml
import glfw

from src.capture import CameraDevice
from src.lens import LensView
from src.scene_state import SceneState
from src.render.mesh import SphereMesh, QuadMesh
from src.render.grid import Grid
from src.render.renderer import Renderer
from src.constants import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    SPHERE_LAT_STEPS,
    SPHERE_LON_STEPS,
    SPHERE_RADIUS,
)
from src.input_handler import InputHandler

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
        
        self.cam_configs = self.config_data.get('cameras', [])
        if not self.cam_configs:
             raise RuntimeError("No cameras defined in config file.")

        # Optional viewer/virtual camera config (separate file)
        self.viewer_config = {
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0,
            'fov': 70.0,
        }
        self.softborder = False
        self.cache_lookup = True
        self.maskblur = 0
        self.sphere_lat_steps = SPHERE_LAT_STEPS
        self.sphere_lon_steps = SPHERE_LON_STEPS
        self.sphere_radius = SPHERE_RADIUS
        try:
            base_dir = os.path.dirname(os.path.abspath(self.config_path))
            viewer_path = os.path.join(base_dir, 'config.yaml')
            if os.path.exists(viewer_path):
                with open(viewer_path, 'r') as f:
                    viewer_data = yaml.safe_load(f) or {}

                cl = viewer_data.get('cache_lookup', True)
                if isinstance(cl, str):
                    self.cache_lookup = cl.strip().lower() in ('1', 'true', 'yes', 'on')
                else:
                    self.cache_lookup = bool(cl)

                mb = viewer_data.get('maskblur', 0)
                try:
                    self.maskblur = max(0, int(mb))
                except Exception:
                    self.maskblur = 0

                sb = viewer_data.get('softborder', False)
                if isinstance(sb, str):
                    self.softborder = sb.strip().lower() in ('1', 'true', 'yes', 'on')
                else:
                    self.softborder = bool(sb)
                view = viewer_data.get('view', viewer_data) or {}
                for k in ('yaw', 'pitch', 'roll', 'fov'):
                    if k in view:
                        self.viewer_config[k] = float(view[k])

                mesh_cfg = viewer_data.get('mesh', viewer_data) or {}
                try:
                    if 'sphere_lat_steps' in mesh_cfg:
                        self.sphere_lat_steps = max(8, int(mesh_cfg['sphere_lat_steps']))
                    if 'sphere_lon_steps' in mesh_cfg:
                        self.sphere_lon_steps = max(8, int(mesh_cfg['sphere_lon_steps']))
                    if 'sphere_radius' in mesh_cfg:
                        self.sphere_radius = float(mesh_cfg['sphere_radius'])
                except Exception as e:
                    print(f"[warn] mesh config ignored: {e}")
        except Exception as e:
            print(f"[warn] Failed to load viewer config.yaml: {e}")

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
        
        for i, cc in enumerate(self.cam_configs):
            dev_id = str(cc.get("id", "0"))
            cam_name = cc.get('name', dev_id)

            enabled_raw = cc.get('enabled', True)
            enabled = enabled_raw
            if isinstance(enabled_raw, str):
                enabled = enabled_raw.strip().lower() not in ('0', 'false', 'no', 'off')
            else:
                enabled = bool(enabled_raw)

            if not enabled:
                print(f"[info] Camera '{cam_name}' ({dev_id}) disabled in config; skipping.")
                continue

            if dev_id in failed_dev_ids:
                print(f"[warn] Skipping camera '{cam_name}' ({dev_id}) (previously failed to open)")
                continue

            # Retrieve or create device
            if dev_id in self.device_registry:
                dev = self.device_registry[dev_id]
                print(f"Reusing existing device {dev_id} for '{cam_name}'")
            else:
                try:
                    print(f"Initializing new device {dev_id} for '{cam_name}'")
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
        self.sphere_mesh = SphereMesh(
            lat_steps=self.sphere_lat_steps,
            lon_steps=self.sphere_lon_steps,
            radius=self.sphere_radius,
            fov_deg=PROJ_FOV,
        )
        self.quad_mesh = QuadMesh()
        self.grid = Grid(extent=25.0, spacing=1.0)
        self.renderer = Renderer(self.softborder)

    def _init_state(self):
        vc = getattr(self, 'viewer_config', None) or {}
        fov = float(vc.get('fov', 70.0))
        orbit_pitch = float(vc.get('orbit_pitch', 10.0))

        self.scene = SceneState(
            yaw=float(vc.get('yaw', 0.0)),
            pitch=float(vc.get('pitch', 0.0)),
            roll=float(vc.get('roll', 0.0)),
            fov=fov,
            view_mode='inside',
            prev_view_mode='inside',
            orbit_radius=14.0,
            orbit_pitch=orbit_pitch,
            orbit_angle_offset=0.0,
        )
        self.default_fov = fov
        self.default_orbit_pitch = orbit_pitch

        self.edit_mode = False
        self.sel_lens_idx = 0
        self.sel_attr_idx = 0

    def reset_view(self):
        self.scene.reset(default_fov=self.default_fov, default_orbit_pitch=self.default_orbit_pitch)

    def set_view_mode(self, mode: str):
        self.scene.set_view_mode(mode)

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
            dev.upload_texture(edit_mode=self.edit_mode or getattr(self.scene, 'view_mode', '') == 'all')

    def _render(self):
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        self.renderer.draw_frame((fb_w, fb_h), self.scene, self.sphere_mesh, self.quad_mesh, self.grid, self.lenses)
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
