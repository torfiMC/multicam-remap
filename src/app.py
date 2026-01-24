import os
import queue
import threading
import concurrent.futures
import base64
import time
import numpy as np
import yaml
import glfw
import cv2
from OpenGL import GL

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
    DEFAULT_FOV,
)
from src.input_handler import InputHandler
from src.webserver import ControlServer

class App:
    def __init__(self, config_path: str, fullscreen: bool = False):
        self.config_path = config_path
        self.fullscreen = fullscreen
        self.state_lock = threading.RLock()
        self._task_queue = queue.Queue()
        self.state_version = 0
        self.control_server = None
        self.stream_fps = 10.0
        self.stream_quality = 80
        self.stream_max_width = 1280
        self._last_stream_time = 0.0
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

        # Web control server (runs on its own thread)
        self.control_server = ControlServer(self)
        self.control_server.start()

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

    def run_on_main(self, func, timeout: float = 5.0):
        """Schedule a callable to run on the render thread and wait for its result."""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._task_queue.put((func, fut))
        return fut.result(timeout=timeout)

    def _process_tasks(self):
        while True:
            try:
                func, fut = self._task_queue.get_nowait()
            except queue.Empty:
                break

            try:
                result = func()
                if fut:
                    fut.set_result(result)
            except Exception as e:
                if fut and not fut.done():
                    fut.set_exception(e)
                print(f"[warn] Scheduled task failed: {type(e).__name__}: {e}")

    def _lens_idx_for_config(self, cfg_idx: int):
        try:
            return self.lens_config_indices.index(cfg_idx)
        except ValueError:
            return None

    def _stop_device_if_unused(self, device):
        if any(lens.camera is device for lens in self.lenses):
            return
        try:
            device.stop()
        except Exception as e:
            print(f"[warn] Failed to stop device: {e}")
        self.devices = [d for d in self.devices if d is not device]
        for key, dev in list(self.device_registry.items()):
            if dev is device:
                self.device_registry.pop(key, None)

    def _remove_lens(self, lens_idx: int):
        lens = self.lenses.pop(lens_idx)
        self.lens_config_indices.pop(lens_idx)
        self.lens_configs.pop(lens_idx)
        try:
            lens.dispose()
        except Exception as e:
            print(f"[warn] Failed to dispose lens: {e}")
        self._stop_device_if_unused(lens.camera)
        self.sel_lens_idx = max(0, min(self.sel_lens_idx, len(self.lenses) - 1)) if self.lenses else 0

    def _add_lens_from_config(self, cfg_idx: int, force_regen: bool = False):
        cfg = self.cam_configs[cfg_idx]
        dev_id = str(cfg.get("id", "0"))
        cam_name = cfg.get('name', dev_id)

        device = self.device_registry.get(dev_id)
        if not device:
            print(f"Initializing new device {dev_id} for '{cam_name}'")
            device = CameraDevice(cfg)
            self.device_registry[dev_id] = device
            self.devices.append(device)
        else:
            print(f"Reusing device {dev_id} for '{cam_name}'")

        lens = LensView(
            device,
            cfg,
            softborder=self.softborder,
            cache_lookup=self.cache_lookup,
            maskblur=self.maskblur,
            force_regen=force_regen,
        )
        print(f"[lens] Activated '{cam_name}' (cfg #{cfg_idx})")
        self.lenses.append(lens)
        self.lens_configs.append(cfg)
        self.lens_config_indices.append(cfg_idx)
        return lens

    def _rebuild_lens(self, cfg_idx: int, force_regen: bool = False):
        lens_idx = self._lens_idx_for_config(cfg_idx)
        if lens_idx is None:
            return self._add_lens_from_config(cfg_idx, force_regen=force_regen)

        old_lens = self.lenses[lens_idx]
        device = old_lens.camera
        try:
            old_lens.dispose()
        except Exception as e:
            print(f"[warn] Failed to dispose old lens: {e}")

        lens = LensView(
            device,
            self.cam_configs[cfg_idx],
            softborder=self.softborder,
            cache_lookup=self.cache_lookup,
            maskblur=self.maskblur,
            force_regen=force_regen,
        )
        self.lenses[lens_idx] = lens
        self.lens_configs[lens_idx] = self.cam_configs[cfg_idx]
        return lens

    def apply_camera_update(self, cfg_idx: int, updates: dict, save: bool = True):
        with self.state_lock:
            if cfg_idx < 0 or cfg_idx >= len(self.cam_configs):
                raise IndexError(f"Camera index {cfg_idx} is out of range")

            cfg = self.cam_configs[cfg_idx]
            lens_idx = self._lens_idx_for_config(cfg_idx)
            lens = self.lenses[lens_idx] if lens_idx is not None else None

            rebuild_lookup = bool(updates.get("rebuild_lookup", False))

            if "enabled" in updates and updates["enabled"] is not None:
                cfg["enabled"] = bool(updates["enabled"])

            old_fov = float(cfg.get("fov", DEFAULT_FOV))
            old_mask = float(cfg.get("mask_mindistance", 0.0))
            old_distortion = str(cfg.get("distortion", "fisheye"))

            changed_fov = False
            changed_mask = False
            changed_distortion = False

            if "fov" in updates and updates["fov"] is not None:
                new_fov = float(updates["fov"])
                changed_fov = abs(new_fov - old_fov) > 1e-6
                cfg["fov"] = new_fov

            if "mask_mindistance" in updates and updates["mask_mindistance"] is not None:
                new_mask = float(updates["mask_mindistance"])
                changed_mask = abs(new_mask - old_mask) > 1e-6
                cfg["mask_mindistance"] = new_mask

            if "distortion" in updates and updates["distortion"]:
                new_dist = str(updates["distortion"])
                changed_distortion = (new_dist != old_distortion)
                cfg["distortion"] = new_dist

            numeric_pose_fields = ("yaw", "pitch", "roll", "orientation")
            for key in numeric_pose_fields:
                if key in updates and updates[key] is not None:
                    cfg[key] = float(updates[key])

            rebuild_lookup = rebuild_lookup or changed_fov or changed_mask or changed_distortion

            enabled_now = bool(cfg.get("enabled", True))

            if not enabled_now and lens_idx is not None:
                self._remove_lens(lens_idx)
                lens = None
            elif enabled_now:
                try:
                    if lens_idx is None:
                        lens = self._add_lens_from_config(cfg_idx, force_regen=rebuild_lookup or changed_fov)
                        lens_idx = self._lens_idx_for_config(cfg_idx)
                    elif rebuild_lookup or changed_fov:
                        lens = self._rebuild_lens(cfg_idx, force_regen=rebuild_lookup or changed_fov)
                except Exception as e:
                    raise RuntimeError(f"Failed to refresh camera '{cfg.get('name', cfg_idx)}': {e}")

                if lens is None:
                    raise RuntimeError(f"Camera '{cfg.get('name', cfg_idx)}' could not be activated (no lens instance).")

            if lens is not None:
                lens.world_yaw = float(cfg.get("yaw", lens.world_yaw))
                lens.world_pitch = float(cfg.get("pitch", lens.world_pitch))
                lens.world_roll = float(cfg.get("roll", lens.world_roll))
                lens.orientation = float(cfg.get("orientation", lens.orientation))
                lens.fov = float(cfg.get("fov", lens.fov))
                lens.mask_mindistance = float(cfg.get("mask_mindistance", getattr(lens, "mask_mindistance", 0.0)))
                lens.distortion = cfg.get("distortion", getattr(lens, "distortion", "fisheye"))

            self.state_version += 1
            if save:
                self.save_config()

            return self.describe_cameras()[cfg_idx]

    def describe_cameras(self):
        with self.state_lock:
            lens_by_cfg = {cfg_idx: (i, lens) for i, cfg_idx in enumerate(self.lens_config_indices) for lens in [self.lenses[i]]}
            cameras = []
            for idx, cfg in enumerate(self.cam_configs):
                lens_tuple = lens_by_cfg.get(idx)
                lens = lens_tuple[1] if lens_tuple else None
                device = lens.camera if lens else None
                cameras.append({
                    "index": idx,
                    "id": str(cfg.get("id", "")),
                    "name": cfg.get("name", f"Cam {idx}"),
                    "enabled": bool(cfg.get("enabled", True)),
                    "active": lens is not None,
                    "yaw": float(lens.world_yaw if lens else cfg.get("yaw", 0.0)),
                    "pitch": float(lens.world_pitch if lens else cfg.get("pitch", 0.0)),
                    "roll": float(lens.world_roll if lens else cfg.get("roll", 0.0)),
                    "orientation": float(lens.orientation if lens else cfg.get("orientation", 0.0)),
                    "fov": float(getattr(lens, "fov", cfg.get("fov", DEFAULT_FOV))),
                    "mask_mindistance": float(cfg.get("mask_mindistance", 0.0)),
                    "distortion": cfg.get("distortion", "fisheye"),
                    "type": cfg.get("type", "single"),
                    "resolution": cfg.get("resolution", []),
                    "actual_resolution": [getattr(device, "actual_w", None), getattr(device, "actual_h", None)] if device else None,
                    "state_version": self.state_version,
                })
            return cameras

    def describe_view(self):
        with self.state_lock:
            scene = getattr(self, 'scene', None)
            return {
                "yaw": float(getattr(scene, 'yaw', 0.0)),
                "pitch": float(getattr(scene, 'pitch', 0.0)),
                "roll": float(getattr(scene, 'roll', 0.0)),
                "fov": float(getattr(scene, 'fov', self.default_fov)),
                "view_mode": getattr(scene, 'view_mode', 'inside'),
                "state_version": self.state_version,
            }

    def apply_view_update(self, updates: dict):
        with self.state_lock:
            scene = getattr(self, 'scene', None)
            if scene is None:
                return self.describe_view()

            if updates.get("reset", False):
                self.reset_view()
                return self.describe_view()

            changed = False
            for key in ("yaw", "pitch", "roll", "fov"):
                if key in updates and updates[key] is not None:
                    setattr(scene, key, float(updates[key]))
                    changed = True

            if changed:
                self.state_version += 1

            return self.describe_view()

    def renderer_status(self):
        with self.state_lock:
            scene = getattr(self, 'scene', None)
            return {
                "view_mode": getattr(scene, 'view_mode', 'inside'),
                "edit_mode": bool(getattr(self, 'edit_mode', False)),
                "fov": getattr(scene, 'fov', 70.0),
                "state_version": self.state_version,
                "active_cameras": len(self.lenses),
                "configured_cameras": len(self.cam_configs),
            }

    def snapshot_cameras(self, max_width: int = 320):
        """Return JPEG snapshots (base64) for active lenses, resized to max_width."""
        with self.state_lock:
            snapshots = []
            for lens_idx, lens in enumerate(self.lenses):
                cam_idx = self.lens_config_indices[lens_idx]
                cfg = self.cam_configs[cam_idx]
                cam = lens.camera
                frame = None
                try:
                    with cam.lock:
                        if getattr(cam, 'last_frame', None) is not None:
                            frame = cam.last_frame.copy()
                except Exception:
                    frame = None

                if frame is None:
                    continue

                h, w = frame.shape[:2]

                # If this lens uses a slice of a dual feed, crop to the slice so snapshots match the rendered portion.
                try:
                    slice_w = int(round(lens.uv_scale_x * w)) if getattr(lens, 'uv_scale_x', 1.0) < 1.0 else w
                    slice_w = max(1, min(w, slice_w))
                    x0 = int(round(getattr(lens, 'uv_offset_x', 0.0) * w))
                    x0 = max(0, min(w - 1, x0))
                    x1 = max(x0 + 1, min(w, x0 + slice_w))
                    frame = frame[:, x0:x1]
                    h, w = frame.shape[:2]
                except Exception:
                    pass

                if w > max_width:
                    scale = max_width / float(w)
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue

                b64 = base64.b64encode(buf.tobytes()).decode('ascii')
                snapshots.append({
                    "index": cam_idx,
                    "name": cfg.get("name", f"Cam {cam_idx}"),
                    "active": True,
                    "image": f"data:image/jpeg;base64,{b64}",
                })
            return {
                "count": len(snapshots),
                "snapshots": snapshots,
                "state_version": self.state_version,
            }

    def reset_view(self):
        self.scene.reset(default_fov=self.default_fov, default_orbit_pitch=self.default_orbit_pitch)
        with self.state_lock:
            self.state_version += 1

    def set_view_mode(self, mode: str):
        self.scene.set_view_mode(mode)

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self._update()
            self._render()
        
        # Cleanup
        if self.control_server:
            self.control_server.stop()
        for dev in self.devices:
            dev.stop()
        glfw.terminate()

    def _update(self):
        self._process_tasks()
        for dev in self.devices:
            dev.update()
            dev.upload_texture(edit_mode=self.edit_mode or getattr(self.scene, 'view_mode', '') == 'all')

    def _render(self):
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        self.renderer.draw_frame((fb_w, fb_h), self.scene, self.sphere_mesh, self.quad_mesh, self.grid, self.lenses)
        self._maybe_stream_frame(fb_w, fb_h)
        glfw.swap_buffers(self.window)

    def _maybe_stream_frame(self, fb_w: int, fb_h: int) -> None:
        if not self.control_server or not self.control_server.has_stream_clients():
            return
        if fb_w <= 0 or fb_h <= 0:
            return

        now = time.time()
        min_interval = 1.0 / max(self.stream_fps, 0.1)
        if (now - getattr(self, '_last_stream_time', 0.0)) < min_interval:
            return

        try:
            raw = GL.glReadPixels(0, 0, fb_w, fb_h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((fb_h, fb_w, 3))
            frame = np.flipud(frame)
            frame = frame[:, :, ::-1]

            if self.stream_max_width and fb_w > self.stream_max_width:
                scale = self.stream_max_width / float(fb_w)
                new_h = max(1, int(fb_h * scale))
                frame = cv2.resize(frame, (self.stream_max_width, new_h), interpolation=cv2.INTER_AREA)

            quality = int(max(1, min(100, getattr(self, 'stream_quality', 80))))
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                self.control_server.broadcast_frame(buf.tobytes())
                self._last_stream_time = now
        except Exception as e:
            print(f"[warn] Stream capture failed: {type(e).__name__}: {e}")

    def save_config(self):
        print(f"[edit] Saving configuration to {self.config_path}...")
        try:
            with self.state_lock:
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
                    cfg['fov'] = float(getattr(lens, 'fov', cfg.get('fov', 0.0)))
                    cfg['mask_mindistance'] = float(getattr(lens, 'mask_mindistance', cfg.get('mask_mindistance', 0.0)))
                    cfg['distortion'] = getattr(lens, 'distortion', cfg.get('distortion', 'fisheye'))
                    cfg['enabled'] = bool(cfg.get('enabled', True))
                
                with open(self.config_path, 'w') as f:
                    yaml.dump({'cameras': self.cam_configs}, f, sort_keys=False)
            print("[edit] Save complete.")
        except Exception as e:
            print(f"[edit] Error saving config: {e}")
