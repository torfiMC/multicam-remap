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
from src.config_loader import load_camera_config, load_viewer_config
from src.device_manager import DeviceManager
from src.pipeline.ffmpeg_recorder import FFmpegRecorder

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

        # Config
        self.cam_configs, self.config_data = load_camera_config(self.config_path)
        viewer_opts = load_viewer_config(self.config_path)
        self.viewer_config = viewer_opts["viewer"]
        mesh_cfg = viewer_opts["mesh"]
        self.softborder = viewer_opts["softborder"]
        self.cache_lookup = viewer_opts["cache_lookup"]
        self.maskblur = viewer_opts["maskblur"]
        self.record_cfg = viewer_opts.get("record", {}) or {}
        self.sphere_lat_steps = mesh_cfg.get("sphere_lat_steps", SPHERE_LAT_STEPS)
        self.sphere_lon_steps = mesh_cfg.get("sphere_lon_steps", SPHERE_LON_STEPS)
        self.sphere_radius = mesh_cfg.get("sphere_radius", SPHERE_RADIUS)
        self.base_dir = os.path.dirname(os.path.abspath(self.config_path))
        self.record_enabled = bool(self.record_cfg.get("enabled", False))
        self.record_path = self.record_cfg.get("path", os.path.join(self.base_dir, "capture.mp4"))
        self.record_fps = float(self.record_cfg.get("fps", 30.0))
        self.recorder = None

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
        self.device_manager = DeviceManager(
            self.cam_configs,
            softborder=self.softborder,
            cache_lookup=self.cache_lookup,
            maskblur=self.maskblur,
        )
        self.device_manager.initialize()
        # Aliases for existing call sites
        self.device_registry = self.device_manager.device_registry
        self.devices = self.device_manager.devices
        self.lenses = self.device_manager.lenses
        self.lens_configs = self.device_manager.lens_configs
        self.lens_config_indices = self.device_manager.lens_config_indices

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

    def apply_camera_update(self, cfg_idx: int, updates: dict, save: bool = True):
        with self.state_lock:
            lens = self.device_manager.apply_camera_update(cfg_idx, updates, save=save)
            if lens is None and bool(self.cam_configs[cfg_idx].get("enabled", True)):
                raise RuntimeError(f"Camera '{self.cam_configs[cfg_idx].get('name', cfg_idx)}' could not be activated.")

            self.sel_lens_idx = max(0, min(self.sel_lens_idx, len(self.lenses) - 1)) if self.lenses else 0
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
                "pano_yaw": float(getattr(scene, 'pano_yaw', 0.0)),
                "pano_pitch": float(getattr(scene, 'pano_pitch', 0.0)),
                "pano_zoom": float(getattr(scene, 'pano_zoom', 1.0)),
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

            for key in ("pano_yaw", "pano_pitch", "pano_zoom"):
                if key in updates and updates[key] is not None:
                    setattr(scene, key, float(updates[key]))
                    changed = True

            if changed:
                scene.clamp_pano()

            if changed:
                self.state_version += 1

            return self.describe_view()

    def describe_recording(self) -> dict:
        with self.state_lock:
            return {
                "enabled": bool(getattr(self, "record_enabled", False)),
                "path": getattr(self, "record_path", None),
                "fps": float(getattr(self, "record_fps", 0.0)),
                "active": bool(getattr(self, "recorder", None)),
            }

    def set_recording(self, enabled: bool):
        with self.state_lock:
            prev_enabled = bool(getattr(self, "record_enabled", False))
            if enabled == prev_enabled:
                return self.describe_recording()

            self.record_enabled = bool(enabled)
            if not enabled:
                if getattr(self, "recorder", None):
                    try:
                        self.recorder.stop()
                    except Exception:
                        pass
                    self.recorder = None
            else:
                # If the path was relative, keep it relative to config dir for predictable output.
                if not os.path.isabs(self.record_path):
                    self.record_path = os.path.join(self.base_dir, self.record_path)

            self.state_version += 1
            return self.describe_recording()

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
                "record": self.describe_recording(),
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
        if getattr(self, "recorder", None):
            self.recorder.stop()
        if hasattr(self, "device_manager"):
            self.device_manager.stop_all()
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
        should_stream = bool(self.control_server and self.control_server.has_stream_clients())
        should_record = bool(self.record_enabled)
        if not (should_stream or should_record):
            return
        if fb_w <= 0 or fb_h <= 0:
            return

        now = time.time()
        target_fps = self.stream_fps if should_stream else self.record_fps
        if should_stream and should_record:
            target_fps = max(self.stream_fps, self.record_fps)
        min_interval = 1.0 / max(target_fps, 0.1)
        if (now - getattr(self, '_last_stream_time', 0.0)) < min_interval:
            return

        try:
            raw = GL.glReadPixels(0, 0, fb_w, fb_h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((fb_h, fb_w, 3))
            frame = np.flipud(frame)
            frame = frame[:, :, ::-1]

            if should_record:
                self._maybe_record_frame(frame)
                if not should_stream:
                    self._last_stream_time = now

            if should_stream and self.stream_max_width and fb_w > self.stream_max_width:
                scale = self.stream_max_width / float(fb_w)
                new_h = max(1, int(fb_h * scale))
                frame = cv2.resize(frame, (self.stream_max_width, new_h), interpolation=cv2.INTER_AREA)

            if should_stream:
                quality = int(max(1, min(100, getattr(self, 'stream_quality', 80))))
                ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if ok:
                    self.control_server.broadcast_frame(buf.tobytes())
                    self._last_stream_time = now
        except Exception as e:
            print(f"[warn] Stream capture failed: {type(e).__name__}: {e}")

    def _ensure_recorder(self, width: int, height: int) -> None:
        if self.recorder or not self.record_enabled:
            return
        try:
            self.recorder = FFmpegRecorder(self.record_path, width, height, fps=self.record_fps)
            print(f"[record] Writing to {self.record_path} @ {self.record_fps} fps")
        except Exception as e:
            print(f"[warn] Recorder disabled: {e}")
            self.record_enabled = False

    def _maybe_record_frame(self, frame_bgr) -> None:
        if not self.record_enabled:
            return
        try:
            h, w = frame_bgr.shape[:2]
            self._ensure_recorder(w, h)
            if self.recorder:
                self.recorder.write_frame(frame_bgr.tobytes())
        except Exception as e:
            print(f"[warn] Recording failed: {e}")
            self.record_enabled = False

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
