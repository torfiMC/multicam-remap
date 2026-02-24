import threading
from typing import Dict, List
from src.capture import CameraDevice
from src.lens import LensView
from src.constants import DEFAULT_FOV


class DeviceManager:
    """Owns cameras/lenses and their lifecycles."""

    def __init__(self, cam_configs: List[dict], softborder: bool, cache_lookup: bool, maskblur: int):
        self.cam_configs = cam_configs
        self.softborder = softborder
        self.cache_lookup = cache_lookup
        self.maskblur = maskblur

        self.device_registry: Dict[str, CameraDevice] = {}
        self.devices: List[CameraDevice] = []
        self.lenses: List[LensView] = []
        self.lens_configs: List[dict] = []
        self.lens_config_indices: List[int] = []
        self._lock = threading.RLock()

    def initialize(self) -> None:
        failed_dev_ids = set()
        with self._lock:
            for i, cc in enumerate(self.cam_configs):
                dev_id = str(cc.get("id", "0"))
                cam_name = cc.get("name", dev_id)

                enabled_raw = cc.get("enabled", True)
                if isinstance(enabled_raw, str):
                    enabled = enabled_raw.strip().lower() not in ("0", "false", "no", "off")
                else:
                    enabled = bool(enabled_raw)

                if not enabled:
                    print(f"[info] Camera '{cam_name}' ({dev_id}) disabled; skipping.")
                    continue

                if dev_id in failed_dev_ids:
                    print(f"[warn] Skipping camera '{cam_name}' ({dev_id}) (previously failed)")
                    continue

                dev = self.device_registry.get(dev_id)
                if dev:
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
                print("[warn] No cameras initialized; running with an empty scene.")

    def add_lens_from_config(self, cfg_idx: int, force_regen: bool = False) -> LensView:
        cfg = self.cam_configs[cfg_idx]
        dev_id = str(cfg.get("id", "0"))
        cam_name = cfg.get("name", dev_id)

        with self._lock:
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
            self.lenses.append(lens)
            self.lens_configs.append(cfg)
            self.lens_config_indices.append(cfg_idx)
            return lens

    def rebuild_lens(self, cfg_idx: int, force_regen: bool = False) -> LensView:
        lens_idx = self._lens_idx_for_config(cfg_idx)
        if lens_idx is None:
            return self.add_lens_from_config(cfg_idx, force_regen=force_regen)

        with self._lock:
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

    def remove_lens(self, lens_idx: int) -> None:
        with self._lock:
            lens = self.lenses.pop(lens_idx)
            self.lens_config_indices.pop(lens_idx)
            self.lens_configs.pop(lens_idx)
            try:
                lens.dispose()
            except Exception as e:
                print(f"[warn] Failed to dispose lens: {e}")
            self._stop_device_if_unused(lens.camera)

    def stop_all(self) -> None:
        with self._lock:
            for dev in self.devices:
                try:
                    dev.stop()
                except Exception as e:
                    print(f"[warn] Failed to stop device: {e}")

    # Utilities

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

    def update_pose_fields(self, cfg_idx: int, updates: dict):
        lens_idx = self._lens_idx_for_config(cfg_idx)
        lens = self.lenses[lens_idx] if lens_idx is not None else None
        cfg = self.cam_configs[cfg_idx]

        if lens is not None:
            lens.world_yaw = float(cfg.get("yaw", lens.world_yaw))
            lens.world_pitch = float(cfg.get("pitch", lens.world_pitch))
            lens.world_roll = float(cfg.get("roll", lens.world_roll))
            lens.orientation = float(cfg.get("orientation", lens.orientation))
            lens.fov = float(cfg.get("fov", lens.fov))
            lens.mask_mindistance = float(cfg.get("mask_mindistance", getattr(lens, "mask_mindistance", 0.0)))
            lens.distortion = cfg.get("distortion", getattr(lens, "distortion", "fisheye"))

        return lens

    def apply_camera_update(self, cfg_idx: int, updates: dict, save: bool = True):
        if cfg_idx < 0 or cfg_idx >= len(self.cam_configs):
            raise IndexError(f"Camera index {cfg_idx} is out of range")

        with self._lock:
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

            for key in ("yaw", "pitch", "roll", "orientation"):
                if key in updates and updates[key] is not None:
                    cfg[key] = float(updates[key])

            rebuild_lookup = rebuild_lookup or changed_fov or changed_mask or changed_distortion
            enabled_now = bool(cfg.get("enabled", True))

            if not enabled_now and lens_idx is not None:
                self.remove_lens(lens_idx)
                lens = None
            elif enabled_now:
                if lens_idx is None:
                    lens = self.add_lens_from_config(cfg_idx, force_regen=rebuild_lookup or changed_fov)
                    lens_idx = self._lens_idx_for_config(cfg_idx)
                elif rebuild_lookup or changed_fov:
                    lens = self.rebuild_lens(cfg_idx, force_regen=rebuild_lookup or changed_fov)

            if lens is not None:
                self.update_pose_fields(cfg_idx, updates)

            # Do not handle saving here; caller persists.
            return lens
