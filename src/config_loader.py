import os
import yaml
from typing import Tuple, Dict, Any
from src.constants import (
    SPHERE_LAT_STEPS,
    SPHERE_LON_STEPS,
    SPHERE_RADIUS,
)


_DEFAULT_VIEWER = {
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "fov": 70.0,
    "orbit_pitch": 10.0,
}

_DEFAULT_MESH = {
    "sphere_lat_steps": SPHERE_LAT_STEPS,
    "sphere_lon_steps": SPHERE_LON_STEPS,
    "sphere_radius": SPHERE_RADIUS,
}


def _as_bool(val, default=False):
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    try:
        return bool(val)
    except Exception:
        return default


def load_camera_config(config_path: str) -> Tuple[list, dict]:
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config file {config_path} not found.")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f) or {}
    cam_configs = config_data.get("cameras", []) or []
    if not cam_configs:
        raise RuntimeError("No cameras defined in config file.")
    return cam_configs, config_data


def load_viewer_config(config_path: str) -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.abspath(config_path))
    viewer_path = os.path.join(base_dir, "config.yaml")

    viewer_cfg = dict(_DEFAULT_VIEWER)
    mesh_cfg = dict(_DEFAULT_MESH)
    softborder = False
    cache_lookup = True
    maskblur = 0
    record_cfg = {}

    if not os.path.exists(viewer_path):
        return {
            "viewer": viewer_cfg,
            "mesh": mesh_cfg,
            "softborder": softborder,
            "cache_lookup": cache_lookup,
            "maskblur": maskblur,
            "record": record_cfg,
        }

    try:
        with open(viewer_path, "r") as f:
            data = yaml.safe_load(f) or {}

        cache_lookup = _as_bool(data.get("cache_lookup", cache_lookup), cache_lookup)
        softborder = _as_bool(data.get("softborder", softborder), softborder)
        try:
            maskblur = max(0, int(data.get("maskblur", maskblur)))
        except Exception:
            maskblur = 0

        view = data.get("view", data) or {}
        for k in ("yaw", "pitch", "roll", "fov", "orbit_pitch"):
            if k in view:
                try:
                    viewer_cfg[k] = float(view[k])
                except Exception:
                    pass

        mesh = data.get("mesh", data) or {}
        for k in ("sphere_lat_steps", "sphere_lon_steps"):
            if k in mesh:
                try:
                    mesh_cfg[k] = max(8, int(mesh[k]))
                except Exception:
                    pass
        if "sphere_radius" in mesh:
            try:
                mesh_cfg["sphere_radius"] = float(mesh["sphere_radius"])
            except Exception:
                pass

        record_cfg = data.get("record", {}) or {}
    except Exception as e:
        print(f"[warn] Failed to load viewer config.yaml: {e}")

    return {
        "viewer": viewer_cfg,
        "mesh": mesh_cfg,
        "softborder": softborder,
        "cache_lookup": cache_lookup,
        "maskblur": maskblur,
        "record": record_cfg,
    }
