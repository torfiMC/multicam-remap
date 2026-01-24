import math
from dataclasses import dataclass
import numpy as np
from src.math_utils import mat4_from_yaw_pitch_roll, mat4_look_at


@dataclass
class SceneState:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    fov: float = 70.0
    view_mode: str = "inside"  # inside | orbit | equirect | all
    prev_view_mode: str = "inside"
    orbit_radius: float = 14.0
    orbit_pitch: float = 10.0
    orbit_angle_offset: float = 0.0

    def reset(self, default_fov: float = 70.0, default_orbit_pitch: float = 10.0):
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.fov = default_fov
        self.orbit_pitch = default_orbit_pitch
        self.orbit_angle_offset = 0.0
        self.view_mode = "inside"
        self.prev_view_mode = "inside"

    def set_view_mode(self, mode: str):
        if mode not in ("inside", "orbit", "equirect", "all"):
            mode = "inside"
        self.prev_view_mode = self.view_mode
        self.view_mode = mode

    def orbit_eye(self) -> np.ndarray:
        ang = math.radians(self.orbit_angle_offset)
        pitch = math.radians(self.orbit_pitch)
        r = self.orbit_radius
        x = r * math.cos(pitch) * math.sin(ang)
        y = r * math.sin(pitch)
        z = r * math.cos(pitch) * math.cos(ang)
        return np.array([x, y, z], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        if self.view_mode == "orbit":
            eye = self.orbit_eye()
            return mat4_look_at(eye, np.zeros(3, dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32))
        # Inside view keeps the legacy roll->pitch->yaw order for user navigation.
        return mat4_from_yaw_pitch_roll(self.yaw, self.pitch, self.roll)

    @property
    def is_sphere(self) -> bool:
        return self.view_mode in ("inside", "orbit")
