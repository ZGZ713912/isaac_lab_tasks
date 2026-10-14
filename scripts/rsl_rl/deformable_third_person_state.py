"""Simulator-independent input and camera geometry for third-person driving."""

from dataclasses import dataclass, field
import math

import numpy as np


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_yaw(quaternion):
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass
class ThirdPersonState:
    vx_max: float = .8
    vy_max: float = .5
    wz_max: float = 1.5
    sensitivity_deg: float = .15
    distance: float = 3.
    pitch_deg: float = 20.
    target_yaw: float = 0.
    right_held: bool = False
    suspended: bool = False
    reset_requested: bool = False
    pressed: set = field(default_factory=set)

    def reset(self, yaw):
        self.pressed.clear()
        self.target_yaw = float(yaw)
        self.right_held = False
        self.suspended = False
        self.reset_requested = False

    def stop(self, yaw, *, suspend=False):
        self.reset(yaw)
        self.suspended = suspend

    def key(self, name, down, yaw):
        if down and name in ("ESCAPE", "L"):
            self.stop(yaw, suspend=name == "ESCAPE")
        elif down and name == "R":
            self.stop(yaw)
            self.reset_requested = True
        elif name in ("W", "S", "A", "D"):
            if down:
                self.suspended = False
                self.pressed.add(name)
            else:
                self.pressed.discard(name)

    def right_button(self, down):
        self.right_held = bool(down)
        if down:
            self.suspended = False

    def mouse_move(self, dx, dy):
        if not self.right_held or self.suspended:
            return
        self.target_yaw = wrap_angle(self.target_yaw - math.radians(self.sensitivity_deg * dx))
        self.pitch_deg = float(np.clip(self.pitch_deg - self.sensitivity_deg * dy, 5., 75.))

    def scroll(self, amount):
        if not self.suspended:
            self.distance = float(np.clip(self.distance * math.exp(-.12 * amount), 1., 8.))

    def command(self, yaw, yaw_rate=0.):
        if self.suspended:
            return (0., 0., 0.)
        forward = int("W" in self.pressed) - int("S" in self.pressed)
        lateral = int("A" in self.pressed) - int("D" in self.pressed)
        scale = max(1., math.hypot(forward, lateral))
        error = wrap_angle(self.target_yaw - yaw)
        wz = 0. if abs(error) <= math.radians(1.) else float(
            np.clip(3. * error - .3 * yaw_rate, -self.wz_max, self.wz_max))
        return (self.vx_max * forward / scale, self.vy_max * lateral / scale, wz)

    def camera_pose(self, position, yaw, cast=None):
        pivot = np.asarray(position, dtype=float) + (0., 0., .12)
        pitch = math.radians(self.pitch_deg)
        direction = np.array((-math.cos(yaw) * math.cos(pitch),
                              -math.sin(yaw) * math.cos(pitch), math.sin(pitch)))
        distance = self.distance
        if cast is not None:
            hit = cast(pivot, direction, distance)
            if hit is not None:
                distance = max(.12, min(distance, hit - .15))
        return pivot + direction * distance, pivot
