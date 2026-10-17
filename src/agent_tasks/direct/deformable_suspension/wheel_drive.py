"""Bounded wheel-speed PI with a shared target slew limit, at the physics rate."""

import torch


class WheelVelocityPI:
    def __init__(self, shape, device, cfg, dtype=torch.float32):
        self.dt = cfg.adrc_dt
        self.kp = cfg.wheel_velocity_kp
        self.ki = cfg.wheel_velocity_ki
        self.limit = cfg.wheel_torque_limit
        self.speed_limit = cfg.wheel_speed_limit
        self.acceleration = cfg.wheel_acceleration_limit
        self.inertia = cfg.wheel_axial_inertia
        if min(self.dt, self.kp, self.limit, self.speed_limit, self.acceleration) <= 0 or self.ki < 0:
            raise ValueError("Wheel PI requires positive limits/dt/kp and nonnegative ki")
        self.target = torch.zeros(shape, device=device, dtype=dtype)
        self.integral = torch.zeros_like(self.target)  # Nm, not an unbounded sum of speed errors
        self.torque = torch.zeros_like(self.target)

    def update(self, requested_speed, measured_speed, contact):
        requested_speed = requested_speed.clamp(-self.speed_limit, self.speed_limit)
        delta = requested_speed - self.target
        # Scale the four-wheel change together to preserve the requested twist direction.
        scale = (self.acceleration * self.dt / delta.abs().amax(-1, keepdim=True).clamp_min(1.0e-9)).clamp(max=1.0)
        target_delta = delta * scale
        self.target.add_(target_delta)
        feedforward = self.inertia * target_delta / self.dt
        error = self.target - measured_speed
        candidate = (self.integral + self.ki * self.dt * error).clamp(-self.limit, self.limit)
        raw = feedforward + self.kp * error + candidate
        saturating_outwards = ((raw > self.limit) & (error > 0)) | ((raw < -self.limit) & (error < 0))
        self.integral.copy_(torch.where(saturating_outwards, self.integral, candidate))
        # An unloaded spinning wheel must not accumulate a traction demand for landing.
        self.integral.masked_fill_(~contact, 0.0)
        self.torque.copy_((feedforward + self.kp * error + self.integral).clamp(-self.limit, self.limit))
        return self.torque

    def reset(self, env_ids):
        self.target[env_ids] = 0.0
        self.integral[env_ids] = 0.0
        self.torque[env_ids] = 0.0
