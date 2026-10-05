"""Experimental current-domain actuator for deformable V2.

The CSV torque field is an uncalibrated current proxy. Only this module maps
protocol counts to the N*m consumed by Isaac's actuated URDF joints. Schema 1
retains the original priors; schema 2 adds the CAD-conditional fitted mechanism
and measured command/feedback timing. Neither is an absolute shaft-torque calibration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

_DEFAULT_MODEL = Path(__file__).with_name("configs") / "deformable_real2sim.json"
URDF_CORNER_ORDER = ("right_front", "left_front", "left_back", "right_back")


class FittedLegMechanism:
    """Current-driven assembly residual, friction and compliant upper stop.

    PhysX still supplies CAD gravity, inertia and ground contact. The identified
    extra inertia must be applied as joint armature by the caller; it must not be
    approximated by a torque using a stale acceleration sample.
    """

    def __init__(self, document, device, dtype=torch.float32):
        if tuple(document.get("joint_order", ())) != URDF_CORNER_ORDER:
            raise ValueError("fitted mechanism must use URDF corners RF, LF, LB, RB")
        if document.get("absolute_shaft_torque_calibrated") is not False:
            raise ValueError("this fit is simulation-equivalent, not calibrated shaft torque")
        zero = float(document["urdf_zero_physical_angle_deg"])*math.pi/180
        if not math.isfinite(zero):
            raise ValueError("invalid fitted physical-angle zero")
        self.physical_angle_zero = zero
        fields = {"gain":"torque_per_count", "armature":"extra_inertia",
                  "viscous":"viscous_friction", "coulomb":"coulomb_friction",
                  "bias":"constant_effort_bias", "residual_cos":"assembly_residual_cos_nm",
                  "residual_sin":"assembly_residual_sin_nm", "stop_angle":"upper_stop_physical_angle_deg",
                  "stop_stiffness":"stop_stiffness_nm_rad", "stop_damping":"stop_compression_damping_nm_s_rad"}
        for name, field in fields.items():
            values = [float(document["models"][corner][field]) for corner in URDF_CORNER_ORDER]
            if not all(math.isfinite(x) for x in values):
                raise ValueError(f"non-finite fitted {field}")
            if name in ("gain", "armature", "viscous", "coulomb", "stop_stiffness", "stop_damping") and min(values) < 0:
                raise ValueError(f"negative passive parameter {field}")
            if name in ("gain", "stop_stiffness") and min(values) <= 0:
                raise ValueError(f"nonpositive fitted {field}")
            setattr(self, name, torch.tensor(values, device=device, dtype=dtype))
        self.stop_q = zero-self.stop_angle*math.pi/180

    def effort(self, current, q, qd, *, torque_scale=1.0):
        depth = (self.stop_q-q).clamp_min(0.)
        contact = self.stop_stiffness*depth + self.stop_damping*(-qd).clamp_min(0.)*(depth > 0)
        friction = self.viscous*qd + self.coulomb*torch.tanh(qd/.02)
        residual = self.bias+self.residual_cos*q.cos()+self.residual_sin*q.sin()
        return self.gain*current*torque_scale-friction-residual+contact


def load_real2sim_model(path: str | Path | None = None) -> dict:
    model_path = Path(path) if path else _DEFAULT_MODEL
    with model_path.open() as handle:
        document = json.load(handle)
    model = document.get("model", document)
    if model.get("schema_version") not in (1, 2):
        raise ValueError("unsupported deformable real2sim model schema")
    mapping = model.get("simulation_effort_mapping", {})
    if mapping.get("effort_unit") != "N*m at the URDF actuated leg joint":
        raise ValueError("real2sim effort mapping must use Isaac joint N*m")
    for key in ("nominal_torque_per_current_raw_nm", "effort_limit_nm"):
        value = mapping.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid real2sim {key}: {value}")
    if model["schema_version"] == 2:
        FittedLegMechanism(model["fitted_mechanism"], "cpu")
    return model


def _range(value, *, integer=False, minimum=0):
    low, high = value
    if not all(math.isfinite(v) for v in (low, high)) or low < minimum or low > high:
        raise ValueError(f"invalid real2sim range: {value}")
    if integer and (int(low) != low or int(high) != high):
        raise ValueError(f"real2sim step counts must be integers: {value}")
    return (int(low), int(high)) if integer else (float(low), float(high))


class Real2SimActuator:
    """1 kHz CAN command hold, delay, motor-current lag and encoder feedback.

    Parameters are sampled independently for each environment and joint at
    reset. Fixed parameter ranges disable parameter variation, not temporal
    sensor noise. Use zero noise ranges for a noise-free deterministic probe.
    """

    def __init__(self, shape, device, cfg, model=None, dtype=torch.float32):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        self.num_envs, self.num_joints = shape
        self.model = model if model is not None else load_real2sim_model(cfg.real2sim_model_path)
        self.mechanism = (FittedLegMechanism(self.model["fitted_mechanism"],device,dtype)
                          if "fitted_mechanism" in self.model else None)
        mapping = self.model["simulation_effort_mapping"]
        self.current_limit = float(getattr(cfg, "real2sim_current_limit", 2048.0))
        self.effort_limit = float(getattr(cfg, "max_leg_torque", mapping["effort_limit_nm"]))
        if self.mechanism is not None:
            self.effort_limit = float(getattr(cfg,"real2sim_total_effort_limit_nm",mapping["effort_limit_nm"]))
        gain = getattr(cfg, "real2sim_torque_per_current_raw", None)
        self.nominal_torque_per_count = float(mapping["nominal_torque_per_current_raw_nm"] if gain is None else gain)
        self.dt = float(cfg.adrc_dt)
        for value in (self.current_limit, self.effort_limit, self.nominal_torque_per_count, self.dt):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("real2sim limits, torque gain and dt must be positive and finite")
        self.command_range = _range(getattr(cfg, "real2sim_command_delay_steps_range", (0, 1)), integer=True)
        self.feedback_range = _range(getattr(cfg, "real2sim_feedback_delay_steps_range", (1, 2)), integer=True)
        self.period_range = _range(getattr(cfg, "real2sim_command_period_steps_range", (2, 2)), integer=True, minimum=1)
        self.feedback_period_range = _range(getattr(cfg, "real2sim_feedback_period_steps_range", (1, 1)), integer=True, minimum=1)
        self.randomize = bool(getattr(cfg, "real2sim_randomize", True))
        self.max_command_delay = self.command_range[1]
        self.max_feedback_delay = self.feedback_range[1]
        self._env = torch.arange(self.num_envs, device=device)[:, None]
        self._joint = torch.arange(self.num_joints, device=device)[None, :]
        self._command_fifo = torch.zeros(self.max_command_delay + 1, *shape, device=device, dtype=dtype)
        self._sensor_fifo = torch.zeros(self.max_feedback_delay + 1, *shape, 3, device=device, dtype=dtype)
        self._command_cursor = self._sensor_cursor = 0
        self._command_delay = torch.zeros(shape, dtype=torch.long, device=device)
        self._feedback_delay = torch.zeros_like(self._command_delay)
        self._command_period = torch.ones_like(self._command_delay)
        self._feedback_period = torch.ones_like(self._command_delay)
        self._feedback_remaining = torch.zeros_like(self._command_delay)
        self._hold_remaining = torch.zeros_like(self._command_delay)
        for name in ("_prepared_current", "_motor_current", "_last_effort", "_backlash_remaining",
                     "_last_nonzero_sign", "_previous_q"):
            setattr(self, name, torch.zeros(shape, device=device, dtype=dtype))
        self._sensor = torch.zeros(*shape, 3, device=device, dtype=dtype)
        self._held_sensor = torch.zeros_like(self._sensor)
        self._parameter_ranges = {
            "_torque_scale": ("real2sim_torque_scale_range", (1., 1.)),
            "_current_noise_std": ("real2sim_current_noise_std_range", (0., 0.)),
            "_angle_noise_std": ("real2sim_angle_noise_std_range", (0., 0.)),
            "_velocity_noise_std": ("real2sim_velocity_noise_std_range", (0., 0.)),
            "_coulomb_friction": ("real2sim_coulomb_friction_range", (0., 0.)),
            "_viscous_friction": ("real2sim_viscous_friction_range", (0., 0.)),
            "_backlash": ("real2sim_backlash_range", (0., 0.)),
            "_current_sensor_noise_std": ("real2sim_current_sensor_noise_std_range", (0., 0.)),
        }
        for name, (field, default) in self._parameter_ranges.items():
            _range(getattr(cfg, field, default))
            setattr(self, name, torch.zeros(shape, device=device, dtype=dtype))
        lag = float(getattr(cfg, "real2sim_torque_lag_tau_s", 0.0))
        self.deadzone = float(getattr(cfg, "real2sim_current_deadzone_raw", 0.0))
        self.engaged_scale = float(getattr(cfg, "real2sim_backlash_engaged_scale", 0.25))
        self.friction_eps = float(getattr(cfg, "real2sim_friction_velocity_eps", 0.02))
        if not all(math.isfinite(v) for v in (lag, self.deadzone, self.engaged_scale, self.friction_eps)):
            raise ValueError("non-finite real2sim actuator parameter")
        if lag < 0 or self.deadzone < 0 or not 0 < self.engaged_scale <= 1 or self.friction_eps <= 0:
            raise ValueError("invalid real2sim actuator parameter")
        self.lag_alpha = 1. if lag == 0 else 1. - math.exp(-self.dt / lag)

    def _sample_steps(self, bounds, shape):
        low, high = bounds
        if self.randomize and low != high:
            return torch.randint(low, high + 1, shape, device=self.device)
        return torch.full(shape, round((low + high) / 2), device=self.device, dtype=torch.long)

    def reset(self, env_ids, q, qd):
        """Reset selected environments using full-batch joint state arrays."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        shape = (len(env_ids), self.num_joints)
        self._command_delay[env_ids] = self._sample_steps(self.command_range, shape)
        self._feedback_delay[env_ids] = self._sample_steps(self.feedback_range, shape)
        self._command_period[env_ids] = self._sample_steps(self.period_range, shape)
        self._feedback_period[env_ids] = self._sample_steps(self.feedback_period_range, shape)
        for name, (field, default) in self._parameter_ranges.items():
            low, high = _range(getattr(self.cfg, field, default))
            value = torch.empty(shape, device=self.device, dtype=self.dtype)
            value.uniform_(low, high) if self.randomize else value.fill_((low + high) / 2)
            getattr(self, name)[env_ids] = value
        self._command_fifo[:, env_ids] = 0
        for name in ("_prepared_current", "_motor_current", "_last_effort", "_backlash_remaining",
                     "_last_nonzero_sign", "_hold_remaining", "_feedback_remaining"):
            getattr(self, name)[env_ids] = 0
        self._previous_q[env_ids] = q[env_ids]
        self._sensor[env_ids, :, 0] = q[env_ids]
        self._sensor[env_ids, :, 1] = qd[env_ids]
        self._sensor[env_ids, :, 2] = 0
        self._sensor_fifo[:, env_ids] = self._sensor[env_ids]
        self._held_sensor[env_ids] = self._sensor[env_ids]

    @property
    def command_current_raw(self):
        """Last quantized CAN command, in configured motor-axis direction."""
        return self._prepared_current

    @property
    def last_effort(self):
        """Actual simulated effort after losses and the final PhysX limit."""
        return self._last_effort

    def sensor_state(self):
        return self._sensor[..., 0], self._sensor[..., 1], self._sensor[..., 2]

    def sensor_measurement(self, q, qd):
        """Return delayed encoder position, velocity and motor current counts.

        Current feedback is independent of torque gain and friction; a real
        robot cannot measure the simulator's ground-truth mechanical effort.
        """
        packet = torch.stack((q+torch.randn_like(q)*self._angle_noise_std,
                              qd+torch.randn_like(qd)*self._velocity_noise_std,
                              self._motor_current+torch.randn_like(q)*self._current_sensor_noise_std),dim=-1)
        ready = self._feedback_remaining <= 0
        self._held_sensor.copy_(torch.where(ready[...,None],packet,self._held_sensor))
        self._feedback_remaining.copy_(torch.where(ready,self._feedback_period-1,self._feedback_remaining-1))
        self._sensor_fifo[self._sensor_cursor] = self._held_sensor
        index = (self._sensor_cursor - self._feedback_delay) % len(self._sensor_fifo)
        self._sensor.copy_(self._sensor_fifo[index, self._env, self._joint])
        self._sensor_cursor = (self._sensor_cursor + 1) % len(self._sensor_fifo)
        return self.sensor_state()

    def apply(self, current_raw, q, qd):
        # std::round in the hardware rounds halves away from zero.
        limited = current_raw.clamp(-self.current_limit, self.current_limit)
        quantized = limited.sign() * (limited.abs() + 0.5).floor()
        quantized = torch.where(quantized.abs() < self.deadzone, 0., quantized)
        ready = self._hold_remaining <= 0
        self._prepared_current.copy_(torch.where(ready, quantized, self._prepared_current))
        self._hold_remaining.copy_(torch.where(ready, self._command_period - 1, self._hold_remaining - 1))
        self._command_fifo[self._command_cursor] = self._prepared_current
        index = (self._command_cursor - self._command_delay) % len(self._command_fifo)
        delayed = self._command_fifo[index, self._env, self._joint]
        self._command_cursor = (self._command_cursor + 1) % len(self._command_fifo)
        noisy = delayed + torch.randn_like(delayed) * self._current_noise_std
        if self.mechanism is None:
            noisy = noisy.clamp(-self.current_limit,self.current_limit)
        self._motor_current += self.lag_alpha * (noisy - self._motor_current)
        motor_effort = self._motor_current * self.nominal_torque_per_count * self._torque_scale
        friction = self._viscous_friction * qd + self._coulomb_friction * torch.tanh(qd / self.friction_eps)
        # Phenomenological reversal loss, not an identified gear-contact model.
        direction = delayed.sign()
        reversal = (direction * self._last_nonzero_sign) < 0
        self._backlash_remaining.copy_(torch.where(reversal, self._backlash, self._backlash_remaining))
        travel = (q - self._previous_q).abs()
        self._backlash_remaining.sub_(travel).clamp_min_(0.)
        engagement = torch.where(self._backlash_remaining > 0, self.engaged_scale, 1.)
        effort = motor_effort*engagement-friction
        if self.mechanism is not None:
            # Command limit caps the motor. The compliant stop is passive and
            # must not be clipped to that same motor-effort bound.
            scale = self.nominal_torque_per_count/self.model["simulation_effort_mapping"]["nominal_torque_per_current_raw_nm"]
            effort = self.mechanism.effort(self._motor_current*engagement,q,qd,torque_scale=self._torque_scale*scale)-friction
        self._last_effort.copy_(effort.clamp(-self.effort_limit,self.effort_limit))
        self._last_nonzero_sign.copy_(torch.where(direction != 0, direction, self._last_nonzero_sign))
        self._previous_q.copy_(q)
        return self._last_effort
