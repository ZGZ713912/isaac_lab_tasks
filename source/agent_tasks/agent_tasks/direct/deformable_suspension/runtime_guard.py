"""Batched deployment guard and causal sensor history, without Isaac imports.

Only joint encoders, gyro, accelerometer, odometry and initial AHRS attitude
enter the guard. Terrain normals and contact forces are deliberately absent.
The equations and defaults match RMCS SuspensionGuard.
"""

import math

import torch
from torch.nn.functional import normalize

GRAVITY = 9.80665


def flat_return_target(previous, desired, velocity, dt, rate_limit, acceleration_limit):
    """Brake before minangle instead of ending a constant-speed ramp abruptly."""
    error = desired - previous
    stopping_speed = (2. * acceleration_limit * error.abs()).sqrt().clamp(max=rate_limit)
    goal = error.sign() * stopping_speed
    next_velocity = velocity + (goal - velocity).clamp(-acceleration_limit * dt, acceleration_limit * dt)
    movement = next_velocity * dt
    crossing = (movement * error >= 0.) & (movement.abs() >= error.abs())
    target = previous + torch.where(crossing, error, movement)
    next_velocity = torch.where(crossing, torch.zeros_like(next_velocity), next_velocity)
    return target, next_velocity


def support_plane(q):
    s = math.sqrt(.5)
    radial = .12921 / s + .029108 * q.cos() + .13694 * q.sin() + .08542
    x = radial * q.new_tensor((s, s, -s, -s))
    y = radial * q.new_tensor((-s, s, s, -s))
    z = .08195 + .029108 * q.sin() - .13694 * q.cos()
    matrix = torch.stack((x, y, torch.ones_like(q)), -1)
    transpose = matrix.transpose(-1, -2)
    plane = torch.linalg.solve(transpose @ matrix, transpose @ z.unsqueeze(-1)).squeeze(-1)
    normal = normalize(torch.cat((-plane[:, :2], torch.ones_like(plane[:, :1])), -1), dim=-1)
    error = (matrix @ plane.unsqueeze(-1)).squeeze(-1).sub(z).abs().amax(-1)
    return normal, error


class SuspensionRuntimeGuard:
    def __init__(self, num_envs, device, *, enter_deg=3., exit_deg=3.5,
                 plane_tolerance_m=.006, confirmation_s=.1, attitude_tau_s=.25,
                 acceleration_tau_s=.05, acceleration_fusion_scale=.5):
        if (not 0 < enter_deg <= exit_deg < 90 or confirmation_s < 0
                or min(plane_tolerance_m, attitude_tau_s, acceleration_tau_s, acceleration_fusion_scale) <= 0
                or not all(math.isfinite(x) for x in (enter_deg, exit_deg, plane_tolerance_m,
                    confirmation_s, attitude_tau_s, acceleration_tau_s, acceleration_fusion_scale))):
            raise ValueError("Invalid suspension guard configuration")
        self.enter_deg, self.exit_deg = enter_deg, exit_deg
        self.plane_tolerance_m, self.confirmation_s = plane_tolerance_m, confirmation_s
        self.attitude_tau_s, self.acceleration_tau_s = attitude_tau_s, acceleration_tau_s
        self.acceleration_fusion_scale = acceleration_fusion_scale
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.flat_hold = self.valid.clone()
        self.flat_candidate = self.valid.clone()
        self.confirmation = torch.zeros(num_envs, device=device)
        self.up = torch.zeros(num_envs, 3, device=device)
        self.up[:, 2] = 1.
        self.velocity = torch.zeros_like(self.up)
        self.acceleration = torch.zeros_like(self.up)
        self.force_filter = torch.zeros_like(self.up)
        self.grade_deg = torch.full_like(self.confirmation, float("nan"))
        self.plane_error_m = self.grade_deg.clone()

    def reset(self, ids):
        self.valid[ids] = False
        self.flat_hold[ids] = False
        self.flat_candidate[ids] = False
        self.confirmation[ids] = 0.
        self.up[ids] = self.up.new_tensor((0., 0., 1.))
        self.velocity[ids] = 0.
        self.acceleration[ids] = 0.
        self.force_filter[ids] = 0.
        self.grade_deg[ids] = float("nan")
        self.plane_error_m[ids] = float("nan")

    def update(self, q, gyro, specific_force, encoder_twist, initial_gravity, dt):
        if not math.isfinite(dt) or not 0 < dt <= .05:
            raise ValueError("Guard needs a finite sensor period <=50ms")
        previous_hold = self.flat_hold.clone()
        valid = torch.cat((q, gyro, specific_force, encoder_twist, initial_gravity), -1).isfinite().all(-1)
        force_norm = specific_force.norm(dim=-1)
        valid &= (force_norm >= .25 * GRAVITY) & (force_norm <= 2.5 * GRAVITY)
        valid &= initial_gravity.norm(dim=-1) > 1.e-9
        fresh = ~self.valid
        velocity = torch.cat((encoder_twist[:, :2], torch.zeros_like(encoder_twist[:, :1])), -1)
        inertial = (velocity - self.velocity) / dt + torch.cross(gyro, velocity, dim=-1)
        alpha = -math.expm1(-dt / self.acceleration_tau_s)
        self.acceleration = torch.where(fresh[:, None], torch.cross(gyro, velocity, dim=-1),
                                       self.acceleration.lerp(inertial, alpha))
        self.force_filter = torch.where(fresh[:, None], specific_force, self.force_filter.lerp(specific_force, alpha))
        self.velocity.copy_(velocity)
        # Rodrigues propagation: body coordinates rotate opposite to the gyro.
        angle = gyro.norm(dim=-1, keepdim=True) * dt
        axis = normalize(gyro, dim=-1)
        propagated = (self.up * angle.cos() - torch.cross(axis, self.up, dim=-1) * angle.sin()
                      + axis * (axis * self.up).sum(-1, keepdim=True) * (1. - angle.cos()))
        self.up = torch.where(fresh[:, None], normalize(-initial_gravity, dim=-1), propagated)
        corrected = self.force_filter - self.acceleration
        magnitude = corrected.norm(dim=-1, keepdim=True)
        valid &= (magnitude[:, 0] >= .65 * GRAVITY) & (magnitude[:, 0] <= 1.35 * GRAVITY)
        alpha = -math.expm1(-dt / self.attitude_tau_s) / (
            1. + (self.acceleration.norm(dim=-1, keepdim=True) / self.acceleration_fusion_scale).square())
        self.up = normalize(self.up * (1. - alpha) + corrected / magnitude.clamp_min(1.e-9) * alpha, dim=-1)
        normal, error = support_plane(torch.nan_to_num(q))
        grade = torch.acos((normal * self.up).sum(-1).clamp(-1., 1.)) * (180. / math.pi)
        planar = valid & grade.isfinite() & (error <= self.plane_tolerance_m)
        candidate = planar & (grade <= self.enter_deg)
        self.flat_candidate.copy_(candidate)
        self.confirmation = torch.where(candidate & ~previous_hold, self.confirmation + dt, 0.)
        self.flat_hold = torch.where(previous_hold, planar & (grade <= self.exit_deg),
                                     candidate & (self.confirmation + 1.e-7 >= self.confirmation_s))
        self.confirmation[self.flat_hold] = 0.
        self.valid.copy_(valid)
        self.grade_deg.copy_(grade)
        self.plane_error_m.copy_(error)
        self.reset(~valid)
        return -self.up.clone(), previous_hold != self.flat_hold


class CausalPolicyHistory:
    """200Hz calls retain 100Hz frame spacing AND a 10ms action feedback delay."""

    def __init__(self, num_envs, history_length, policy_period_s, history_period_s, device):
        ratio = history_period_s / policy_period_s
        stride = round(ratio)
        if stride < 1 or abs(stride - ratio) > 1.e-7:
            raise ValueError("History period must be an integer multiple of the policy period")
        self.stride = stride
        self.frames = torch.zeros(num_envs, (history_length - 1) * stride + 1, 32, device=device)
        self.feedback = torch.zeros(num_envs, stride, 4, device=device)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.valid[ids] = False
        self.frames[ids] = 0.
        self.feedback[ids] = 0.

    def push(self, frame, applied_action):
        fresh = ~self.valid
        self.feedback = torch.roll(self.feedback, -1, dims=1)
        self.feedback[:, -1] = applied_action
        self.feedback[fresh] = 0.
        frame = frame.clone()
        frame[:, 22:26] = self.feedback[:, 0]
        self.frames = torch.roll(self.frames, -1, dims=1)
        self.frames[:, -1] = frame
        self.frames[fresh] = frame[fresh, None]
        self.valid[:] = True
        return self.frames[:, ::self.stride].clone()
