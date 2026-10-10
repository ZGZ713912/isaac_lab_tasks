"""Sensor-only geometric training teacher, never part of exported MLP inference.

Encoders estimate the supporting wheel plane; IMU gravity supplies world up.
The teacher searches reachable attitude corrections within the calibrated leg
stroke and CAD clearance envelope. Its targets still pass through the normal
actuator and target-rate limits in Isaac Sim.
"""

import math

import torch
from torch import nn


class GeometryTeacher(nn.Module):
    def __init__(self, fallback, utilities, *, lower, upper, baseline, bottom_bounds,
                 clearance=.006, target_step=.02, damping_s=.05, correction_limit=.04,
                 correction_smoothing=1.0, yaw_support_gate=False):
        super().__init__()
        if (not 0 <= lower < baseline < upper or clearance <= 0 or target_step <= 0
                or damping_s < 0 or correction_limit <= 0
                or not math.isfinite(correction_smoothing) or not 0 < correction_smoothing <= 1
                or type(yaw_support_gate) is not bool):
            raise ValueError("Invalid geometric teacher stroke or margins")
        self.fallback = fallback.eval().requires_grad_(False)
        self.utilities = utilities
        self.lower, self.upper, self.baseline = lower, upper, baseline
        self.clearance, self.target_step, self.damping_s = clearance, target_step, damping_s
        self.correction_limit = correction_limit
        self.correction_smoothing = correction_smoothing
        self.yaw_support_gate = yaw_support_gate
        grid = torch.linspace(lower, upper, 65)
        centers, _ = utilities.wheel_geometry(grid[:, None].expand(-1, 4))
        self.register_buffer("grid", grid)
        self.register_buffer("grid_centers", centers)
        self.register_buffer("bottom_bounds", torch.tensor(bottom_bounds))

    def _interval(self, normal, margin=None):
        heights = self.utilities.WHEEL_RADIUS - torch.einsum("...d,kid->...ki", normal, self.grid_centers)
        lo = heights.amin(-2).amax(-1)
        hi = heights.amax(-2).amin(-1)
        b = self.bottom_bounds
        bottom = (normal[..., 0] * torch.where(normal[..., 0] >= 0, b[0], b[1])
                  + normal[..., 1] * torch.where(normal[..., 1] >= 0, b[2], b[3])
                  + normal[..., 2] * b[4])
        lo = lo.maximum((self.clearance if margin is None else margin) - bottom)
        return lo, hi, heights

    @staticmethod
    def _rotated_normal(normal, axis, angle):
        c, s = angle.cos().unsqueeze(-1), angle.sin().unsqueeze(-1)
        return normal * c - torch.cross(axis, normal, dim=-1) * s + axis * (axis * normal).sum(-1, keepdim=True) * (1-c)

    def targets(self, raw):
        latest = raw.reshape(raw.shape[0], -1, 32)[:, -1]
        q = latest[:, 10:14].clamp(self.lower, self.upper)
        centers, _ = self.utilities.wheel_geometry(q)
        centered = centers - centers.mean(1, keepdim=True)
        xy, z = centered[..., :2], centered[..., 2:]
        slope = torch.linalg.solve(xy.transpose(1, 2) @ xy + 1.e-8 * torch.eye(2, device=q.device),
                                   xy.transpose(1, 2) @ z).squeeze(-1)
        normal = torch.nn.functional.normalize(torch.cat((-slope, torch.ones_like(slope[:, :1])), -1), dim=-1)
        plane_residual = (centered * normal[:, None]).sum(-1).abs().amax(-1)
        current_height = self.utilities.WHEEL_RADIUS - (centers.mean(1) * normal).sum(-1)
        up = torch.nn.functional.normalize(-latest[:, 7:10], dim=-1)
        omega = 2 * latest[:, 4:7]
        grade = torch.acos((normal * up).sum(-1).clamp(-1, 1)) * (180/math.pi)
        moving = torch.maximum(latest[:, 1:3].norm(dim=-1)/.4, latest[:, 3].abs()/.125)
        moving = torch.maximum(moving, omega[:, :2].norm(dim=-1)/.25).clamp(0, 1)
        margin = self.clearance + max(0., .018-self.clearance) * ((grade-10)/7).clamp(0, 1) * moving
        # Prediction reduces correction as the body is already approaching up.
        up = torch.nn.functional.normalize(up - self.damping_s * torch.cross(omega, up, dim=-1), dim=-1)
        angle = torch.atan2(up[:, :2].norm(dim=-1), up[:, 2])
        axis = torch.stack((-up[:, 1], up[:, 0], torch.zeros_like(up[:, 0])), -1)
        axis = torch.nn.functional.normalize(axis, dim=-1)
        fractions = torch.linspace(0, 1, 17, device=q.device)
        normals = self._rotated_normal(normal[:, None], axis[:, None], angle[:, None] * fractions)
        lo, hi, _ = self._interval(normals, margin[:, None])
        feasible = (lo <= hi) & (normals[..., 2] > .7)
        fraction = torch.where(feasible, fractions, -torch.ones_like(lo)).amax(-1)
        valid = (fraction >= 0) & (plane_residual < .004) & (up[:, 2] > .7)
        left, right = fraction.clamp_min(0), (fraction + 1/16).clamp(0, 1)
        for _ in range(12):
            mid = .5 * (left + right)
            n = self._rotated_normal(normal, axis, angle * mid)
            l, h, _ = self._interval(n, margin)
            ok = (l <= h) & (n[:, 2] > .7)
            left, right = torch.where(ok, mid, left), torch.where(ok, right, mid)
        goal_normal = self._rotated_normal(normal, axis, angle * left)
        lo, hi, heights = self._interval(goal_normal, margin)
        height = current_height.maximum(lo).minimum(hi)
        difference = heights - height[:, None, None]
        crossings = difference[:, :-1] * difference[:, 1:] <= 0
        midpoint = .5 * (self.grid[:-1] + self.grid[1:])
        distance = (midpoint[None, :, None] - q[:, None]).abs()
        index = torch.where(crossings, distance, torch.full_like(distance, 1.e6)).argmin(1)
        h0 = torch.gather(heights, 1, index[:, None]).squeeze(1)
        h1 = torch.gather(heights, 1, (index + 1)[:, None]).squeeze(1)
        denominator = h1 - h0
        denominator = torch.where(denominator.abs() > 1.e-8, denominator, torch.ones_like(denominator))
        weight = ((height[:, None] - h0) / denominator).clamp(0, 1)
        goal = self.grid[index] + weight * (self.grid[index+1] - self.grid[index])
        valid = valid & crossings.any(1).all(-1) & torch.isfinite(goal).all(-1)
        return goal, valid, dict(correction_fraction=left, plane_residual=plane_residual,
                                 goal_normal=goal_normal, normal_height=height, clearance_margin=margin,
                                 estimated_grade_deg=grade)

    def forward(self, raw):
        goal, valid, details = self.targets(raw)
        latest = raw.reshape(raw.shape[0], -1, 32)[:, -1]
        q = latest[:, 10:14].clamp(self.lower, self.upper)
        fallback = self.fallback(raw).clamp(-1, 1)
        fallback_target = self.utilities.suspension_target(fallback, q.new_full((len(q), 1), self.baseline),
                                                           self.upper, self.lower)
        # Encoders alone do not give common height a restoring gain. Retain the
        # demonstrated actor's height anchor while correcting leg differences.
        centers, _ = self.utilities.wheel_geometry(fallback_target)
        normal = details["goal_normal"]
        height = (self.utilities.WHEEL_RADIUS - (centers * normal[:, None]).sum(-1)).mean(-1)
        lo, hi, heights = self._interval(normal, details["clearance_margin"])
        height = height.maximum(lo).minimum(hi)
        difference = heights-height[:, None, None]
        crossings = difference[:, :-1]*difference[:, 1:] <= 0
        midpoint = .5*(self.grid[:-1]+self.grid[1:])
        distance = (midpoint[None, :, None]-q[:, None]).abs()
        index = torch.where(crossings, distance, torch.full_like(distance, 1.e6)).argmin(1)
        h0 = torch.gather(heights, 1, index[:, None]).squeeze(1)
        h1 = torch.gather(heights, 1, (index+1)[:, None]).squeeze(1)
        denominator = h1-h0
        denominator = torch.where(denominator.abs()>1.e-8, denominator, torch.ones_like(denominator))
        fraction = ((height[:, None]-h0)/denominator).clamp(0, 1)
        goal = self.grid[index]+fraction*(self.grid[index+1]-self.grid[index])
        valid = valid & crossings.any(1).all(-1)
        # Do not limit relative to the measured encoder: that caps position
        # error before the inner PID can support gravity, so the legs collapse.
        # The environment applies its existing rate limit to the setpoint,
        # independently of measurement lag, along with current/stroke limits.
        difference = goal-fallback_target
        # A separate clip on each leg distorts the supporting plane. Scale the
        # complete posture correction together so differential legs retain
        # their coordinated direction within the same per-leg amplitude cap.
        scale = self.correction_limit / difference.abs().amax(-1, keepdim=True).clamp_min(self.correction_limit)
        # Real rollouts show that steep grades need the demonstrated support
        # reserve. Fade the experimental leveling correction at 10--12 degrees;
        # never trade away their support for a geometric attitude witness.
        grade_gate = ((12-details["estimated_grade_deg"])/2).clamp(0, 1)
        if self.yaw_support_gate:
            # Ten-degree spin starts were the remaining contact failure in
            # actual rollouts. Retain the base's demonstrated support there.
            # Commands are already sensor ABI inputs: yaw is scaled by .25.
            # Allow half a degree for the encoder-plane estimate. A tiny
            # nonzero gate must not retain a large previous correction.
            reserve = ((details["estimated_grade_deg"]-8)/1.5).clamp(0, 1)
            yaw_fraction = (latest[:, 3].abs()/.125).clamp(0, 1)
            grade_gate = grade_gate*(1-reserve*yaw_fraction)
        correction = grade_gate[:, None]*scale*difference
        if self.correction_smoothing < 1:
            frames = raw.reshape(raw.shape[0], -1, 32)
            # Previous applied action already belongs to the sensor ABI. Use
            # it to retain only the slow corrective part, while the base MLP
            # keeps its full-rate restoring response and gravity support.
            previous_raw = torch.cat((frames[:, :1], frames[:, :-1]), 1).flatten(1)
            previous_base = self.fallback(previous_raw).clamp(-1, 1)
            previous_base_target = self.utilities.suspension_target(
                previous_base, q.new_full((len(q), 1), self.baseline), self.upper, self.lower)
            previous_target = self.utilities.suspension_target(
                latest[:, 22:26], q.new_full((len(q), 1), self.baseline), self.upper, self.lower)
            previous_correction = (previous_target-previous_base_target).clamp(-self.correction_limit,
                                                                              self.correction_limit)
            previous_loaded = frames[:, -2, 18:22].abs().mean(-1) > .02
            previous_correction = torch.where(previous_loaded[:, None], previous_correction,
                                               torch.zeros_like(previous_correction))
            correction = previous_correction.lerp(correction, self.correction_smoothing)
            correction = torch.where(grade_gate[:, None] > 0, correction, torch.zeros_like(correction))
        target = fallback_target + correction
        target = target.clamp(self.lower, self.upper)
        geometric = self.utilities.suspension_action(target, q.new_full((len(q), 1), self.baseline),
                                                     self.upper, self.lower)
        # Preserve the demonstrated flat-ground feedback instead of making
        # common leg height a neutrally stable integrator. Unequal legs expose
        # a leveled slope even when its body tilt is already near zero.
        tilt = torch.atan2(latest[:, 7:9].norm(dim=-1), -latest[:, 9])
        sloped = (q.amax(-1) - q.amin(-1) > math.radians(6)) | (tilt > math.radians(3))
        loaded = latest[:, 18:22].abs().mean(-1) > .02
        return torch.where((valid & sloped & loaded)[:, None], geometric, fallback)
