"""Batched RMCS DeformableJointController: physical-angle TD + ESO + NLESF.

URDF q is zero at the high posture: alpha = alpha_max - q (radians).
Motor torque is positive along URDF q, hence negative physical-alpha acceleration.
"""

import torch


class LegADRC:
    def __init__(self, shape, device, cfg, dtype=torch.float32):
        self.cfg = cfg
        if cfg.adrc_dt <= 0 or cfg.adrc_b0 == 0 or cfg.adrc_delta <= 0:
            raise ValueError("ADRC requires positive dt/delta and nonzero b0")
        self.x1 = torch.zeros(shape, device=device, dtype=dtype)
        self.x2 = torch.zeros_like(self.x1)
        self.z1 = torch.zeros_like(self.x1)
        self.z2 = torch.zeros_like(self.x1)
        self.z3 = torch.zeros_like(self.x1)
        self.last_u = torch.zeros_like(self.x1)
        self.applied_u = torch.zeros_like(self.x1)

    def reset(self, env_ids, q, q_target):
        self.x1[env_ids] = self.cfg.leg_max_physical_angle - q_target
        self.z1[env_ids] = self.cfg.leg_max_physical_angle - q
        for state in (self.x2, self.z2, self.z3, self.last_u, self.applied_u):
            state[env_ids] = 0.0

    def update(self, q, q_target):
        c = self.cfg
        h = c.adrc_dt
        measurement = c.leg_max_physical_angle - q
        target = c.leg_max_physical_angle - q_target
        # Raw RMCS mode uses published output; calibrated mode feeds back motor saturation.
        error = self.z1 - measurement
        self.z1 += h * (self.z2 - 3.0 * c.adrc_eso_w0 * error)
        observer_u = self.applied_u if getattr(c, "adrc_feedback_applied_torque", False) else self.last_u
        self.z2 += h * (self.z3 + c.adrc_b0 * observer_u - 3.0 * c.adrc_eso_w0**2 * error)
        self.z3 += h * (-c.adrc_eso_w0**3 * error)
        self.z3.clamp_(-c.adrc_z3_limit, c.adrc_z3_limit)

        # RL supplies no finite reference velocity, so RMCS always takes the TD path.
        d = c.adrc_td_r * c.adrc_td_h**2
        a0 = c.adrc_td_h * self.x2
        y = self.x1 - target + a0
        a1 = torch.sqrt(d * (d + 8.0 * y.abs()))
        a = torch.where(y.abs() > d, a0 + y.sign() * (a1 - d) * 0.5, a0 + y)
        fh = torch.where(a.abs() <= d, -c.adrc_td_r * a / d, -c.adrc_td_r * a.sign())
        fh = fh.clamp(-c.adrc_td_max_acc, c.adrc_td_max_acc)
        self.x1 += c.adrc_td_h * self.x2
        self.x2 += c.adrc_td_h * fh
        self.x2.clamp_(-c.adrc_td_max_vel, c.adrc_td_max_vel)

        e1, e2 = self.x1 - self.z1, self.x2 - self.z2
        fal1 = torch.where(e1.abs() <= c.adrc_delta, e1 / c.adrc_delta**(1.0 - c.adrc_alpha1),
                           e1.abs().pow(c.adrc_alpha1) * e1.sign())
        fal2 = torch.where(e2.abs() <= c.adrc_delta, e2 / c.adrc_delta**(1.0 - c.adrc_alpha2),
                           e2.abs().pow(c.adrc_alpha2) * e2.sign())
        output = ((c.adrc_k1 * fal1 + c.adrc_k2 * fal2 - self.z3) / c.adrc_b0).clamp(
            c.adrc_u_min, c.adrc_u_max)
        self.last_u.copy_((c.adrc_kt * output).clamp(c.adrc_output_min, c.adrc_output_max))
        self.applied_u.copy_(self.last_u.clamp(-c.max_leg_torque, c.max_leg_torque))
        return self.applied_u
