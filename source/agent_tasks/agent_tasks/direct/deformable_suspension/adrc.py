"""Batched RMCS DeformableJointController: physical-angle TD + ESO + NLESF.

Physical angle is alpha = calibrated_zero - q (radians). Legacy tasks retain
their historical offset; V3 uses the CAD rod direction as calibrated_zero.
Motor torque is positive along URDF q, hence negative physical-alpha acceleration.
"""

import math

import torch


class LegADRC:
    def __init__(self, shape, device, cfg, dtype=torch.float32):
        self.cfg = cfg
        if cfg.adrc_dt <= 0 or cfg.adrc_b0 == 0 or cfg.adrc_delta <= 0:
            raise ValueError("ADRC requires positive dt/delta and nonzero b0")
        self.current_mode = getattr(cfg, "adrc_output_domain", "torque") == "current_raw"
        self.current_scale = float(getattr(cfg, "adrc_controller_output_to_current_raw", 1.0))
        self.angle_zero = float(getattr(cfg, "leg_physical_angle_zero", cfg.leg_max_physical_angle))
        if not math.isfinite(self.angle_zero):
            raise ValueError("ADRC physical-angle zero must be finite")
        if self.current_mode and (not math.isfinite(self.current_scale) or self.current_scale <= 0):
            raise ValueError("ADRC controller-to-current scale must be finite and positive")
        self.x1 = torch.zeros(shape, device=device, dtype=dtype)
        self.x2 = torch.zeros_like(self.x1)
        self.z1 = torch.zeros_like(self.x1)
        self.z2 = torch.zeros_like(self.x1)
        self.z3 = torch.zeros_like(self.x1)
        self.last_u = torch.zeros_like(self.x1)
        self.applied_u = torch.zeros_like(self.x1)

    def reset(self, env_ids, q, q_target):
        self.x1[env_ids] = self.angle_zero - q_target
        self.z1[env_ids] = self.angle_zero - q
        for state in (self.x2, self.z2, self.z3, self.last_u, self.applied_u):
            state[env_ids] = 0.0

    def update(self, q, q_target, *, applied_current_raw=None, target_physical_velocity=None):
        c = self.cfg
        h = c.adrc_dt
        measurement = self.angle_zero - q
        target = self.angle_zero - q_target
        error = self.z1 - measurement
        self.z1 += h * (self.z2 - 3.0 * c.adrc_eso_w0 * error)
        if self.current_mode:
            # Hardware feeds back its most recently prepared, quantized CAN
            # command, including saturation and the command hold.  It is not
            # the new servo request, motor current feedback, or PhysX effort.
            prepared_current = self.applied_u if applied_current_raw is None else applied_current_raw
            observer_u = prepared_current / self.current_scale
        else:
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
        if target_physical_velocity is None:
            self.x1 += c.adrc_td_h * self.x2
            self.x2 += c.adrc_td_h * fh
            self.x2.clamp_(-c.adrc_td_max_vel, c.adrc_td_max_vel)
            reference_angle, reference_velocity = self.x1, self.x2
        else:
            # Identification sweeps supply a finite physical-angle velocity;
            # RMCS bypasses TD and leaves its hidden state unchanged per joint.
            supplied = torch.as_tensor(target_physical_velocity,device=q.device,dtype=q.dtype)
            finite = torch.isfinite(supplied)
            self.x1 += torch.where(finite,0.,c.adrc_td_h*self.x2)
            next_x2 = (self.x2+c.adrc_td_h*fh).clamp(-c.adrc_td_max_vel,c.adrc_td_max_vel)
            self.x2.copy_(torch.where(finite,self.x2,next_x2))
            reference_angle = torch.where(finite,target,self.x1)
            reference_velocity = torch.where(finite,supplied,self.x2)

        e1, e2 = reference_angle - self.z1, reference_velocity - self.z2
        fal1 = torch.where(e1.abs() <= c.adrc_delta, e1 / c.adrc_delta**(1.0 - c.adrc_alpha1),
                           e1.abs().pow(c.adrc_alpha1) * e1.sign())
        fal2 = torch.where(e2.abs() <= c.adrc_delta, e2 / c.adrc_delta**(1.0 - c.adrc_alpha2),
                           e2.abs().pow(c.adrc_alpha2) * e2.sign())
        output = (c.adrc_k1 * fal1 + c.adrc_k2 * fal2 - self.z3) / c.adrc_b0
        # RMCS current-domain ADRC observes and publishes protocol counts.  The
        # conversion to Isaac's joint N*m is intentionally performed by
        # Real2SimActuator after this method.  The historical torque-domain
        # path keeps its existing final N*m limit for old checkpoints.
        if self.current_mode:
            current_limit = float(getattr(c, "real2sim_current_limit", 2048.0))
            self.last_u.copy_(c.adrc_kt * output)
            raw_output = self.last_u * self.current_scale
            self.applied_u.copy_(raw_output.clamp(-current_limit, current_limit))
        else:
            output = output.clamp(c.adrc_u_min, c.adrc_u_max)
            self.last_u.copy_((c.adrc_kt * output).clamp(c.adrc_output_min, c.adrc_output_max))
            self.applied_u.copy_(self.last_u.clamp(-c.max_leg_torque, c.max_leg_torque))
        return self.applied_u
