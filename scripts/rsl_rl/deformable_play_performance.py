"""Wall-clock measurements for bounded deformable play diagnostics."""

from collections import defaultdict
import json
import math
from pathlib import Path
import time


def configure_render_rate(cfg, hz):
    if not math.isfinite(hz) or hz <= 0:
        raise ValueError("Render rate must be positive and finite")
    cfg.sim.render_interval = max(cfg.decimation, math.ceil(1.0 / (cfg.sim.dt * hz)))


class PlayPerformance:
    def __init__(self, env, enabled=False):
        self.env = env
        self.enabled = enabled
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.steps = 0
        self.originals = []
        self.motion = []

    def _wrap(self, owner, name, label):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))

        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.seconds[label] += time.perf_counter() - start
                self.calls[label] += 1

        setattr(owner, name, measured)

    def start(self):
        if self.enabled:
            self._wrap(self.env.sim, "render", "render")
            self._wrap(self.env.sim, "step", "physics")
            for name, label in (("_apply_action", "controller"),
                                ("_get_observations", "observations"),
                                ("_get_rewards", "rewards"), ("_get_dones", "terminations")):
                self._wrap(self.env, name, label)
            for owner, name, label in ((self.env._leg_adrc, "update", "controller_adrc"),
                                      (self.env._leg_actuator, "apply", "controller_motor"),
                                      (self.env._wheel_drive, "update", "controller_wheel"),
                                      (self.env, "_wheel_geometry_w", "wheel_geometry"),
                                      (self.env, "_ground_height", "ground_height"),
                                      (self.env.scene, "update", "scene_update"),
                                      (self.env.scene, "write_data_to_sim", "scene_write")):
                if owner is not None:
                    self._wrap(owner, name, label)
        self.started = time.perf_counter()

    def record_policy(self, start):
        if self.enabled:
            # Complete the tiny actor call before attributing later work to physics.
            import torch
            if str(self.env.device).startswith("cuda"):
                torch.cuda.synchronize(self.env.device)
            self.seconds["policy"] += time.perf_counter() - start
            self.calls["policy"] += 1

    def record_step(self):
        self.steps += 1
        if self.enabled and self.steps % max(1, round(.25 / self.env.step_dt)) == 0:
            import torch
            data = self.env.robot.data
            q = data.joint_pos[0, self.env._legs_idx]
            self.motion.append(dict(step=self.steps, simulation_s=self.steps*self.env.step_dt,
                command=self.env.cmd_buf[0].tolist(), velocity=data.root_link_lin_vel_b[0].tolist(),
                position=data.root_link_pos_w[0].tolist(), quaternion=data.root_link_quat_w[0].tolist(),
                yaw_rate=data.root_ang_vel_b[0, 2].item(),
                leg_targets_deg=torch.rad2deg(self.env.cfg.leg_physical_angle_zero-self.env.leg_target[0]).tolist(),
                tilt_deg=float(torch.rad2deg(torch.atan2(data.projected_gravity_b[0, :2].norm(),
                                                       -data.projected_gravity_b[0, 2]))),
                leg_angles_deg=torch.rad2deg(self.env.cfg.leg_physical_angle_zero-q).tolist(),
                contact_count=int((self.env.wheel_normal_forces[0]>
                                   self.env.cfg.wheel_contact_force_threshold).sum().item()),
                slip_m_s=self.env._last_wheel_slip[0].abs().mean().item()))

    def progress(self):
        wall = max(time.perf_counter()-self.started, 1.e-9)
        sim = self.steps*self.env.step_dt
        return (f"[perf] sim={sim:.3f}s wall={wall:.2f}s realtime={sim/wall:.3f}x "
                f"policy_steps={self.steps/wall:.1f}/s "
                f"policy={1000*self.seconds['policy']/max(1,self.calls['policy']):.2f}ms "
                f"controller={self.seconds['controller']/wall:.0%} "
                f"physics={self.seconds['physics']/wall:.0%} render={self.seconds['render']/wall:.0%}")

    def finish(self, output=None, extra=None):
        wall = time.perf_counter()-self.started
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        report = dict(device=str(self.env.device), steps=self.steps, wall_s=wall,
            simulation_s=self.steps*self.env.step_dt,
            realtime_factor=self.steps*self.env.step_dt/max(wall,1.e-9),
            physics_dt=self.env.physics_dt, policy_dt=self.env.step_dt,
            render_interval=self.env.cfg.sim.render_interval,
            phase_seconds=dict(self.seconds), phase_calls=dict(self.calls), motion=self.motion)
        report.update(extra or {})
        if self.enabled:
            summary = {key: value for key, value in report.items() if key not in ("motion", "field", "third_person")}
            if "field" in report:
                summary["field"] = {key: report["field"][key] for key in
                    ("collision_meshes", "query_triangles", "physics_scene_count", "resets", "physics_probes")}
            if "third_person" in report:
                summary["third_person"] = {key: report["third_person"][key] for key in
                    ("input_counts", "target_yaw", "live_input")}
            print("[perf-summary] " + json.dumps(summary, ensure_ascii=False), flush=True)
        if output:
            path=Path(output)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        return report
