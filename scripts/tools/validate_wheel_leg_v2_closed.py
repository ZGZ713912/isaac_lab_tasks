#!/usr/bin/env python3
"""Run a small Isaac Lab dynamics probe for Wheel_leg_V2 closures."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for package in ("agent_world", "agent_tasks", "agent_rl"):
    package_root = REPO_ROOT / "source" / package
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--dt", type=float, default=1.0 / 240.0)
parser.add_argument(
    "--usd",
    type=Path,
    default=REPO_ROOT
    / "source/agent_world/agent_world/assets/usd_files/Wheel_leg_V2/Wheel_leg_V2_closed_gas_spring.usd",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaacsim.core.prims import RigidPrim

from agent_world.actuators.wheel_leg_v2_gas_spring import WheelLegV2GasSpringModel


FOUR_BAR = (
    ("LL_link4", "L_link3", (0.00827712, -0.00724487, -0.36980174), (0.23542012, 0.08457446, -0.37359826)),
    ("RR_link4", "R_link3", (0.00565100, -0.00943867, 0.01919846), (-0.08800131, 0.23572013, 0.02300155)),
)
SPRINGS = (("LLL_link1", "LLL_link2"), ("RRR_link1", "RRR_link2"))


def quat_apply_np(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q_xyz = quat[1:]
    q_w = quat[0]
    t = 2.0 * np.cross(q_xyz, vector)
    return vector + q_w * t + np.cross(q_xyz, t)


def frame_point(view: RigidPrim, local: tuple[float, float, float]) -> np.ndarray:
    pos, quat = view.get_world_poses()
    pos = pos[0].detach().cpu().numpy() if torch.is_tensor(pos) else pos[0]
    quat = quat[0].detach().cpu().numpy() if torch.is_tensor(quat) else quat[0]
    return pos + quat_apply_np(quat, np.asarray(local, dtype=np.float32))


def main() -> None:
    if not args_cli.usd.is_file():
        raise FileNotFoundError(args_cli.usd)

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)
    sim_cfg.physx.min_position_iteration_count = 16
    sim_cfg.physx.max_position_iteration_count = 64
    sim_cfg.physx.min_velocity_iteration_count = 4
    sim_cfg.physx.max_velocity_iteration_count = 16
    sim_cfg.physx.enable_external_forces_every_iteration = True
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, -1.5, 1.0], [0.0, 0.0, 0.0])
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    asset_cfg = sim_utils.UsdFileCfg(usd_path=str(args_cli.usd.resolve()))
    asset_cfg.func("/World/Robot", asset_cfg, translation=(0.0, 0.0, 0.4))
    sim.reset()
    four_bar_views = [(RigidPrim(f"/World/Robot/{a}"), RigidPrim(f"/World/Robot/{b}")) for a, b, _, _ in FOUR_BAR]
    spring_views = [(RigidPrim(f"/World/Robot/{a}"), RigidPrim(f"/World/Robot/{b}")) for a, b in SPRINGS]
    for view_a, view_b in four_bar_views + spring_views:
        view_a.initialize()
        view_b.initialize()
    model = WheelLegV2GasSpringModel()
    max_loop_error = 0.0
    min_length = float("inf")
    max_length = 0.0
    min_force = float("inf")
    max_force = 0.0

    for step in range(args_cli.steps):
        for view_a, view_b in spring_views:
            p0 = frame_point(view_a, (0.0, 0.0, 0.0))
            p1 = frame_point(view_b, (0.0, 0.0, 0.0))
            v0 = view_a.get_velocities()[0, :3]
            v1 = view_b.get_velocities()[0, :3]
            if torch.is_tensor(v0):
                v0 = v0.detach().cpu().numpy()
                v1 = v1.detach().cpu().numpy()
            delta = p1 - p0
            length = float(np.linalg.norm(delta))
            axis = delta / max(length, 1.0e-8)
            length_rate = float(np.dot(v1 - v0, axis))
            length_tensor = torch.tensor([length], dtype=torch.float32)
            axis_tensor = torch.tensor(axis, dtype=torch.float32).unsqueeze(0)
            rate_tensor = torch.tensor([length_rate], dtype=torch.float32)
            force = float(model.force_magnitude(length_tensor, rate_tensor)[0])
            force_vector = torch.as_tensor((force * axis).reshape(1, 3), dtype=torch.float32, device=sim.device)
            view_a.apply_forces(force_vector * -1.0, is_global=True)
            view_b.apply_forces(force_vector, is_global=True)
            min_length = min(min_length, length)
            max_length = max(max_length, length)
            min_force = min(min_force, force)
            max_force = max(max_force, force)

        sim.step()

        for (view_a, view_b), (_, _, local0, local1) in zip(four_bar_views, FOUR_BAR):
            p0 = frame_point(view_a, local0)
            p1 = frame_point(view_b, local1)
            max_loop_error = max(max_loop_error, float(np.linalg.norm(p0 - p1)))

        if not simulation_app.is_running():
            break

    print("Wheel_leg_V2 closure probe")
    print(f"  four-bar max pivot error: {max_loop_error:.6e} m")
    print(f"  gas-spring length range: {min_length:.6f} .. {max_length:.6f} m")
    print(f"  gas-spring force range: {min_force:.3f} .. {max_force:.3f} N")
    if max_loop_error > 2.0e-3:
        raise RuntimeError(f"four-bar closure error too large: {max_loop_error} m")


if __name__ == "__main__":
    main()
    simulation_app.close()
