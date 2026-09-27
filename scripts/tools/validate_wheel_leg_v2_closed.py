#!/usr/bin/env python3
"""Validate the authored Wheel_leg_V2 closed-chain USD in Isaac Sim.

Loads ``Wheel_leg_V2.usd`` as an articulation (closures excluded from the
articulation), runs gravity, and measures the four-bar closure error and the
gas-spring travel.  Reference bundle reports ~0.73 mm closure error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for package in ("agent_world", "agent_tasks", "agent_rl"):
    package_root = REPO_ROOT / "source" / package
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

# NOTE: numpy/scipy must be imported after AppLauncher (Isaac Sim bundles its
# own numpy; importing the environment numpy first breaks scipy binaries and
# prevents GUI extensions from starting).
from isaaclab.app import AppLauncher

ASSET_DIR = REPO_ROOT / "source/agent_world/agent_world/assets/usd_files/Wheel_leg_V2"
DEFAULT_USD = ASSET_DIR / "Wheel_leg_V2.usd"
DEFAULT_CONSTRAINTS = ASSET_DIR / "constraints.json"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
parser.add_argument("--constraints", type=Path, default=DEFAULT_CONSTRAINTS)
parser.add_argument("--steps", type=int, default=480)
parser.add_argument("--dt", type=float, default=1.0 / 240.0)
parser.add_argument("--spawn-z", type=float, default=0.30)
parser.add_argument("--error-tolerance", type=float, default=2.0e-3)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402


def load_constraints(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    closures = [(c["name"], c["body0"], c["body1"], c["local_pos0_m"], c["local_pos1_m"]) for c in data["closures"]]
    springs = [(s["name"], s["body0"], s["body1"], s["nominal_length_m"]) for s in data["gas_springs"]]
    return closures, springs


def robot_cfg(usd_path: Path, spawn_z: float) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            copy_from_source=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, spawn_z),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "passive": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
                effort_limit=0.0,
                velocity_limit=0.0,
            )
        },
    )


def body_ids(robot: Articulation, name: str) -> int:
    indices, _ = robot.find_bodies(name)
    if len(indices) != 1:
        raise RuntimeError(f"expected one body {name}, got {indices}")
    return int(indices[0])


def main() -> None:
    closures, springs = load_constraints(args_cli.constraints)

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)
    sim_cfg.physx.min_position_iteration_count = 8
    sim_cfg.physx.max_position_iteration_count = 32
    sim_cfg.physx.min_velocity_iteration_count = 4
    sim_cfg.physx.max_velocity_iteration_count = 16
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.0, -1.2, 0.6], [0.0, 0.0, 0.0])

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    robot = Articulation(robot_cfg(args_cli.usd.resolve(), args_cli.spawn_z))
    sim.reset()
    robot.reset()
    dt = args_cli.dt

    closure_ids = [(name, body_ids(robot, a), body_ids(robot, b),
                    torch.tensor(p0, dtype=torch.float32, device=sim.device),
                    torch.tensor(p1, dtype=torch.float32, device=sim.device))
                   for name, a, b, p0, p1 in closures]
    spring_ids = [(name, body_ids(robot, a), body_ids(robot, b), nominal)
                  for name, a, b, nominal in springs]

    def point(body: int, local: torch.Tensor) -> torch.Tensor:
        return robot.data.body_pos_w[:, body] + quat_apply(robot.data.body_quat_w[:, body], local.expand(1, -1))

    max_error = {name: 0.0 for name, *_ in closures}
    final_error = {}
    spring_min = {name: float("inf") for name, *_ in springs}
    spring_max = {name: 0.0 for name, *_ in springs}

    for _ in range(args_cli.steps):
        sim.step()
        robot.update(dt)
        for name, b0, b1, p0, p1 in closure_ids:
            err = float(torch.norm(point(b0, p0) - point(b1, p1)))
            max_error[name] = max(max_error[name], err)
            final_error[name] = err
        for name, b0, b1, nominal in spring_ids:
            length = float(torch.norm(point(b0, torch.zeros(3, device=sim.device)) -
                                      point(b1, torch.zeros(3, device=sim.device))))
            spring_min[name] = min(spring_min[name], length)
            spring_max[name] = max(spring_max[name], length)
        if not simulation_app.is_running():
            break

    print("\nWheel_leg_V2 closed-chain validation")
    print(f"  base height at end: {float(robot.data.root_pos_w[0, 2]):.4f} m")
    for name, *_ in closures:
        print(f"  {name:14s} max closure error = {max_error[name]*1000:8.4f} mm   final = {final_error[name]*1000:8.4f} mm")
    for name, *_ in springs:
        print(f"  {name:14s} length range = [{spring_min[name]*1000:.2f}, {spring_max[name]*1000:.2f}] mm")
    worst = max(max_error.values())
    print(f"  worst closure error = {worst*1000:.4f} mm (reference ~0.7261 mm, tolerance {args_cli.error_tolerance*1000:.1f} mm)")
    if worst > args_cli.error_tolerance:
        raise SystemExit(f"[FAIL] closure error {worst*1000:.3f} mm exceeds tolerance")


if __name__ == "__main__":
    main()
    simulation_app.close()
