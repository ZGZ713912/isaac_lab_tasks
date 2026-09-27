#!/usr/bin/env python3
"""Interactive closed-chain viewer for Wheel_leg_V2.

Opens ``Wheel_leg_V2.usd`` with gravity disabled, fixes the base, and lets you
drive ``L_joint1`` with a position target from a small omni.ui window while
the four-bar closures and the gas-spring prismatic follow.  Live readouts show
the closure errors and gas-spring lengths so the closed chain can be judged
directly.
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

# NOTE: numpy/scipy must NOT be imported before AppLauncher.  Isaac Sim bundles
# its own numpy; importing the environment numpy first makes scipy binaries
# incompatible and breaks extension startup (GUI closes immediately).
from isaaclab.app import AppLauncher

ASSET_DIR = REPO_ROOT / "source/agent_world/agent_world/assets/usd_files/Wheel_leg_V2"
DEFAULT_USD = ASSET_DIR / "Wheel_leg_V2.usd"
DEFAULT_CONSTRAINTS = ASSET_DIR / "constraints.json"
URDF = ASSET_DIR / "urdf/urdf_v5.0.urdf"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
parser.add_argument("--constraints", type=Path, default=DEFAULT_CONSTRAINTS)
parser.add_argument("--dt", type=float, default=1.0 / 240.0)
parser.add_argument("--target-joint", type=str, default="L_joint1")
parser.add_argument("--slider-min", type=float, default=-1.5)
parser.add_argument("--slider-max", type=float, default=1.5)
parser.add_argument("--stiffness", type=float, default=80.0)
parser.add_argument("--damping", type=float, default=4.0)
parser.add_argument("--passive-damping", type=float, default=1.0)
parser.add_argument("--max-effort", type=float, default=100.0)
parser.add_argument("--drive-mode", choices=["position", "effort"], default="position")
parser.add_argument("--torque", type=float, default=20.0)
parser.add_argument("--no-fix-base", action="store_true", default=False)
parser.add_argument("--no-markers", action="store_true", default=False)
parser.add_argument("--max-seconds", type=float, default=0.0, help="auto-exit after N seconds (0 = until closed)")
parser.add_argument("--self-test-steps", type=int, default=0, help="headless: steps to run before exit")
parser.add_argument("--self-test-amplitude", type=float, default=0.5, help="headless: target sweep amplitude (rad)")
parser.add_argument(
    "--self-test-stop-after",
    type=int,
    default=0,
    help="GUI regression test: stop the timeline after N steps to verify the app stays closable",
)
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
    closures = [
        (c["name"], c["body0"], c["body1"], np.array(c["local_pos0_m"]), np.array(c["local_pos1_m"]))
        for c in data["closures"]
    ]
    springs = [(s["name"], s["body0"], s["body1"]) for s in data["gas_springs"]]
    return closures, springs


def urdf_joint_names(path: Path):
    import xml.etree.ElementTree as ET

    return [j.get("name") for j in ET.parse(path).getroot().findall("joint")]


def robot_cfg(usd_path: Path, fix_base: bool, target_joint: str, passive: list[str]) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            copy_from_source=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=fix_base,
                enabled_self_collisions=False,
                solver_position_iteration_count=24,
                solver_velocity_iteration_count=12,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "drive": ImplicitActuatorCfg(
                joint_names_expr=[target_joint],
                stiffness=args_cli.stiffness if args_cli.drive_mode == "position" else 0.0,
                damping=args_cli.damping if args_cli.drive_mode == "position" else 0.0,
                effort_limit_sim=args_cli.max_effort,
                velocity_limit_sim=100.0,
            ),
            "passive": ImplicitActuatorCfg(
                joint_names_expr=passive,
                stiffness=0.0,
                damping=args_cli.passive_damping,
                effort_limit_sim=args_cli.max_effort,
                velocity_limit_sim=100.0,
            ),
        },
    )


def main() -> None:
    closures, springs = load_constraints(args_cli.constraints)
    all_joints = urdf_joint_names(URDF)
    passive = [j for j in all_joints if j != args_cli.target_joint]

    # Isaac Lab's SimulationContext registers a timeline STOP callback whose body
    # is a blocking ``while not is_playing(): render()`` loop, which makes an
    # interactive GUI unclosable once the timeline is stopped.  NOTE: reset()
    # sets this flag back to False, so it must be re-applied AFTER reset().
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)
    sim_cfg.gravity = (0.0, 0.0, 0.0)
    sim_cfg.physx.min_position_iteration_count = 12
    sim_cfg.physx.max_position_iteration_count = 48
    sim_cfg.physx.min_velocity_iteration_count = 4
    sim_cfg.physx.max_velocity_iteration_count = 16
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([0.9, -1.1, 0.5], [0.0, 0.0, 0.0])

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    robot = Articulation(robot_cfg(args_cli.usd.resolve(), not args_cli.no_fix_base, args_cli.target_joint, passive))
    sim.reset()
    robot.reset()
    # reset() re-enables the blocking stop callback; disable it again.
    sim._disable_app_control_on_stop_handle = True
    dt = args_cli.dt

    def body_id(name: str) -> int:
        ids, _ = robot.find_bodies(name)
        if len(ids) != 1:
            raise RuntimeError(f"expected one body {name}, got {ids}")
        return int(ids[0])

    target_ids, target_names = robot.find_joints(args_cli.target_joint)
    target_id = int(target_ids[0])

    closure_ids = [
        (name, body_id(a), body_id(b),
         torch.tensor(p0, dtype=torch.float32, device=sim.device),
         torch.tensor(p1, dtype=torch.float32, device=sim.device))
        for name, a, b, p0, p1 in closures
    ]
    spring_ids = [(name, body_id(a), body_id(b)) for name, a, b in springs]

    def point(body: int, local: torch.Tensor) -> torch.Tensor:
        return robot.data.body_pos_w[:, body] + quat_apply(robot.data.body_quat_w[:, body], local.expand(1, -1))

    markers = None
    if not args_cli.no_markers:
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        markers = {}
        for key, color in (("body0", (1.0, 0.15, 0.15)), ("body1", (0.15, 0.55, 1.0))):
            markers[key] = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/ClosureMarkers_{key}",
                    markers={"point": sim_utils.SphereCfg(
                        radius=0.008,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    )},
                )
            )

    window = None
    target_model = None
    labels: dict = {}
    if not args_cli.headless:
        import omni.ui as ui

        window = ui.Window("Wheel_leg_V2 Joint Viewer", width=440, height=340)
        with window.frame:
            with ui.VStack(spacing=6):
                ui.Label(f"重力: OFF    基座: {'固定' if not args_cli.no_fix_base else '自由'}", height=22)
                with ui.HStack(height=24):
                    ui.Label(f"{args_cli.target_joint} 目标角", width=150)
                    target_model = ui.FloatField(width=90).model
                    target_model.set_value(0.0)
                    ui.FloatSlider(
                        min=args_cli.slider_min,
                        max=args_cli.slider_max,
                        step=0.01,
                        model=target_model,
                    )

                def reset_all():
                    if target_model is not None:
                        target_model.set_value(0.0)

                ui.Button("全部归零", clicked_fn=reset_all, height=26)
                ui.Separator()
                for name in ("L_four_bar_P", "R_four_bar_P", "L_four_bar_E", "R_four_bar_E",
                             "L_gas_spring", "R_gas_spring", "angle"):
                    labels[name] = ui.Label("", height=20)

    print(f">>> viewer ready: target joint = {args_cli.target_joint} ({target_names})")
    print(f">>> gravity off, fix_base={not args_cli.no_fix_base}, markers={markers is not None}")
    print(f">>> stop-handler disabled: {sim._app_control_on_stop_handle is None or sim._disable_app_control_on_stop_handle}")

    import time as _time

    started = _time.time()
    step = 0
    worst = 0.0
    idle = 0
    angle_range = [float("inf"), float("-inf")]
    while simulation_app.is_running():
        if args_cli.max_seconds and (_time.time() - started) >= args_cli.max_seconds:
            print(f">>> max-seconds {args_cli.max_seconds} reached, exiting")
            break
        # When the timeline is paused/stopped, only refresh rendering so the UI
        # and window-close events keep working (never block on sim.step()).
        if not sim.is_playing():
            sim.render()
            idle += 1
            if args_cli.self_test_stop_after and idle % 240 == 0:
                print(f"[test] idle heartbeat {idle}: app still responsive after stop")
            continue

        target = 0.0
        if target_model is not None:
            try:
                target = float(target_model.get_value_as_float())
            except Exception:  # noqa: BLE001
                target = 0.0
        elif args_cli.self_test_steps:
            # headless self-test: sweep the target to exercise the closed chain
            import math as _math

            period = max(60, args_cli.self_test_steps // 2)
            target = args_cli.self_test_amplitude * _math.sin(2.0 * _math.pi * step / period)
        targets = robot.data.default_joint_pos.clone()
        targets[:, target_id] = target
        if args_cli.drive_mode == "effort":
            efforts = torch.zeros_like(targets)
            efforts[:, target_id] = args_cli.torque
            robot.set_joint_effort_target(efforts)
        else:
            robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        if args_cli.self_test_steps and step % 60 == 0:
            print(
                f"[step {step:4d}] target={target:+.4f} "
                f"{args_cli.target_joint}={float(robot.data.joint_pos[0, target_id]):+.4f}"
            )

        errors = {}
        points0 = []
        points1 = []
        for name, b0, b1, p0, p1 in closure_ids:
            pb0 = point(b0, p0)[0]
            pb1 = point(b1, p1)[0]
            points0.append(pb0)
            points1.append(pb1)
            errors[name] = float(torch.norm(pb0 - pb1))
            worst = max(worst, errors[name])

        if markers is not None:
            markers["body0"].visualize(translations=torch.stack(points0))
            markers["body1"].visualize(translations=torch.stack(points1))

        if labels:
            for name in errors:
                labels[name].text = f"{name} 闭合误差: {errors[name] * 1000:8.4f} mm"
            zero = torch.zeros(3, device=sim.device)
            for name, b0, b1 in spring_ids:
                length = float(torch.norm(point(b0, zero) - point(b1, zero)))
                labels[name].text = f"{name} 长度: {length * 1000:8.3f} mm"
            labels["angle"].text = f"{args_cli.target_joint} 当前角: {float(robot.data.joint_pos[0, target_id]):+.4f} rad"

        step += 1
        if args_cli.self_test_stop_after and step == args_cli.self_test_stop_after:
            import omni.timeline

            print(f"[test] calling timeline.stop() at step {step}", flush=True)
            omni.timeline.get_timeline_interface().stop()
            print(f"[test] stop() returned, is_playing={sim.is_playing()}", flush=True)
        angle_now = float(robot.data.joint_pos[0, target_id])
        angle_range[0] = min(angle_range[0], angle_now)
        angle_range[1] = max(angle_range[1], angle_now)
        if args_cli.self_test_steps and step >= args_cli.self_test_steps:
            break

    if args_cli.self_test_steps:
        print("\nself-test summary")
        for name, b0, b1, p0, p1 in closure_ids:
            err = float(torch.norm(point(b0, p0) - point(b1, p1)))
            print(f"  {name:14s} final closure error = {err * 1000:8.4f} mm")
        print(f"  worst closure error over run = {worst * 1000:8.4f} mm")
        print(f"  {args_cli.target_joint} angle range = [{angle_range[0]:+.4f}, {angle_range[1]:+.4f}] rad")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(">>> interrupted by user, closing")
    finally:
        try:
            if simulation_app.is_running():
                simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
