"""Bounded or interactive Wheel-leg V4 policy play on a static RMUC field."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "scripts/rsl_rl", *(ROOT / "source" / p for p in ("agent_world", "agent_tasks", "agent_rl"))):
    sys.path.insert(0, str(directory))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Robotics-Wheel-Leg-V4-Flat-Play-v0",
                    choices=("Robotics-Wheel-Leg-V4-Flat-v0", "Robotics-Wheel-Leg-V4-Flat-Play-v0"))
policy_source = parser.add_mutually_exclusive_group(required=True)
policy_source.add_argument("--checkpoint")
policy_source.add_argument("--onnx", help="ONNX actor; inference does not load a .pt checkpoint")
parser.add_argument("--policy-config", default=None, help="Matching training run or params directory for ONNX")
parser.add_argument("--field-root", default=str(ROOT / "assets/terrains/RMUC2026_V2_gray_fence_revision_03"))
parser.add_argument("--spawn-xy", type=float, nargs=2, default=None)
parser.add_argument("--spawn-yaw", type=float, default=0.)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--render_hz", type=float, default=50.)
parser.add_argument("--camera-mode", choices=("follow", "fixed"), default="follow")
parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "WZ", "HEIGHT"), default=None)
parser.add_argument("--profile_output", default=None)
parser.add_argument("--capture-output", default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
if args.num_envs != 1 or args.max_steps < 0:
    parser.error("field play requires --num_envs 1 and nonnegative --max_steps")
if not all(math.isfinite(v) for v in [args.spawn_yaw, args.render_hz, *(args.spawn_xy or []), *(args.command or [])]):
    parser.error("commands, spawn and render_hz must be finite")
if args.render_hz <= 0:
    parser.error("render_hz must be positive")
model = Path(args.onnx or args.checkpoint).expanduser().resolve()
if not model.is_file():
    parser.error("policy file does not exist")
if args.policy_config and not args.onnx:
    parser.error("--policy-config requires --onnx")
if not (Path(args.field_root).expanduser() / "asset_package.json").is_file():
    parser.error("field-root must contain asset_package.json; see docs/rmuc-field-play.md")

from play_lifecycle import PlayLifecycle

shutdown = PlayLifecycle(timeout=30.)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
shutdown.bind_app(simulation_app)

import torch
import yaml
import agent_world  # noqa: F401
import agent_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from agent_tasks.direct.wheel_leg_v4.field_env import WheelLegFieldEnv
from deformable_play_performance import configure_render_rate
from deformable_third_person_state import ThirdPersonState, quaternion_yaw
from keyboard_controller import Se3KeyboardMobile, Se3KeyboardMobileCfg
from onnx_policy import OnnxPolicy, load_policy_config, resolve_policy_config


def main():
    torch.set_num_threads(1)
    print("[play] Checking saved Wheel-leg V4 configuration", flush=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    params = resolve_policy_config(model, args.policy_config) if args.onnx else model.parent / "params"
    if args.onnx:
        saved, agent = load_policy_config(params)
    else:
        saved = yaml.load((params / "env.yaml").read_text(), Loader=yaml.BaseLoader)
        agent = yaml.safe_load((params / "agent.yaml").read_text())
    for name in ("observation_space", "state_space", "action_space", "num_obs_hist", "num_single_obs", "decimation",
                 "v4_leg_position_scale", "v4_wheel_velocity_scale", "v4_leg_kp", "v4_leg_kd", "v4_wheel_kd"):
        if name not in saved or float(saved[name]) != float(getattr(cfg, name)):
            raise ValueError(f"Checkpoint/task contract mismatch: {name}")
    if float(saved["sim"]["dt"]) != cfg.sim.dt:
        raise ValueError("Checkpoint/task physics timestep mismatch")
    if not args.onnx and (agent.get("class_name") != "OnPolicyRunner" or agent["policy"]["class_name"] != "ActorCritic"):
        raise ValueError("This field entry supports the Wheel-leg V4 ActorCritic checkpoint")
    agent["device"] = args.device
    if args.onnx:
        policy = OnnxPolicy(model, observation_dim=cfg.observation_space, action_dim=cfg.action_space,
                            history_length=cfg.num_obs_hist, frame_dim=cfg.num_single_obs, saved_env=saved)
        print("[policy] " + json.dumps(policy.audit()), flush=True)
    cfg.play = True
    cfg.play_keep_done_reset = True
    cfg.episode_length_s = 1.e6
    cfg.seed = args.seed
    cfg.field_root, cfg.field_spawn_xy, cfg.field_spawn_yaw = args.field_root, args.spawn_xy, args.spawn_yaw
    cfg.events = None
    cfg.self_obs_noise_cfg = None
    cfg.use_joint_vel_random_start = False
    cfg.commands.debug_vis = False
    cfg.commands.heading_command = False
    cfg.commands.rel_standing_envs = 0.
    cfg.commands.rel_heading_envs = 0.
    cfg.commands.resampling_time_range = (1.e9, 1.e9)
    cfg.commands.ranges.lin_vel_x = (0., 0.)
    cfg.commands.ranges.lin_vel_y = (0., 0.)
    cfg.commands.ranges.ang_vel_z = (0., 0.)
    cfg.play_ang_vel_z_debug_vis = False
    cfg.play_wheel_material_debug = False
    configure_render_rate(cfg, args.render_hz)
    print("[play] Building field environment", flush=True)
    raw = WheelLegFieldEnv(cfg=cfg)
    shutdown.bind_env(raw)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.get("clip_actions"))
    if not args.onnx:
        runner = OnPolicyRunner(env, agent, log_dir=None, device=args.device)
        print(f"[policy] Loading {model}", flush=True)
        runner.load(str(model))
        policy = runner.get_inference_policy(device=args.device)
    # Manual commands must survive the inherited command generator's compute().
    raw.command_generator.compute = lambda dt: None
    obs, _ = env.reset()
    camera = ThirdPersonState()
    controller = None
    reset_requested = []
    if not args.headless and args.command is None:
        controller = Se3KeyboardMobile(Se3KeyboardMobileCfg(
            pos_sensitivity=.8, rot_sensitivity=1.5, height_sensitivity=.05, sim_device=args.device))
        controller.set_env(env)
        controller.set_default_height(float(cfg.default_height_cmd))
        controller.set_height_range((.15, .35))
        controller.add_callback("R", lambda: reset_requested.append(True))
        controller.add_callback("ESCAPE", controller.reset)
        controller.reset()
        import carb.settings
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        viewport_id = getattr(viewport, "viewport_id", None) or getattr(viewport, "id", None)
        if viewport_id is not None:
            carb.settings.get_settings().set("/exts/omni.kit.manipulator.camera/viewportMode", [viewport_id, "orbit"])
        print("[controls] W/S 前后；A/D 转向；Z/X 调高度；L 停止；R 重置。", flush=True)
    rows, steps, resets = [], 0, 0
    start = time.perf_counter()
    while simulation_app.is_running() and not shutdown.requested:
        if reset_requested:
            obs, _ = env.reset()
            controller.reset()
            reset_requested.clear()
        if controller is not None:
            controller.advance_step(raw.step_dt)
        else:
            vx, wz, height = args.command or (0., 0., float(cfg.default_height_cmd))
            raw.command_generator.command[:] = raw.command_generator.command.new_tensor((vx, 0., wz))
            raw.height_cmd[:] = height
        if not args.headless and (args.camera_mode == "follow" or steps == 0):
            eye, pivot = camera.camera_pose(raw.robot.data.root_pos_w[0].tolist(),
                quaternion_yaw(raw.robot.data.root_quat_w[0].tolist()), raw.field_surface.segment)
            raw.viewport_camera_controller.update_view_location(eye=eye, lookat=pivot)
        with torch.inference_mode():
            actions = policy(obs)
            if not torch.isfinite(actions).all():
                raise RuntimeError("Nonfinite policy actions")
            obs, _, dones, _ = env.step(actions)
        steps += 1
        if bool(dones.any()):
            resets += 1
            if controller is not None:
                controller.reset()
        if steps == 1 or steps % max(1, round(.25 / raw.step_dt)) == 0:
            row = dict(step=steps, position=raw.robot.data.root_pos_w[0].tolist(),
                       velocity_body=raw.robot.data.root_lin_vel_b[0].tolist(),
                       command=raw._v4_command()[0].tolist(), height_m=float(raw._base_height()[0]),
                       wheel_contact=raw._wheel_contact_flags()[0].tolist(),
                       projected_gravity=raw.robot.data.projected_gravity_b[0].tolist())
            if args.max_steps or args.profile_output:
                rows.append(row)
            print("[motion] " + json.dumps(row), flush=True)
        if args.max_steps and steps >= args.max_steps:
            break
    wall = time.perf_counter()-start
    result = dict(policy=policy.audit() if args.onnx else dict(backend="pytorch", checkpoint=str(model),
                      checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest()),
                  task=args.task, policy_steps=steps, simulation_seconds=steps*raw.step_dt,
                  wall_seconds=wall, real_time_factor=steps*raw.step_dt/max(wall, 1.e-9),
                  physics_dt=cfg.sim.dt, policy_dt=raw.step_dt, render_interval=cfg.sim.render_interval,
                  automatic_resets=resets, field=raw.field_audit(), motion=rows)
    if args.onnx:
        result["policy"]["config_directory"] = str(params)
    else:
        result.update(checkpoint=str(model), checkpoint_sha256=result["policy"]["checkpoint_sha256"])
    if args.profile_output:
        output = Path(args.profile_output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    if args.capture_output and not args.headless:
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        output = Path(args.capture_output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        capture = capture_viewport_to_file(get_active_viewport(), str(output))
        task = simulation_app.run_coroutine(capture.wait_for_result(), run_until_complete=False)
        deadline = time.perf_counter()+10.
        while not task.done() and time.perf_counter() < deadline:
            simulation_app.update()
        if not task.done():
            task.cancel()
            raise RuntimeError("Viewport capture timed out")
        if not task.result():
            raise RuntimeError("Viewport capture returned no color buffer")
        import omni.kit.renderer_capture
        omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        if not output.is_file():
            raise RuntimeError("Viewport capture did not write an image")
    if controller is not None:
        controller.__del__()
    print(f"[play] {steps} steps; {result['real_time_factor']:.3f}x; {resets} automatic resets", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[play] Interrupted; closing", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        shutdown.close()
