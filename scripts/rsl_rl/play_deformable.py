# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
# =============================================================================

"""Deformable 主动悬挂键盘 play：加载 checkpoint + GUI + 键盘遥控查看训练效果。

键位：
    W/S  前进/后退 (vx)      A/D  左移/右移 (vy)
    X/Z  自旋 +/-(ωz, 增量，松开保持)   Q  查询/切换基准角   L  全部归零

用法（仓库根目录，需 GUI）：
    python scripts/rsl_rl/play_deformable.py \
        --task=Robotics-Deformable-Suspension-Guarded-History-MLP-Real2Sim-v4 \
        --checkpoint=logs/rsl_rl/deformable_guarded_mlp5_v4/<run>/model_999.pt \
        --grade-deg=0 --fixed_camera --debug_motion --profile \
        --device=cpu
"""

"""Launch Isaac Sim Simulator first."""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
for _pkg in ("agent_world", "agent_tasks", "agent_rl"):
    _pkg_root = os.path.join(_REPO_ROOT, "source", _pkg)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
_RSL_RL_SCRIPTS = os.path.join(_REPO_ROOT, "scripts", "rsl_rl")
if _RSL_RL_SCRIPTS not in sys.path:
    sys.path.insert(0, _RSL_RL_SCRIPTS)

import argparse
import json
import math
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard play for deformable active suspension.")
parser.add_argument(
    "--task",
    type=str,
    default="Robotics-Deformable-Suspension-Rough-Keyboard-Play-History-Transformer-v2",
    help="Task name (keyboard play variant).",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_XXXX.pt")
parser.add_argument("--num_envs", type=int, default=1, help="Must be 1 for keyboard play.")
parser.add_argument("--vx_max", type=float, default=None, help="Forward cap: third-person/field 0.8, ordinary play 3.0 m/s.")
parser.add_argument("--vy_max", type=float, default=None, help="Lateral cap: third-person/field 0.5, ordinary play 1.6 m/s.")
parser.add_argument("--wz_max", type=float, default=None, help="Yaw cap: third-person/field 1.5, ordinary play 2*pi rad/s.")
parser.add_argument("--wz_step", type=float, default=0.6, help="Yaw-rate increment per Z/X press (rad/s).")
parser.add_argument("--no_vis", action="store_true", default=False, help="Disable play visualization (contact/level/HUD).")
parser.add_argument("--fixed_camera", action="store_true", help="Keep the camera stationary between resets to see world displacement.")
parser.add_argument("--camera-mode", choices=("follow", "fixed", "third-person"), default=None)
parser.add_argument("--field-root", type=str, default=None, help="RMUC package directory containing asset_package.json.")
parser.add_argument("--spawn-xy", nargs=2, type=float, default=None, metavar=("X", "Y"))
parser.add_argument("--spawn-yaw", type=float, default=0., help="Initial field heading in source-world degrees.")
parser.add_argument("--teleop-replay", type=str, default=None, help="Bounded diagnostic replay of third-person input events.")
parser.add_argument("--capture-output", type=str, default=None, help="Save the final GUI viewport for validation.")
parser.add_argument("--verify-input-api", action="store_true", help="Probe native Kit input during bounded GUI replay.")
parser.add_argument("--debug_motion", action="store_true", help="Print commanded and measured motion every 0.5 simulation seconds.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many policy steps; 0 runs until closed.")
parser.add_argument("--render_hz", type=float, default=50.0, help="GUI refresh rate in simulation time; control timing is preserved.")
parser.add_argument("--cpu_threads", type=int, default=1, help="Torch CPU threads for this single-robot play process.")
parser.add_argument("--profile", action="store_true", help="Measure policy, physics/control and rendering wall time.")
parser.add_argument("--profile_output", type=str, default=None, help="Save timing and motion measurements as JSON.")
parser.add_argument("--benchmark_motion", action="store_true", help="Run a bounded simulated standing/forward/lateral/spin command sequence.")
parser.add_argument("--legacy_wrench_api", action="store_true", help="Use the original external-wrench wrapper for performance comparisons.")
parser.add_argument("--training_metrics", action="store_true", help="Compute unused training rewards/statistics during play for comparisons.")
parser.add_argument("--seed", type=int, default=None, help="Repeatable simulation seed for diagnostics.")
parser.add_argument(
    "--grade-deg",
    type=float,
    default=None,
    help="Override the periodic terrain with a constant 0..20 degree ramp for visual validation.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli, hydra_args = parser.parse_known_args()
if args_cli.fixed_camera and args_cli.camera_mode not in (None, "fixed"):
    parser.error("--fixed_camera conflicts with --camera-mode")
args_cli.camera_mode = args_cli.camera_mode or ("fixed" if args_cli.fixed_camera else "follow")
args_cli.fixed_camera = args_cli.camera_mode == "fixed"
for name, ordinary, driving in (("vx_max", 3., .8), ("vy_max", 1.6, .5), ("wz_max", 2*math.pi, 1.5)):
    if getattr(args_cli, name) is None:
        setattr(args_cli, name, driving if args_cli.field_root or args_cli.camera_mode == "third-person" else ordinary)
    if not math.isfinite(getattr(args_cli, name)) or getattr(args_cli, name) <= 0:
        parser.error(f"{name} must be positive and finite")
if args_cli.num_envs != 1 or args_cli.max_steps < 0:
    parser.error("keyboard play requires --num_envs=1 and nonnegative --max_steps")
if not math.isfinite(args_cli.render_hz) or args_cli.render_hz <= 0 or args_cli.cpu_threads < 1:
    parser.error("render_hz must be positive and finite; cpu_threads must be positive")
if args_cli.benchmark_motion and not args_cli.max_steps:
    parser.error("--benchmark_motion requires a bounded --max_steps")
if not os.path.isfile(args_cli.checkpoint):
    parser.error("checkpoint file does not exist")
if args_cli.field_root and args_cli.grade_deg is not None:
    parser.error("--field-root cannot be combined with --grade-deg")
if not args_cli.field_root and (args_cli.spawn_xy is not None or args_cli.spawn_yaw != 0.):
    parser.error("--spawn-xy/--spawn-yaw require --field-root")
if not all(math.isfinite(value) for value in [args_cli.spawn_yaw, *(args_cli.spawn_xy or [])]):
    parser.error("spawn coordinates and heading must be finite")
replay_events = []
if args_cli.teleop_replay:
    if args_cli.camera_mode != "third-person" or not args_cli.max_steps or args_cli.benchmark_motion:
        parser.error("--teleop-replay requires bounded third-person play without --benchmark_motion")
    replay_events = json.loads(Path(args_cli.teleop_replay).read_text())
    if not isinstance(replay_events, list) or any(event.get("type") not in ("key", "right", "move", "scroll")
            or type(event.get("step")) is not int or not 0 <= event["step"] < args_cli.max_steps for event in replay_events):
        parser.error("replay must contain valid events with policy steps inside --max_steps")
    replay_events.sort(key=lambda event: event["step"])
if args_cli.field_root:
    package = Path(args_cli.field_root).expanduser().resolve() / "asset_package.json"
    if not package.is_file():
        parser.error(f"field package not found: {package}")
if args_cli.verify_input_api and (not args_cli.teleop_replay or args_cli.headless):
    parser.error("--verify-input-api requires bounded GUI --teleop-replay")

sys.argv = [sys.argv[0]] + hydra_args
from play_lifecycle import PlayLifecycle

shutdown = PlayLifecycle(timeout=30. if args_cli.field_root else 8.)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
shutdown.bind_app(simulation_app)

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402

import agent_world  # noqa: F401,E402
import agent_tasks  # noqa: F401,E402
import agent_rl.rsl_rl.modules  # noqa: F401,E402  注册 ActorCriticTransformer 到 OnPolicyRunner
import agent_rl.rsl_rl.algorithms  # noqa: F401,E402  注册 DiagnosticPPO 到 OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from keyboard_controller_deformable import DeformableKeyboard, DeformableKeyboardCfg  # noqa: E402
from deformable_play_vis import DeformablePlayVis  # noqa: E402
from agent_tasks.direct.deformable_suspension import cfg_utils as du  # noqa: E402
from scripts.utils.deformable_checkpoint import (  # noqa: E402
    configure_recovery_precision,
    validate_deformable_checkpoint,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(args_cli.cpu_threads)
third_person_controller = None


def _disable_viewport_wasd() -> None:
    """把视口相机从 fly 切到 orbit，避免 WASD 抢机器人遥控键。

    Kit 的 viewportMode 格式是 [viewport_id, "fly"|"orbit"]（见
    omni.kit.viewport.window tests）。纯字符串 set 不会真正切模式。
    """
    try:
        import carb

        settings = carb.settings.get_settings()
        viewport_id = None
        try:
            from omni.kit.viewport.utility import get_active_viewport

            vp = get_active_viewport()
            viewport_id = getattr(vp, "viewport_id", None) or getattr(vp, "id", None)
        except Exception:  # noqa: BLE001
            viewport_id = None

        if viewport_id is not None:
            settings.set(
                "/exts/omni.kit.manipulator.camera/viewportMode",
                [viewport_id, "orbit"],
            )
            print(f"[INFO] 视口相机已切到 orbit（viewport_id={viewport_id}，禁用 WASD fly）")
        else:
            # 兜底：部分版本接受全局字符串
            settings.set("/exts/omni.kit.manipulator.camera/viewportMode", "orbit")
            print("[INFO] 视口相机已设为 orbit（全局，未取到 viewport_id）")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 无法切换视口相机模式: {exc}")


def camera_follow(env) -> None:
    """相机跟随机器人（移植 play.py，减少手动转视角需求）。"""
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
    camera_offset = torch.tensor([-3.0, 0.0, 0.5], dtype=torch.float32, device=env.device)
    camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0), pos=robot_pos.unsqueeze(0), quat=robot_quat.unsqueeze(0)
    ).squeeze(0)
    window_size = 50
    camera_follow.smooth_camera_positions.append(camera_pos)
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)
    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(), lookat=robot_pos.cpu().numpy()
    )


def _configure_grade(env_cfg, grade_deg: float | None) -> None:
    """Make a long, constant ramp before Isaac Lab builds the terrain."""
    if grade_deg is None:
        return
    if not math.isfinite(grade_deg) or not 0.0 <= grade_deg <= 20.0:
        raise ValueError("--grade-deg must be finite and in [0, 20]")
    generator = getattr(env_cfg.terrain, "terrain_generator", None)
    sub_terrains = getattr(generator, "sub_terrains", None) if generator is not None else None
    sub = sub_terrains.get("periodic_slope") if sub_terrains else None
    if sub is None:
        raise ValueError("--grade-deg requires a periodic_slope terrain task")
    # A 20 m segment keeps the vehicle on one uphill ramp instead of hiding the
    # behavior at a short crest/flat transition.
    sub.angle_range = (grade_deg, grade_deg)
    sub.angle_choices = None
    sub.segment_length = 20.0
    env_cfg.boundary_reset_enabled = False
    env_cfg.spawn_dir_jitter = False
    env_cfg.spawn_phase_stratify = False
    if grade_deg > 5.0:
        # A horizontal, four-contact reset may be impossible on a steep ramp.
        env_cfg.best_effort_leveling = True
        env_cfg.best_effort_tilt_weight = 6.0
        env_cfg.termination_roll_deg = 45.0
        env_cfg.termination_pitch_deg = 45.0


def _place_on_grade_ramp(env) -> None:
    """Put the single play environment well inside the uphill part of the ramp."""
    unwrapped = env.unwrapped
    if not getattr(unwrapped, "_periodic", False):
        return
    # The periodic profile starts at x + _profile_x_offset.  Phase 90 m with a
    # 20 m segment is 10 m into the 0..20 m uphill section.
    origins = unwrapped.scene.env_origins
    origins[:, 0] = 90.0 - float(unwrapped._profile_x_offset)
    origins[:, 1] = 0.0
    origins[:, 2] = 0.0


def _print_suspension_reference(env) -> None:
    """Print the configured q reference and its physical/height interpretation."""
    unwrapped = env.unwrapped
    q_choices = tuple(float(q) for q in getattr(unwrapped.cfg, "q_cmd_choices", ()))
    q_cmd = float(unwrapped.q_cmd[0].item()) if hasattr(unwrapped, "q_cmd") else None
    if not q_choices and q_cmd is None:
        return
    q_ref = q_cmd if q_cmd is not None else q_choices[0]
    zero = float(getattr(unwrapped.cfg,"leg_physical_angle_zero",du.PHYSICAL_MAX_ANGLE))
    q_limit = float(unwrapped.cfg.leg_target_upper_limit)
    print(
        "[INFO] 悬挂基准: "
        f"q_cmd={q_ref:.4f} rad, physical={math.degrees(zero-q_ref):.2f} deg, "
        f"base_h={float(du.q_to_base_height(q_ref)):.4f} m; "
        f"q_target_max={q_limit:.4f} rad, base_h={float(du.q_to_base_height(q_limit)):.4f} m",
        flush=True,
    )
    print(
        "[INFO] q 越大车体越低；q_cmd 是零动作基准，q_target_max 是目标范围上限，"
        "不是同一个量。",
        flush=True,
    )


def main() -> None:
    global third_person_controller
    _disable_viewport_wasd()
    # ---- env + agent config -------------------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.play = True
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    _configure_grade(env_cfg, args_cli.grade_deg)
    resume_path = os.path.abspath(args_cli.checkpoint)
    validate_deformable_checkpoint(env_cfg, resume_path)
    # Render less often without changing 1ms physics, 200Hz policy or history.
    from deformable_play_performance import configure_render_rate
    configure_render_rate(env_cfg, args_cli.render_hz)
    # Training tasks can also be played with their original control contract.
    # Keep keyboard commands from being resampled or scaled by the curriculum.
    env_cfg.external_cmd_override = True
    env_cfg.commands_world_frame = False
    env_cfg.spawn_dir_stratify = False
    env_cfg.spawn_dir_jitter = False
    env_cfg.boundary_reset_enabled = False
    env_cfg.episode_length_s = 1.0e6
    if getattr(env_cfg, "runtime_guard_enabled", False):
        env_cfg.guard_training_ground_truth = False
    if args_cli.field_root:
        env_cfg.field_root = args_cli.field_root
        env_cfg.field_spawn_xy = args_cli.spawn_xy
        env_cfg.field_spawn_yaw = args_cli.spawn_yaw
    # Match the initial visible asset to the configured low-body reference.
    # Reset still solves per-corner contact on terrain before the first action.
    q_ref = float(env_cfg.q_cmd_choices[0])
    env_cfg.robot_cfg.init_state.joint_pos = {
        **{name: q_ref for name in du.ORDERED_LEG_JOINT_NAMES + du.ORDERED_WS_JOINT_NAMES},
        **{name: -q_ref for name in du.ORDERED_UPPER_LEG_JOINT_NAMES},
        **{name: 0.0 for name in du.ORDERED_WHEEL_JOINT_NAMES},
    }
    initial_pos = env_cfg.robot_cfg.init_state.pos
    env_cfg.robot_cfg.init_state.pos = (
        initial_pos[0], initial_pos[1],
        float(du.q_to_base_height(q_ref)) + max(0.0, env_cfg.reset_height_buffer),
    )

    from scripts.utils.deformable_checkpoint import load_deformable_agent_config
    agent_cfg = load_deformable_agent_config(resume_path, args_cli.device)
    configure_recovery_precision(agent_cfg["policy"], agent_cfg["algorithm"])
    if hasattr(env_cfg, "policy_history_length"):
        env_cfg.policy_history_length = agent_cfg["policy"].get("history_length", 1)
        env_cfg.observation_space = 32 * env_cfg.policy_history_length

    # ---- environment --------------------------------------------------------
    if args_cli.field_root:
        from agent_tasks.direct.deformable_suspension.field_env import DeformableFieldEnv
        env = DeformableFieldEnv(cfg=env_cfg, render_mode=None)
    else:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    shutdown.bind_env(env)
    if args_cli.grade_deg is not None:
        _place_on_grade_ramp(env)
        print(f"[INFO] 固定坡度 play: {args_cli.grade_deg:.1f}°，已放置在 20 m 上坡段内部", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
    if not args_cli.legacy_wrench_api:
        # The legacy wrapper only converts inputs and forwards to this same
        # composer, while logging a deprecation warning every physics step.
        env.unwrapped.robot.set_external_force_and_torque = (
            env.unwrapped.robot.permanent_wrench_composer.set_forces_and_torques
        )
    if not args_cli.training_metrics:
        # Reward/direction statistics are not read by the frozen inference
        # policy. Action-history updates remain in _pre_physics_step.
        play_rewards = torch.zeros(env.unwrapped.num_envs, device=env.device)
        env.unwrapped._get_rewards = lambda: play_rewards.zero_()
    obs, _ = env.reset()
    _print_suspension_reference(env)
    q = env.unwrapped.robot.data.joint_pos[0, env.unwrapped._legs_idx]
    print(f"[INFO] 策略运行前初始腿角 q={q.tolist()}", flush=True)
    # env 创建后 viewport 已就绪，再确保一次 orbit（防 App 启动时未生效）
    _disable_viewport_wasd()

    keyboard = None if args_cli.headless or args_cli.camera_mode == "third-person" else DeformableKeyboard(
        DeformableKeyboardCfg(
            vx_max=args_cli.vx_max,
            vy_max=args_cli.vy_max,
            wz_max=args_cli.wz_max,
            wz_step=args_cli.wz_step,
            q_choices=tuple(env.unwrapped.cfg.q_cmd_choices),
        )
    )
    if keyboard is not None:
        keyboard.set_env(env)
        keyboard.reset()
        print(f"[INFO] {keyboard}")
    if args_cli.camera_mode == "third-person":
        from deformable_third_person import ThirdPersonController
        third_person_controller = ThirdPersonController(env, vx_max=args_cli.vx_max, vy_max=args_cli.vy_max,
            wz_max=args_cli.wz_max, live_input=not args_cli.headless and not args_cli.teleop_replay)
        print(f"[INFO] {third_person_controller}", flush=True)

    # ---- visualization (A 接触 + B 水平 + C HUD) ----
    play_vis = None
    if not args_cli.no_vis and not getattr(args_cli, "headless", False):
        play_vis = DeformablePlayVis(env)
        print("[INFO] 可视化已开启：红球=接触, 绿杆=世界竖直/红杆=车身z轴, HUD=姿态/高度/接触力")

    # ---- runner + checkpoint ------------------------------------------------
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- reset + loop -------------------------------------------------------
    camera_follow.smooth_camera_positions = []
    if not getattr(args_cli, "headless", False):
        if third_person_controller is not None:
            third_person_controller.update_camera()
        else:
            camera_follow(env)
    input_probe = None
    if args_cli.verify_input_api:
        from deformable_input_probe import probe_kit_input
        for _ in range(3):
            simulation_app.update()
        input_probe = probe_kit_input(third_person_controller)
        print("[input-api-probe] " + json.dumps(input_probe), flush=True)
        # Kit layout updates can advance its timeline without the 1 kHz drive
        # loop. Start the measured rollout from a fresh pose and observation.
        with torch.inference_mode():
            obs, _ = env.reset()
        third_person_controller.reset()
        third_person_controller.update_camera()
    print("[INFO] play 已启动" if third_person_controller is not None else
          "[INFO] play 已启动：W/S 前后，A/D 横移，X/Z 自旋，Q 查询/切换基准，L 归零")
    print("[INFO] 请确保 Isaac Sim 窗口有焦点才能接收键盘输入")
    print(f"[INFO] 相机模式：{args_cli.camera_mode}")
    motion_tick = 0
    motion_interval = max(1, round(0.5 / env.unwrapped.step_dt))
    motion_origin = env.unwrapped.robot.data.root_pos_w[0].clone()
    from deformable_play_performance import PlayPerformance
    performance = PlayPerformance(env.unwrapped, enabled=args_cli.profile)
    performance.start()
    benchmark_commands = ((0., 0., 0.), (args_cli.vx_max, 0., 0.),
                          (0., args_cli.vy_max, 0.), (0., 0., args_cli.wz_max),
                          (0., 0., -args_cli.wz_max))
    benchmark_period = max(1, args_cli.max_steps // len(benchmark_commands))
    last_visual_step = -env.unwrapped.cfg.sim.render_interval
    last_report_wall = time.perf_counter()
    replay_index = 0
    if args_cli.field_root and args_cli.profile:
        probes = env.unwrapped.verify_physics_queries()
        print("[field-probes] " + json.dumps(probes), flush=True)
        if not all(probe["passed"] for probe in probes):
            raise RuntimeError("Field collision/query verification failed")

    while simulation_app.is_running() and not shutdown.requested:
        if keyboard is not None:
            keyboard.apply()
        if third_person_controller is not None:
            while replay_index < len(replay_events) and replay_events[replay_index]["step"] == motion_tick:
                third_person_controller.dispatch(replay_events[replay_index])
                replay_index += 1
            reset_obs = third_person_controller.apply()
            if reset_obs is not None:
                obs = reset_obs
                motion_origin = env.unwrapped.robot.data.root_pos_w[0].clone()
        if args_cli.benchmark_motion:
            index = min(len(benchmark_commands)-1, motion_tick // benchmark_period)
            env.unwrapped.cmd_buf[:] = torch.tensor(benchmark_commands[index], device=env.device)
        policy_start = time.perf_counter()
        with torch.inference_mode():
            actions = policy(obs)
            performance.record_policy(policy_start)
            obs, _, dones, _ = env.step(actions)
        # USD marker writes and camera GPU->CPU transfers follow the GUI cadence.
        if env.unwrapped._sim_step_counter - last_visual_step >= env.unwrapped.cfg.sim.render_interval:
            if play_vis is not None:
                play_vis.update()
            if not getattr(args_cli, "headless", False) and not args_cli.fixed_camera:
                if third_person_controller is not None:
                    third_person_controller.update_camera()
                else:
                    camera_follow(env)
            last_visual_step = env.unwrapped._sim_step_counter
        motion_tick += 1
        performance.record_step()
        if args_cli.profile and time.perf_counter() - last_report_wall >= 2.0:
            print(performance.progress(), flush=True)
            last_report_wall = time.perf_counter()
        if args_cli.debug_motion and motion_tick % motion_interval == 0:
            u = env.unwrapped
            data = u.robot.data
            cmd = u.cmd_buf[0].tolist()
            vel = data.root_link_lin_vel_b[0, :2].tolist()
            delta = (data.root_pos_w[0, :2] - motion_origin[:2]).tolist()
            q = data.joint_pos[0, u._legs_idx]
            target = u.leg_target[0]
            physical_zero = float(getattr(u.cfg, "leg_physical_angle_zero", du.PHYSICAL_MAX_ANGLE))
            physical_angles = torch.rad2deg(physical_zero - q).tolist()
            gravity = data.projected_gravity_b[0]
            tilt = math.degrees(math.atan2(float(gravity[:2].norm()), -float(gravity[2])))
            loads = u.wheel_normal_forces[0]
            contacts = int((loads > u.cfg.wheel_contact_force_threshold).sum())
            print(
                f"[motion] cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                f"vel_b=({vel[0]:+.3f},{vel[1]:+.3f}) "
                f"wz={data.root_ang_vel_b[0, 2].item():+.3f} "
                f"delta_xy_w=({delta[0]:+.3f},{delta[1]:+.3f}) m "
                f"q=[{q.min().item():.3f},{q.max().item():.3f}] "
                f"target=[{target.min().item():.3f},{target.max().item():.3f}] "
                f"physical_deg=[{','.join(f'{angle:.1f}' for angle in physical_angles)}] "
                f"body_top={u.body_top_height[0].item() * 1000:.1f}mm "
                f"q_cmd={u.q_cmd[0].item():.3f} tilt={tilt:.2f}deg "
                f"contact={contacts}/4 min_load={loads.min().item():.1f}N "
                f"slip={u._last_wheel_slip[0].abs().mean().item():.3f}m/s",
                flush=True,
            )
        if bool(dones.any()):
            if keyboard is not None:
                keyboard.reset()
            if third_person_controller is not None:
                third_person_controller.reset()
            motion_origin = env.unwrapped.robot.data.root_pos_w[0].clone()
            camera_follow.smooth_camera_positions = []
            if not getattr(args_cli, "headless", False):
                if third_person_controller is not None:
                    third_person_controller.update_camera()
                else:
                    camera_follow(env)
            print("[INFO] 环境重置，键盘命令已归零")
        if args_cli.max_steps and motion_tick >= args_cli.max_steps:
            break
    extras = {}
    if args_cli.field_root:
        extras["field"] = env.unwrapped.field_audit()
        extras["field"]["physics_probes"] = getattr(env.unwrapped, "field_physics_probes", [])
    if third_person_controller is not None:
        extras["third_person"] = third_person_controller.audit()
        extras["third_person"]["native_input_probe"] = input_probe
    performance.finish(args_cli.profile_output, extra=extras)
    if args_cli.capture_output and not args_cli.headless:
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        capture_path = Path(args_cli.capture_output).resolve()
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        capture = capture_viewport_to_file(get_active_viewport(), str(capture_path))
        task = simulation_app.run_coroutine(capture.wait_for_result(), run_until_complete=False)
        deadline = time.perf_counter() + 10.
        while not task.done() and time.perf_counter() < deadline:
            simulation_app.update()
        if not task.done():
            task.cancel()
            raise RuntimeError("Viewport capture did not complete within 10 seconds")
        if not task.result():
            raise RuntimeError("Viewport capture returned no color buffer")
        import omni.kit.renderer_capture
        omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        if not capture_path.is_file():
            raise RuntimeError("Viewport capture finished without writing its image")
        print(f"[capture] {capture_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(">>> interrupted by user, closing", flush=True)
    except Exception:
        # SimulationApp.close may terminate Python before the pending
        # exception is displayed. Keep the actual cause visible first.
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            if third_person_controller is not None:
                third_person_controller.close()
        finally:
            shutdown.close()
