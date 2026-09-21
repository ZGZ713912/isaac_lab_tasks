# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V2 合同（contract）加载与校验。
#
# 设计参考 V40 训练仓 wheeled-biped-rl-train/src/wheeled_tasks/v40/contract.py，
# 但针对 V2 闭链机器人做了简化与适配：
#   - 控制面是 6 个真实电机树关节：髋 L_joint1/R_joint1、膝 LL_joint1/RR_joint1、
#     轮 L_joint3/R_joint3；
#   - 其余 12 个树关节为被动闭链，不进入合同观测/动作；
#   - 资产为已 authored 的 Wheel_leg_V2.usd（闭链已在 USD 内），不再运行时转 URDF。
#
# 本模块不依赖 Torch / Isaac，可单独做 CPU 静态检查。
# =============================================================================

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from agent_world import AssetPath

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONTRACT = PACKAGE_DIR / "contracts" / "own_wheel_leg_v2.json"
ROUND2_CONTRACT = PACKAGE_DIR / "contracts" / "own_wheel_leg_v2_round2.json"

CONTRACT_V1_ID = "own-wheel-leg-v2-jointspace-h5-v1"
CONTRACT_V2_ID = "own-wheel-leg-v2-jointspace-h5-v2"
CONTRACT_IDS = {CONTRACT_V1_ID, CONTRACT_V2_ID}

# 6 个电机树关节的固定顺序（left hip/knee/wheel, right hip/knee/wheel）
ORDER = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]
LEG_INDICES = [0, 1, 3, 4]
HIP_INDICES = [0, 3]
KNEE_INDICES = [1, 4]
WHEEL_INDICES = [2, 5]


def is_round2(contract: dict) -> bool:
    """v2 profile = 观测噪声 + 持续倾倒终止 + reset 根速度随机。"""
    identity = contract["contract_id"]
    if identity not in CONTRACT_IDS:
        raise ValueError("Unsupported Wheel_leg_V2 contract identity")
    return identity == CONTRACT_V2_ID


def _finite(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _numbers(values, size, label, positive=False):
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"{label} must contain {size} numbers")
    return [_finite(x, label, positive) for x in values]


def _interval_list(values, label):
    """接受 [lo,hi] 或 [[lo,hi],...]，返回 [[lo,hi],...] 并校验 lo<=hi。"""
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be an interval or a nonempty interval list")
    scalars = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    pairs = [values] if scalars else values
    out = []
    for pair in pairs:
        lo, hi = _numbers(pair, 2, label)
        if lo > hi:
            raise ValueError(f"{label} has a reversed interval")
        out.append([lo, hi])
    return out


def validate_contract(c: dict) -> dict:
    """拒绝维度/单位/内部不一致的合同；不修改输入。"""
    try:
        if type(c["schema_version"]) is not int or c["schema_version"] != 1:
            raise ValueError("Unsupported contract schema_version")
        if c["contract_id"] not in CONTRACT_IDS or c["robot_id"] != "wheel_leg_v2":
            raise ValueError("Unsupported Wheel_leg_V2 contract identity")
        if c["control_frame"] != "Xforward_Yleft_Zup":
            raise ValueError("Policy requires canonical body axes")

        timing, obs, joints, action = c["timing"], c["observations"], c["joints"], c["actions"]
        dt = _finite(timing["physics_dt"], "physics_dt", True)
        policy_dt = _finite(timing["policy_dt"], "policy_dt", True)
        decimation = timing["decimation"]
        if (isinstance(decimation, bool) or not isinstance(decimation, int) or decimation < 1
                or not math.isclose(dt * decimation, policy_dt, abs_tol=1e-12)):
            raise ValueError("Physics and policy clocks do not match decimation")

        dims = [obs["single_dim"], obs["history_length"], obs["actor_dim"], obs["critic_dim"], action["dimension"]]
        if dims != [25, 5, 125, 29, 6]:
            raise ValueError("Contract requires single25/history5/actor125/critic29/action6")
        if obs["history_order"] != "oldest_to_newest_current_last" or obs["initial_history"] != "repeat_first_valid_observation":
            raise ValueError("Unsupported history semantics")
        if obs["empirical_normalization"] is not False or obs["wheel_position_observed"] is not False:
            raise ValueError("Unsupported observation/normalization contract")
        for key in ("angular_velocity", "joint_position", "joint_velocity"):
            _finite(obs["scales"][key], key, True)
        _numbers(obs["scales"]["command"], 3, "command scales", True)
        _finite(obs["clip"], "observation clip", True)

        if joints["action_order"] != ORDER:
            raise ValueError("Joint action order mismatch")
        if joints["leg_indices"] != LEG_INDICES or joints["hip_indices"] != HIP_INDICES:
            raise ValueError("Joint leg/hip index mismatch")
        if joints["knee_indices"] != KNEE_INDICES or joints["wheel_indices"] != WHEEL_INDICES:
            raise ValueError("Joint knee/wheel index mismatch")
        if set(joints["continuous"]) != {"L_joint1", "R_joint1", "L_joint3", "R_joint3"}:
            raise ValueError("Hips and wheels must stay continuous")
        _numbers(joints["nominal_positions"], 6, "nominal positions")
        _finite(joints["hip_soft_deviation"], "hip work deviation", True)
        # knee_hard_limits 允许为空：V2 URDF 全部 continuous，机械限位待标定。
        hard = joints.get("knee_hard_limits", {})
        if not isinstance(hard, dict):
            raise ValueError("knee_hard_limits must be a mapping")
        for name, bounds in hard.items():
            if name not in ORDER:
                raise ValueError(f"unknown knee joint: {name}")
            _numbers(bounds, 2, f"{name} hard limits")

        _numbers(action["leg_position_scales"], 4, "leg action scales", True)
        _finite(action["clip"], "action clip", True)
        _finite(action["wheel_velocity_scale"], "wheel action scale", True)

        leg, wheel = c["actuators"]["leg"], c["actuators"]["wheel"]
        for module in (leg, wheel):
            for name in ("kd", "effort_limit"):
                _finite(module[name], name, True)
            if _finite(module["armature"], "armature") < 0:
                raise ValueError("Armature cannot be negative")
        _finite(leg["kp"], "leg kp", True)
        eta = _finite(wheel["gearbox_efficiency"], "gearbox efficiency", True)
        if eta > 1 or wheel["curve_side"] != "motor":
            raise ValueError("Wheel curve must be motor-side and efficiency <=1")
        speeds, torques = wheel["motor_speed_rad_s"], wheel["motor_torque_nm"]
        if len(speeds) < 2 or len(speeds) != len(torques):
            raise ValueError("Motor curve needs matching samples")
        speeds = _numbers(speeds, len(speeds), "motor speeds")
        torques = _numbers(torques, len(torques), "motor torques")
        if speeds[0] != 0 or any(b <= a for a, b in zip(speeds, speeds[1:])) or any(t < 0 for t in torques):
            raise ValueError("Motor curve needs increasing speeds and nonnegative torques")

        _finite(c["asset"]["nominal_base_height"], "nominal height", True)
        commands = c["commands"]
        _finite(commands["resample_seconds"], "resample interval", True)
        _finite(commands.get("special_mode_min_episode_seconds", 0.0), "special mode min episode")
        for stage in commands["stages"].values():
            for name in ("vx", "wz", "height"):
                _interval_list(stage[name], name)
            for low, high in _interval_list(stage["height"], "height"):
                if not 0.10 <= low <= high <= 0.50:
                    raise ValueError("height command outside conservative V2 domain")
            if is_round2(c):
                probability = _finite(stage.get("standing_probability", 0.0), "standing probability")
                if not 0.0 <= probability <= 1.0:
                    raise ValueError("Standing probability must be within [0,1]")
            modes = stage.get("special_modes", {})
            if not isinstance(modes, dict):
                raise ValueError("special_modes must be a mapping")
            for mode_name, mode in modes.items():
                if not isinstance(mode, dict):
                    raise ValueError(f"special mode {mode_name} must be a mapping")
                rel = _finite(mode.get("rel_envs", 0.0), f"{mode_name} rel_envs")
                if not 0.0 <= rel <= 1.0:
                    raise ValueError(f"special mode {mode_name} rel_envs must be within [0,1]")
                start = mode.get("iteration_start", 0)
                end = mode.get("iteration_end", -1)
                if isinstance(start, bool) or not isinstance(start, int) or start < 0:
                    raise ValueError(f"special mode {mode_name} iteration_start must be a nonnegative int")
                if isinstance(end, bool) or not isinstance(end, int) or end < -1:
                    raise ValueError(f"special mode {mode_name} iteration_end must be int >= -1")
                for name in ("vx", "wz", "height"):
                    if name in mode:
                        _interval_list(mode[name], f"{mode_name}.{name}")
                if "resample_seconds" in mode:
                    _finite(mode["resample_seconds"], f"{mode_name}.resample_seconds", True)

        for key in ("sigma_velocity", "sigma_yaw", "sigma_yaw_square", "sigma_height"):
            _finite(c["rewards"][key], key, True)
        for value in c["rewards"]["weights"].values():
            _finite(value, "reward weight")
        for key in ("contact_force_threshold", "max_tilt_deg", "min_base_height", "episode_seconds"):
            _finite(c["termination"][key], key, True)

        policy = c["policy"]
        if policy["class_name"] != "ActorCritic" or policy["empirical_normalization"] is not False:
            raise ValueError("Requires ordinary non-normalizing ActorCritic")
        for key in ("actor_hidden_dims", "critic_hidden_dims"):
            dims = policy[key]
            if not isinstance(dims, list) or not 1 <= len(dims) <= 6 or any(type(x) is not int or not 1 <= x <= 2048 for x in dims):
                raise ValueError("Invalid policy hidden dimensions")
        if policy["activation"] not in {"elu", "relu", "tanh"}:
            raise ValueError("Unsupported policy activation")

        if is_round2(c):
            if timing != {"physics_dt": 0.005, "decimation": 2, "policy_dt": 0.01}:
                raise ValueError("V2 profile requires the declared 100Hz policy clock")
            noise = obs.get("noise", {})
            for key in ("angular_velocity", "gravity", "joint_position", "joint_velocity"):
                _finite(noise[key], f"noise.{key}", True)
            _finite(c["termination"]["failure_gravity_z"], "failure_gravity_z")
            _finite(c["termination"]["failure_seconds"], "failure_seconds", True)
            _numbers(c["reset"]["root_velocity_range"], 2, "root velocity range")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Incomplete Wheel_leg_V2 contract: {error}") from error
    return c


def load_contract(path=None) -> dict:
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    data = json.loads(Path(path or DEFAULT_CONTRACT).read_text(encoding="utf-8"), parse_constant=reject)
    return validate_contract(data)


def contract_digest(c: dict) -> str:
    validate_contract(c)
    return hashlib.sha256(
        json.dumps(c, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def asset_paths(c: dict) -> dict:
    """把合同里的相对资产路径解析成绝对路径（基于 agent_world.AssetPath）。"""
    directory = Path(AssetPath) / c["asset"]["directory"]
    return {
        "directory": directory,
        "usd": directory / c["asset"]["usd"],
        "urdf": directory / c["asset"]["urdf"],
        "constraints": directory / c["asset"]["constraints"],
    }


def load_constraints(c: dict) -> dict:
    """读取 constraints.json（闭链 + 气弹簧），供 env 做闭合误差监控。"""
    path = asset_paths(c)["constraints"]
    return json.loads(path.read_text(encoding="utf-8"))
