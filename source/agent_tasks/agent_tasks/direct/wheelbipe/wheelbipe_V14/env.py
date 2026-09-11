# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 【文件总览·给初学者】本文件定义 V14 机器人的强化学习"环境类" WheelbipeV14Env。
# 在 Isaac Lab 的 direct（直接）工作流中，环境类就是一台"仿真机房"：
#   1) 启动时：按配置搭好场景（机器人、地形、传感器），并初始化各种内部变量；
#   2) 训练时：每个策略步被依次调用：
#        _pre_physics_step / _apply_action（把策略输出的动作发给机器人）
#        → 物理仿真推进 → _get_dones（判断摔倒/超时）→ _reset_idx（重置摔倒的机器人）
#        → _get_rewards（算奖励，公式在父类里，权重来自 env_cfg.py）
#        → _get_observations（拼出策略吃的观测向量）；
#   3) 本类继承 V13 环境（它再往上继承 Isaac Lab 的 DirectRLEnv）。
#      V14 只重写/新增了：云台（头部）控制、粗糙地形高度课程学习、速度轨迹录制；
#      其余通用逻辑（腿/轮电机控制、奖励公式、观测拼接）都在父类里。
# 配套的 env_cfg.py 是"参数表"，本类通过 self.cfg.xxx 随时读取参数。
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations  # 让类型注解允许"先引用后定义"的现代写法

import atexit   # 程序退出钩子：注册"进程结束前自动执行"的函数，这里用于退出前关闭 CSV 文件
import csv      # Python 自带 CSV 读写库，把速度/reward 轨迹写成表格文件
import os       # 标准库：取进程号 pid（拼进输出文件名，避免多进程同名覆盖）
from pathlib import Path  # 面向对象的路径工具，比字符串拼路径更安全方便
from datetime import datetime  # 取当前时间，生成带时间戳的文件名

import torch    # PyTorch：观测/动作/奖励全都是 GPU 上的张量，几十个环境并行计算
import isaaclab.sim as sim_utils  # Isaac Lab 仿真工具：这里用到球体标记物的配置类
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # 可视化标记：在机器人头顶画球提示"小陀螺平移模式"
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi  # 数学工具：四元数→欧拉角；把角度规范到 [-π, π]

from agent_tasks.direct.wheelbipe.wheelbipe_V13.env import WheelbipeV13Env  # 父类：V13 环境，腿/轮控制与奖励公式等通用逻辑都在这里
from agent_tasks.direct.wheelbipe.wheelbipe_V14.cfg_utils import V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG  # 粗糙地形"高度课程"的默认参数
from agent_tasks.direct.wheelbipe.wheelbipe_V14.env_cfg import WheelbipeV14FlatEnvCfg  # 本环境配套的默认配置类（参数表）
from scripts.utils.velocity_trace_html import build_reward_signs, build_velocity_trace_html  # 工具：把录制数据渲染成交互式 HTML 图表


def get_rough_height_offset_curriculum_cfg(cfg) -> dict:
    """Return normalized rough height-offset curriculum settings."""
    # 读取"粗糙地形高度课程"参数：没配置就用默认值，缺的键用默认值补齐
    raw_cfg = getattr(cfg, "rough_height_offset_curriculum_cfg", None)  # 从配置对象取原始参数（可能没配，为 None）
    normalized = dict(V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG)   # 拷贝一份默认参数作为底本
    if isinstance(raw_cfg, dict):   # 用户确实配置了参数字典时
        normalized.update(raw_cfg)  # 用用户的值覆盖同名默认键（其余键保留默认）
    return normalized               # 返回补齐后的完整参数字典


def get_rough_terrain_boundary_reset_cfg(cfg) -> dict:
    # 同上：读取"跑到地形边界就重置"的参数并补齐默认值
    raw_cfg = getattr(cfg, "rough_terrain_boundary_reset_cfg", None)  # 原始参数（可能为 None）
    normalized = {                     # 默认：功能关闭、边界余量 0.5 米
        "enabled": False,
        "margin": 0.5,
        "use_inner_terrain_area": True,
    }
    if isinstance(raw_cfg, dict):      # 用户配置了才覆盖
        normalized.update(raw_cfg)
    return normalized


def get_training_progress_steps_per_iteration(cfg) -> int:
    """Resolve PPO steps-per-iteration used to extrapolate training progress from env steps."""
    # 解析"PPO 每轮迭代包含多少个环境步"，用于把环境步数换算成训练轮次(iteration)
    curriculum_cfg = get_rough_height_offset_curriculum_cfg(cfg)  # 先看课程参数里有没有配
    for attr in ("training_progress_steps_per_iteration",):       # 再看配置类上有没有这个属性
        steps = int(getattr(cfg, attr, 0))                        # 取值（默认 0 表示没配）
        if steps > 0:                                             # 配了有效值就直接用
            return steps
    return int(curriculum_cfg["steps_per_iteration"])             # 都没配则退回课程参数里的值（通常 24）


def get_extrapolated_training_iteration(env) -> int:
    """Extrapolate current training iteration from runner anchor + ``common_step_counter``."""
    # 外推当前训练轮次：环境本身不知道 runner 跑到第几轮，
    # 所以用"runner 上次报数时的轮次 + 之后累计的环境步数 ÷ 每轮步数"来推算。
    runner_iteration = int(getattr(env, "_training_iteration", 0))  # runner 最近一次同步进来的轮次（锚点）
    steps_per_iteration = get_training_progress_steps_per_iteration(getattr(env, "cfg", env))  # 每轮多少个环境步
    step_iteration = int(getattr(env, "_training_iteration_base", runner_iteration)) + (  # 锚点轮次 + 增量：
        max(
            int(getattr(env, "common_step_counter", 0))          # 环境启动以来累计的总步数
            - int(getattr(env, "_training_progress_step_base", 0)),  # 减去锚点时刻的步数 = 锚点之后的步数
            0,                                                   # 防止出现负数
        )
        // steps_per_iteration                                   # 步数整除每轮步数 = 经过了多少轮
    )
    return max(runner_iteration, step_iteration, 0)              # 取两者较大者（保证轮次只进不退）


class WheelbipeV14Env(WheelbipeV13Env):
    """V14 机器人环境：在 V13 基础上增加云台控制、地形课程、轨迹录制。"""

    cfg: WheelbipeV14FlatEnvCfg  # 类型注解：本环境持有的配置对象类型（参数表）

    def __init__(self, cfg: WheelbipeV14FlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)  # 先让父类把场景、机器人、传感器、命令等全部建好

        # —— 训练进度（课程学习用）——
        self._training_iteration = 0            # runner 同步进来的当前训练轮次
        self._training_iteration_base = 0       # 轮次锚点：与步数锚点配套，用于外推
        self._training_progress_step_base = 0   # 步数锚点：记录"上次同步轮次时的总步数"
        self._mean_total_reward: float | None = None  # runner 同步来的平均总奖励（目前仅记录）
        self._rough_height_offset_curriculum_last_level = -1   # 上次课程等级（-1 表示还没打印过日志）
        self._rough_height_offset_curriculum_last_iteration = -1  # 上次课程等级对应的轮次
        self._apply_rough_height_offset_curriculum(self.robot._ALL_INDICES, force=True)  # 启动时立即把所有环境摆到课程对应的初始地形等级

        # —— 找到关键部件的索引（之后每步都按索引批量读写，避免每步按名字查找）——
        self._wheel_link_idx, _ = self.robot.find_bodies(".*_wheel_link")  # 左右轮子 link 的索引（算轮子高度/接触用）
        self._gimbal_yaw_link_idx, _ = self.robot.find_bodies("gimbal_yaw_link")  # 云台 yaw 连杆索引（算云台朝向用）
        self._guide_link_idx = []          # V14 已去掉 guide 机构，这里保留空列表兼容父类逻辑
        self._use_gimbal = self._is_gimbal_enabled()  # 读配置判断本任务是否启用云台
        if self._use_gimbal:               # 启用云台：找到 yaw / pitch 两个关节的索引
            self._gimbal_yaw_idx, self._gimbal_yaw_joint_names = self.robot.find_joints(
                self.cfg.gimbal_yaw_name   # 配置里写的 yaw 关节名（正则）
            )
            self._gimbal_pitch_idx, self._gimbal_pitch_joint_names = self.robot.find_joints(
                self.cfg.gimbal_pitch_name  # pitch 关节名
            )
        else:                              # 不启用云台：索引留空，相关逻辑会自动跳过
            self._gimbal_yaw_idx, self._gimbal_yaw_joint_names = [], []
            self._gimbal_pitch_idx, self._gimbal_pitch_joint_names = [], []
        self._gimbal_idx = list(self._gimbal_yaw_idx) + list(self._gimbal_pitch_idx)  # 云台全部关节索引
        self._ordered_leg_joint_idx = self._resolve_names_to_indices(  # 把配置要求的 12 个腿关节名换成索引
            self.cfg.ordered_leg_joint_names,
            self.robot.joint_names,
            kind="joint",
        )
        self._ordered_leg_body_idx = self._resolve_names_to_indices(   # 同理，12 个腿连杆 body 名换成索引
            self.cfg.ordered_leg_body_names,
            self.robot.body_names,
            kind="body",
        )
        self.reorder_reset_joint_idx = list(self._ordered_leg_joint_idx)  # 告诉父类：重置时按这个顺序摆腿关节
        self._undesired_contact_link_idx = self._find_contact_sensor_indices(  # "不希望接触"的部件索引（碰到就罚分）
            [
                "base_link",        # 车体
                ".*_rear1_link",    # 各腿连杆
                ".*_rear2_link",
                ".*_front1_link",
                ".*_front2_link",
                ".*_front3_link",
                ".*_front4_link",
                "gimbal_yaw_link",  # 云台
                "gimbal_pitch_link",
                ".*_guide_link",    # V14 无此部件，正则匹配为空
            ]
        )
        self._desired_contact_link_idx = self._find_contact_sensor_indices([".*_wheel_link"])  # "希望接触"的部件：只有轮子贴地是正常的

        # —— 重置时检测"这些部位是否触地"的索引（触地说明机器人翻车/卡住）——
        reset_contact_body_names = [          # 默认这些部位触地即判定异常
                                    "base_link",
                                    "gimbal_yaw_link",
                                    "gimbal_pitch_link",
                                    ".*_guide_link",
                                    ]
        terrain_cfg = getattr(self.cfg, "terrain", None)  # 看看任务用什么地形
        if self._use_gimbal and getattr(terrain_cfg, "terrain_type", None) != "plane":  # 有云台且非平地时
            reset_contact_body_names = ["gimbal_yaw_link", "gimbal_pitch_link"]  # 粗糙地形上车身可能正常贴坡，只查云台触地
        self._reset_contact_link_idx = self._find_contact_sensor_indices(
            reset_contact_body_names          # 存成索引，重置逻辑里用
        )

        # —— 云台相关的内部状态张量（形状都是 [环境数, ...]，GPU 上批量并行）——
        self._gimbal_pitch_target = torch.full(      # pitch 关节的位置目标张量
            (self.num_envs, len(self._gimbal_pitch_idx)),
            fill_value=float(getattr(self.cfg, "gimbal_pitch_target_pos", 0.0)),  # 目标角度来自配置（默认 -0.5 rad，抬头/低头固定角）
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_yaw_velocity_target = torch.zeros(   # yaw 关节的速度目标（自旋模式用：让头部持续旋转）
            (self.num_envs, len(self._gimbal_yaw_idx)),
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_heading_target_w = torch.zeros(      # yaw 的"世界系朝向"目标（航向锁定模式用）
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_heading_target_initialized = torch.zeros(  # 每个 env 是否已采样过航向目标（布尔）
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._ensure_gimbal_heading_pd_gain_tensors()     # 创建航向 PD 控制的 kp/kd 增益张量
        self._gimbal_spin_translate_lin_vel_yaw = torch.zeros(  # "小陀螺平移"目标速度（在云台 yaw 坐标系下的 vx, vy）
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_spin_translate_height_cmd = torch.zeros(   # 小陀螺平移模式同时采样的身高指令
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_spin_translate_sin_heading = torch.zeros(  # 平移方向的正弦（方向用 sin/cos 表示，避免角度突变）
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_spin_translate_cos_heading = torch.ones(   # 平移方向的余弦（初值 1 对应方向角 0）
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_spin_translate_active = torch.zeros(       # 每个 env 是否正处于"小陀螺平移"模式（布尔掩码）
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._gimbal_spin_translate_last_command_counter = torch.full(  # 记录各 env 上次命令计数，用于检测"命令已重新采样"
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._gimbal_spin_translate_marker: VisualizationMarkers | None = None  # 可视化标记物句柄（Play 模式才创建）
        self._create_gimbal_spin_translate_marker()             # 按配置决定是否创建头顶提示球
        self._gimbal_yaw_actuator_default_stiffness = None      # 备份 yaw 执行器默认刚度（航向 PD 模式要临时改成 0）
        self._gimbal_yaw_actuator_default_damping = None        # 备份 yaw 执行器默认阻尼
        self._capture_gimbal_yaw_actuator_gains()               # 现在就把默认增益备份下来
        self._validate_v14_bookkeeping()                        # 校验机器人部件数量是否符合 V14 预期（防资产改坏）
        self._reset_gimbal_joints(self.robot._ALL_INDICES)      # 启动时把云台摆到初始状态
        # —— 速度轨迹录制（Play 时可选：把速度/高度/各项 reward 写 CSV 并生成 HTML 图）——
        self._velocity_trace_initialized = False                # 录制器是否已初始化（懒初始化：第一次用时才建文件）
        self._velocity_trace_selected_env_id: int | None = None  # 选中的录制对象环境号（锁定某个 agent 持续记录）
        self._velocity_trace_last_sample_time = -1.0e9          # 上次采样时刻（负大数保证第一步就采样）
        self._velocity_trace_last_html_time = -1.0e9            # 上次写 HTML 的时刻（控制 HTML 刷新频率）
        self._velocity_trace_file = None                        # 打开的 CSV 文件句柄
        self._velocity_trace_writer = None                      # csv.DictWriter 写入器
        self._velocity_trace_rows: list[dict[str, float | int | str]] = []  # 内存里也存一份行数据，供生成 HTML
        if self._is_velocity_trace_enabled():                   # 配置里开了录制才注册退出钩子
            atexit.register(self._close_velocity_trace)         # 程序退出时自动收尾（写最终 HTML、关文件）

    def _is_gimbal_enabled(self) -> bool:
        # 判断是否启用云台：优先看显式开关 use_gimbal；没配则看有没有给云台关节名
        use_gimbal = getattr(self.cfg, "use_gimbal", None)
        if use_gimbal is not None:
            return bool(use_gimbal)
        return bool(getattr(self.cfg, "gimbal_yaw_name", None)) or bool(
            getattr(self.cfg, "gimbal_pitch_name", None)
        )

    def set_training_progress(
        self,
        iteration: int | None = None,
        mean_total_reward: float | None = None,
    ) -> None:
        """Receive optional runner-side progress for curriculum scheduling."""

        # 训练 runner 每轮回调本方法，把"真实轮次/平均奖励"告诉环境，供课程学习调度
        if iteration is not None:                     # 收到轮次时：更新轮次与两个锚点
            self._training_iteration = int(iteration)
            self._training_iteration_base = int(iteration)
            self._training_progress_step_base = int(getattr(self, "common_step_counter", 0))  # 锚定此刻的总步数
        if mean_total_reward is not None:             # 收到平均奖励时：仅记录
            self._mean_total_reward = float(mean_total_reward)
        self._sync_command_generator_training_iteration()  # 把推算的轮次同步给命令生成器（某些模式按轮次启停）

    def _get_training_iteration(self) -> int:
        # 对外提供"当前训练轮次"：用锚点+步数外推，见 get_extrapolated_training_iteration
        return get_extrapolated_training_iteration(self)

    def _sync_command_generator_training_iteration(self) -> None:
        """Push extrapolated training iteration to special-mode command generators."""
        # 把推算出的轮次传给命令生成器（若它支持），特殊模式靠它决定何时开始训练
        command_gen = getattr(self, "command_generator", None)      # 环境的命令生成器（采样速度指令）
        if command_gen is not None and hasattr(command_gen, "set_training_iteration"):
            command_gen.set_training_iteration(self._get_training_iteration())

    def _rough_height_offset_curriculum_enabled(self) -> bool:
        # 粗糙地形高度课程是否启用（配置开关）
        return bool(get_rough_height_offset_curriculum_cfg(self.cfg)["enabled"])

    def _get_rough_height_offset_curriculum_iteration(self) -> int:
        # 课程推进以"训练轮次"为时钟：直接取外推轮次
        curriculum_cfg = get_rough_height_offset_curriculum_cfg(self.cfg)
        steps_per_iteration = int(curriculum_cfg["steps_per_iteration"])
        if steps_per_iteration <= 0:                    # 未配置每轮步数时退回 runner 轮次
            return max(int(getattr(self, "_training_iteration", 0)), 0)
        return self._get_training_iteration()

    def _get_rough_height_offset_curriculum_level(self) -> tuple[int, float, int]:
        # 计算当前课程等级：(等级号, 0~1 的进度比例, 当前轮次)
        # 课程逻辑：每 interval 轮升一级，难度随等级上升（地形起伏加大）
        curriculum_cfg = get_rough_height_offset_curriculum_cfg(self.cfg)
        num_levels = max(int(curriculum_cfg["num_levels"]), 1)   # 总级数（至少 1 级）
        if num_levels <= 1:                             # 只有 1 级 = 没有课程，难度恒定
            return 0, 1.0, self._get_rough_height_offset_curriculum_iteration()

        iteration = self._get_rough_height_offset_curriculum_iteration()  # 当前轮次
        max_iteration = int(curriculum_cfg["max_iteration"])  # 超过该轮次后等级封顶
        interval = int(curriculum_cfg["interval"])            # 每隔多少轮升一级
        capped_iteration = min(iteration, max(max_iteration, 0))  # 轮次封顶
        level = num_levels - 1 if interval <= 0 else capped_iteration // max(interval, 1)  # 轮次整除间隔 = 等级
        level = int(max(0, min(level, num_levels - 1)))       # 夹在 [0, 总级数-1] 内
        scale = float(level) / float(num_levels - 1)          # 归一化难度 0~1（日志/地形缩放用）
        return level, scale, iteration

    def _apply_rough_height_offset_curriculum(
        self,
        env_ids: torch.Tensor | None,
        *,
        force: bool = False,
    ) -> None:
        # 把指定 env 的出生点搬到"当前课程等级"对应的地形上（课程学习的核心）
        if not self._rough_height_offset_curriculum_enabled():  # 课程没开就什么都不做
            return

        terrain = getattr(self, "terrain", None)              # 地形导入器对象
        terrain_origins = getattr(terrain, "terrain_origins", None)  # 每个(等级,地形类型)地块的原点坐标表
        terrain_levels = getattr(terrain, "terrain_levels", None)    # 每个 env 当前所在等级
        terrain_types = getattr(terrain, "terrain_types", None)      # 每个 env 当前所在地形类型
        env_origins = getattr(terrain, "env_origins", None)          # 每个 env 的出生点坐标
        if terrain_origins is None or terrain_levels is None or terrain_types is None or env_origins is None:
            return                                            # 缺任何一项（如平地任务）就无法做课程，直接返回

        env_ids_t = self._as_env_ids_tensor(env_ids)          # None → 全部环境的索引张量
        if env_ids_t.numel() == 0:                            # 没有环境要处理
            return

        level, scale, iteration = self._get_rough_height_offset_curriculum_level()  # 当前应处的课程等级
        max_level = int(terrain_origins.shape[0]) - 1         # 地形实际支持的最高等级（防越界）
        num_types = int(terrain_origins.shape[1])             # 同一等级有几种地形变体
        level = min(level, max_level)
        if level < 0 or num_types <= 0:
            return
        curriculum_cfg = get_rough_height_offset_curriculum_cfg(self.cfg)
        num_levels = max(int(curriculum_cfg["num_levels"]), 1)
        terminal_level = min(num_levels - 1, max_level)       # 课程最终等级（封顶后的）
        random_up_to_current = bool(curriculum_cfg["random_reset_up_to_current_level"])  # 是否在 0~当前级随机
        random_after_max = bool(curriculum_cfg["random_reset_after_max"])  # 毕业后是否全等级随机（巩固）
        random_reset = random_up_to_current or (random_after_max and level >= terminal_level)  # 本次是否随机分配等级
        random_max_level = level if random_up_to_current else max_level  # 随机时的等级上限

        current_levels = terrain_levels[env_ids_t]            # 这些 env 现在所在的等级
        if not random_reset and not force and current_levels.numel() > 0 and torch.all(current_levels == level):
            return                                            # 等级没变化且非强制：无需搬运，跳过（省性能）

        if random_reset:                                      # 随机模式：给每个 env 掷骰子定等级
            reset_levels = torch.randint(
                0,
                random_max_level + 1,                         # 0 ~ 上限（含）
                (env_ids_t.numel(),),
                device=env_ids_t.device,
                dtype=terrain_levels.dtype,
            )
            randomize_type = bool(curriculum_cfg["randomize_type_on_random_reset"])  # 是否连地形类型也随机
            if randomize_type:
                terrain_types[env_ids_t] = torch.randint(     # 随机挑一种地形变体
                    0,
                    num_types,
                    (env_ids_t.numel(),),
                    device=env_ids_t.device,
                    dtype=terrain_types.dtype,
                )
        else:                                                 # 普通模式：所有 env 统一搬到当前等级
            reset_levels = torch.full(
                (env_ids_t.numel(),), level, device=env_ids_t.device, dtype=terrain_levels.dtype
            )

        terrain_levels[env_ids_t] = reset_levels              # 写回各 env 的等级
        terrain.env_origins[env_ids_t] = terrain_origins[reset_levels.long(), terrain_types[env_ids_t].long()]  # 出生点 = 对应(等级,类型)地块的原点
        if level != self._rough_height_offset_curriculum_last_level:  # 等级发生变化时打印一次日志
            print(
                "[TerrainCurriculum] V14 rough height_offset_range "
                f"scale={scale:.2f}, level={level}, iteration={iteration}, "
                f"random_reset={random_reset}, random_max_level={random_max_level}"
            )
        self._rough_height_offset_curriculum_last_level = level      # 记住本次等级（下次比较用）
        self._rough_height_offset_curriculum_last_iteration = iteration

    def _append_rough_height_offset_curriculum_log(self) -> None:
        # 把课程状态写进训练日志（TensorBoard 可见），方便观察课程推进
        if not self._rough_height_offset_curriculum_enabled():
            return
        level, scale, iteration = self._get_rough_height_offset_curriculum_level()
        self.extras.setdefault("log", {})                     # extras["log"] 是传给 runner 的日志字典
        self.extras["log"]["TerrainCurriculum/height_offset_scale"] = scale       # 归一化难度
        self.extras["log"]["TerrainCurriculum/height_offset_level"] = float(level)  # 当前等级
        self.extras["log"]["TerrainCurriculum/iteration"] = float(iteration)        # 对应轮次

    # ───────────────────────────────────────────────────────────────────────
    # 速度轨迹录制：Play 时把某个 agent 的速度/高度/各项 reward 按固定频率写 CSV，
    # 并周期性生成"鼠标悬停联动"的交互式 HTML 图表，用于分析策略表现。
    # ───────────────────────────────────────────────────────────────────────

    def _get_velocity_trace_cfg(self) -> dict:
        # 读取录制功能的配置字典（没配则返回空字典）
        cfg = getattr(self.cfg, "velocity_trace_cfg", {}) or {}
        return dict(cfg) if isinstance(cfg, dict) else {}

    def _is_velocity_trace_enabled(self) -> bool:
        # 录制功能开关
        cfg = self._get_velocity_trace_cfg()
        return bool(cfg.get("enabled", False))

    def _ensure_velocity_trace(self) -> None:
        # 懒初始化录制器：只在第一次真正录数据时创建文件（避免 Play 不录制时也建文件）
        if self._velocity_trace_initialized:
            return
        self._velocity_trace_initialized = True

        cfg = self._get_velocity_trace_cfg()
        csv_path = Path(str(cfg.get("csv_path", "logs/debug/velocity_trace.csv")))  # CSV 输出路径
        html_path = Path(str(cfg.get("html_path", csv_path.with_suffix(".html"))))  # HTML 输出路径（默认同名 .html）
        if bool(cfg.get("unique_path", False)):               # 需要防重名时：追加时间戳+进程号
            suffix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"
            csv_path = csv_path.with_name(f"{csv_path.stem}_{suffix}{csv_path.suffix}")
            html_path = html_path.with_name(f"{html_path.stem}_{suffix}{html_path.suffix}")
        csv_path.parent.mkdir(parents=True, exist_ok=True)    # 确保输出目录存在
        html_path.parent.mkdir(parents=True, exist_ok=True)
        self._velocity_trace_csv_path = csv_path              # 记住最终路径（日志打印用）
        self._velocity_trace_html_path = html_path
        self._velocity_trace_max_rows = max(int(cfg.get("max_rows", 20000)), 1)  # 内存中最多保留多少行（防撑爆内存）
        self._velocity_trace_reward_keys = tuple(getattr(self.cfg, "rewards", {}).keys())  # 各奖励项的名字（每项记一列）

        self._velocity_trace_file = csv_path.open("w", newline="")  # 打开 CSV 文件准备写入
        fieldnames = [                                        # CSV 表头：时间/环境/地形
            "sim_time_s",         # 仿真累计时间
            "episode_time_s",     # 本局已进行时间
            "env_id",             # 被记录的环境号
            "terrain",            # 所在地形名
            "cmd_x",              # 指令速度 x
            "cmd_y",              # 指令速度 y
            "cmd_yaw",            # 指令偏航/第 3 列指令
            "vel_x_b",            # 实测机体系速度 x
            "vel_y_b",            # 实测机体系速度 y
            "yaw_rate_b",         # 实测偏航角速度
            "height_cmd",         # 身高指令
            "height_obs",         # 实测身高
            "height_relative",    # 相对地面身高
            "height_reward_ref",  # 奖励参考身高
            "airborne",           # 是否腾空
        ]
        fieldnames += ["reward_total"] + [f"reward_{key}" for key in self._velocity_trace_reward_keys]  # 加上总奖励和每项奖励列
        self._velocity_trace_writer = csv.DictWriter(self._velocity_trace_file, fieldnames=fieldnames)
        self._velocity_trace_writer.writeheader()             # 写表头
        self._velocity_trace_file.flush()                     # 立即落盘表头
        print(f"[VelocityTrace] CSV: {csv_path}")             # 告诉用户文件在哪
        print(f"[VelocityTrace] HTML: {html_path}")

    def _close_velocity_trace(self) -> None:
        # 程序退出时的收尾：写最终 HTML、刷盘、关文件（异常也要保证文件关闭）
        if getattr(self, "_velocity_trace_file", None) is not None:
            try:
                self._write_velocity_trace_html()             # 退出前生成最终 HTML
                self._velocity_trace_file.flush()
                self._velocity_trace_file.close()
            except Exception:
                pass                                          # 收尾失败不影响主流程
            self._velocity_trace_file = None

    def _get_velocity_trace_terrain_name(self, env_id: int) -> str:
        # 查询某环境当前所在地形的名字（CSV 里的 terrain 列）
        manager = self._get_terrain_task_manager()            # "地形任务"管理器（若启用）
        if manager is not None and manager.enabled:
            masks = manager.get_task_masks(self.robot.data.root_pos_w)  # 各任务地形的空间掩码
            for name, mask in masks.items():
                if bool(mask[env_id].item()):                 # 该环境落在哪个任务区
                    return str(name)

        terrain_command_manager = getattr(self, "_terrain_command_manager", None)  # 退而求其次：地形命令管理器
        if terrain_command_manager is not None and getattr(terrain_command_manager, "enabled", False):
            key_indices = terrain_command_manager.get_current_terrain_key_indices()  # 各环境当前地形键序号
            key_idx = int(key_indices[env_id].item())
            terrain_keys = getattr(terrain_command_manager, "terrain_keys", ())
            if 0 <= key_idx < len(terrain_keys):
                return str(terrain_keys[key_idx])
        return "unknown"                                      # 都查不到就标 unknown

    def _select_velocity_trace_env(self) -> int | None:
        # 挑出"录谁"：优先用配置指定的 env 号；否则锁定已选/按地形挑第一个
        cfg = self._get_velocity_trace_cfg()
        requested_env = cfg.get("env_id", cfg.get("agent_index", None))  # 配置里指定的环境号（若有）
        if requested_env is not None:
            env_id = int(requested_env)
            if 0 <= env_id < self.num_envs:                   # 合法就直接用
                return env_id
            return None                                       # 非法则不录

        selected = self._velocity_trace_selected_env_id       # 之前已选中的环境号
        lock_agent = bool(cfg.get("lock_agent", True))        # 是否锁定不换人
        if selected is not None and lock_agent and 0 <= selected < self.num_envs:
            return selected                                   # 锁定模式下一直录同一个

        terrain_name = cfg.get("terrain_name", None)          # 没指定 env 时：按地形名挑
        if terrain_name:
            mask = self.get_terrain_name_mask(str(terrain_name))  # 处于该地形的 env 掩码
            candidates = mask.nonzero(as_tuple=False).squeeze(-1)  # 候选环境号列表
            if candidates.numel() == 0:
                return selected if selected is not None and lock_agent else None  # 没找到就沿用旧的
            selected = int(candidates[0].item())              # 取第一个
        else:
            selected = 0                                      # 连地形都没指定就录 0 号

        self._velocity_trace_selected_env_id = selected       # 记住选择
        print(f"[VelocityTrace] selected env_id={selected}, terrain={self._get_velocity_trace_terrain_name(selected)}")
        return selected

    def _record_velocity_trace(self) -> None:
        # 每个策略步调用：按采样周期抓一帧数据写成 CSV 一行
        if not self._is_velocity_trace_enabled():
            return
        self._ensure_velocity_trace()                         # 首次调用时初始化文件
        cfg = self._get_velocity_trace_cfg()

        sim_time = float(getattr(self, "common_step_counter", 0)) * float(self.step_dt)  # 当前仿真时间(秒)
        sample_dt = max(float(cfg.get("sample_dt", self.step_dt)), float(self.step_dt))  # 采样周期(不小于策略步长)
        if sim_time - self._velocity_trace_last_sample_time + 1.0e-9 < sample_dt:
            return                                            # 还没到采样点：跳过本步

        env_id = self._select_velocity_trace_env()            # 确定录哪个环境
        if env_id is None:
            return

        airborne_state = getattr(self, "height_reward_airborne_state", None)  # 父类维护的腾空标志
        effective_height_cmd = self._get_effective_height_cmd()  # 当前生效的身高指令
        obs_height = self._get_observed_height()              # 观测里的身高
        if self._use_absolute_height() or self._use_leg_length_height():
            relative_obs_height = obs_height                  # 绝对高度口径：直接用
        else:
            relative_obs_height = obs_height - self.ground_z_est  # 相对口径：减去地面高度估计
        wheel_height_w = self.robot.data.body_pos_w[:, self._wheel_link_idx, 2]  # 两轮的世界系高度
        height_reward_ref = self._get_height_reward_reference_height(relative_obs_height, wheel_height_w)  # 奖励用的参考身高
        reward_terms = getattr(self, "_last_reward_terms", {}) or {}  # 父类缓存的"本步各项奖励"
        total_reward = getattr(self, "_last_total_reward", None)      # 缓存的总奖励
        row = {                                               # 组装 CSV 一行
            "sim_time_s": sim_time,
            "episode_time_s": float(self.episode_length_buf[env_id].item()) * float(self.step_dt),  # 本局时间=步数×步长
            "env_id": int(env_id),
            "terrain": self._get_velocity_trace_terrain_name(env_id),
            "cmd_x": float(self.command[env_id, 0].item()),   # 速度指令 x
            "cmd_y": float(self.command[env_id, 1].item()),   # 速度指令 y
            "cmd_yaw": float(self.command[env_id, 2].item()), # 第 3 列指令（偏航或高度，见命令生成器）
            "vel_x_b": float(self.robot.data.root_lin_vel_b[env_id, 0].item()),  # 实测速度 x
            "vel_y_b": float(self.robot.data.root_lin_vel_b[env_id, 1].item()),  # 实测速度 y
            "yaw_rate_b": float(self.robot.data.root_ang_vel_b[env_id, 2].item()),  # 实测角速度
            "height_cmd": float(effective_height_cmd[env_id].item()) if hasattr(self, "height_cmd") else 0.0,
            "height_obs": float(obs_height[env_id].item()),
            "height_relative": float(relative_obs_height[env_id].item()),
            "height_reward_ref": float(height_reward_ref[env_id].item()),
            "airborne": int(bool(airborne_state[env_id].item())) if airborne_state is not None else 0,
            "reward_total": float(total_reward[env_id].item()) if total_reward is not None else 0.0,
        }
        for key in getattr(self, "_velocity_trace_reward_keys", ()):  # 把每个奖励项展开成单独一列
            value = reward_terms.get(key, None)
            row[f"reward_{key}"] = float(value[env_id].item()) if value is not None else 0.0
        self._velocity_trace_rows.append(row)                 # 存内存（供 HTML 用）
        if len(self._velocity_trace_rows) > self._velocity_trace_max_rows:
            self._velocity_trace_rows = self._velocity_trace_rows[-self._velocity_trace_max_rows :]  # 超限就丢最老的
        if self._velocity_trace_writer is not None:
            self._velocity_trace_writer.writerow(row)         # 写 CSV
            self._velocity_trace_file.flush()                 # 立即落盘（崩溃也不丢数据）
        self._velocity_trace_last_sample_time = sim_time      # 记录本次采样时刻

        html_update_dt = float(cfg.get("html_update_interval_s", 1.0))  # HTML 刷新周期（默认 1 秒）
        if sim_time - self._velocity_trace_last_html_time + 1.0e-9 >= html_update_dt:
            self._write_velocity_trace_html()                 # 到点就重新生成 HTML
            self._velocity_trace_last_html_time = sim_time

    def _write_velocity_trace_html(self) -> None:
        # 把内存里的行数据渲染成交互式 HTML 并写盘
        html_path = getattr(self, "_velocity_trace_html_path", None)
        if html_path is None:
            return
        reward_signs = build_reward_signs(getattr(self.cfg, "rewards", {}), rows=self._velocity_trace_rows)  # 每项奖励的正负号(热图配色用)
        html = build_velocity_trace_html(self._velocity_trace_rows, reward_signs)  # 生成完整 HTML 页面
        html_path.write_text(html)

    def _get_observations(self) -> dict:
        # 每步取观测：先走父类通用流程（拼接本体观测/噪声/延迟/堆叠等），再顺手录一帧数据
        observations = super()._get_observations()
        self._record_velocity_trace()
        return observations

    def _reset_idx(self, env_ids):
        # 环境重置钩子：env_ids 是"需要重置"的环境号列表
        self._apply_rough_height_offset_curriculum(env_ids)   # ① 先按课程等级安排出生地形
        super()._reset_idx(env_ids)                           # ② 父类做通用重置（摆位姿、清计数、重采命令等）
        self._reset_gimbal_joints(env_ids)                    # ③ V14 额外重置云台关节
        self._append_rough_height_offset_curriculum_log()     # ④ 课程状态写入日志

    def _get_rough_terrain_boundary_time_out(self) -> torch.Tensor:
        # 检测哪些环境"跑出了地形边界"（返回布尔张量）
        cfg = get_rough_terrain_boundary_reset_cfg(self.cfg)
        if not bool(cfg["enabled"]):                          # 功能没开：全部返回 False
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        terrain_cfg = getattr(self.cfg, "terrain", None)
        terrain_gen = getattr(terrain_cfg, "terrain_generator", None)  # 地形生成器（知道地块尺寸）
        if terrain_gen is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        size = getattr(terrain_gen, "size", None)             # 单个地块的 (长, 宽) 米数
        if size is None or len(size) < 2:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        half_x = 0.5 * float(getattr(terrain_gen, "num_rows", 0)) * float(size[0])  # 整片地形的半长（行数×地块长/2）
        half_y = 0.5 * float(getattr(terrain_gen, "num_cols", 0)) * float(size[1])  # 整片地形的半宽
        if not bool(cfg["use_inner_terrain_area"]):           # 若允许算上外围缓冲带
            border_width = float(getattr(terrain_gen, "border_width", 0.0))
            half_x += border_width
            half_y += border_width

        margin = max(float(cfg["margin"]), 0.0)               # 边界安全余量：离边多远就算越界
        half_x = max(half_x - margin, 0.0)
        half_y = max(half_y - margin, 0.0)
        root_pos_w = self.robot.data.root_pos_w               # 各环境车体的世界坐标
        return (torch.abs(root_pos_w[:, 0]) > half_x) | (torch.abs(root_pos_w[:, 1]) > half_y)  # x 或 y 超界 → True

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 结束判定钩子：返回 (是否终止terminate, 是否超时time_out)
        terminate, time_out = super()._get_dones()            # 父类：摔倒/姿态异常→terminate；时间到→time_out
        # Boundary resets are treated as time-outs so they do not contribute to termination reward.
        time_out |= self._get_rough_terrain_boundary_time_out()  # 越界按"超时"处理（不计入摔倒罚分，只换地方重来）
        return terminate, time_out

    def _resolve_names_to_indices(
        self,
        ordered_names: tuple[str, ...] | list[str],
        available_names: list[str],
        *,
        kind: str,
    ) -> list[int]:
        # 工具：按给定名字顺序查出索引列表；缺名字直接报错（尽早发现资产/配置不匹配）
        name_to_idx = {name: idx for idx, name in enumerate(available_names)}  # 名字→索引 的查找表
        missing_names = [name for name in ordered_names if name not in name_to_idx]  # 找不到的名字
        if missing_names:
            raise RuntimeError(
                f"Wheelbipe V14 {kind} discovery failed, missing names: {missing_names}"
            )
        return [name_to_idx[name] for name in ordered_names]  # 按要求的顺序返回索引

    def _validate_v14_bookkeeping(self) -> None:
        # 启动自检：核对 V14 机器人的关节/连杆数量是否符合预期，防止资产改动后悄悄出错
        expected_joint_name = "left_front2_joint"             # 抽查一个必须存在的关节
        expected_body_name = "left_front2_link"               # 和一条必须存在的连杆
        if expected_joint_name not in self.robot.joint_names:
            raise RuntimeError(f"Wheelbipe V14 is missing required joint: {expected_joint_name}")
        if expected_body_name not in self.robot.body_names:
            raise RuntimeError(f"Wheelbipe V14 is missing required body: {expected_body_name}")

        spring_count = len(self._spring_idx) if self._spring_idx is not None else 0  # 弹簧关节数量
        checks = {                                            # 名称 → (实际数量, 期望数量)
            "legs_act": (len(self._legs_act_idx), 4),         # 4 个主动腿关节（髋部电机）
            "wheel": (len(self._wheel_idx), 2),               # 2 个轮电机
            "spring2": (spring_count, 2),                     # 2 个弹簧关节
            "ordered_leg_joint_idx": (len(self._ordered_leg_joint_idx), 12),  # 12 个有序腿关节
            "ordered_leg_body_idx": (len(self._ordered_leg_body_idx), 12),    # 12 条有序腿连杆
        }
        if self._use_gimbal:                                  # 启用云台时再查云台两个关节
            checks["gimbal_yaw"] = (len(self._gimbal_yaw_idx), 1)
            checks["gimbal_pitch"] = (len(self._gimbal_pitch_idx), 1)
        bad_checks = [
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in checks.items()
            if actual != expected                             # 收集所有不匹配项
        ]
        if bad_checks:
            raise RuntimeError(
                "Wheelbipe V14 asset discovery mismatch: " + ", ".join(bad_checks)  # 一次性报出所有问题
            )

    def _sample_gimbal_yaw_velocity_targets(self, env_ids: torch.Tensor) -> None:
        # 给指定环境重采 yaw 关节的速度目标（默认云台按恒速自旋；训练时随机速度=域随机化）
        if len(self._gimbal_yaw_idx) == 0 or env_ids.numel() == 0:
            return
        low, high = getattr(self.cfg, "gimbal_yaw_velocity_range", (-1.0, 1.0))  # 速度采样范围（配置可改）
        if high < low:                                        # 区间写反了就纠正
            low, high = high, low
        if abs(high - low) < 1.0e-6:                          # 区间宽度≈0：退化为恒定值
            self._gimbal_yaw_velocity_target[env_ids].fill_(float(low))
            return
        sampled = torch.empty(                                # 在区间内均匀随机采样
            (env_ids.numel(), len(self._gimbal_yaw_idx)),
            dtype=torch.float,
            device=self.device,
        ).uniform_(float(low), float(high))
        self._gimbal_yaw_velocity_target[env_ids] = sampled

    def _get_gimbal_heading_control_cfg(self) -> dict:
        # 读取"云台航向锁定"配置（PD 控制头部朝向固定方向），没配返回空字典
        cfg = getattr(self.cfg, "gimbal_heading_control_cfg", {}) or {}
        return dict(cfg) if isinstance(cfg, dict) else {}

    def _is_gimbal_heading_control_enabled(self) -> bool:
        # 航向锁定是否启用（开关打开且确实存在 yaw 关节）
        cfg = self._get_gimbal_heading_control_cfg()
        return bool(cfg.get("enabled", False)) and len(self._gimbal_yaw_idx) > 0

    def _get_gimbal_spin_translate_cfg(self) -> dict:
        # 读取"小陀螺平移"模式配置（边自旋边平移）
        cfg = getattr(self.cfg, "gimbal_spin_translate_cfg", {}) or {}
        return dict(cfg) if isinstance(cfg, dict) else {}

    def _is_gimbal_spin_translate_enabled(self) -> bool:
        # 小陀螺平移是否可用：模式开关打开（且若要求航向锁定则锁定也必须开）
        cfg = self._get_gimbal_spin_translate_cfg()
        if not bool(cfg.get("enabled", False)):
            return False
        if bool(cfg.get("require_heading_control", True)) and not self._is_gimbal_heading_control_enabled():
            return False
        return len(self._gimbal_yaw_idx) > 0

    def _is_gimbal_spin_translate_marker_enabled(self) -> bool:
        # 头顶提示球只在 Play 演示且开了调试可视化时显示
        return bool(getattr(self.cfg, "play", False)) and bool(
            getattr(self.cfg, "play_gimbal_spin_translate_debug_vis", False)
        )

    def _create_gimbal_spin_translate_marker(self) -> None:
        # 创建可视化标记：两个球形（active=醒目绿球，inactive=半径近 0 的隐形球）
        self._gimbal_spin_translate_marker = None
        if not self._is_gimbal_spin_translate_marker_enabled():
            return                                            # 不需要可视化就不建
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/gimbal_spin_translate_marker",  # 标记物在场景中的挂载路径
            markers={
                "inactive": sim_utils.SphereCfg(              # "关闭"状态：几乎不可见的小黑球
                    radius=0.001,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.0, 0.0),
                        emissive_color=(0.0, 0.0, 0.0),
                    ),
                ),
                "active": sim_utils.SphereCfg(                # "激活"状态：绿色发光球
                    radius=float(getattr(self.cfg, "play_gimbal_spin_translate_marker_radius", 0.12)),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 1.0, 0.55),
                        emissive_color=(0.0, 0.35, 0.12),
                        metallic=0.0,
                        roughness=0.18,
                    ),
                ),
            },
        )
        self._gimbal_spin_translate_marker = VisualizationMarkers(marker_cfg)  # 实例化标记物
        self._gimbal_spin_translate_marker.set_visibility(True)

    def _update_gimbal_spin_translate_marker(self) -> None:
        # 每步刷新标记物：处于小陀螺平移模式的环境头顶显示绿球，否则隐藏
        marker = getattr(self, "_gimbal_spin_translate_marker", None)
        if marker is None:
            return
        if not self._is_gimbal_spin_translate_marker_enabled():
            marker.set_visibility(False)                      # 不该显示时整体隐藏
            return
        marker.set_visibility(True)
        active = getattr(                                     # 当前激活掩码（防御式：属性可能还没建）
            self,
            "_gimbal_spin_translate_active",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        positions = self.robot.data.root_pos_w.clone()        # 以车体位置为基准
        positions[:, 2] += float(getattr(self.cfg, "play_gimbal_spin_translate_marker_height", 0.85))  # 抬高到头顶
        positions[~active, 2] = -1000.0                       # 未激活的挪到地下（看不见）
        marker_indices = active.to(dtype=torch.long)          # 激活→用 active 球(1)，未激活→inactive 球(0)
        marker.visualize(translations=positions, marker_indices=marker_indices)

    def _get_command_special_mode_mask(self, mode_name: str) -> torch.Tensor:
        # 查询哪些环境正处于名为 mode_name 的"特殊命令模式"（返回布尔掩码）
        command_generator = getattr(self, "command_generator", None)
        special_mode_id = getattr(command_generator, "special_mode_id", None)  # 各环境当前模式编号
        mode_names = tuple(getattr(command_generator, "_mode_names", ()))      # 编号→模式名 对照表
        if command_generator is None or special_mode_id is None or len(mode_names) == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 没有特殊模式机制
        try:
            mode_idx = mode_names.index(mode_name)            # 模式名→编号
        except ValueError:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 该模式不存在
        return special_mode_id == mode_idx                    # 环境的模式编号等于目标 → True

    def _get_gimbal_heading_control_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        # 判断指定环境这一步是否该用"航向锁定"控制 yaw
        if not self._is_gimbal_heading_control_enabled() or env_ids.numel() == 0:
            return torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        cfg = self._get_gimbal_heading_control_cfg()
        if not bool(cfg.get("apply_only_in_special_mode", False)):
            return torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)  # 不限模式：全程启用
        mode_name = str(                                      # 限定模式：只在小陀螺平移模式里启用
            cfg.get(
                "special_mode_name",
                self._get_gimbal_spin_translate_cfg().get("special_mode_name", "gimbal_spin_translate"),
            )
        )
        return self._get_command_special_mode_mask(mode_name)[env_ids]

    def _ensure_gimbal_heading_pd_gain_tensors(self) -> None:
        # 保证航向 PD 的 kp/kd 张量存在、尺寸=环境数、且在正确的设备上
        cfg = self._get_gimbal_heading_control_cfg()
        kp = float(cfg.get("kp", 2.0))                        # 默认比例增益
        kd = float(cfg.get("kd", 0.15))                       # 默认微分增益
        if not hasattr(self, "_gimbal_heading_kp") or self._gimbal_heading_kp.shape[0] != self.num_envs:
            self._gimbal_heading_kp = torch.full(             # 首次创建：全部填默认 kp
                (self.num_envs,),
                kp,
                dtype=torch.float,
                device=self.device,
            )
        else:
            self._gimbal_heading_kp = self._gimbal_heading_kp.to(device=self.device, dtype=torch.float)  # 已存在：仅纠正设备/类型（域随机化事件可能改过值）
        if not hasattr(self, "_gimbal_heading_kd") or self._gimbal_heading_kd.shape[0] != self.num_envs:
            self._gimbal_heading_kd = torch.full(             # kd 同理
                (self.num_envs,),
                kd,
                dtype=torch.float,
                device=self.device,
            )
        else:
            self._gimbal_heading_kd = self._gimbal_heading_kd.to(device=self.device, dtype=torch.float)

    def _get_gimbal_yaw_actuator(self):
        # 拿到 yaw 关节的执行器对象（管刚度/阻尼/力矩限制的那个组件）
        actuators = getattr(self.robot, "actuators", None)
        if not isinstance(actuators, dict):
            return None
        return actuators.get("gimbal_yaw", None)

    def _capture_gimbal_yaw_actuator_gains(self) -> None:
        # 备份 yaw 执行器的默认刚度/阻尼（航向 PD 模式需要先把它们清零，退出时再还原）
        actuator = self._get_gimbal_yaw_actuator()
        if actuator is None:
            return
        if hasattr(actuator, "stiffness"):
            self._gimbal_yaw_actuator_default_stiffness = actuator.stiffness.detach().clone()  # 克隆一份避免被原地修改
        if hasattr(actuator, "damping"):
            self._gimbal_yaw_actuator_default_damping = actuator.damping.detach().clone()

    def _set_gimbal_yaw_actuator_gains_for_heading_control(self, env_ids: torch.Tensor, enabled: bool) -> None:
        # 切换 yaw 执行器增益：航向 PD 用"纯力矩"控制，所以位置式的刚度/阻尼必须清零
        actuator = self._get_gimbal_yaw_actuator()
        if actuator is None or env_ids.numel() == 0:
            return
        if enabled:                                           # 进入 PD 模式：清零（力矩全由我们的 PD 给出）
            if hasattr(actuator, "stiffness"):
                actuator.stiffness[env_ids] = 0.0
            if hasattr(actuator, "damping"):
                actuator.damping[env_ids] = 0.0
            return
        if hasattr(actuator, "stiffness") and self._gimbal_yaw_actuator_default_stiffness is not None:
            actuator.stiffness[env_ids] = self._gimbal_yaw_actuator_default_stiffness[env_ids]  # 退出 PD 模式：还原默认刚度
        if hasattr(actuator, "damping") and self._gimbal_yaw_actuator_default_damping is not None:
            actuator.damping[env_ids] = self._gimbal_yaw_actuator_default_damping[env_ids]      # 还原默认阻尼

    @staticmethod
    def _sample_uniform_range(
        value_range: tuple[float, float] | list[float],
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        # 工具：在 [low, high] 内均匀采 count 个数（区间反了自动纠正；宽度≈0 时返回常量）
        low, high = float(value_range[0]), float(value_range[1])
        if high < low:
            low, high = high, low
        if abs(high - low) < 1.0e-6:
            return torch.full((count,), low, dtype=torch.float, device=device)
        return torch.empty(count, dtype=torch.float, device=device).uniform_(low, high)

    def _sample_gimbal_heading_targets(self, env_ids: torch.Tensor) -> None:
        # 给指定环境采样"航向锁定目标"（世界系朝向角，弧度）
        if env_ids.numel() == 0 or not hasattr(self, "_gimbal_heading_target_w"):
            return
        cfg = self._get_gimbal_heading_control_cfg()
        mode = str(cfg.get("target_mode", "hold_reset_heading"))  # 三种目标来源模式
        if mode == "fixed":                                   # 固定朝向：都用配置给的同一个角度
            self._gimbal_heading_target_w[env_ids] = float(cfg.get("fixed_heading", 0.0))
        elif mode == "sampled":                               # 随机朝向：区间内均匀采（训练时增加多样性）
            heading_range = cfg.get("heading_range", (-torch.pi, torch.pi))
            self._gimbal_heading_target_w[env_ids] = self._sample_uniform_range(
                heading_range,
                int(env_ids.numel()),
                self.device,
            )
        else:                                                 # 默认 hold_reset_heading：锁住重置瞬间的朝向
            self._gimbal_heading_target_w[env_ids] = self.robot.data.heading_w[env_ids]
        self._gimbal_heading_target_w[env_ids] = wrap_to_pi(self._gimbal_heading_target_w[env_ids])  # 规范到 [-π, π]
        if hasattr(self, "_gimbal_heading_target_initialized"):
            self._gimbal_heading_target_initialized[env_ids] = True  # 标记"已采样"，PD 控制可以开工

    def _sample_gimbal_spin_translate_velocity(self, env_ids: torch.Tensor) -> None:
        # 给指定环境采样"小陀螺平移"的目标：云台系速度、方向、以及可选的身高指令
        if env_ids.numel() == 0 or not hasattr(self, "_gimbal_spin_translate_lin_vel_yaw"):
            return
        cfg = self._get_gimbal_spin_translate_cfg()
        if "lin_vel_yaw_speed_range" in cfg or "lin_vel_yaw_heading_range" in cfg:
            # 风格 A："速度大小 + 方向角"的参数化
            speed_range = cfg.get("lin_vel_yaw_speed_range", (0.0, 0.4))    # 速度大小范围
            heading_range = cfg.get("lin_vel_yaw_heading_range", (-torch.pi, torch.pi))  # 方向角范围
            if (
                isinstance(speed_range, (list, tuple))
                and len(speed_range) > 0
                and isinstance(speed_range[0], (list, tuple))
            ):
                # Multi-segment: list of (min, max) tuples, randomly select one segment per env
                # 多段区间：如 [(0,0.04),(0.25,0.5)]，每个环境先随机选一段、再在段内均匀采
                n_segments = len(speed_range)
                seg_idx = torch.randint(0, n_segments, (int(env_ids.numel()),), device=self.device)  # 每环境选段
                speed = torch.empty(int(env_ids.numel()), dtype=torch.float, device=self.device)
                for i, seg in enumerate(speed_range):
                    mask = seg_idx == i                          # 选中第 i 段的环境
                    n = mask.sum().item()
                    if n > 0:
                        speed[mask] = self._sample_uniform_range(seg, n, self.device)  # 段内采样
                speed = torch.clamp(speed, min=0.0)              # 速度大小不为负
            else:
                # Single tuple: (min, max)
                speed = torch.clamp(                             # 单一区间直接采
                    self._sample_uniform_range(speed_range, int(env_ids.numel()), self.device),
                    min=0.0,
                )
            speed_deadzone = max(float(cfg.get("lin_vel_yaw_speed_deadzone", 0.0)), 0.0)  # 死区：低于该速度视为"不移动"
            if speed_deadzone > 0.0:
                speed = torch.where(speed < speed_deadzone, torch.zeros_like(speed), speed)  # 死区内直接置 0
            heading = self._sample_uniform_range(heading_range, int(env_ids.numel()), self.device)  # 平移方向角
            self._gimbal_spin_translate_sin_heading[env_ids] = torch.sin(heading)  # 方向用 sin/cos 存（观测友好）
            self._gimbal_spin_translate_cos_heading[env_ids] = torch.cos(heading)
            self._gimbal_spin_translate_lin_vel_yaw[env_ids, 0] = speed * torch.cos(heading)  # 方向分解到 x
            self._gimbal_spin_translate_lin_vel_yaw[env_ids, 1] = speed * torch.sin(heading)  # 和 y 分量
        else:
            # 风格 B：直接对 vx / vy 各自采样（方向是采样结果的副产品）
            x_range = cfg.get("lin_vel_x_yaw_range", (-0.4, 0.4))
            y_range = cfg.get("lin_vel_y_yaw_range", (-0.4, 0.4))
            vx = self._sample_uniform_range(x_range, int(env_ids.numel()), self.device)
            vy = self._sample_uniform_range(y_range, int(env_ids.numel()), self.device)
            self._gimbal_spin_translate_lin_vel_yaw[env_ids, 0] = vx
            self._gimbal_spin_translate_lin_vel_yaw[env_ids, 1] = vy
            heading_raw = torch.atan2(vy, vx)                     # 由速度向量反推方向角
            self._gimbal_spin_translate_sin_heading[env_ids] = torch.sin(heading_raw)
            self._gimbal_spin_translate_cos_heading[env_ids] = torch.cos(heading_raw)
        height_range = cfg.get("lin_vel_yaw_height_range", None)  # 同步采样的身高指令范围（可选）
        if height_range is not None:
            if (
                isinstance(height_range, (list, tuple))
                and len(height_range) > 0
                and isinstance(height_range[0], (list, tuple))
            ):
                # 身高也支持多段区间（逻辑同速度多段）
                n_segments = len(height_range)
                seg_idx = torch.randint(0, n_segments, (int(env_ids.numel()),), device=self.device)
                height_cmd = torch.empty(int(env_ids.numel()), dtype=torch.float, device=self.device)
                for i, seg in enumerate(height_range):
                    mask = seg_idx == i
                    n = mask.sum().item()
                    if n > 0:
                        height_cmd[mask] = self._sample_uniform_range(seg, n, self.device)
                self._gimbal_spin_translate_height_cmd[env_ids] = height_cmd
            else:
                self._gimbal_spin_translate_height_cmd[env_ids] = self._sample_uniform_range(
                    height_range, int(env_ids.numel()), self.device
                )
        else:
            self._gimbal_spin_translate_height_cmd[env_ids] = 0.0  # 没配则 0（表示不覆盖身高指令）

    def _get_gimbal_yaw_joint_angle_wrapped(self) -> torch.Tensor:
        # 读 yaw 关节当前角度，规范到 [-π, π]（连续自旋时角度会超界）
        if len(self._gimbal_yaw_idx) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        yaw_pos = self.joint_pos[:, self._gimbal_yaw_idx[0]]  # 所有环境该关节的角度
        return wrap_to_pi(yaw_pos)

    def _get_gimbal_yaw_link_heading_w(self) -> torch.Tensor:
        # 云台朝向：优先用 yaw 连杆的世界系偏航角（真实头部朝向）
        if len(self._gimbal_yaw_link_idx) > 0:
            _, _, yaw = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self._gimbal_yaw_link_idx[0]])  # 四元数→欧拉角取 yaw
            return wrap_to_pi(yaw)
        return self.robot.data.heading_w                      # 没有云台时退回车体朝向

    def _get_gimbal_yaw_link_ang_vel_z_w(self) -> torch.Tensor:
        # 云台的偏航角速度（PD 的 D 项要用）
        if len(self._gimbal_yaw_link_idx) > 0 and hasattr(self.robot.data, "body_ang_vel_w"):
            return self.robot.data.body_ang_vel_w[:, self._gimbal_yaw_link_idx[0], 2]  # 连杆角速度 z 分量
        if hasattr(self.robot.data, "root_ang_vel_w"):
            return self.robot.data.root_ang_vel_w[:, 2]       # 退回车体角速度
        return self.robot.data.root_ang_vel_b[:, 2]           # 最后退回机体系角速度

    def _apply_gimbal_heading_pd(self, env_ids: torch.Tensor) -> None:
        # 云台航向 PD 控制：力矩 = kp × 朝向误差 - kd × 角速度，直接下发力矩目标
        if not self._is_gimbal_heading_control_enabled() or env_ids.numel() == 0:
            return
        if hasattr(self, "_gimbal_heading_target_initialized"):
            init_mask = self._gimbal_heading_target_initialized[env_ids]
            if not torch.all(init_mask):                      # 还没采过目标的先补采
                self._sample_gimbal_heading_targets(env_ids[~init_mask])
        cfg = self._get_gimbal_heading_control_cfg()
        heading_error = wrap_to_pi(                           # 朝向误差 = 目标角 - 当前角（规范到 ±π）
            self._gimbal_heading_target_w[env_ids] - self._get_gimbal_yaw_link_heading_w()[env_ids]
        )
        heading_rate = self._get_gimbal_yaw_link_ang_vel_z_w()[env_ids]  # 当前角速度（阻尼项）
        self._ensure_gimbal_heading_pd_gain_tensors()         # 确保 kp/kd 张量就绪
        effort = self._gimbal_heading_kp[env_ids] * heading_error - self._gimbal_heading_kd[env_ids] * heading_rate  # 经典 PD 律
        max_effort = float(cfg.get("max_effort", 2.0))
        if max_effort > 0.0:
            effort = torch.clamp(effort, -max_effort, max_effort)  # 力矩限幅（防头发疯）
        self.robot.set_joint_effort_target(                   # 下发为关节力矩目标
            effort.unsqueeze(-1),
            joint_ids=self._gimbal_yaw_idx,
            env_ids=env_ids,
        )

    def _get_gimbal_spin_translate_mode_mask(self) -> torch.Tensor:
        # 哪些环境当前处于小陀螺平移特殊模式
        if not self._is_gimbal_spin_translate_enabled():
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mode_name = str(self._get_gimbal_spin_translate_cfg().get("special_mode_name", "gimbal_spin_translate"))
        return self._get_command_special_mode_mask(mode_name)

    def _update_gimbal_spin_translate_samples(self, active_mask: torch.Tensor) -> None:
        # 维护小陀螺平移的采样：刚进入模式 / 命令重采时更新目标；退出模式时清零
        if not hasattr(self, "_gimbal_spin_translate_last_command_counter"):
            return
        command_counter = getattr(self.command_generator, "command_counter", None)  # 命令生成器的重采计数
        if command_counter is None:
            resample_mask = active_mask & ~self._gimbal_spin_translate_active  # 没有计数器：只在"新进入模式"时采
        else:
            command_counter = command_counter.to(device=self.device, dtype=torch.long)
            resample_mask = active_mask & (                   # 激活中且命令计数变了 = 命令被重采了
                self._gimbal_spin_translate_last_command_counter != command_counter
            )
        resample_ids = resample_mask.nonzero(as_tuple=False).flatten()
        if resample_ids.numel() > 0:
            self._sample_gimbal_spin_translate_velocity(resample_ids)  # 为这些环境采新目标
            if command_counter is not None:
                self._gimbal_spin_translate_last_command_counter[resample_ids] = command_counter[resample_ids]  # 记住对应计数
        inactive_ids = (~active_mask).nonzero(as_tuple=False).flatten()
        if inactive_ids.numel() > 0:
            self._gimbal_spin_translate_lin_vel_yaw[inactive_ids] = 0.0  # 退出模式：目标清零
            self._gimbal_spin_translate_last_command_counter[inactive_ids] = -1
        self._gimbal_spin_translate_active = active_mask.clone()  # 更新激活掩码缓存

    def _apply_gimbal_spin_translate_command(self) -> None:
        # 把小陀螺平移的目标写进 self.command：车身命令 = 云台系速度旋转到车体系
        active_mask = self._get_gimbal_spin_translate_mode_mask()
        self._update_gimbal_spin_translate_samples(active_mask)   # 先更新采样
        active_ids = active_mask.nonzero(as_tuple=False).flatten()
        if active_ids.numel() == 0:
            return
        if not bool(self._get_gimbal_spin_translate_cfg().get("project_to_body_command", True)):
            self.command[active_ids, 0:2] = 0.0               # 不投影模式：直接清掉平移指令（纯自旋）
            return
        yaw_angle = self._get_gimbal_yaw_joint_angle_wrapped()[active_ids]  # 云台相对车体的自转角
        cos_yaw = torch.cos(yaw_angle)
        sin_yaw = torch.sin(yaw_angle)
        vel_yaw = self._gimbal_spin_translate_lin_vel_yaw[active_ids]  # 云台系目标速度 (vx_yaw, vy_yaw)
        # 二维旋转：把"云台系"速度旋到"车体系"，车身就始终朝云台指向平移
        self.command[active_ids, 0] = cos_yaw * vel_yaw[:, 0] - sin_yaw * vel_yaw[:, 1]
        self.command[active_ids, 1] = sin_yaw * vel_yaw[:, 0] + cos_yaw * vel_yaw[:, 1]
        if hasattr(self, "_gimbal_spin_translate_height_cmd"):
            height_cmd = self._gimbal_spin_translate_height_cmd[active_ids]
            override_mask = height_cmd > 0.0                  # >0 才覆盖（0 表示"不指定"）
            if torch.any(override_mask):
                self.command[active_ids[override_mask], 2] = height_cmd[override_mask]  # 把高度指令写入 command 第 3 列

    def _get_gimbal_spin_translate_measured_lin_vel_yaw(self) -> torch.Tensor:
        # 实测速度反投影回云台系（上面是"指令云台系→车体系"，这是它的逆变换）
        yaw_angle = self._get_gimbal_yaw_joint_angle_wrapped()
        cos_yaw = torch.cos(yaw_angle)
        sin_yaw = torch.sin(yaw_angle)
        vel_b = self.robot.data.root_lin_vel_b[:, :2]         # 车体系实测速度 (vx, vy)
        vel_yaw = torch.zeros_like(vel_b)
        vel_yaw[:, 0] = cos_yaw * vel_b[:, 0] + sin_yaw * vel_b[:, 1]  # 旋转矩阵的逆（转置）
        vel_yaw[:, 1] = -sin_yaw * vel_b[:, 0] + cos_yaw * vel_b[:, 1]
        return vel_yaw

    def _get_gimbal_spin_translate_reward_terms(self) -> dict[str, torch.Tensor]:
        # 小陀螺平移模式的专属奖励项：追踪指令速度/方向、罚超速/罚乱动
        # 返回 dict：奖励项名 → 每环境一个标量（最终乘以 env_cfg 里的权重）
        zero = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if not hasattr(self, "_gimbal_spin_translate_active") or not torch.any(self._gimbal_spin_translate_active):
            return {                                          # 没有环境在该模式：所有项返回 0（省计算）
                "gimbal_spin_track_lin_vel_yaw_frame": zero,
                "gimbal_spin_track_lin_speed": zero,
                "gimbal_spin_track_lin_heading": zero,
                "gimbal_spin_lin_vel_yaw_square": zero,
                "gimbal_spin_lin_speed_overshoot": zero,
                "gimbal_spin_heading_error_square": zero,
                "gimbal_spin_stand_still_lin_vel": zero,
                "gimbal_spin_track_lin_heading_v2": zero,
                "gimbal_spin_heading_error_square_v2": zero,
            }

        active = self._gimbal_spin_translate_active.float()   # 激活掩码转 0/1（用于屏蔽非激活环境）
        cmd_yaw = self._gimbal_spin_translate_lin_vel_yaw     # 云台系指令速度
        meas_yaw = self._get_gimbal_spin_translate_measured_lin_vel_yaw()  # 云台系实测速度
        err_yaw = cmd_yaw - meas_yaw                          # 速度误差向量
        err_yaw_sq = torch.sum(torch.square(err_yaw), dim=-1) # 误差平方和 ‖v_cmd - v_meas‖²

        speed_cmd = torch.linalg.norm(cmd_yaw, dim=-1)        # 指令速率（向量长度）
        speed_meas = torch.linalg.norm(meas_yaw, dim=-1)      # 实测速率
        speed_err = speed_cmd - speed_meas                    # 速率误差

        heading_cmd = torch.atan2(cmd_yaw[:, 1], cmd_yaw[:, 0])   # 指令方向角
        heading_meas = torch.atan2(meas_yaw[:, 1], meas_yaw[:, 0])  # 实测方向角
        heading_err = wrap_to_pi(heading_cmd - heading_meas)      # 方向误差（规范 ±π）
        # Vector-based heading error: single atan2(cross, dot), naturally in [-π, π]
        # 方向误差 v2：直接用两向量叉积/点积的 atan2（结果天然在 ±π，少一次 wrap）
        heading_err_v2 = torch.atan2(
            cmd_yaw[:, 0] * meas_yaw[:, 1] - cmd_yaw[:, 1] * meas_yaw[:, 0],  # 叉积 z 分量
            cmd_yaw[:, 0] * meas_yaw[:, 0] + cmd_yaw[:, 1] * meas_yaw[:, 1],  # 点积
        )
        heading_gate = (                                      # 方向奖励的"门"：速度太小时方向没意义，不计方向分
            (speed_cmd > float(getattr(self.cfg, "gimbal_spin_heading_cmd_speed_min", 0.1)))   # 指令速度够大
            & (speed_meas > float(getattr(self.cfg, "gimbal_spin_heading_meas_speed_min", 0.05)))  # 实测速度够大
        ).float()

        lin_vel_sigma = max(float(getattr(self.cfg, "gimbal_spin_lin_vel_yaw_sigma", 0.25)), 1.0e-6)   # 各奖励的 σ 尺度参数（控制曲线陡缓）
        speed_sigma = max(float(getattr(self.cfg, "gimbal_spin_lin_speed_sigma", 0.25)), 1.0e-6)
        heading_sigma = max(float(getattr(self.cfg, "gimbal_spin_lin_heading_sigma", 0.25)), 1.0e-6)
        lin_square_sigma = float(getattr(self.cfg, "gimbal_spin_lin_vel_yaw_square_sigma", 1.0))
        overshoot_sigma = float(getattr(self.cfg, "gimbal_spin_lin_speed_overshoot_sigma", 1.0))
        heading_square_sigma = float(getattr(self.cfg, "gimbal_spin_heading_error_square_sigma", 1.0))
        stand_still_speed_threshold = float(                  # 指令速度低于它 = "要求站住"
            getattr(
                self.cfg,
                "gimbal_spin_stand_still_speed_threshold",
                self._get_gimbal_spin_translate_cfg().get("lin_vel_yaw_speed_deadzone", 0.05),
            )
        )
        stand_still_mask = (speed_cmd <= max(stand_still_speed_threshold, 0.0)).float()  # 站住掩码（1=要求站住）

        return {
            # exp(-误差²/σ)：误差越小奖励越接近 1（追踪指令速度向量）
            "gimbal_spin_track_lin_vel_yaw_frame": torch.exp(-err_yaw_sq / lin_vel_sigma) * active,
            # 追踪速率大小（不管方向）
            "gimbal_spin_track_lin_speed": torch.exp(-torch.square(speed_err) / speed_sigma) * active,
            # 追踪方向角（有速度门控 + 站住时不计）
            "gimbal_spin_track_lin_heading": torch.exp(-torch.square(heading_err) / heading_sigma) * active * heading_gate * (1.0 - stand_still_mask),
            # 二次型速度误差惩罚（σ²‖误差‖²）
            "gimbal_spin_lin_vel_yaw_square": (lin_square_sigma ** 2) * err_yaw_sq * active,
            # 超速惩罚：只罚"实测比指令快"的部分（max(·,0)）
            "gimbal_spin_lin_speed_overshoot": torch.square(
                torch.clamp(speed_meas - speed_cmd, min=0.0) * overshoot_sigma
            ) * active * (1.0 - stand_still_mask),
            # 方向误差二次惩罚（同上带门控）
            "gimbal_spin_heading_error_square": torch.square(heading_err * heading_square_sigma) * active * heading_gate * (1.0 - stand_still_mask),
            # 要求站住时：实测速度的 L1 惩罚（动得越多罚越重）
            "gimbal_spin_stand_still_lin_vel": torch.sum(torch.abs(meas_yaw), dim=-1) * active * stand_still_mask,
            # v2 版方向追踪奖励（数值上更稳定的方向误差）
            "gimbal_spin_track_lin_heading_v2": torch.exp(-torch.square(heading_err_v2) / heading_sigma) * active * heading_gate * (1.0 - stand_still_mask),
            # v2 版方向误差惩罚
            "gimbal_spin_heading_error_square_v2": torch.square(heading_err_v2 * heading_square_sigma) * active * heading_gate * (1.0 - stand_still_mask),
        }

    def _apply_gimbal_targets(self, env_ids: torch.Tensor | None = None) -> None:
        # 每步把云台控制量下发给仿真：pitch 走位置目标；yaw 按"航向锁定 / 恒速自旋"二选一
        if len(self._gimbal_idx) == 0:
            return
        env_ids_t = self._as_env_ids_tensor(env_ids)          # None → 全部环境
        if env_ids_t.numel() == 0:
            return
        if len(self._gimbal_pitch_idx) > 0:                   # pitch：始终位置伺服到固定角
            pitch_targets = self._gimbal_pitch_target[env_ids_t]
            pitch_zero_vel = torch.zeros_like(pitch_targets)  # 同时把速度目标置 0（抑制抖动）
            self.robot.set_joint_position_target(
                pitch_targets,
                joint_ids=self._gimbal_pitch_idx,
                env_ids=env_ids_t,
            )
            self.robot.set_joint_velocity_target(
                pitch_zero_vel,
                joint_ids=self._gimbal_pitch_idx,
                env_ids=env_ids_t,
            )
        if len(self._gimbal_yaw_idx) > 0:                     # yaw：分两拨环境，两种控制方式
            heading_mask = self._get_gimbal_heading_control_mask(env_ids_t)  # 谁用航向锁定
            heading_ids = env_ids_t[heading_mask]
            velocity_ids = env_ids_t[~heading_mask]           # 其余用恒速自旋
            if heading_ids.numel() > 0:
                self._set_gimbal_yaw_actuator_gains_for_heading_control(heading_ids, enabled=True)  # 清零位置增益
                self._apply_gimbal_heading_pd(heading_ids)    # 下发 PD 力矩
            if velocity_ids.numel() > 0:
                self._set_gimbal_yaw_actuator_gains_for_heading_control(velocity_ids, enabled=False)  # 还原位置增益
                yaw_vel_targets = self._gimbal_yaw_velocity_target[velocity_ids]  # 采样的自旋速度
                yaw_zero_effort = torch.zeros(                # 力矩目标置 0（交给速度伺服去控）
                    (velocity_ids.numel(), len(self._gimbal_yaw_idx)),
                    dtype=torch.float,
                    device=self.device,
                )
                self.robot.set_joint_effort_target(
                    yaw_zero_effort,
                    joint_ids=self._gimbal_yaw_idx,
                    env_ids=velocity_ids,
                )
                self.robot.set_joint_velocity_target(         # 下发速度目标 → 执行器内部速度环跟随
                    yaw_vel_targets,
                    joint_ids=self._gimbal_yaw_idx,
                    env_ids=velocity_ids,
                )

    def _reset_gimbal_joints(self, env_ids: torch.Tensor | None = None) -> None:
        # 重置云台：直接写关节状态到仿真 + 重采目标 + 立即下发一次控制
        if len(self._gimbal_idx) == 0:
            return
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return
        self._sample_gimbal_yaw_velocity_targets(env_ids_t)   # 先重采 yaw 自旋速度目标
        if len(self._gimbal_pitch_idx) > 0:                   # pitch：摆到固定目标角、速度 0
            pitch_joint_pos = self._gimbal_pitch_target[env_ids_t]
            pitch_joint_vel = torch.zeros_like(pitch_joint_pos)
            self.robot.write_joint_state_to_sim(              # 直接改仿真里的关节状态（瞬移式重置）
                pitch_joint_pos,
                pitch_joint_vel,
                self._gimbal_pitch_idx,
                env_ids_t,
            )
        if len(self._gimbal_yaw_idx) > 0:                     # yaw：按是否航向锁定分两种摆法
            heading_cfg = self._get_gimbal_heading_control_cfg()
            reset_with_heading_control = self._is_gimbal_heading_control_enabled() and not bool(
                heading_cfg.get("apply_only_in_special_mode", False)
            )
            if reset_with_heading_control:                    # 航向锁定模式：采个新朝向目标，把 yaw 关节瞬移到对准位置
                self._sample_gimbal_heading_targets(env_ids_t)
                yaw_joint_pos = wrap_to_pi(                   # 关节角 = 目标朝向 - 车体朝向
                    self._gimbal_heading_target_w[env_ids_t] - self.robot.data.heading_w[env_ids_t]
                ).unsqueeze(-1)
                yaw_joint_vel = torch.zeros_like(yaw_joint_pos)
            else:                                             # 自旋模式：yaw 归零、按采样的速度自旋
                yaw_joint_pos = torch.zeros(
                    (env_ids_t.numel(), len(self._gimbal_yaw_idx)),
                    dtype=torch.float,
                    device=self.device,
                )
                yaw_joint_vel = self._gimbal_yaw_velocity_target[env_ids_t]
                if hasattr(self, "_gimbal_heading_target_initialized"):
                    self._gimbal_heading_target_initialized[env_ids_t] = False  # 清"已采样"标志（下次用时重采）
            self.robot.write_joint_state_to_sim(
                yaw_joint_pos,
                yaw_joint_vel,
                self._gimbal_yaw_idx,
                env_ids_t,
            )
        self._sample_gimbal_spin_translate_velocity(env_ids_t)  # 重采小陀螺平移目标（备用）
        self._gimbal_spin_translate_active[env_ids_t] = False   # 重置后先不处于该模式
        self._gimbal_spin_translate_last_command_counter[env_ids_t] = -1
        self._apply_gimbal_targets(env_ids_t)                   # 立刻下发一次云台控制

    def _on_command_updated(self) -> None:
        # 命令重新采样后的回调：走父类流程 + 刷新小陀螺平移的命令投影
        super()._on_command_updated()
        self._apply_gimbal_spin_translate_command()

    def _postprocess_reward_terms(self, reward_terms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # 奖励后处理钩子：父类算完各项奖励后，这里做模式相关的调整
        reward_terms = super()._postprocess_reward_terms(reward_terms)
        if hasattr(self, "_gimbal_spin_translate_active") and torch.any(self._gimbal_spin_translate_active):
            keep_mask = (~self._gimbal_spin_translate_active).float()  # 1=非该模式(保留)，0=该模式(清零)
            suppressed_terms = getattr(                            # 该模式下要作废的"普通模式"奖励项
                self.cfg,
                "gimbal_spin_suppressed_reward_terms",
                (
                    "track_lin_vel_xy",            # 普通速度追踪类奖励
                    "track_lin_vel_xy_soft",
                    "track_lin_vel_xy_tight",
                    "track_lin_vel_xy_huge_gap",
                    "track_lin_vel_xy_square",
                    "stand_still_lin_vel",
                ),
            )
            for name in suppressed_terms:
                if name in reward_terms:
                    reward_terms[name] = reward_terms[name] * keep_mask  # 模式内的环境该项清零（乘 0）
        reward_terms.update(self._get_gimbal_spin_translate_reward_terms())  # 加入小陀螺专属奖励项
        return reward_terms

    def _get_ctrl_mode_obs_raw(self) -> torch.Tensor:
        # 控制模式观测（7 维槽位）：告诉策略"当前是什么模式、指令是什么"
        # 普通布局 = [normal, stair, slope, recover, jump, height_target, state_time]
        # 小陀螺平移模式下改写为 = [普通标志, 模式标志, 指令速度, sin/cos方向, sin/cos云台角]
        obs = super()._get_ctrl_mode_obs_raw()
        if obs.shape[-1] < 7:
            return obs                                        # 槽位不足 7 维：放弃改写
        active_mask = self._gimbal_spin_translate_active
        if not torch.any(active_mask):
            return obs                                        # 没有环境在该模式：观测保持父类结果
        obs = obs.clone()                                     # 克隆后再改（避免污染父类缓存）
        active_ids = active_mask.nonzero(as_tuple=False).flatten()
        obs[active_ids, :7] = 0.0                             # 先清零这 7 维
        obs[active_ids, 1] = 1.0                              # 第 1 维=1：告诉策略"我在小陀螺平移模式"
        vel_yaw = self._gimbal_spin_translate_lin_vel_yaw[active_ids]  # 云台系指令速度
        speed = torch.linalg.norm(vel_yaw, dim=-1)            # 指令速率
        obs[active_ids, 2] = speed                            # 第 2 维：指令速度大小
        if bool(self._get_gimbal_spin_translate_cfg().get("use_sampled_heading_obs", True)):
            obs[active_ids, 3] = self._gimbal_spin_translate_sin_heading[active_ids]  # 采样时存的 sin 方向
            obs[active_ids, 4] = self._gimbal_spin_translate_cos_heading[active_ids]  # 和 cos 方向
        else:
            heading = torch.atan2(vel_yaw[:, 1], vel_yaw[:, 0])  # 由速度向量现算方向
            obs[active_ids, 3] = torch.sin(heading)
            obs[active_ids, 4] = torch.cos(heading)
        if bool(self._get_gimbal_spin_translate_cfg().get("zero_heading_in_deadzone", False)):
            deadzone = max(float(self._get_gimbal_spin_translate_cfg().get("lin_vel_yaw_speed_deadzone", 0.0)), 0.0)
            zero_heading_mask = speed <= deadzone             # 死区内（=不动）：方向观测置 0
            if torch.any(zero_heading_mask):
                obs[active_ids[zero_heading_mask], 3] = 0.0
                obs[active_ids[zero_heading_mask], 4] = 0.0
        yaw_angle = self._get_gimbal_yaw_joint_angle_wrapped()[active_ids]  # 云台自转角
        obs[active_ids, 5] = torch.sin(yaw_angle)             # 第 5/6 维：sin/cos(云台角)
        obs[active_ids, 6] = torch.cos(yaw_angle)             # 让策略知道"头已经转到哪了"
        return obs

    def _apply_action(self) -> None:
        # 动作下发钩子（每策略步一次）：父类控制腿/轮 → 本类追加云台相关控制
        super()._apply_action()
        self._apply_gimbal_spin_translate_command()           # 若在小陀螺模式：改写 self.command
        self._apply_gimbal_targets(self.robot._ALL_INDICES)   # 下发云台 pitch/yaw 控制量
        self._update_gimbal_spin_translate_marker()           # 刷新头顶提示球

    def _custom_reset_random(self, env_ids):
        # 自定义随机重置钩子：父类摆完机器人后，再把云台也摆好
        super()._custom_reset_random(env_ids)
        self._reset_gimbal_joints(env_ids)
