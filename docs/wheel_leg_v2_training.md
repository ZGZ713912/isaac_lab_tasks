# Wheel_leg_V2 训练说明（移植自 V40 完成训练栈）

本任务把 [wheeled-biped-rl-train](../../../wheeled-biped-rl-train)（V40 完成训练仓）
的 **contract 驱动 DirectRLEnv** 架构移植到本仓库的闭链轮腿机器人 `Wheel_leg_V2`。

## 1. 机器人差异

| 项 | V40（串联开链） | Wheel_leg_V2（闭链） |
|---|---|---|
| 树关节 | 6 revolute | 18 revolute |
| 闭链 | 无 | 4 × `SphericalJoint`（四杆）+ 2 × `PrismaticJoint`（气弹簧），均 `excludeFromArticulation` |
| 驱动关节 | `L_joint1/2/3`,`R_joint1/2/3` | **髋 `L_joint1/R_joint1`、膝 `LL_joint1/RR_joint1`、轮 `L_joint3/R_joint3`** |
| 被动关节 | 无 | `LL_*`/`RR_*`（四杆）、`LLL_*`/`RRR_*`（气弹簧），共 12 个，effort 恒 0 |
| 资产 | 运行时 URDF→USD + 运行时碰撞过滤 | 直接加载已 authored 的 `Wheel_leg_V2.usd`（闭链在 USD 内） |

V2 的闭链树结构中，髋电机位于 `L_joint1/R_joint1`，膝电机位于四杆支链的
`LL_joint1/RR_joint1`。`L_joint2/R_jonit2` 是主链中的被动关节，由闭合约束跟随，
不能再作为膝电机输入。训练动作和观测现在直接使用这 6 个电机树关节。

**奖励设计**：以 V40 权重/核为基础（`velocity/yaw/height/upright/vertical_velocity/action_rate/
effort/knee_soft_limit`），增加前倾约束 `pitch/pitch_rate`，并在车身前倾时衰减速度奖励；同时移植 V1 的防弹跳项 `wheel_hop`（轮心离地高度超 r+tol 的平方）、
`wheel_slip`（轮底切向滑移²，仅触地轮）、`leg_joint_osc`（腿关节速度相对 EMA 的偏差²）、
`action_smoothness_leg`（腿动作二阶差分²）；`velocity`/`yaw` 追踪在轮子离地时被
**接触门控**（`velocity_contact_gate`）清零，避免"腾空拿速度分"。`lateral_velocity`、
`zero_command_translation`、`termination_penalty` 权重为 0，无 alive bonus。见
`contracts/own_wheel_leg_v2.json` 的 `rewards`。

## 2. 文件

```
source/agent_world/agent_world/assets/wheel_leg_V2.py            # ArticulationCfg（指向已构建 USD）
source/agent_tasks/agent_tasks/direct/wheel_leg_v2/
  __init__.py                                                    # gym 注册
  env_cfg.py                                                     # DirectRLEnvCfg + stage 子类
  env.py                                                         # WheelLegV2Env（闭链适配）
  core.py                                                        # 纯张量 obs/history/action/reward（移植 V40）
  contract.py                                                    # 合同加载/校验
  contracts/own_wheel_leg_v2.json                                # v1 profile（默认，无噪声）
  contracts/own_wheel_leg_v2_round2.json                         # v2 profile（噪声 + 持续倾倒）
  slope.py                                                       # 周期坡面解析求高/梯度（Torch）
  jump.py                                                        # 跳跃相位/弹道/辅助力/奖励（wheelbipe 语义）
  agents/rsl_rl_ppo_cfg.py                                       # PPO 配置（移植 V40）
```

已注册任务（`scripts/list_envs.py` 可列）：

| task id | stage | 说明 |
|---|---|---|
| `Robotics-Wheel-Leg-V2-Stand-v0` | stand | 零速度、固定高度站立（首训建议） |
| `Robotics-Wheel-Leg-V2-Height-v0` | height | 零速度、变高 |
| `Robotics-Wheel-Leg-V2-Flat-v0` | locomotion | 统一速度/转向/高度指令（v1 profile，无噪声） |
| `Robotics-Wheel-Leg-V2-Flat-Round2-v0` | locomotion | V40 完成 profile（噪声+持续倾倒+reset 速度随机） |
| `Robotics-Wheel-Leg-V2-Slope-v0` | locomotion | 周期坡面 10–17°（round2 profile） |
| `Robotics-Wheel-Leg-V2-Slope-Steep-v0` | locomotion | 周期坡面 17–25°（round2 profile） |
| `Robotics-Wheel-Leg-V2-Jump-v0` | jump | 原地/行进跳跃（随机触发+弹道参考+辅助力） |
| `Robotics-Wheel-Leg-V2-Stand-Play-v0` | stand | 单环境演示 |
| `Robotics-Wheel-Leg-V2-Flat-Play-v0` | locomotion | 单环境演示 |
| `Robotics-Wheel-Leg-V2-Flat-Round2-Play-v0` | locomotion | round2 单环境演示 |
| `Robotics-Wheel-Leg-V2-Slope-Play-v0` | locomotion | 周期坡面 10–17° 单环境演示 |
| `Robotics-Wheel-Leg-V2-Slope-Steep-Play-v0` | locomotion | 周期坡面 17–25° 单环境演示 |
| `Robotics-Wheel-Leg-V2-Jump-Play-v0` | jump | 跳跃单环境演示 |

## 3. 运行

```bash
conda activate isaaclab

# 0) 资产闭链回归（应通过，worst closure error ~0.7mm）
python scripts/tools/validate_wheel_leg_v2_closed.py --headless --device cuda:0

# 1) 端到端 smoke（少环境少迭代）
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V2-Stand-v0 \
    --num_envs=64 --max_iterations=2 --headless --device=cuda:0

# 2) 站立训练
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V2-Stand-v0 \
    --num_envs=1024 --max_iterations=20000 --headless --device=cuda:0

# 3) 站稳后统一 locomotion
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V2-Flat-v0 \
    --num_envs=1024 --max_iterations=20000 --headless --device=cuda:0

# 4) 斜坡课程（round2 profile）：先缓坡，再陡坡
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V2-Slope-v0 \
    --num_envs=1024 --max_iterations=20000 --headless --device=cuda:0
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V2-Slope-Steep-v0 \
    --num_envs=1024 --max_iterations=20000 --headless --device=cuda:0

# 5) 跳跃（随机触发 + 弹道参考 + 辅助力衰减）
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V2-Jump-v0 \
    --num_envs=1024 --max_iterations=20000 --headless --device=cuda:0
```

日志输出到 `logs/rsl_rl/wheel_leg_v2_flat_direct/<时间戳>/`。

## 4. 合同要点

- **观测 25** = `ang_vel_b3 | proj_grav_b3 | cmd(vx,wz,height)3 | 四腿相对名义角4 | 六关节速度6 | 上一动作6`；
  history 5 帧 → actor 125；critic 29 = 25 + 真值线速度3 + 真实车高1。
- **动作 6** = `[L_joint1,LL_joint1,L_joint3,R_joint1,RR_joint1,R_joint3]` 的归一化输出；
  腿目标 = nominal + 0.5·a，轮目标 = 10·a（rad/s）。动作 clip 100；轮速目标仍由当前策略输出和控制器裁剪共同决定。
- **名义位形** = URDF 零位 `q=0`（SolidWorks 装配位形，闭链在 q=0 自洽），即 nominal 全 0。
- **时钟** = 200Hz 物理 / 100Hz 策略（dt 0.005 × decimation 2）。
- **执行器** = effort 模式：腿 `kp=60,kd=2,effort=40`；轮 `kd=0.2` + M3508 11:1 torque-speed 曲线
  （effort 上限 3.84 N·m）。被动关节 effort=0。
- **闭链看门狗** = `env_cfg.closure_error_tolerance`（默认 5mm）；超阈值视为约束发散，安全 reset，
  并在 TensorBoard 记录 `Closure/*_mm` 与 `Geometry/closure_max_mm`。

## 5. 首版未决 / 待标定项

1. **膝/髋机械限位**：V2 URDF 全部 `continuous`（USD 限位 ±3.4e38，无硬止挡）。
   合同的 `knee_hard_limits` 目前为空 = 不做膝目标夹紧、`knee_soft_limit` 奖励恒 0。
   机械标定后按 `"LL_joint1": [lo,hi], "RR_joint1": [lo,hi]` 填入即可自动生效。
2. **站立高度**：`nominal_base_height=0.23`。接入气弹簧后实测零动作静平衡高 ≈0.231 m
   （原无弹簧 ≈0.2167 m）。若结构/气弹簧/力曲线改动，用 `Geometry/base_height_m` 重新校准。
3. **气弹簧**：已接入 `WheelLegV2GasSpringModel`（BKB0.45-063-172，10MPa，260→380 N 线性，
   行程 109–172 mm，阻尼 0）。因为气弹簧是 `excludeFromArticulation` 的棱柱 loop joint，
   **不能下发 effort**，改为每物理子步按 `constraints.json` 的 `gas_springs` 锚点求长度/轴向，
   对 `LLL/RRR_link1,2` 施等大反向轴向力（`env._apply_gas_spring_forces`），
   并在 TensorBoard 记录 `Spring/length_m`、`Spring/force_n`。标定后可在
   `env_cfg.gas_spring_force_at_min_n/max_n/damping_n_s_per_m` 调整。
4. **复制策略**：`replicate_physics=True` + `clone_in_fabric=False`。
   实测 PhysX 物理复制会正确复制闭链 loop joints（多环境闭合误差 <0.001mm），
   4096 env 可跑（~3s/iter）；而 `clone_in_fabric=True` 会让接触传感器初始化失败
   （`Failed to initialize contact reporter for specified bodies`），必须保持 False。
   早期用 `replicate_physics=False` 时 4096 env 会因逐环境克隆 USD 打爆内存而卡死。
5. **电机参数标定**：当前已按实际电机所在关节建模，但减速器参数、闭链传动扭矩和编码器零位仍需实机标定。

## 6. 与 V40 的对应

| V40 文件 | 本仓库对应 |
|---|---|
| `wheeled_tasks/v40/core.py` | `wheel_leg_v2/core.py`（去掉 knee 限位硬编码，改可空） |
| `wheeled_tasks/v40/contract.py` | `wheel_leg_v2/contract.py`（去掉 manifest/research 重机制） |
| `wheeled_tasks/direct/v40_serial/env_cfg.py` | `wheel_leg_v2/env_cfg.py` |
| `wheeled_tasks/direct/v40_serial/env.py` | `wheel_leg_v2/env.py`（+ 18 关节/闭链/气弹簧适配） |
| `wheeled_tasks/agents/v40_ppo_cfg.py` | `wheel_leg_v2/agents/rsl_rl_ppo_cfg.py` |
| `contracts/own_v40_v1/v2.json` | `contracts/own_wheel_leg_v2[_round2].json` |

## 7. 斜坡地形（周期坡面，课程两段）

- 地形用 `agent_world.terrains.HfCustomPeriodicSlopeTerrainCfg`：单张大 tile（150×150 m），
  剖面沿 x，周期 = 4×1 m（上坡/平/下坡/平），每周期独立随机坡角；`use_terrain_origins=False`，
  各 env 按 `env_spacing` 铺在同一张坡面上（与 `deformable_suspension` 同款）。
- 求高与地形网格同源：`wheel_leg_v2/slope.py` 提供 `periodic_slope_height_torch` /
  `periodic_slope_gradient_torch`。`_base_height`、`_wheel_clearance`（接触门控 / `wheel_hop`）
  都改为相对**解析地面高度**，而非 `env_origins.z`。
- spawn：按坡角对齐 pitch、随机 yaw，落到地面 + `slope_spawn_drop_m`(4cm) 后自然落地；
  越界（离 tile 边缘 <3m）按 time_out 重置。
- 课程：`Slope-v0`(10–17°) → `Slope-Steep-v0`(17–25°)，均用 round2 profile。

## 8. 跳跃（wheelbipe 语义）

实现在 `wheel_leg_v2/jump.py`（纯 Torch `JumpController`），语义照搬
`wheelbipe/state_machines/{jump_takeoff,airborne}.py`，仅 `stage == "jump"` 生效：

1. **触发**：`trigger_rate_per_s` 随机 + 外部 `request_jump()`；要求 episode 时长≥
   `jump_min_episode_time_s`、冷却结束。
2. **弹道参考**：采样峰值高度 `jump_peak_height_range`，解析求离地速度、蹬伸时间、总时长
   （`gravity`），相位 `IDLE→PUSH→TUCK→IDLE`。
3. **辅助力**：PUSH 内对 `base_link` 施世界 +Z 力
   `force_z + missing_vel_gain·max(v_rel − vz, 0)`（上限 `max_force_z`）；
   概率 `jump_assist_prob_start→end` 按 `common_step_counter` 衰减
   （`jump_assist_decay_iterations × jump_steps_per_iteration`），用于 bootstrap。
4. **奖励**（contract `rewards.weights`）：`jump_push_track`、`jump_push_max_vel`、
   `jump_peak_track`（退出事件）、`jump_air_time`、`airborne_landing_down_vel`。
5. **日志**：`Jump/{PUSH,TUCK,active,assist}_frac`、`assist_prob`、`assist_force_n`、
   `active_max_height_m`、`Reward/jump_*`。
6. **未做**（后续）：腾空高度奖励参考覆盖（airborne height override）、落地轨迹参考、
   域随机化、`ctrl_mode_obs` 通道。
