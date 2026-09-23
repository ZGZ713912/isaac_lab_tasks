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
   （原无弹簧 ≈0.2167 m；注意该 0.231 是**旧的反向力曲线**下测得）。若结构/气弹簧/力曲线改动，
   用 `Geometry/base_height_m` 重新校准。
3. **气弹簧**：已接入 `WheelLegV2GasSpringModel`（BKB0.45-063-172，10MPa，线性，
   **347 N@109mm（全压）→ 279 N@172mm（全伸）**，力随伸长而减小，行程 109–172 mm，阻尼 0）。
   注：2026-09-22 前代码把两端力接反（260@109 / 380@172），已修正；旧 ckpt 与旧静平衡高度作废。
   因为气弹簧是 `excludeFromArticulation` 的棱柱 loop joint，
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

## 9. 小陀螺（高速自旋）课程（移植 wheelbipe special_mode）

目标：先稳住 `±2 rad/s` 自旋，再按 wheelbipe 全课程练到 `2π–4.5π rad/s`。

### 9.1 奖励改动（`contracts/*.json` + `core.py`）
- `yaw`：`σ_yaw` 0.5 → **0.25**，权重 1.0 → **1.5**；新增 **`yaw_square =
  (wz_err·0.5)²`（权重 −1.0）**，对齐 wheelbipe `track_ang_vel_z_square`，给大偏航误差强梯度。
- 新增 **`ang_vel_xy = ω_x²+ω_y²`（权重 −0.05）**，自旋时压横滚/俯仰晃动。
- `termination_penalty` **保持 0**：实测设为 −50 会把早期 baseline 学习速度拖慢 ~20×
  （episode length iter100：86 vs 1999）。摔落代价由终止本身 + `pitch/upright` 惩罚承担。
- 用户新增的前倾约束 `pitch/pitch_rate` 与速度-姿态门控 `velocity_upright_gate`
  （`pitch_gate_sigma`）保留；注意 wheelbipe 该门控默认 `σ=0.1` 且默认关闭，
  V2 当前取 `σ=0.02` 偏硬，若早期学不动可放宽。
- 摩擦：`sim.physics_material` 与地面材质 μ 0.5 → **0.9**（包胶轮），否则轮端 ~2.0 N·m 就超摩擦打滑。

### 9.2 命令课程（`contracts/*.json` 的 `commands.stages.locomotion.special_modes`）
- 普通档 `wz` 仍 `±2.0`（先稳 ±2）；自旋档 `vx≈±0.1`（解耦平移）。
- `spin_low`：`wz=±[2π, 3.25π]`，`rel_envs=0.15`，`iteration_start=2000`，`resample_seconds=8`。
- `spin_mid`：`wz=±[3.25π, 4.5π]`，`rel_envs=0.15`，`iteration_start=4000`，`resample_seconds=8`。
- 进入特殊模式要求本局已进行 ≥ `special_mode_min_episode_seconds`（默认 5s）。
- 采样实现见 `env.py::_sample_commands` / `_sample_interval`：按 `iteration = common_step_counter //
  iteration_steps + iteration_offset` 外推轮次（`iteration_steps` 必须等于 PPO 的
  `num_steps_per_env=48`）；**resume 续训要设 `env_cfg.iteration_offset=已训轮次`，否则课程从头开始**。

### 9.3 检查点/动力学一致性（重要）
- 气弹簧现在是默认开启（`gas_spring_enabled=True`）；**所有 2026-09-20 22:32 之前训练的
  ckpt 都是“无气簧”动力学，不能直接用来评估/play**，否则会莫名失稳。
- 用 `scripts/rsl_rl/diagnose_wheel_leg_v2.py` 做定量验收时要选对 `--checkpoint`
  （locomotion run，不是 stand run）：曾出现用 stand 模型测 spin 得到 `mean_wz≈0.05`
  被误判为“不会自旋”。`BINS` 里已有 `spin_max=(0,2.0)`；高速档可自行加 `wz=2π/3π/4π`。

### 9.4 运行
```bash
# 训练（气簧 ON，含自旋课程）
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V2-Flat-v0 \
    --num_envs=1024 --max_iterations=20000 --headless --device=cuda:0

# 定量诊断（对固定指令跑稳态）
python scripts/rsl_rl/diagnose_wheel_leg_v2.py \
    --task=Robotics-Wheel-Leg-V2-Flat-Play-v0 \
    --checkpoint=logs/rsl_rl/wheel_leg_v2_flat_direct/<ts>/model_XXXX.pt \
    --num_envs=32 --warmup=100 --steps=400 --headless --device=cuda:0
```

## 10. 轮速控制链诊断与左右轮符号修复（2026-09-21）

### 10.1 问题
平地前进时腿长期置后、pitch 稳不住、实际速度远低于指令（`vx≈0.06~0.12` vs 指令 `~1.0`），
平移与自转都表现差。`effort` 惩罚仅约 `-2e-5/step`，不是它在限制轮子。

### 10.2 只读几何结论（用 URDF 纯 FK 算出，无需仿真）
- 零位下 `L_joint3` 转轴在 base 系为 **+Y**，`R_joint3` 为 **−Y**；因为腿关节全为横轴，
  该方向在整个工作空间内恒定（已用 hip/knee ±0.5 rad 验证）。
- 直行（`ω_y>0`）需要 `q̇_L>0, q̇_R<0`（**左右反号**）；自转需要 `q̇_L>0, q̇_R>0`（**左右同号**）。
  即 joint 动作空间里「同号=自转、反号=直行」，**与直觉相反**。
- 零位轮心 `x = −0.033 m`，而实测 `−0.031~−0.035 m` ⇒ **“腿置后”就是名义位形，不是策略漂移**，
  因此不要加“轮心回 x=0”的奖励（会把正确姿态当误差）。
- 零位质心在接地点**前方 13 mm** ⇒ 静力前倾失稳，必须靠轮子主动配平（配平力矩 ≈1.8 N·m）。
- 轮半径 0.06 m 正确（`L_link3.STL` 直径实测 0.120 m）。整车 13.886 kg（base 11.62 kg）。

### 10.3 probe 实测（`scripts/tools/wheel_leg_v2_wheel_probe.py`）
无策略直接驱动轮关节速度目标（`action[2]/action[5]`，腿维置 0 保持名义位形）。
**修复前**（无气簧，自由 base，`wheel_velocity_scale=10`）：

| 关节目标 (L,R) | 实际关节速 | 世界 ω_y (L,R) | vx | wz |
|---|---|---|---|---|
| (+10,+10) | (+5.4,+5.9) | +2.8 / **−2.6** | +0.10 | **−1.53** |
| (+10,−10) | (+5.2,−5.5) | +5.2 / **+5.6** | **+0.34** | +0.03 |
| (−10,−10) | (−6.1,−5.3) | −2.6 / **+2.8** | +0.08 | **+1.53** |
| (−10,+10) | (−5.4,+5.5) | −5.4 / **−5.5** | **−0.26** | +0.02 |
| (+30,−30) | (+18.7,−18.3) | +18.6/+18.4 | **+0.76** | +0.05 |
| (+60,−60) | (+21.9,−23.6) | +21.5/+23.4 | **+1.05** | +0.01 |

⇒ **同号关节目标 = 纯自转，反号 = 纯直行**，与 10.2 的推导完全一致。

### 10.4 修复
- `core.decode_targets`：`wheel = clipped[wheel_indices] * wheel_velocity_scale * wheel_joint_sign`；
  合同新增 `actions.wheel_joint_sign = [1.0, -1.0]`（顺序 = `wheel_indices` = `[L_joint3, R_joint3]`），
  `contract.py` 校验“长度=2 且每项 ±1”。
- 语义变为 **正动作 = 车轮正向滚动 = 车体前进**（左右轮一致）；自转 = 左右动作反号。
- **修复后复测**：`act(+1,+1)` → `vx=+0.342, wz=+0.031`（直行）；
  `act(+1,−1)` → `wz=−1.533`（自转）；`act(−1,−1)` → `vx=−0.260`（后退）；
  `act(+6,+6)` → `vx=+1.05`。✅ 符合标准约定。
- **所有旧 ckpt 的轮动作维语义已失效，必须新开 run 从零重训**。

### 10.5 轮速伺服 / 力矩-转速包络（probe 实测）
- `kd` 扫描 0.2 / 1.0 / 3.0：轮速都在 **≈30.7 rad/s 饱和**（固定 base，气簧关），
  目标 60/100/200 结果完全相同 ⇒ 饱和由 **M3508 力矩-转速曲线包络 + 滑移负载**决定，
  **不是 `kd` 太软**。曲线：`motor_speed_rad_s=[0, 999.06]`、`motor_torque_nm=[0.3489, 0]`
  ⇒ 关节空载上限 `999.06/11 = 90.8 rad/s`（轮缘 5.45 m/s），但带载会大幅提前饱和。
- 低指令下滚动良好：`act=1.0`（目标 10 rad/s）→ `vx=0.342, slip=0.043`；
  `act=6.0`（目标 60 rad/s）→ `slip=0.965`（**烧胎**）。
- 结论：**课程 `vx` 由 ±2.0 收到 ±1.0**（`locomotion.vx`，两个合同同步改），
  先把无滑移滚动学稳，再逐步放开。

### 10.6 气簧注意事项
气簧力在名义位形约 290 N（修正后线性模型；按产品图真实曲线约 326 N），
**开环（无策略、腿动作置 0）会主导姿态**：probe 中 `zero` 动作就有
`pitch=+20.8°`，加轮动作后到 `−32°`。因此**气簧 A/B 必须在闭环（带策略）下做**，
开环 probe 不能用来判断气簧好坏。

### 10.7 诊断工具
- `scripts/tools/wheel_leg_v2_wheel_probe.py`：无策略轮系 probe。
  `--fix-base`（隔离伺服/力矩）、`--no-gas-spring`、`--wheel-kd`、`--settle/--measure`；
  输出 `logs/debug/wheel_leg_v2_wheel_probe_*.json`（含每 10 步 trace：vx/wz/ω_y）。
- 新增训练日志：`Orientation/{pitch,roll}_deg`、`Orientation/pitch_rate_rad_s`、
  `Geometry/wheel_x_rel_root_{left,right,min}_m`。
- 注意：`contact_sensor.net_forces_w` 的 x 分量在探针中恒为 0（匀速滚动时水平净力本就≈0），
  **不要**用它判断推进方向，用 `vx/wz` 与轮世界 `ω_y`。
