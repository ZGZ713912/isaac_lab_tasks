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
effort/knee_soft_limit`），并移植 V1 的防弹跳项 `wheel_hop`（轮心离地高度超 r+tol 的平方）、
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
  agents/rsl_rl_ppo_cfg.py                                       # PPO 配置（移植 V40）
```

已注册任务（`scripts/list_envs.py` 可列）：

| task id | stage | 说明 |
|---|---|---|
| `Robotics-Wheel-Leg-V2-Stand-v0` | stand | 零速度、固定高度站立（首训建议） |
| `Robotics-Wheel-Leg-V2-Height-v0` | height | 零速度、变高 |
| `Robotics-Wheel-Leg-V2-Flat-v0` | locomotion | 统一速度/转向/高度指令（v1 profile，无噪声） |
| `Robotics-Wheel-Leg-V2-Flat-Round2-v0` | locomotion | V40 完成 profile（噪声+持续倾倒+reset 速度随机） |
| `Robotics-Wheel-Leg-V2-Stand-Play-v0` | stand | 单环境演示 |
| `Robotics-Wheel-Leg-V2-Flat-Play-v0` | locomotion | 单环境演示 |
| `Robotics-Wheel-Leg-V2-Flat-Round2-Play-v0` | locomotion | round2 单环境演示 |

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
```

日志输出到 `logs/rsl_rl/wheel_leg_v2_flat_direct/<时间戳>/`。

## 4. 合同要点

- **观测 25** = `ang_vel_b3 | proj_grav_b3 | cmd(vx,wz,height)3 | 四腿相对名义角4 | 六关节速度6 | 上一动作6`；
  history 5 帧 → actor 125；critic 29 = 25 + 真值线速度3 + 真实车高1。
- **动作 6** = `[L_joint1,LL_joint1,L_joint3,R_joint1,RR_joint1,R_joint3]` 的归一化输出；
  腿目标 = nominal + 0.5·a，轮目标 = 10·a（rad/s）。动作 clip 100。
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
2. **站立高度**：`nominal_base_height=0.22`。实测零动作静平衡高 ≈0.2167 m
   （轮子承重 ~68/75 N，闭链误差 <0.05mm，无腿碰地；URDF 零位 FK 估算轮底在 base 下方
   约 0.2376 m）。若结构/气弹簧改动，用 `Geometry/base_height_m` 重新校准。
3. **气弹簧**：第一版被动、不加力（纯 prismatic 约束）。
   力曲线模型在 `source/agent_world/agent_world/actuators/wheel_leg_v2_gas_spring.py`，
   后续可接入。
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
