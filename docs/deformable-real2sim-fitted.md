# Deformable Real2Sim 条件拟合版 V3

这版已用实际 CSV 拟合电流驱动机构，并在 Isaac Sim/PhysX 中回放。它替代早期
`25/2048 Nm/count` 先验，作为独立的 `Real2Sim-v3` 训练任务。轴端真实力矩仍没有
测力标定；这里的 N·m 是以 CAD 质量、惯量和几何为前提的**仿真等效量**。

## 数据、模型与适用范围

审计五份 CSV，共 175481 行，四份完成；23:01:48 的 923 行不完整记录不参加拟合。
源文件 SHA256、模型参数、拟合区间和检查结果保存在模型 JSON 内。

- 悬空 21:23:46、22:29:29 的慢扫、快扫及 1 Hz 段用于拟合。
- 接地慢扫前 60 s 用于约束整体载荷下的电流尺度；后 60 s 和完整快扫用于检查。
- 23:03:11 整份记录未参加数值参数拟合，但已经用于诊断及选择模型结构，**不是盲测**。
- 输入采用 `current_raw`，不用 CSV 中未标定的 `/torque` 代理。

机构模型包含电流到关节力矩的增益、附加转子惯量、黏性/库仑摩擦、随角度变化的
装配残差，以及单边弹性限位和压缩阻尼。接近 75° 时，记录存在很大的电流而角度
几乎不动；省略该限位会使开环阶跃预测发散。附加惯量直接写入 PhysX joint armature，
避免用滞后的加速度“补偿惯量”。摩擦和装配残差是等效参数，不能据此认定某个零件磨损。

| 量 | 拟合/实现 |
| --- | --- |
| 共同有效电流增益 | 0.0216634 N·m/count |
| 指令限幅 | ±2048 counts，即名义电机输出约 ±44.37 N·m |
| 各腿附加惯量 | 0.0290–0.0336 kg·m² |
| 库仑摩擦 | 0.579–0.644 N·m |
| 黏性摩擦 | 0.104–0.189 N·m·s/rad |
| 限位位置 | 各腿约 74.25–75.48°，记录间存在漂移 |
| 指令/反馈报文周期 | 2 ms，分别保持；控制器每 1 ms 更新 |
| 报文相位 | 编码器报文在相邻的控制 tick 生效，与电流准备错开 1 ms |
| 电流响应等效一阶时间常数 | 各腿 4.51–4.86 ms，训练标称 4.6 ms |

一阶时间常数来自 5 ms 采样的命令/反馈拟合，中间未记录的 2 ms 报文使用插值近似。
不能从这些数据把纯传输延迟和电流内环滞后单独辨识出来。电流响应残差也含采样与
插值误差，不能全部当成白噪声。传感器噪声采用静止段量级，训练保留 ±10% 增益变化
作为额外鲁棒性先验。回差默认关闭；现有单侧角度反馈无法将其与弹性/摩擦唯一分离。

控制链与 RMCS 相同：`b0=-1`，内部 ADRC 输出乘 5.74635241301908 得到 current_raw；
ESO 接收最近已准备、量化、限幅的命令。扫频/阶跃提供有限参考速度，绕过 TD；RL
没有参考速度，使用目标速率限制及 TD。反馈报文在两次采样间保持，不虚构 1 kHz 编码器。

相位不能忽略：新准备电流的 CSV 行中，准备时刻的反馈年龄中位数约 1.59 ms；相邻
控制 tick 的反馈时间戳则比电流准备晚约 0.41 ms。将二者错误同步会令同样 ADRC
参数下的阶跃响应过慢。修正的是报文生效相位，没有改动 metadata 中的控制增益。

## 角度及接线映射

CAD 连杆向量是 `(0.029108, 0, -0.13694)`，所以 `q=0` 对应约 **78°**，75° 是命令
上限，不能同时当作 CAD 零位。V3 使用 `q = atan2(0.13694,0.029108) - physical_angle`。
17° 软基准对应 `q=1.0646473 rad`，reset 直接使用该姿态；训练目标为 16–75°，其中
16° 保留约 6.54 mm 名义底盘离地量，17° 仍为低车身软奖励基准。原 V1/V2 的映射保留。

URDF `joint_leg_1..4` 的实际顺序是 **右前、左前、左后、右后**，由 +X 前、+Y 左的
根关节位置确定；CSV 顺序是左前、左后、右后、右前。所有逐腿参数已转换到 URDF 顺序。

V3 的 action contract 为 `minangle_physical_v3`，观测为 `current_fraction_v2`。
Actor 仍为 8×32 维历史，critic 为 40 维；电流槽位不能与旧力矩观测混用。
每个训练目录保存完整模型快照及 SHA256；play/eval 校验契约和快照后加载。

## 已完成检查

离线 URDF 约化动力学与实际 USD/PhysX 比较，最大惯量差 `5.71e-8 kg·m²`、最大重力
力矩差 `1.74e-6 N·m`。因此该等效拟合确实针对当前 Isaac 资产，而非另一个机构模型。

Isaac 开环检查只在每个 0.5 s 窗口开始时读取实测角度/速度，随后只重放电流，未来的
实测状态不参与控制。436 个不重叠窗口的角度 RMS 误差范围如下，完整逐腿指标在输出 JSON：

| 实验段 | 四腿角度 RMS 误差范围 |
| --- | --- |
| 慢扫 | 1.91–2.25° |
| 快扫 | 2.09–2.83° |
| 1 Hz 压力段 | 3.45–6.04° |
| 阶跃 | 2.28–4.58° |

另外完成了 29.995 s 的 **reference-only 闭环回放**，只输入日志的目标角/目标速度，
初始角度/速度只设置一次，之后不使用实测状态反馈。CSV 没有记录 ESO 隐状态，因此
前 5 s 作为初始化段排除。名义 4.6 ms 电流滞后下，后 25 s 的结果为：

| 腿 | 角度 RMS | 角度绝对误差 P95 | 电流 RMS 误差 |
| --- | ---: | ---: | ---: |
| 右前 | 1.863° | 4.493° | 298 counts |
| 左前 | 0.343° | 0.703° | 112 counts |
| 左后 | 0.503° | 1.222° | 94 counts |
| 右后 | 0.347° | 0.822° | 70 counts |

角度响应已较接近，但限位处电流的松弛过程仍未复现，右前最明显；不能用角度曲线接近
推断整个执行器已完全一致。高负载循环后限位位置发生变化，现有静态弹性限位不能
区分塑性、迟滞、预紧变化和编码器零位变化。比较图同时保留电流曲线，避免掩盖该差异。

![闭环角度与电流对比](../outputs/deformable_real2sim_20261005/fitted_closed_loop_comparison.png)

接地留出段的共同运动合力矩 RMS 残差约 2.03–2.07 N·m，对照合力矩 RMS 约 17.6 N·m，
约为 12%。这是无额外支撑、四轮共同接地、近似同步慢运动下的虚功检查，不等同于
逐轮载荷或整车动态预测验收。限位附近、高速和温度/装配漂移仍是主要误差来源。

20° 零动作 RL 集成检查：4 环境、200 步，没有终止/重置，四轮接触比例 100%，
初始四腿均为实际 17°，观测和奖励有限。车体倾角仍约 20°；这只是接入检查，**不是
策略调平成功**。训练能力应单独使用 POLICY/ZERO 配对的 10°/17°/20° 评估。

最终测试：91 项回归通过，随后 ADRC 有限参考速度状态处理修正的 38 项针对性复查
通过（与前者重叠，不相加）。完成 3 轮、96 个 transition 的 GPU PPO 接入测试，72 个
策略张量和优化器状态均有限。保存目录：
`logs/rsl_rl/deformable_real2sim_fitted_precision_v3/2026-10-05_10-45-53_fitted_real2sim_integration/`。
`model_2.pt` 已在 20° play 中重新加载、运行 50 步并以退出码 0 关闭，只用于接入验证。
完整验证清单为 `outputs/deformable_real2sim_20261005/fitted_verification.json`。

## 命令

完整训练（此命令与有限轮数集成验证不同）：

```bash
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Robotics-Deformable-Suspension-BestEffort-Precision-Real2Sim-v3 \
  --headless --num_envs 64 --device cuda:0 --max_iterations 10000
```

模型保存在 `logs/rsl_rl/deformable_real2sim_fitted_precision_v3/`。把 `checkpoint` 设置为
该目录中实际完成训练的模型后，用 20° 坡道播放：

```bash
checkpoint=/absolute/path/to/run/model_N.pt
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python scripts/rsl_rl/play_deformable.py \
  --task Robotics-Deformable-Suspension-Rough-Keyboard-Play-Real2Sim-v3 \
  --checkpoint "$checkpoint" --grade-deg 20 --device cuda:0 \
  --vx_max 0.8 --vy_max 0.5 --wz_max 1.5 --fixed_camera --debug_motion
```

重现数据拟合与回放：

```bash
python scripts/tools/deformable_assembly_fit.py \
  --output outputs/deformable_real2sim_20261005/dynamics_fit_v3.json
python scripts/tools/deformable_current_fit.py \
  --output outputs/deformable_real2sim_20261005/current_fit.json
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python scripts/tools/deformable_fitted_isaac_replay.py \
  --fit outputs/deformable_real2sim_20261005/dynamics_fit_v3.json \
  --csv /home/noir/Documents/workspace/deformable阶越/deformable_angle_sweep_suspended_2026-10-04_23-03-11_run1.csv \
  --output outputs/deformable_real2sim_20261005/isaac_fitted_replay_v3_full.json \
  --windows-per-stage 300 --headless --device cuda:0
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python scripts/tools/deformable_fitted_closed_loop.py \
  --csv /home/noir/Documents/workspace/deformable阶越/deformable_angle_sweep_suspended_2026-10-04_23-03-11_run1.csv \
  --output outputs/deformable_real2sim_20261005/fitted_closed_loop_phase1.json \
  --headless --device cuda:0
```

主要依据：[MIT 机器人系统辨识](https://underactuated.mit.edu/sysid.html) 中的惯量、摩擦
及动力学残差回归方法，以及 [Isaac Lab 执行器说明](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/actuators.html)
中的显式力矩边界与 armature。实际 API 以本机 Isaac Lab 2.3.2 源码为准。用户提供的
飞书页面重定向到登录页，未获得正文，没有作为已读依据。
