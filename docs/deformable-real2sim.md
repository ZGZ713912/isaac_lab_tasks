# Deformable 电流域 Real2Sim（2026-10-05）

当前完成了数据审计和电流域仿真接入。CSV 支持确认控制器参数、通信节拍和闭环响应，
**尚未得到独立验证的电流到关节力矩标定，也没有辨识出磨损、摩擦或回差的唯一参数。**
本任务可用于模型消融及后续拟合，现阶段不应把长程训练结果称为已验证的实车策略。

本轮验证：83 项针对性回归测试通过；4 环境、每轮 8 步的 3 轮 GPU PPO 测试完成，
保存的 72 个策略张量全部有限，优化器有 72 项参数状态。随后成功重新加载快照模型，
平地运行至少 50 步，再用 SIGINT 正常退出（0.82 s）。训练测试目录为
`logs/rsl_rl/deformable_real2sim_precision_current_v1/2026-10-05_09-20-28_real2sim_integration_smoke/`，
其中 `model_2.pt` 仅用于接入检查，不能当作已收敛策略。

## 数据与统计

审计 5 份 CSV，共 175481 行；CSV 与 metadata 都记录 SHA256。
四份记录正常完成，23:01:48 的阶跃记录只有 923 行且 `completed=false`，保留在
报告中但不参与紧凑模型的节拍统计。所有记录 `write_failed=false`、`dropped_samples=0`。

| 记录 | 行数 | 用途 |
| --- | ---: | --- |
| 21:23:46 悬空慢扫 | 41678 | 闭环低频响应 |
| 21:28:44 接地慢扫 | 39456 | 接地负载差异 |
| 22:29:29 悬空慢扫加高速 | 43681 | 饱和和边界压力检查 |
| 23:01:48 阶跃 | 923 | 未完成，只审计 |
| 23:03:11 阶跃 | 49743 | 独立闭环阶跃诊断，尚未通过仿真对齐 |

CSV 采样约 5 ms；CAN 指令周期约 **2.000 ms**，反馈周期约 **1.999 ms**。
计算方式为正的时间戳差除以序号增量，避免把跳过的数据包误当成 4–6 ms 的周期。
反馈年龄中位数约 **1.017 ms**，它包含采样相位，不能直接等同于纯传输延迟。
分位数采用固定种子的均匀蓄水池近似，均值、标准差使用所有有限样本。

完整阶跃记录每关节有 5 次可分析跳变。10–90% 时间计算为 `t90-t10`，窗口不跨越
下一次跳变；稳定尾段取每次保持末尾 2 s 并避开下一跳变前 0.5 s。

| 腿 | 10–90% 时间中位数 | 超调中位数 | 稳态角标准差中位数 |
| --- | ---: | ---: | ---: |
| 左前 | 0.310 s | 4.92° | 0.00221° |
| 左后 | 0.315 s | 4.80° | 0.00492° |
| 右后 | 0.320 s | 5.00° | 0.00275° |
| 右前 | 0.315 s | 4.88° | 0.00181° |

这些是闭环及方向混合统计，逐方向结果在审计 JSON；不能把尾段角波动全当作编码器
噪声，也不能把闭环上升时间当作电机时间常数。高速段出现 ±2048 电流饱和及角度越界，
不作为线性小信号电机拟合样本。接地与悬空的低角度电流差约为
左前 −266、左后 −506、右后 −258、右前 −517 counts，混合了载荷、几何和摩擦，
不能单独解释为关节磨损。

重现审计（仓库根目录）：

```bash
/home/noir/miniconda3/envs/isaaclab/bin/python scripts/tools/deformable_real2sim_identify.py \
  --csv /home/noir/Documents/workspace/deformable扫频数据慢速/*.csv \
  --csv /home/noir/Documents/workspace/deformable扫频数据慢速加高速悬空/*.csv \
  --csv /home/noir/Documents/workspace/deformable阶越/*.csv \
  --output source/agent_tasks/agent_tasks/direct/deformable_suspension/configs/deformable_real2sim.json \
  --report outputs/deformable_real2sim_20261005/audit.json
```

## 控制和观测

URDF `q = 75° - physical_angle`，`physical_angle` 表示腿相对水平的角度。
RL 残差目标仍相对 17° minangle，并保留目标速率限制。ADRC 使用 `b0=-1`、1 ms 步长，
控制器内部输出乘 **5.74635241301908** 才成为 LK `current_raw`。ESO 输入为上一次
已经量化、限幅、保持的 CAN 指令除以该系数，避免反馈未执行的请求值或 PhysX 力矩。
扫频日志提供有限的目标速度，会绕过 TD；RL 无目标速度，经过限速和 TD，因此两条
参考输入链不能直接用同一个阶跃时间评判。

电流指令限幅 ±2048、按 C++ `std::round` 方式量化，再经过 2 ms 指令保持、
传输延迟、电流一阶滞后和噪声。单独映射成 Isaac 需要的 N·m 后施加摩擦和回差损失。
reset 按环境清空指令、传感器历史、ESO 和滞后状态，防止上一回合残留力矩。

| 参数 | 当前处理 | 证据状态 |
| --- | --- | --- |
| 指令保持周期 | 2 ms | 日志时间戳/序号支持 |
| 反馈年龄 | 1–2 ms 延迟近似 | 日志支持年龄量级；未复现每个 CAN 包 |
| CAN 传输延迟 | 0–1 ms | 未辨识先验 |
| 电流一阶滞后 | 3 ms | 未辨识先验 |
| Nm/count | 25/2048，末端限幅 25 Nm | 未标定仿真先验 |
| 摩擦、回差、增益变化、噪声范围 | 每环境/每关节 reset 随机化 | 敏感性先验；不是磨损拟合结论 |

日志的 `/torque` 是约 `0.1740234 Nm/count` 的历史电流代理，metadata 明确表示
没有轴端扭矩标定；它不直接送给 PhysX。`25/2048` 也不是由数据拟合或由“25 Nm
限值”推导出来的真实常数，只是现有仿真尺度的初值。需要在明确的惯量/负载假设下
拟合有效电流响应，或补充扭矩/负载标定，再在未用于拟合的阶跃和接地记录上验证。

Actor 每帧 32 维，历史 8 帧共 256 维，critic 40 维。Actor 的 `[18:22]` 从旧模型的
力矩比例改为延迟的**电机电流/2048**，避免暴露仿真的真实机械力矩。元数据为
`real2sim_observation_version=current_fraction_v1`。原有 0–2 个 policy step 的整帧延迟
仍存在，它模拟上层观测链，与 1 kHz 电机反馈延迟分开。

训练、play 和评估会拒绝旧力矩观测 checkpoint 与新电流任务混用。每次新训练保存
`params/real2sim_model.json` 和 SHA256，后续播放读取该快照，避免默认 JSON 改动后
静默替换旧模型的执行器配置。参数随机化范围仍由相应训练/评估任务决定。

## 训练及播放

以下是完整训练的启动命令；本轮只执行有限步集成验证，没有启动 10000 轮训练：

```bash
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Robotics-Deformable-Suspension-BestEffort-Precision-Real2Sim-v2 \
  --headless --num_envs 64 --max_iterations 10000 --device cuda:0
```

训练目录为 `logs/rsl_rl/deformable_real2sim_precision_current_v1/`。
平地消融任务为 `Robotics-Deformable-Suspension-Flat-Real2Sim-v2`；Rough 和
BestEffort-Real2Sim-v2 使用 `deformable_real2sim_current_v1` 目录。

新模型训练完成后，设置 `checkpoint` 为真实文件路径：

```bash
checkpoint=/absolute/path/to/deformable_real2sim_precision_current_v1/<run>/model_<N>.pt
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python scripts/rsl_rl/play_deformable.py \
  --task Robotics-Deformable-Suspension-Rough-Keyboard-Play-Real2Sim-v2 \
  --checkpoint "$checkpoint" --grade-deg 20 --device cuda:0 \
  --vx_max 0.8 --vy_max 0.5 --wz_max 1.5 --fixed_camera --debug_motion
```

Real2Sim play 固定参数，但每步仍有随机噪声。`deformable_suspension_eval.py` 则固定
参数并关闭时间噪声用于 POLICY/ZERO 配对比较，在输出中标明；这不是随机扰动鲁棒性
验收。20° 超出 Foundation 的 0–5° 分布，必须同时报告倾角、四轮接触、限位/饱和、
跟踪和物理终止，不能只观察车是否还在动。

20° 零动作集成探针：4 环境、200 步，初始四腿均为 `q=1.01229`，观测/奖励有限，
进程正常退出。共发生 4 次终止/超时重置，settling 后平均倾角约 20.08°、
四轮接触比例 67%，因此**不能称为通过调平或稳定性验收**。数据在
`outputs/deformable_real2sim_20261005/smoke20.json`；该探针没有加载任何策略。

提供的飞书页面未成功获取（502），本次实现依据本地 CSV、metadata 和 RMCS 控制器
源码，没有把未读的页面列作算法依据。
