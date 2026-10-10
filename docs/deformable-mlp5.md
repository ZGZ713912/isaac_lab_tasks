# Deformable 5 帧历史 MLP

2026-10-09：昨日追加 10000 轮后的 `model_10549.pt` 未通过验收，10° 的
Isaac Sim 重跑再次得到最差倾角 P95 11.940°。本次修复采用独立训练种子的
Transformer 闭环示教、5 帧 MLP 蒸馏、必要时的学生状态再示教，以及低噪声
PPO 优化。后续记录见 [MLP 恢复与验证](deformable-mlp5-recovery.md)。
下面的旧长训流程保留为实验记录，流程完成不代表该模型可用。

2026-10-08 将后续 deformable 训练切换到展平历史的 MLP。训练任务为
`Robotics-Deformable-Suspension-Support-Leveling-Joint-History-MLP-Real2Sim-v3`。
该任务继承最新的 Joint 联合支撑调平奖励、0/5/10/17/20° 混合地形、
运动课程、执行器拟合模型和碰撞终止条件，仅改变观测历史长度。

Actor 的 5 帧包含本次观测与之前 4 帧，每帧 32 维，按旧到新拼成 160 维。
控制周期为 10 ms，历史跨度为 40 ms，传感器自身延迟仍由原模型处理。
同一控制步的重复观测读取不会再次推进历史；reset 用首个有效观测填满该
环境的历史，不混入上个 episode。Critic 仍读取当前 40 维特权观测。

| 项目 | 配置 |
| --- | --- |
| 策略类 | ActorCriticSuspensionMLP |
| Actor | 160 → 256 → 128 → 64 → 4 |
| Critic | 40 → 256 → 128 → 64 → 1 |
| 激活 | ELU |
| 初始探索标准差 / 下限 | 0.3 / 0.03，log 参数 |
| 学习率 / 熵系数 | 1e-4 / 0.001 |
| 每轮采样 / PPO epochs | 每环境 24 步 / 3 |
| 动作与动力学 | minangle_physical_v3 / current_fraction_v2，拟合 real2sim |

新 MLP 从头训练，Transformer checkpoint 不能直接作为它的权重起点。
训练入口会拒绝 MLP 的网络类别、历史长度、层宽或激活不匹配，避免只加载
少数碰巧同形的张量而误认为完成了迁移。旧 Transformer 任务继续保留，
便于读取既有训练结果。

## 大训练

以下命令从头训练；环境数可根据显存和实测采样速度调整。

```bash
cd /home/noir/Documents/workspace/example/wheeled-legged_RL
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python -B -u \
  scripts/rsl_rl/train.py \
  --task Robotics-Deformable-Suspension-Support-Leveling-Joint-History-MLP-Real2Sim-v3 \
  --num_envs 512 --max_iterations 10000 --seed 47 --device cuda:0 --headless
```

日志目录为 `logs/rsl_rl/deformable_real2sim_support_leveling_mlp5_v3/`。
训练模型必须保留同一运行目录的 `params/agent.yaml`、`params/env.yaml` 和
`params/real2sim_model.json`，play 与评估从这些快照恢复网络及物理合同。

如果要接着本次 MLP 短训练继续优化，在上述命令上添加
`--checkpoint <MLP运行目录>/model_49.pt --resume_training`。
这会恢复优化器，从第 50 轮继续执行 `--max_iterations` 指定的额外更新数。
没有 `--resume_training` 时，只加载兼容权重，优化器和迭代计数重建。
不要向新任务传入旧 Transformer 的 checkpoint。

## 键盘 play 与评估

```bash
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python \
  scripts/rsl_rl/play_deformable.py \
  --task Robotics-Deformable-Suspension-Rough-Keyboard-Play-History-MLP-Real2Sim-v3 \
  --checkpoint <MLP运行目录>/model_49.pt --device cuda:0 \
  --vx_max 0.8 --vy_max 0.5 --wz_max 1.5
```

play 支持 `--grade-deg 10`、`17` 或 `20` 查看固定长坡。
下例评估所有六类运动指令；将坡度改为 0、5、10、17、20 可复用既有大坡
诊断。5 帧观测不会改变 3°、四轮接触和物理终止的验收条件。

```bash
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python -B \
  scripts/tools/deformable_suspension_eval.py \
  --task Robotics-Deformable-Suspension-Support-Leveling-Joint-History-MLP-Real2Sim-v3 \
  --checkpoint <MLP运行目录>/model_49.pt \
  --policy-only --history 5 --grade-deg 10 --num_envs 16 --steps 600 \
  --command-profile play --command-frame body --device cuda:0
```

## 本次短训练验证

本次验证输出位于 `outputs/deformable_mlp5_short_20261008/`。
50 轮、128 环境、seed 47 的运行用于验证历史、PPO 更新、模型保存与重载。
从头训练的 50 轮成绩不代表调平、大坡或实车部署通过；策略能力仍需长训及
匹配评估。`validation.json` 记录实际结果，`train.log` 保留完整日志。

`network_benchmark.json` 比较新 5 帧 MLP 与原 8 帧 Transformer 的纯 actor
CPU 推理时间（单线程、batch 1/128）。它不包含仿真、GPU 或 PPO 更新时间，
不能把网络加速倍数当成整个训练的加速倍数。

实际短训完成了 50 次更新、153600 次环境转移，actor 与 critic 各 8 个参数
张量均有变化，68 个权重及优化器张量均有限。严格权重重载、100 步无 GUI
play、5° 驻车/前进各 200 步的评估入口全部正常完成，相关 194 项测试通过。
单轮平均采样 6.17 秒、PPO 更新 0.052 秒。单车纯 actor CPU 推理中位数为
MLP 0.0305 ms、旧 8 帧 Transformer 0.2749 ms。

当前短训模型没有通过策略验收：5° 驻车的接地率约 85.3%、倾角 P95 6.09°、
物理终止 1 次；前进接地率 92.5%、P95 5.96°、物理终止 0 次。
这些仅是 4 环境、2 秒诊断，不能用于判断最终能力或优劣。

## 已授权的自动训练流程

用户随后授权继续验证联合奖励并执行大训练。当前流程为：

1. 从 MLP `model_49.pt` 恢复优化器，继续 500 轮、128 环境。
2. 使用 0/5/10/17/20°、六类车体系 play 指令，每类 16 环境、600 步评估。
3. 继续 10000 轮、512 环境，保留优化器、噪声和命令课程的训练进度。
4. 再做同条件五坡度评估，记录严格 3° 成绩和大坡诊断，不放宽验收。

所有仿真串行执行；中间出现技术失败、非有限 checkpoint 或模型合同不匹配
会停止流程并记录失败。训练正常时，即便短阶段的严格水平目标未达标，也会
继续已授权的训练预算。`status=completed` 仅代表流程完成，实际达标另见
`strict_horizontal_goal_achieved` 和各坡度 assessment。

```bash
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python -B -u \
  scripts/tools/deformable_mlp_train.py \
  --checkpoint logs/rsl_rl/deformable_real2sim_support_leveling_mlp5_v3/2026-10-08_20-17-57_mlp5_short_20261008/model_49.pt \
  --output-dir outputs/deformable_mlp5_joint_long_20261008 --run
```

该流程已启动，不能重复使用同一输出目录。`status.json` 持续更新实际进度，
每项作业有独立日志、源码快照与数值训练指标，不自动生成图表。
按正常续训计数，500 轮阶段终点为 `model_549.pt`，大训终点为
`model_10549.pt`；最终文件路径以 `status.json` 的实际记录为准。
