# V3 联合支撑调平奖励

2026-10-08 的 Mixed 长训完成 2500 次更新、512 个环境，最终模型为
`logs/rsl_rl/deformable_real2sim_support_leveling_mixed_v3/2026-10-08_06-03-36_support_leveling_mixed_short_20261008_060329_910497/model_2499.pt`。
名义模型评估覆盖 0/5/10/17/20°、六类车体系 play 命令，每类 16 个姿态、6 s。
最差场景倾角 P95 分别为 0.082°、1.630°、5.041°、11.385°、14.188°。
五坡度物理终止、越界均为零；20° 最低场景四轮接触率为 98.580%。
0/5° 通过严格水平检查，10/17/20° 未通过，整体调平验收拒绝。
完整报告位于 `outputs/deformable_long2499_joint_reward_review_20261008/assessment.json`。

训练前后窗口的总奖励由约 5034 增到 5343，同时平均 episode 长度由约
1354 增到 1444 步。残余倾角由约 3.66° 增到 3.90°，接触且水平比例仍约
65%。总奖励增加不能直接解释成调平精度提高。
同一次训练第 500 轮与第 2499 轮的配对 10° 评估中，最差场景 P95 从
4.974° 变成 5.041°，六类命令中的五类 P95 变差。约 2000 次额外更新没有
降低这个最差场景误差；该观察只覆盖一次训练的名义 10° 条件。

旧 Mixed 同时使用接触奖励、线性倾角代价、二次倾角代价、单轴水平指数奖励、
逐轮 gap/load 代价和轮载均衡奖励。最后一项继承自 SafeMobility，权重为 4；
四轮接触并不要求四轮载荷相等。这些独立项目允许相互补偿，也增加了坡上调平的干扰。

## 联合目标

新任务是 `Robotics-Deformable-Suspension-Support-Leveling-Joint-Real2Sim-v3`。
`--stage joint` 使用单独的 experiment 和输出目录，旧 Mixed 配置保留用于对照。

对每一步计算：

```text
load_deficit = max_i(max(20 N - force_i, 0) / (20 N - contact_threshold))
gap_deficit  = max_i(max(gap_i - 1 mm, 0) / 3 mm)
S = 1 / (1 + load_deficit + gap_deficit)
Q = 1 / (1 + (tilt / 10 degrees)^2) / (1 + clearance_cost)
joint_score = S * (1 + Q) - 1
reward += 80 * joint_score
```

`contact_threshold` 来自任务已有配置，当前为 3 N。`clearance_cost` 使用原有
按坡度、运动和姿态速度计算的软离地余量，作为联合水平收益的一部分。
该分数在 [-1, 1] 内，20 N 以上的不均匀载荷没有代价。
最弱轮支撑和最大间隙都进入同一个支撑质量，而不是用其他轮的高载荷取平均掩盖它。
没有接触力时，缩小离地间隙仍会提高分数。

任一轮力不超过接触阈值时，S 不超过 0.5，即使车身完全水平，联合分数也不超过零。
四轮达到最低载荷、间隙在容差内时，有限倾角和有限离地代价的分数为正。
因此在这个主奖励内，失去一轮接触不能用完全水平来超过受支撑的倾斜姿态。
这不是对总回报、可行姿态、PPO 收敛或实际车辆的数学保证。

Joint 关闭旧接触、轮载均衡、二次倾角、单轴水平、线性倾角、独立 gap/load 和
独立软离地代价，避免重复计算。碰撞/擦地惩罚与终止、速度/牵引指标、平地低位偏好继续保留。
原来的可行切坡 reset 仍启用。探索噪声 0.015、学习率 5e-5、Mixed 地形和
命令课程保持原配置；没有同时进行参数搜索。

新增训练指标为 `dynamic/joint_leveling_score`、`dynamic/joint_support_quality` 和
`dynamic/joint_level_quality`，原始倾角、接触、轮载、间隙、限位、速度误差仍保留。
这些训练特权量没有添加到 actor 输入；8×32 历史、40 维 critic、4 维动作、
16–75° 物理动作范围和执行器模型不变。

## 下一轮对照短训

```bash
cd /home/noir/Documents/workspace/example/wheeled-legged_RL
/home/noir/miniconda3/envs/isaaclab/bin/python -B -u \
  /home/noir/Documents/workspace/example/wheeled-legged_RL/scripts/tools/deformable_leveling_short_train.py \
  --stage joint --iterations 500 --num-envs 512 --seed 47 --noise-std 0.015 \
  --checkpoint /home/noir/Documents/workspace/example/wheeled-legged_RL/logs/rsl_rl/deformable_real2sim_support_leveling_mixed_v3/2026-10-08_06-03-36_support_leveling_mixed_short_20261008_060329_910497/model_2499.pt \
  --run
```

去掉 `--run` 仅打印命令。只加载策略权重，优化器与探索噪声重新初始化。
完成后使用新输出目录的 `next_commands.txt` 执行同条件五坡度评估，再做验收。
本次长训作为直接对照；3°、接触率、物理终止和速度检查不因新奖励而放宽。
先观察联合目标是否提高 10° 的精度且保留 0/5° 与陡坡支撑，再决定是否加大训练预算。
新实现的测试与少量 PPO 集成运行不代表新策略效果已经改善。
本次验证为 117 项针对性 CPU 测试通过，以及从长训模型热启动的 32 环境、
2 次 PPO 更新正常完成；保存权重与优化器的 288 个张量均有限，观测、动作、
物理行程、执行器模型、时序和命令配置的兼容性检查全部通过。
2 轮检查模型只作为集成产物，不作为待部署或下一轮的选定模型。

## 可行性边界

当前 V3 16–75° 行程和 6 mm 保守车身间隙扫描中，10° 各朝向存在 0–2.814°
的四轮接触几何见证；17° 为 6.061–9.802°，20° 为 9.061–12.802°。
17/20° 的全部 24 个朝向没有水平四轮接触高度交集，这一检查在应用车身
包络限制之前已失败。非零倾角见证不代表经过认证的全局最优或动态可行性。
结果依赖当前 CAD、关节零点、行程和地形模型；若实车确有水平姿态，需要校核这些模型量。
奖励无法创造模型中不存在的几何行程，也不能把保守模型的结论直接当作实车结论。

若联合目标短训仍在相同精度处停滞，应转向几何可行姿态求解、受约束动作投影或
可行姿态示教，再进行匹配评估；不继续仅靠增大倾角/接触权重或堆叠训练轮数。
