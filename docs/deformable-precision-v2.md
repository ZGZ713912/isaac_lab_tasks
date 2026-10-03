# Deformable 精修与大坡评估

本轮以 `deformable_minangle_residual_v2/2026-10-03_00-00-56/model_9999.pt`
为共同起点，在 Foundation 的 0–5° 地形上做两组独立的 1000 轮精修。
每组使用 64 个环境，学习率 `1e-4`、熵系数 `0.001`，加载权重后将动作
噪声标准差从约 `0.495` 重设为 `0.15`。重新初始化优化器，并直接使用完整
运动指令范围；不继承原训练的迭代计数。

| 组别 | Task | orientation_x/y_exp_sigma |
| --- | --- | --- |
| low_entropy | BestEffort-Foundation-v2 | 0.05 |
| tight_tilt | BestEffort-Precision-v2 | 0.004 |

两组保持相同的网络、历史长度、动作合同、接触奖励、动力学和安全终止条件。
噪声重设只用于带 `--checkpoint` 的权重精修，不能与 `--resume_training`
一起使用。单独设置 `init_noise_std` 会被加载的 checkpoint 噪声参数覆盖。

## 批量执行

必须使用 Isaac Lab 的 Python；省略 `--run` 只输出命令，不创建实验。
两组训练以及全部评估串行运行，避免多个仿真争用 GPU。

```bash
LD_LIBRARY_PATH=/home/noir/.local/lib \
  /home/noir/miniconda3/envs/isaaclab/bin/python \
  scripts/tools/deformable_precision_pipeline.py \
  --checkpoint logs/rsl_rl/deformable_minangle_residual_v2/2026-10-03_00-00-56/model_9999.pt \
  --output-dir logs/debug/deformable_precision_20261003 \
  --run
```

已经运行的目录不能重复启动；新实验需要新的 `--output-dir`。
每个子进程有独立日志，`status.json` 记录进程、进度、失败原因、源模型和
代码摘要。默认每个子进程最长 4 小时，超时或失败会停止流程。

训练后，原模型和两个新模型使用同一评估环境、种子、16 个环境、每场景
600 步，比较静止、前进、侧移、正向自旋、反向自旋和复合运动。
排序首先考虑物理安全和四轮接触率是否达到 98%，然后比较最差场景的
“四轮贴地且倾角小于 3°”比例和倾角 P95。原模型参与排序；选出的模型
是否通过 Foundation 验收单独记录为 `foundation_accepted`。
`status=completed` 表示流程完成，不能代替验收通过。

之后自动对原模型与选出的模型测试 10°、17°、20°，仍使用上述六个场景。
大坡成绩没有通过也会保留诊断结果。最终生成 `report.md`、各组评估 JSON
及 `geometric_reference.json`。

## 单独测试指定坡度

`--grade-deg` 使用连续 20 米长坡，禁用网格边界重置，保留物理安全终止。
每一步检查四轮仍在同一上坡平面内，避免将坡顶、坡底或换坡后的数据混入。
该入口最多支持 24 个环境、600 步；默认保留 ZERO 对照，下面仅评估模型。

```bash
LD_LIBRARY_PATH=/home/noir/.local/lib \
  /home/noir/miniconda3/envs/isaaclab/bin/python -B \
  scripts/tools/deformable_suspension_eval.py \
  --task Robotics-Deformable-Suspension-BestEffort-Foundation-v2 \
  --checkpoint logs/rsl_rl/deformable_minangle_residual_v2/2026-10-03_00-00-56/model_9999.pt \
  --num_envs 16 --steps 600 --policy-only --grade-deg 20 --device cuda:0
```

将 `20` 改为 `10` 或 `17` 即可测试其他坡度。结果记录四轮接触与小于 3°
联合达标率、倾角分布、物理终止、腿关节和目标接近限位的比例、腿目标跟踪
误差、目标变化速率饱和、轮胎纵向滑移、实际速度及指令速度。

## 尽力范围的参考

```bash
/home/noir/miniconda3/envs/isaaclab/bin/python \
  scripts/tools/deformable_feasibility.py --best-effort --slopes 10 17 20
```

几何扫描覆盖每个坡度的 24 个朝向，要求四轮球面同时接触、腿关节位于
`[0, Q_LOW]`、底盘至少有 6 mm 法向净空，并使用实际车体网格的底部边界。
多起点数值求解得到的是可行姿态参考；非零结果不能当成已证明的全局最优
或动态能力极限。真实训练结果还受扭矩、轮胎摩擦和腿运动速度影响。

本次预扫描得到的可行姿态倾角范围为：10° 坡 0–2.62°，17° 坡
6.13–9.61°，20° 坡 9.13–12.61°。10° 有 3/24 个朝向通过严格水平
可行性检查，17° 和 20° 为 0/24。因此大坡需要同时报告绝对 3° 目标和
相对于几何参考的表现，不能只用是否小于 3° 判断模型有没有尽力。

本次已通过 58 项 CPU 回归测试、3 轮 GPU 精修启动验证和 20° 场景的
100 步初始化验证。完整精修和大坡能力结论以批量流程生成的报告为准。
