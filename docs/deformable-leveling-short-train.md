# V3 车身调平短训

这次准备的是新的 `Robotics-Deformable-Suspension-Leveling-Real2Sim-v3` 任务。
配置和 CPU 检查已完成，训练由用户启动；目前没有新策略效果或水平达标结论。

## 启动

```bash
python /home/noir/Documents/workspace/example/wheeled-legged_RL/scripts/tools/deformable_leveling_short_train.py --run
```

入口脚本使用本机 IsaacLab Python 和 `run_gui.sh`，无需先切换 conda 环境。
默认从上轮全姿态双参考 `model_100.pt` 加载权重，重新初始化优化器与探索噪声，
不会初始化冻结参考策略，也不恢复旧优化器或迭代号。
默认为 **256 环境、每轮 24 步、101 次 PPO 更新**，共 620544 个 transition，
保存第 0/50/100 轮；最后模型位于新的
`logs/rsl_rl/deformable_real2sim_leveling_v3/<本次运行>/model_100.pt`。
启动参数可改，例如 `--num-envs 128 --iterations 201`；末尾去掉 `--run` 只打印命令，
不会创建目录或启动仿真。

训练完成后自动导出 TensorBoard 的 PNG/PDF 曲线和原始标量 JSON。
终端会打印本轮独有的输出目录：
`outputs/deformable_leveling_short_<上海时间戳>/`。其中包含源码快照、`status.json`、
`charts/training_curves.png/pdf`、`charts/training_metrics.json` 和 `next_commands.txt`。
后者保存这个新 checkpoint 的 **20° play、五坡度评估、调平验收** 三条完整命令。
不会替换上轮选定模型或自动接受短训结果；状态是“训练完成，等待调平评估”。

## 训练目标的修改

| 项目 | 上轮最终配置 | 调平短训配置 |
| --- | --- | --- |
| 冻结参考动作损失 | 权重 2、所有姿态启用 | 关闭 |
| 直接低位 actor 辅助损失 | 权重 1 | 关闭 |
| 起始动作噪声 / 下限 | 0.01 / 0.005 | 0.08 / 0.03 |
| 连续倾角代价 | 12 × 倾角（rad） | 60 × 倾角（rad），不按接触关闭 |
| 二次倾角代价 | 12 × 重力水平投影平方 | 40 × 重力水平投影平方 |
| 单轴水平奖励 | 指数尺度 0.004、权重 1.5 | 指数尺度 0.04、权重 2 |
| 最低角全地形偏好 | 权重 2 | 关闭；保留按地形衰减的平地偏好 |
| 255 mm 软限高 | 所有地形 | 只用于平地，5° 前衰减到零 |
| 软离地余量 | 12 mm | 6 mm，与几何诊断边界一致 |
| 动作变化惩罚 | -2 | -0.5 |
| 驻车命令比例配置 | 35% | 75% |

角度、速度及电流硬限制没有扩大；执行器拟合 JSON、ADRC 时序、四轮接触奖励、
擦地/碰撞终止保持原约束。Actor 仍为 8×32 历史、critic 40 维、悬挂动作 4 维。
动作约定仍为 `minangle_physical_v3`，可请求 16–75°；没有在 play 中暗中添加调平控制器。

这轮以驻车及慢速修正为主，其余命令范围为 vx ±0.3、vy ±0.2 m/s、wz ±0.5 rad/s。
评估和 play 命令仍使用 vx 0.8、vy 0.5 m/s、wz 1.5 rad/s，以检查原先的运动能力是否退化。
增加奖励/探索可能带来接触和运动退化，必须读取评估，不能靠 reward 上升判定成功。

## 调平验收

训练后先执行 `next_commands.txt` 中的 benchmark，再执行 leveling_check。
默认评估包含 0/5/10/17/20°，每坡度六类命令、每类 16 个环境、6 s，
名义模型、seed 1234、车体系 play 命令，与上轮参考报告配对。
验收逐场景核对模型、seed、时长、坐标系和初始状态哈希，不混合不同条件。

- 所有坡度：无物理终止、无越界，计入失败后的四轮接触率至少 98%。
- 0/5/10°：每种命令均通过现有严格水平验收，包括倾角 RMS 和 P95 均小于 3°，
  接触且水平比例、episode 成功率至少 95%。
- 17/20°：每种命令的倾角 P95 相比上轮降低至少 20%，并保留接触条件。
  这是大坡调平改善门槛，不能称为完全水平。
- 验收 JSON 还记录速度误差、滑移、腿角限位及电流饱和，暴露能力退化。

当前几何诊断在 20° 坡存在约 9.1–12.8° 的残余倾角可行姿态；这些是保守几何见证，
不是动态保证或全局最优证明。不能承诺只靠此次 reward 修改在全部 20° 朝向下达到 0°。
`strict_horizontal_goal_achieved` 只有五坡度都通过严格水平检查才会为 true。
旧模型作为候选已被新验收工具拒绝，接地/驻坡合格不能绕过水平目标。

本次离线检查记录在 `outputs/deformable_leveling_preflight_20261006/`。
它只验证源码、启动流程和 checkpoint 兼容性，不代表已经完成短训或实车验证。
