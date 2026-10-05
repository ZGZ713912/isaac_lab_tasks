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

## GUI play 固定坡度

键盘 play 也支持同一个 `--grade-deg` 覆盖。它把车辆放到 20 m 长上坡段的内部，
并在终端每 0.5 s 打印 `q`、`leg_target` 和 `q_cmd`，用来区分 reset 初值、策略动作
和目标限速造成的高度变化：

```bash
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python -B \
  scripts/rsl_rl/play_deformable.py \
  --task=Robotics-Deformable-Suspension-Rough-Keyboard-Play-History-Transformer-v2 \
  --checkpoint=logs/rsl_rl/deformable_foundation_precision_v2/2026-10-04_01-16-08_deformable_precision_20261003_tight_tilt/model_999.pt \
  --device=cuda:0 --num_envs=1 --grade-deg=20 \
  --vx_max=0.8 --vy_max=0.5 --wz_max=1.5 \
  --fixed_camera --debug_motion
```

该 play 入口的 20° 坡度是超出本次 Foundation 训练分布的压力测试；画面中的车身
不应被当作已经通过 20° 主动调平验收。

`run_gui.sh` 会使用邻近的 `IsaacLab-v2.3.2` 源码；其他安装位置可设
`ISAACLAB_PATH=/absolute/path/to/IsaacLab`。`--max_steps=200` 可用于有界检查。
省略 `--grade-deg` 时仍使用任务原来的周期坡面。

当前 minangle 基准是腿相对水平 **17°**，对应 URDF `q_cmd=1.01229 rad`。
`q` 越大车体越低；`Q_LOW=1.0563 rad` 是底盘净空限位，对应约 14.48°，
不等于 minangle。零动作目标等于基准，但策略可以输出残差抬腿，目标还有速率限制。
低车身软奖励约束 `max(q)` 向基准靠近，允许其余腿为调平而抬起，权重也不能保证
每条腿始终处于基准。**修改奖励不会改变已经冻结的模型动作。**

现在 play 的可见资产以基准腿角创建，并在加载策略前完成地形接触 reset；20°
无法水平四轮接触时，改用贴合坡面的初始化。2026-10-05 实测初始四腿均为
`1.01229 rad`，加载最新策略后静止命令下约为 `0.4–0.52 rad`：这部分抬高是
模型主动输出的结果，需要重新训练或改进策略，不能用换初始姿态代替。

Ctrl+C、SIGTERM、关闭终端产生的 SIGHUP 都会进入清理；窗口关闭也会触发清理。
play 禁用本机 IsaacLab 的 STOP 后无限渲染回调，退出超过 8 秒会结束本次 play
进程，再次发送终止信号可立即退出。兜底只作用于本次进程，不按进程名批量 kill。
实际 GUI 测试中执行 Hyprland 的 `closewindow`（Super+Q 对应的关闭动作）后正常
退出，exit code 为 0；Python 子进程测试还覆盖了持有 GIL 的原生死锁和清理卡住。
真实 Real2Sim play 在 50 步后收到 SIGINT，0.82 s 内退出，exit code 为 0。
此次没有遗留测试进程；SIGHUP 路径另有独立子进程测试覆盖。

## 2026-10-05 核实的训练结果

`logs/debug/deformable_precision_20261003/status.json` 为 `failed`。
两组 1000 轮精修都正常完成并保存 `model_999.pt`；三组 Foundation 评估也已完成。
流程选中 tight_tilt，但 `foundation_accepted=false`。下表取各组六个场景中的最差值，
每个场景 16 个环境、600 步；三组 Foundation 物理终止均为 0。

| 模型 | 最低四轮接触率 | 最低四轮接触且倾角 <3° 比例 | 最高倾角 P95 |
| --- | ---: | ---: | ---: |
| 原 model_9999 | 98.61% | 62.60% | 4.55° |
| low_entropy model_999 | 99.00% | 62.50% | 4.59° |
| tight_tilt model_999 | 98.64% | 62.61% | 4.67° |

精修没有显示出明确的整体改善。10° 原模型评估完成，17° 原模型评估被 SIGKILL
结束（exit -9，日志不能单独确定是谁发出的信号）；原流水线的 20° 评估未完成。
不要把预扫描几何参考或初始化 smoke 当作最新策略的大坡验收。

本轮另对最新 tight_tilt 模型补测固定 20° 坡面：4 环境、每场景 200 步（2 s），
去除最初 0.5 s settling，固定种子与名义参数；两场景均无物理终止。

| 场景 | 四轮接触率 | 倾角 P95 | 接触且 <3° 比例 | 世界坐标平均 vx |
| --- | ---: | ---: | ---: | ---: |
| 静止 | 100.00% | 21.47° | 0% | −0.028 m/s |
| 前进，命令 1 m/s | 98.67% | 21.92° | 0% | 0.755 m/s |

前进平均绝对滑移约 0.041 m/s、腿目标跟踪误差约 0.0118 rad、腿力矩饱和比例 0。
这说明短程内能贴坡行驶，但车身仍随坡倾斜，没有通过主动调平验收。该短程检查也
不能替代六场景、600 步的完整大坡评估。原始结果在
`outputs/deformable_real2sim_20261005/latest_policy20.json`。

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

## 基于扫频 CSV 的 Real2Sim

数据审计、当前模型的适用边界、独立训练任务和播放命令见
[deformable-real2sim.md](deformable-real2sim.md)。Real2Sim 当前是含未标定动力学先验的
实验分支，不能把其集成测试当作实车迁移能力证明。
