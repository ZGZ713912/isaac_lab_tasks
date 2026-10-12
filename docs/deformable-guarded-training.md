# Deformable 200 Hz 平地约束训练

本轮工作针对 2026-10-12 提出的“小训观察方向，有问题迭代、通过后大训”。
任务为 `Robotics-Deformable-Suspension-Guarded-History-MLP-Real2Sim-v4`。
旧 V3 任务、参考 checkpoint 和实车 Transformer ONNX 均保留。

训练从已有接地/调平特性的 H5 MLP 初始化，不转换 Transformer 权重。
参考 checkpoint 为 `2026-10-10_20-32-54_guarded_motion_specialized_v17_20261010_long_01/model_999.pt`，
SHA256 为 `2c2425519924706ca01b5e50e0c9e3293011da8936d68ec3c4b5b6e89863ec03`。
该 MLP 包含三个专家和冻结路由；本轮维持这个结构，允许专家学习新约束。

策略周期为 5 ms，物理/ADRC 周期为 1 ms。5 个历史帧仍每 10 ms 取样，
覆盖 40 ms，上一动作观测仍延迟 10 ms。PPO 每环境每更新采集 48 步，
物理时间仍为 240 ms。奖励按秒保持原尺度；gamma 与 lambda 改为原值的
平方根，保持折扣与 GAE 的物理时间尺度。

平地识别只用关节编码器、陀螺仪、比力、编码器里程计和初始 AHRS 方向。
支撑平面估计、匹配相位的比力/加速度滤波、角速度传播、3° 进入/3.5° 退出、
6 mm 共面误差与 100 ms 确认均与 RMCS 运行时对应。仿真在 200 Hz 读取
模拟传感器；RMCS 组件在 1 kHz 更新。模拟加速度计由运动状态生成比力，
真实世界速度、地形法线与接触力不加入 actor。法线和力只用于物理评估/
原有训练奖励；`training_flat_hold` 只供学习算法使用，不进入 actor/critic。

平地锁定时实际执行的四动作固定为零，四腿目标等于 17°，继续使用
2 rad/s 的目标速率限制。模式进出和环境重置清空历史与动作反馈。新增
损失同时让网络四腿各自学零动作，并在这些平地样本上关闭旧策略参考
约束，防止旧的 19° 姿态与新基准竞争。硬投影负责目标角保证，辅助损失
负责网络输出向基准收敛，两者的验收分开记录。

初始对照以 16 个姿态/工况并行测试静止、前进、横移、正反旋转、动态
运动、加速刹车和前后反向，各 12 s。实际完成的修正后报告在
`outputs/deformable_guarded_200hz_20261012/iteration_01/baseline_flat/`。
全部工况接地率 100%、平地锁定率 100%、锁定退出数 0，目标角误差 0。
实际关节误差 P95 最大 0.104°；反向等动态过程中存在约 0.25° 的瞬态
伺服误差。网络原始输出平均仍相当于偏离基准约 2.6–3.5°。
这些是仿真结果，不能替代实车机械角度与 IMU 标定验收。

第一份平地预检 `baseline_flat/` 曾用斜坡窗口边界检测裁剪完全平坦的
地面，前进样本出现两次人工窗口越界，物理终止为零。旧数据保留；修正
平地窗口检测后重新执行完整对照，不把旧报告当作通过记录。

短训预算为 128 环境、100 次 PPO 更新，种子 67。物理准入同时检查：
目标严格为 minangle、实际角误差、锁定覆盖、静止对角摆动、接地与终止。
另检查原始 actor 的平地误差是否减少至少 30% 或已低于 0.1°。
这些阈值在训练前固定。随后用相同种子、初始状态、时长和工况配对
复验 0/5/10/17/20° 的接地、倾斜与物理终止，保留陡坡原有能力限制。
第二个种子 2941 检查 0/2/2.9° 和带传感器/执行器随机性的平地。

100 次小训已完成，checkpoint 为
`logs/rsl_rl/deformable_guarded_mlp5_v4/2026-10-12_20-40-45_iteration_01_short/model_99.pt`，
SHA256 为 `eb2ef1ddecba87bf2ea793d1e73350d3601f6ba15ba435f34f3785fd26e076c1`。
24 个 actor 权重/偏置张量发生变化，最大参数变化 0.004487；路由保持不变。
前/后 10 次更新的平地学习损失均值为 0.31301/0.0487。
`iteration_01/short_assessment.json` 的平地/网络方向准入通过：静止时原始
actor 平均目标角误差由 3.155° 降至 0.0465°，其他工况降至 0.113–0.353°。
八工况的执行目标误差为 0、接地率 100%、锁定退出/终止为零，实际关节
误差 P95 最大 0.1104°。这项通过只表示平地小训方向正确；完整短训晋级
还需要配对坡面和第二种子的边界/噪声评估通过。

全部短训准入通过后，脚本才运行 512 环境、10,000 次大训练。每 1,000
次更新复查平地与配对坡面，失败即停止晋级、保留候选，不覆盖已通过的
模型。源文件快照、SHA256、每个子任务命令/日志、实际权重变化和评估
结果均记录在同一目录。实时进度以 `iteration_01/status.json` 为准；
本说明不把计划更新数当作已完成更新数。

工作命令：

```sh
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python -B -u \
  scripts/tools/deformable_guarded_train.py \
  --checkpoint logs/rsl_rl/deformable_real2sim_recovered_routed_mlp5_v3/2026-10-10_20-32-54_guarded_motion_specialized_v17_20261010_long_01/model_999.pt \
  --output-dir outputs/deformable_guarded_200hz_20261012/iteration_01 \
  --short-iterations 100 --short-envs 128 --iterations 10000 \
  --chunk-iterations 1000 --num-envs 512 --device cuda:0 --run
```

现有目录不能重复覆盖。查看状态/日志后选择新迭代目录。脚本不启动车辆、
不写回 RMCS，也不自动替换实车 ONNX。
