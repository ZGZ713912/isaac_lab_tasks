# RMUC2026 场地 play

地图已安装在仓库的 `assets/terrains/RMUC2026_V2_gray_fence_revision_03/`。
启动不依赖 Downloads，也无需修改 Isaac Sim 安装目录。
原下载目录仍保留；地图包整体复制，所有相对 USD 引用保留。

## 运行

在仓库根目录执行，每次只开一辆车：

```bash
# deformable：Guarded MLP v4，沿用已验证的第三人称驾驶
./play_rmuc.sh deformable

# wheel-leg：V4，跟随相机和键盘输入（默认 checkpoint 仅训练站立）
./play_rmuc.sh wheel-leg
```

也可以在其他目录使用 `play_rmuc.sh` 的绝对路径，启动器会切回仓库再运行。
默认使用 `/home/noir/miniconda3/envs/isaaclab/bin/python`、CPU 单车及 50 Hz 仿真渲染。
用 `--python PATH` 或 `ISAACLAB_PYTHON` 指定其他兼容环境。
CPU 仿真仍需要 NVIDIA GPU 渲染。

## 指定 ONNX 或 PT 模型

同一个脚本选择车型，`--checkpoint` 后直接传模型路径。后缀为 `.onnx` 时使用 ONNX Runtime，
后缀为 `.pt` 时沿用 checkpoint 推理。ONNX 模式不读取 `.pt` 权重。

```bash
./play_rmuc.sh deformable --checkpoint /absolute/path/to/model.onnx
./play_rmuc.sh wheel-leg --checkpoint /absolute/path/to/model.onnx
```

也可把 `--checkpoint` 写成 `--onnx`（或 `--model`）；这两种写法专用于 ONNX。
例如，当前已验收 deformable 大训练模型对应的 ONNX：

```bash
./play_rmuc.sh deformable --checkpoint \
  logs/rsl_rl/deformable_guarded_mlp5_v4/2026-10-14_12-40-18_training_long_05/exported/policy.onnx

./play_rmuc.sh wheel-leg --checkpoint \
  logs/rsl_rl/wheel_leg_v4_v5_230/2026-09-22_23-32-52/exported/model_19999.onnx
```

模型需要配套的环境配置，因为 ONNX 张量维度本身不能说明电流/力矩语义、动作零点和历史时序。
通常维持原训练目录即可自动找到，无需额外参数：

```text
run/
  exported/model.onnx
  params/env.yaml
  params/agent.yaml
  params/real2sim_model.json   # Real2Sim deformable 需要
```

也支持 `model.onnx` 与 `params/` 在同一层。单独放置模型时显式指定对应的 run 或 params 目录：

```bash
./play_rmuc.sh deformable --checkpoint /other/location/model.onnx \
  --policy-config /absolute/path/to/matching/run
```

可以另加 `--task TASK_ID` 选择同车型的其他兼容任务。默认 deformable 使用 Guarded v4、
wheel-leg 使用 V4 230D；不能把其他车、其他动作/观测合同的模型直接混用。
deformable 支持环境现有 32D 帧的 MLP 扁平历史和 Transformer rank-3 历史；
wheel-leg V4 支持 5×46D 历史。输入/输出名称由 ONNX 自动读取，均须为 float32、单输入单输出。
观测历史由环境按原时序维护，ONNX 适配器只重排为模型需要的维度，不再堆叠或缓存历史。
有额外隐藏状态输入的模型和缺少明确归一化合同的归一化模型会报错。
推理使用一个 CPU 线程；动作裁剪仍沿用保存的配置，车辆控制、相机和按键与 PT 模式相同。
如果 ONNX 使用外部 `.data` 权重文件，需保持其原相对路径。

首次使用某个路径时可追加 `--dry-run` 核对命令；追加 `--max_steps 1000` 可有界运行并自动退出。

wheel-leg 目录原有的 `exported/policy.onnx` 与 `model_19999.pt` 数值不一致，已保留原件；
上面的 `model_19999.onnx` 是从该 checkpoint 另行导出并验证的文件。

默认 checkpoint 固定到以下文件，启动器不会搜索或改用另一个模型：

| 车辆 | 任务 | checkpoint |
| --- | --- | --- |
| deformable | `Robotics-Deformable-Suspension-Guarded-History-MLP-Real2Sim-v4` | `logs/rsl_rl/deformable_guarded_mlp5_v4/2026-10-14_12-40-18_training_long_05/model_999.pt` |
| wheel-leg | `Robotics-Wheel-Leg-V4-Flat-Play-v0` | `logs/rsl_rl/wheel_leg_v4_v5_230/2026-09-22_23-32-52/model_19999.pt` |

使用同任务的其他训练结果：

```bash
./play_rmuc.sh deformable --checkpoint /absolute/path/to/run/model_XXXX.pt
./play_rmuc.sh wheel-leg --checkpoint /absolute/path/to/v4/run/model_XXXX.pt
```

checkpoint 必须保留相邻的 `params/agent.yaml` 和 `params/env.yaml`；deformable 还需其原有 real2sim 模型文件。
入口会验证任务、观测/动作与控制时序合同。wheel-leg 的这个入口专用于 V4 的 230D 观测、6D 动作模型，
其他 wheel-leg 版本需要对应的环境适配，不能混用 checkpoint。
场地入口只运行推理，不训练或覆盖已有模型。

注意：当前默认 wheel-leg V4 checkpoint 的保存配置为 `rel_standing_envs=1.0`，
线速度与角速度的训练范围均为 0，所以这里只验收它在地图上的加载和站立回放。
键盘指令可以送入策略，但该 checkpoint 没有接受巡航指令训练，不能保证跟踪移动指令。
deformable 默认模型可用于地图驾驶；若要查看其他 V4 训练结果，使用上面的 `--checkpoint` 参数。

## 操作

先点击 Isaac Sim 场地视口。

| 车辆 | 平移与转向 | 其他 |
| --- | --- | --- |
| deformable | W/S 前后，A/D 左右平移；按住 Q/E 左右转车头，松开后完成最后朝向 | 相机跟随实际车头；右键鼠标也可转向/调俯仰；滚轮缩放；L 停止；R 重置；Esc 释放输入 |
| wheel-leg V4 | W/S 前后，A/D 左右转向 | Z/X 调车高；L 停止；R 重置；Esc 清空指令 |

wheel-leg 使用两轮差速转向。跟随相机位于实际车头后方，保持世界竖直，遇到已有碰撞体自动缩短距离。
wheel-leg 可用 `--camera-mode fixed` 固定相机。
deformable 的完整第三人称控制说明见 [deformable 场地驾驶](deformable-rmuc-play.md)。

两种车均可追加 `--spawn-xy X Y --spawn-yaw DEG`。
坐标是资产原始世界米制坐标，地面包含负 Z 和轻微倾斜，启动器不会平移、缩放地图。
默认在主地面寻找全部车轮有支撑且周围无遮挡的点，按轮心几何和半径设置初始位置。
无法支撑的指定点会报错。

## 场地与传感器

实际加载 `asset_package.json` 指定的静态碰撞 USD。
加载按 Isaac Lab 的 [USD 资产工作流](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html) 引用资产，
只保留框架的一个 PhysicsScene，不写回原始 USD。

地面、坡面和围栏继承各自任务原有的物理材质。
deformable 沿用实际碰撞网格的轮胎/悬挂查询。
wheel-leg 将全部已启用的碰撞三角形合并成一个不可见、没有物理碰撞的高度查询 Mesh，
使本机的 [RayCaster](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.sensors.html#ray-caster) 能通过单 Mesh 看到完整地图；实际法向接触和轮胎摩擦仍由场地 PhysX 碰撞体提供。
车高改为相对当前位置的真实地面，避免把地图的负 Z 偏移当作车高变化。
轮腿的控制器、动作尺度、物理步长和历史帧数保留原任务值。

地图的碰撞覆盖仍然不完整：资产有 376 个启用碰撞 Mesh，部分装饰物可能被穿过。
接入地图和检查出生点不等于验证了全场所有障碍，也不代表平地训练策略已学会越障。

## 有界验证

查看实际命令而不启动仿真：

```bash
./play_rmuc.sh deformable --dry-run
./play_rmuc.sh wheel-leg --dry-run
```

短回放并自动退出：

```bash
./play_rmuc.sh deformable --max_steps 800 --benchmark_motion --profile \
  --profile_output outputs/rmuc_field_install_20261016/deformable-replay.json

./play_rmuc.sh wheel-leg --max_steps 600 \
  --profile_output outputs/rmuc_field_install_20261016/wheel-leg-replay.json
```

wheel-leg 还可使用 `--command VX WZ HEIGHT` 固定指令；指定后不启用键盘控制。
`--capture-output PATH.png` 保存最后的 GUI 场景。
各进程退出时会关闭环境及 SimulationApp；退出超时监控只处理本次 play 的进程。

2026-10-16 的资产校验和实测记录位于 `outputs/rmuc_field_install_20261016/`。
汇总为 `validation.json`，失败的早期启动日志保留，不计入通过结果。

| 项目 | 结果 |
| --- | --- |
| 地图安装 | 910 个文件、约 866 MiB；原包与项目副本 SHA256 全部一致 |
| 针对性回归 | 30 项通过，含双轮/四轮出生、Q/E 转向、相反键抵消、失焦清空、相机与退出处理 |
| wheel-leg V4 GUI | 600 步、6 s 仿真，0.499×；无自动重置，正常退出；只验收加载和站立回放 |
| deformable 键盘 GUI | 2,400 步、12 s 仿真，0.199×，约 9.96 次渲染更新/墙钟秒；正常退出 |
| Q/E 转向 | 已验证跨 ±180°、松键后停止累加并完成目标；正反转最终误差均约 0.99° |
| 相机与 W 方向 | 相机水平朝向跟随实际车头；W 位移 0.629 m，与车头偏差 0.005° |
| 接触与 minangle | 每 0.25 s 记录的 48 个样本均四轮接触，四腿目标严格 17°；允许 2 s 回落后的实际角误差 P95 0.100° |
| 碰撞与重置 | 地面、坡面、围栏、无支撑及无效出生点检查通过；R 清空历史与驱动，无额外环境重置 |
| 输入与进程 | Kit 虚拟设备验证 Q/E 路由，默认导航收到驾驶事件数为 0；结束后无 play、训练或退出监控残留 |

输入接口通过程序注入验证，未量化人手操作手感；渲染更新频率也不等于屏幕实际呈现帧率。

### ONNX 场地验证（2026-10-16）

`outputs/rmuc_onnx_play_20261016/validation.json` 汇总直接 ONNX 推理记录：

- 50 项针对性测试通过，包含扁平/rank-3 历史顺序、TensorDict 仅取 actor 观测、两车路径选择，
  以及错误维度、历史、观测语义、动作时序、额外状态和非有限值拒绝。
- 每个模型与对应 PT 比较 33 组输入；deformable 最大误差 `7.16e-7`，wheel-leg 最大误差 `7.16e-6`。
- 两个测试目录只含 ONNX 和配套 params，没有 PT 文件；启动使用 `--checkpoint model.onnx`，
  运行报告均为 `onnxruntime` 后端，推理调用数与策略步数一致。
- deformable 完成 1,000 步 / 5 s 仿真，0.205×；WASD/QE、L、R 回放通过，
  20 个记录点均四轮接触且四腿目标为 17°，历史重置检查通过。
- wheel-leg V4 完成 400 步 / 4 s 仿真，0.508×，没有自动重置；此模型仍只验收站立回放。
- 两次 GUI 正常退出，结束后无 play、训练或退出监控进程残留。

ONNX Python 接口参考 [ONNX Runtime 官方文档](https://onnxruntime.ai/docs/api/python/api_summary.html)。
