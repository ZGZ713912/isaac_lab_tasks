# wheeled-legged_RL 脚本与命令行参数总览

> 本文档整理仓库 `scripts/` 目录（以及根目录 `run_gui.sh`、`ubuntu_setup/`）中所有脚本的用途，以及它们支持的全部 `--` 命令行参数。可直接复制到飞书文档。

---

## 一、脚本总览

### 1. scripts/rsl_rl/ — 强化学习核心

| 脚本 | 用途 |
| --- | --- |
| `train.py` | 训练入口。启动 Isaac Sim，按 `--task` 从 registry 加载 env/agent 配置，`gym.make` 建环境并包成 RSL-RL VecEnv，创建 runner 后执行 `runner.learn()`。支持 `--checkpoint` 权重初始化（finetune）或 `--resume_training` 真续训，并把配置 dump 成 `logs/.../params/env.yaml`、`agent.yaml`。 |
| `play.py` | 推理 / 演示入口。加载 `--checkpoint` 跑策略，支持键盘控制、录像、实时绘图、滑移诊断、地形识别打印，并把策略导出为 `exported/policy.pt` / `policy.onnx`（BarlowTwins / DreamWaq / HIM / NP3O 各有导出分支）。 |
| `cli_args.py` | 参数定义与配置覆盖工具（被 train / play / eval 复用）。注册 `--experiment_name`、`--run_name` 等 RSL-RL 参数，并提供 `update_rsl_rl_cfg()` 把命令行值写回 agent 配置（含 CMoE / MoE 超参覆盖）。 |
| `keyboard_controller.py` | 自定义键盘设备 `Se3KeyboardMobile`。W/S 前后速度、A/D 转向、Z/X 增量调高、Q 触发一次手动跳跃窗口、L 重置；兼容 manager 与 direct 两种环境模式，把命令写入 `command` / `height_cmd`。 |

### 2. scripts/ — 辅助入口

| 脚本 | 用途 |
| --- | --- |
| `list_envs.py` | 列出所有已注册的 `Robotics-*` 任务（任务名、entry point、cfg），用来确认 `--task` 该填什么。 |
| `view_robot.py` | 不放策略，用零动作步进环境，纯看模型 / 场景是否加载正常（打印 `env.observation_space` / `action_space`）。其中实验性的 `_spawn_physics_balls_in_envs`（随机物理球）当前被注释。 |
| `eval_checkpoint.py` | 无头定量评估。跑 checkpoint 统计：episode 长度、存活率（time_out 占比）、平均 reward、速度跟踪误差 vx / vy / omega_z、平均车体高度，并输出终端表格 + CSV。 |

### 3. scripts/tools/ — 资产与 URDF

| 脚本 | 用途 |
| --- | --- |
| `prepare_wheel_leg_v1_urdf.py` | 把 SolidWorks 导出的 `Wheel_leg_V1` URDF 规范化到 RL 约定：根坐标绕 Z +90 度、统一左右关节正方向、修正 `effort` / `velocity` 限幅；可选把轮子碰撞体换成解析圆柱。 |
| `check_wheel_leg_v1.py` | Wheel_leg_V1 标定探针（不训练）：检查 USD 关节 / 连杆 / 质量惯量、零位姿态；按指定或网格扫描 `(q1, q2)` 站立姿态，看能存活多少步，输出 stdout + JSON。 |
| `convert_wheel_leg_urdf.sh` | 调 IsaacLab 的 `convert_urdf.py` 把 `urdf_V4.0.urdf` 转成 `Wheel_leg_V1.usd`，并校验 `base.usd` 是否真的含网格（防止只导出 3.5KB 空骨架）。 |
| `convert_wheel_leg_urdf_cylinder.sh` | 圆柱轮碰撞体变体转 USD（独立目录 `Wheel_leg_V1_cylwheel/`），供解析圆柱 vs convex hull 的 A/B 对比。 |
| `convert_deformable_urdf.sh` | 把 `deformable_infantry` URDF 转成 `deformable_suspension.usd`。 |

### 4. scripts/utils/ — 可视化

| 脚本 | 用途 |
| --- | --- |
| `velocity_trace_html.py` | 纯 Python 生成自包含 HTML：速度 / 偏航 / 高度曲线 + reward 正负热力图，支持缩放、拖拽、悬浮查看。 |
| `export_velocity_trace_html.py` | 命令行把 trace CSV 转成上面的 HTML。 |
| `realtime_plotter.py` | play 模式下的 matplotlib 实时绘图（高度轨迹、轮电机功率、腿关节力矩、弹簧力）。 |

### 5. 根目录 / 环境脚本

| 脚本 | 用途 |
| --- | --- |
| `run_gui.sh` | 本机 GUI 启动包装：补 `libxml2.so.2` 兼容链、清空被 isaacgym 污染的 `LD_LIBRARY_PATH`、去掉 `WAYLAND_DISPLAY` 强制走 X11。用法：`./run_gui.sh python scripts/rsl_rl/play.py ...`。 |
| `ubuntu_setup/00_system.sh` | 系统 + NVIDIA 驱动安装（A10 无头服务器）。 |
| `ubuntu_setup/01_conda.sh` | 建 conda 环境、装 Isaac Sim / Lab / RSL-RL 依赖。 |
| `ubuntu_setup/02_fixes.sh` | 重建被 `.gitignore` 误伤的 `agent_rl/rsl_rl/env/` 包，并给入口脚本补 `sys.path`。 |
| `ubuntu_setup/03_smoke.sh` | 无头冒烟验证：查 GPU、列任务、64 env x 2 iter 试训。 |
| `ubuntu_setup/04_train.sh` | 正式无头训练 + 提示评估命令。 |

---

## 二、所有 `--` 参数

### 1. train.py 与 play.py 共用（来自 `cli_args.py` 的 rsl_rl 组）

| 参数 | 含义 |
| --- | --- |
| `--experiment_name` | 日志实验目录名（`logs/rsl_rl/<name>/`）。 |
| `--run_name` | 日志目录后缀，格式 `{时间戳}_{run_name}`。 |
| `--resume` | 占位参数，未接续训逻辑。 |
| `--load_run` | 占位参数，未接续训逻辑。 |
| `--checkpoint` | 要加载的 `.pt` 路径。 |
| `--logger` | 日志器：`wandb` / `tensorboard` / `neptune`。 |
| `--log_project_name` | wandb / neptune 项目名。 |
| `--cmoe_router_temperature` | [CMoE] 覆盖策略 `moe_router_temperature`。 |
| `--cmoe_aux` | [CMoE] 覆盖 `cmoe_expert_value_loss_coef`（per-expert value 辅助损失）。 |
| `--moe_load_balancing_coef` | [MoE / CMoE] 覆盖负载均衡系数。 |
| `--clip_actions` | 覆盖 agent 动作裁剪幅度。 |

> 说明：`--seed` 由 train / play 各自定义；`update_rsl_rl_cfg` 中当 `seed == -1` 时随机取 0 到 10000。

### 2. train.py 专属

| 参数 | 含义 |
| --- | --- |
| `--task` | 任务名（如 `Robotics-Wheelbipe-V14-Flat-v0`）。 |
| `--num_envs` | 并行环境数。 |
| `--seed` | 随机种子。 |
| `--max_iterations` | 训练迭代数。 |
| `--resume_training` | 真续训：从 `--checkpoint` 恢复优化器和迭代计数并接着训。 |
| `--video` | 训练中录像。 |
| `--video_length` | 录像长度（默认 200 步）。 |
| `--video_interval` | 录像间隔（默认 2000 步）。 |
| `--distributed` | 多 GPU / 多节点训练。 |

### 3. play.py 专属

| 参数 | 含义 |
| --- | --- |
| `--task` | 任务名。 |
| `--num_envs` | 并行环境数。 |
| `--video` | 录制一个视频。 |
| `--video_length` | 录制长度。 |
| `--real-time` | 尽量按真实时间步进。 |
| `--disable_fabric` | 禁用 fabric，走 USD I/O。 |
| `--keyboard` | 启用键盘控制（单环境、关 timeout、开手动跳跃）。 |
| `--keyboard_episode_length_s` | 键盘模式覆盖 episode 长度（秒）。 |
| `--plot` | 启用 matplotlib 实时曲线（仅单环境）。 |
| `--expand_obs_dims` | 把观测 pad 成四维后导出 ONNX。 |
| `--max_steps` | 跑多少步后停止，0 表示直到关闭。 |
| `--slip_debug` | 打印轮地滑移诊断。 |
| `--slip_debug_interval` | 滑移打印间隔步数。 |
| `--slip_wheel_radius` | 诊断用轮半径。 |
| `--slip_speed_threshold` | 判定为滑移的接触点水平速度阈值。 |
| `--barlow_twins_jit` | BarlowTwins TorchScript 路径，仅推理 `MlpBarlowTwinsActor`。 |
| `--dreamwaq_print_code_vel` | DreamWaq 下打印 cenet 的 `code_vel`。 |
| `--dreamwaq_code_vel_interval` | 上述打印间隔。 |

### 4. AppLauncher 标准参数（Isaac Lab 提供，train / play / view_robot / eval 自动带）

| 参数 | 含义 |
| --- | --- |
| `--headless` | 无头运行（无显示器训练必加）。 |
| `--device` | 设备，如 `cuda:0` / `cpu`。 |
| `--enable_cameras` | 启用相机（录像时自动开启）。 |
| `--livestream` | 直播模式：`0` / `1` / `2`。 |
| `--rendering_mode` | 渲染模式：`quality` / `performance`。 |
| `--width` / `--height` | 视口分辨率。 |
| `--name` | 应用名称。 |
| `--verbose` / `--info` | 日志详细程度。 |
| `--experience` | 指定 kit experience 文件。 |
| `--kit_args` | 透传给 Kit 的额外参数。 |
| `--xr` | 启用 XR。 |
| `--visualizer` | 可视化器后端（如 `kit` / `newton` / `rerun`）。 |
| `--cpu` | 强制使用 CPU。 |

### 5. eval_checkpoint.py

| 参数 | 含义 |
| --- | --- |
| `--task` | 任务名（默认 `Robotics-Wheelbipe-V14-Flat-Play-v0`）。 |
| `--checkpoint` | checkpoint 路径（必填）。 |
| `--num_envs` | 并行环境数。 |
| `--episodes` | 评估 episode 总数。 |
| `--max_steps` | 每个 episode 最大步数，0 表示用环境默认长度。 |
| `--seed` | 随机种子。 |
| `--csv` | CSV 输出路径（默认 `logs/debug/eval_<时间戳>.csv`）。 |

> 另外强制 headless，并自动带 AppLauncher 标准参数。

### 6. view_robot.py

| 参数 | 含义 |
| --- | --- |
| `--task` | 任务名。 |
| `--num_envs` | 并行环境数。 |
| `--disable_fabric` | 禁用 fabric，走 USD I/O。 |
| `--num_balls` | 生成物理球数量（功能当前被注释）。 |
| `--ball_radius_min` / `--ball_radius_max` | 球半径范围，单位 m。 |
| `--ball_spawn_height` | 球生成高度，单位 m。 |
| `--ball_grid_spacing` | 球网格间距，单位 m。 |
| `--ball_density` | 球材料密度，单位 kg/m^3。 |
| `--ball_friction` | 球摩擦系数。 |
| `--ball_restitution` | 球弹性恢复系数。 |

### 7. check_wheel_leg_v1.py

| 参数 | 含义 |
| --- | --- |
| `--task` | 要探测的任务名（默认 `Robotics-Wheel-Leg-V1-Flat-v0`）。 |
| `--num_envs` | 环境数（建议保持 1，便于读数）。 |
| `--steps` | 每个姿态仿真步数。 |
| `--settle` | 首次测量前的落地步数。 |
| `--pose NAME=VALUE` | 关节姿态覆盖，可重复，如 `--pose L_joint1=-1.06`。 |
| `--grid` | 扫描一组 `(q1, q2)` 姿态。 |
| `--seed` | 随机重置种子。 |
| `--out` | JSON 输出路径（默认 `logs/debug/wheel_leg_v1_probe.json`）。 |

### 8. prepare_wheel_leg_v1_urdf.py

| 参数 | 含义 |
| --- | --- |
| `--wheel-collision` | 轮子碰撞体类型：`mesh`（STL 网格，默认）或 `cylinder`（解析圆柱）。 |

### 9. export_velocity_trace_html.py

| 参数 | 含义 |
| --- | --- |
| `csv_path` | 位置参数，输入的 trace CSV 路径。 |
| `-o` / `--html-path` | 输出 HTML 路径（默认与 CSV 同名 `.html`）。 |
| `--reward-signs-json` | 提供 reward 列正负号或 cfg reward scales 的 JSON。 |

### 10. convert_*.sh 内部调用的 IsaacLab convert_urdf.py

| 参数 | 含义 |
| --- | --- |
| `--joint-target-type none` | 脚本内固定传入，表示不做关节 drive 目标。 |
| 其余参数 | 通过 `"$@"` 透传给 `convert_urdf.py`。 |

---

## 三、常用命令速查

```bash
# 训练（无头）
conda activate isaaclab
PYTHONPATH=$PWD python scripts/rsl_rl/train.py \
    --task=Robotics-Wheelbipe-V14-Flat-v0 \
    --num_envs=4096 --max_iterations=20000 --headless --device=cuda:0

# 续训
PYTHONPATH=$PWD python scripts/rsl_rl/train.py \
    --task=Robotics-Wheelbipe-V14-Flat-v0 \
    --checkpoint=logs/rsl_rl/<exp>/<ts>/model_1000.pt \
    --resume_training --headless

# 定量评估（无头）
python scripts/eval_checkpoint.py \
    --task=Robotics-Wheelbipe-V14-Flat-Play-v0 \
    --checkpoint=logs/rsl_rl/<exp>/<ts>/model_1999.pt \
    --num_envs=16 --episodes=200 --device=cuda:0 --headless

# 本机 GUI + 键盘演示
./run_gui.sh python scripts/rsl_rl/play.py \
    --task=Robotics-Wheelbipe-V14-Flat-Play-v0 \
    --num_envs=1 --checkpoint=<model.pt> --keyboard

# 列出所有任务
python scripts/list_envs.py

# 看曲线
tensorboard --logdir logs/rsl_rl/wheelbipe_v14_2_flat_direct
```

---

## 四、注意事项

- `.script` 目录不存在，本文档对应的是仓库 `scripts/` 目录。
- `train.py` 的 `--resume` 与 `--load_run` 目前是占位参数，真正续训用 `--resume_training` + `--checkpoint`。
- 无显示器（A10）训练必须加 `--headless`；本机 GUI 用 `run_gui.sh` 包装启动。
- `log_project_name` 仅在 `--logger` 为 `wandb` 或 `neptune` 时生效。
