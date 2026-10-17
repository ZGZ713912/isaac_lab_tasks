# wheeled-legged_RL

26 赛季轮腿步兵强化学习训练仓库。基于 **Isaac Sim + Isaac Lab + RSL-RL** 的轮腿（wheelbipe）机器人强化学习训练代码。

包含：

- **任务**：Wheelbipe V14 系列平地 / 粗糙地形任务（含 Play 配置）
- **算法**：普通 PPO、[DreamWaQ](https://github.com/Manaro-Alpha/DreamWaQ)、HIMLoco（HIM）、NP3O（BarlowTwins）
- **状态机**：腾空-落地状态机（airborne / jump_takeoff / step_up / stair）、小陀螺平移模式
- **脚本**：训练、play（键盘控制 / 实时可视化 / 速度轨迹录制）、环境列表
- **预训练模型**：`models/pretrained/` 下的示例权重

## Features

本软件面向轮腿（wheelbipe）机器人的强化学习训练与部署，主要功能：

- **多任务训练**：Wheelbipe V14 系列平地 / 粗糙地形训练与 Play 任务（见 [Available Tasks](#available-tasks)）。
- **多种 RL 算法**：内置 PPO、DreamWaQ（隐式地形想象）、HIMLoco（历史轨迹估计）、NP3O（BarlowTwins + 安全约束）。
- **状态机**：腾空-落地状态机（airborne / jump_takeoff / step_up / stair）与小陀螺平移模式。
- **训练 / 推理脚本**：一键训练（`train.py`）、键盘控制 play（`play.py --keyboard`）、实时可视化（`--plot`）。
- **效果分析**：速度 / reward 轨迹录制，导出 CSV 与交互式 HTML。
- **部署导出**：策略导出为 TorchScript（`.pt`）/ ONNX，便于实机部署。
- **预训练模型**：`models/pretrained/` 下提供示例权重。

## 安装与环境

统一安装和环境要求见[项目首页](../../README.md#安装)。在现有 Isaac Lab 环境中，从仓库根目录执行 `python -m pip install --no-deps -e .`；无需分别安装三个包。

以下命令均从仓库根目录运行。历史预训练权重位于本地 `models/pretrained/`，不随 clone 提供。

## Usage

### List RL tasks

```bash
python scripts/list_envs.py
```

### Environment visualisation

```bash
python scripts/view_robot.py --task=<task_name> --num_envs=1 --device=cpu

# V14 rough play 可视化
python scripts/view_robot.py --task=Robotics-Wheelbipe-V14-Rough-Play-v0 --num_envs=1
```

### Train

```bash
python scripts/rsl_rl/train.py --task=<task_name> --num_envs=4096 --max_iterations=20000 --device=cuda:0 --headless
```

### Play

```bash
# 基础模式
python scripts/rsl_rl/play.py --task=<task_name> --num_envs=16 --checkpoint=<model_path>

# 键盘控制（W/S/A/D 控制，Z/X 调高度，Q 触发跳跃）
python scripts/rsl_rl/play.py --task=<task_name> --num_envs=1 --checkpoint=<model_path> --keyboard

# 实时数据可视化（需要 matplotlib）
python scripts/rsl_rl/play.py --task=<task_name> --num_envs=1 --checkpoint=<model_path> --keyboard --plot

# V14 rough play 推荐查看命令
python scripts/rsl_rl/play.py --task=Robotics-Wheelbipe-V14-Rough-Play-v0 --num_envs=1 --checkpoint=<model_path> --keyboard
```

### Play on the RMUC2026 field

```bash
./play_rmuc.sh deformable  # Guarded MLP v4，WASD 移动、Q/E 转向，第三人称跟随
./play_rmuc.sh wheel-leg   # Wheel-leg V4，跟随相机；默认 checkpoint 仅训练站立

# 用同一入口指定 ONNX（也接受 --onnx PATH），不读取 .pt 权重
./play_rmuc.sh deformable --checkpoint /absolute/path/to/model.onnx
./play_rmuc.sh wheel-leg --checkpoint /absolute/path/to/model.onnx
```

地图安装在 `assets/terrains/RMUC2026_V2_gray_fence_revision_03/`，已脱离 Downloads。
模型选择、操作及回放验证见 [RMUC 场地 play](../rmuc-field-play.md)。

### Record V14 rough speed/reward curves

`WheelbipeV14RoughEnvCfg_v1_Play` 中已启用 `velocity_trace_cfg`，会在 play 时自动选择指定地形中的一个 agent，记录速度、高度和当前配置下的 reward contribution，并导出 CSV 与交互式 HTML。

```bash
# 录制 10s 速度轨迹；环境步长为 0.02s，因此 10s 对应 --max_steps=500
python scripts/rsl_rl/play.py \
  --task=Robotics-Wheelbipe-V14-Rough-Play-v1 \
  --num_envs=64 \
  --checkpoint=<model_path> \
  --device=cuda:0 \
  --headless \
  --max_steps=500
```

输出文件默认在 `logs/debug/` 下，文件名自动带时间戳和 pid：

```text
logs/debug/rough_v1_play_velocity_trace_<timestamp>_pid<pid>.csv
logs/debug/rough_v1_play_velocity_trace_<timestamp>_pid<pid>.html
```

记录列包括：

- 基础信息：`sim_time_s`、`episode_time_s`、`env_id`、`terrain`、`airborne`
- 速度：`cmd_x/cmd_y/cmd_yaw`、`vel_x_b/vel_y_b/yaw_rate_b`
- 高度：`height_cmd`、`height_obs`、`height_relative`、`height_reward_ref`
- reward：`reward_total` 和所有 `reward_<term>` 列

HTML 会把速度/角速度、高度和 reward heatmap 分成三块展示；鼠标 hover 联动所有图表。也可以从 CSV 离线重新导出：

```bash
python scripts/utils/export_velocity_trace_html.py \
  logs/debug/rough_v1_play_velocity_trace_<timestamp>_pid<pid>.csv
```

高度观测可选裁剪，默认关闭，配置项为 `height_obs_clip_enabled` 和 `height_obs_clip_range`。更多实现说明见 `scripts/utils/velocity_trace_html.py` 与 `WheelbipeV14RoughEnvCfg_v1_Play.velocity_trace_cfg`。

## Available Tasks

| 任务 | 说明 |
| --- | --- |
| `Robotics-Wheelbipe-V14-Flat-v0` | 平地 PPO |
| `Robotics-Wheelbipe-V14-Flat-v1` | 平地 PPO + 落地预训练 |
| `Robotics-Wheelbipe-V14-Flat-v2` | 平地 PPO + 小陀螺平移 |
| `Robotics-Wheelbipe-V14-Flat-Play-v0` | 平地 PPO Play |
| `Robotics-Wheelbipe-V14-Flat-Play-v2` | 平地 PPO + 小陀螺平移 Play |
| `Robotics-Wheelbipe-V14-Rough-v0` | 粗糙地形 PPO + 小陀螺 |
| `Robotics-Wheelbipe-V14-Rough-v1` | 粗糙地形 PPO + 跑场 |
| `Robotics-Wheelbipe-V14-Rough-Play-v0` | 粗糙地形 PPO + 小陀螺 Play |
| `Robotics-Wheelbipe-V14-Rough-Play-v1` | 粗糙地形 PPO + 跑场 Play |
| `Robotics-Wheelbipe-V14-Flat-DreamWaQ-v0` | 平地 DreamWaQ |
| `Robotics-Wheelbipe-V14-Flat-DreamWaQ-Play-v0` | 平地 DreamWaQ Play |
| `Robotics-Wheelbipe-V14-Flat-HIM-v0` | 平地 HIMLoco（HIM） |
| `Robotics-Wheelbipe-V14-Flat-HIM-Play-v0` | 平地 HIMLoco Play |
| `Robotics-Wheelbipe-V14-Flat-NP3OBarlow-v0` | 平地 NP3O（BarlowTwins） |
| `Robotics-Wheelbipe-V14-Flat-NP3OBarlow-Play-v0` | 平地 NP3O Play |

## Algorithms

- **PPO**（common）：`src/agent_rl/rsl_rl/`
- **DreamWaQ**：`algorithms/ppo_dreamwaq.py` + `modules/actor_critic_dreamwaq.py` + `runners/on_policy_runner_dreamwaq.py`
- **HIMLoco（HIM）**：`algorithms/ppo_him.py` + `modules/actor_critic_him.py` + `modules/him_estimator.py`
- **NP3O（BarlowTwins）**：`algorithms/np3o.py` + `modules/actor_critic_balowtwins.py`

## State Machines

位于 `src/agent_tasks/direct/wheelbipe/state_machines/`，由 `manager/mdp/state_machine/` 驱动：

- **腾空-落地状态机**：`airborne.py` / `jump_takeoff.py` / `step_up.py` / `stair.py`
- **小陀螺平移模式**：平移状态下保持自旋、移动稳定

## Pretrained Examples

`models/pretrained/` 下提供示例权重（模型文件为 `.pt`，用 `torch.load` 可查看结构）。26 赛季机器人的预训练模型位于 `models/pretrained/26_infantry/`：

```bash
# 平地示例
python scripts/rsl_rl/play.py --task=Robotics-Wheelbipe-V14-Flat-Play-v0 --num_envs=1 --checkpoint=./models/pretrained/26_infantry/flat_and_rotation/<model>.pt --keyboard

# V14 rough play 查看
python scripts/view_robot.py --task=Robotics-Wheelbipe-V14-Rough-Play-v0 --num_envs=1
```

## 项目结构

目录职责、三包边界及旧路径迁移表见[项目架构](../development/architecture.md)。

## System Architecture & Data Flow

### 软硬件系统框图

```mermaid
flowchart TB
    subgraph HW["机器人硬件"]
        ACT["轮 / 腿执行器"]
        SENS["IMU / 编码器"]
        CTRL["控制器 / 上位机"]
    end

    subgraph TRAIN["仿真训练"]
        SIM["Isaac Sim"] --> ENV["Isaac Lab 环境（agent_tasks）"]
        ENV <--> ALG["RL 算法（agent_rl）"]
        ALG --> CKPT["策略权重（.pt）"]
    end

    subgraph DEPLOY["实机部署"]
        CKPT --> INF["推理（.pt / .onnx）"]
        INF --> CTRL
        CTRL --> ACT
        SENS --> CTRL
    end
```

### 训练数据流

```mermaid
flowchart LR
    OBS["观测 obs"] --> POL["策略网络 policy"]
    POL --> A["动作 action"]
    A --> E["环境 / 机器人"]
    E --> R["奖励 reward"]
    R --> BUF["Rollout 存储"]
    BUF --> UPD["PPO 更新"]
    UPD --> POL
```

## Software Architecture

```mermaid
flowchart TB
    subgraph APP["应用层"]
        TRAIN["train.py"]
        PLAY["play.py"]
    end
    subgraph ALG2["算法层"]
        PPO["PPO / DreamWaQ / HIM / NP3O"]
        RUN["OnPolicyRunner"]
        MOD["actor-critic / 估计器"]
    end
    subgraph ENV3["环境层"]
        TASK["agent_tasks：任务 / 状态机"]
        WORLD["agent_world：资产 / 地形 / 执行器"]
    end
    subgraph BASE["基础设施层"]
        ISL["Isaac Lab"]
        ISS["Isaac Sim"]
    end
    TRAIN --> PPO
    PLAY --> RUN
    PPO --> RUN
    RUN --> MOD
    RUN --> TASK
    TASK --> WORLD
    WORLD --> ISL
    ISL --> ISS
```

## RoadMap

- [ ] Sim2Real 实机部署与调参（域随机化、系统辨识）
- [ ] 支持更多地形与任务（斜坡、楼梯、复杂粗糙地形）
- [ ] 更多 RL 算法与基线（PPO 变体、RMA 等）
- [ ] 训练性能优化（多 GPU、混合精度）
- [ ] 自动化评测与报告（定量指标、可视化）
- [ ] 文档与教程完善（原理详解、视频演示）

## Citation

If you find this project useful in your research, please consider citing:

```
@software{wheeled_legged_rl2026,
  author = {Zhang, Zhirui and Cui, Yu},
  title = {wheeled-legged_RL: Reinforcement Learning for Wheeled-legged Robots},
  url = {https://github.com/scutrobotlab/wheeled-legged_RL},
  year = {2026}
}
```

## Acknowledgements

This project uses or derives from the following open-source projects. We are grateful to their authors. License terms of third-party code are listed in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

- [Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [HIMLoco](https://github.com/InternRobotics/HIMLoco)
- [DreamWaQ](https://github.com/Manaro-Alpha/DreamWaQ)
- [ddt_rl_isaacgym](https://github.com/DDTRobot/ddt_rl_isaacgym)
