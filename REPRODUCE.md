# 复现手册 — wheeled-legged_RL（26 赛季轮腿步兵 RL 训练）

本仓库基于 **Isaac Sim 5.1 + Isaac Lab 2.3 + RSL-RL**。本机已按 README 安装并验证完毕，
按本文档即可在 **isaaclab** conda 环境里自己跑一遍训练。

## 0. 环境准备（已完成，可跳过）

| 项 | 状态 |
| --- | --- |
| conda 环境 `isaaclab`（Python 3.11, torch 2.10+cu128, isaacsim 5.1.0.0, isaaclab 2.3.2） | ✅ 已有 |
| `isaaclab_rl` / `isaaclab_tasks`（从 `~/IsaacLab` v2.3.2 tag 可编辑安装到 `~/IsaacLab-v2.3.2`） | ✅ 已装 |
| `agent_world` / `agent_tasks` / `agent_rl`（仓库 `source/` 下可编辑安装） | ✅ 已装 |
| 缺失的 `source/agent_rl/agent_rl/rsl_rl/env/` 包（被 `.gitignore` 的 `env/` 规则误伤，从未入库） | ✅ 已重建 |
| 入口脚本 `sys.path` 修复（`train.py`/`play.py`/`view_robot.py`/`list_envs.py` 自动把仓库根目录加入 path） | ✅ 已修 |
| 端到端 smoke train（CPU 4 envs，2 个 PPO iteration） | ✅ 通过 |

> ⚠️ **提交注意**：`source/agent_rl/agent_rl/rsl_rl/env/` 仍被 `.gitignore` 的 `env/` 规则忽略，
> 提交时需 `git add -f source/agent_rl/agent_rl/rsl_rl/env/`。

## 1. 前置检查（每次跑之前）

```bash
conda activate isaaclab
nvidia-smi          # 必须能看到 RTX 4060 Ti 且显存可用（若报错见第 5 节）
ls /dev/nvidia*     # 应存在 nvidia0 / nvidiactl / nvidia-uvm 等
```

## 2. 训练（平地 PPO，仓库当前配置）

在**仓库根目录** `/home/noir/Documents/workspace/wheeled-legged_RL` 下执行：

```bash
conda activate isaaclab

# 无界面训练（推荐，4096 envs；显存紧张就降到 2048）
python scripts/rsl_rl/train.py --task=Robotics-Wheelbipe-V14-Flat-v0 \
    --num_envs=4096 --max_iterations=20000 --headless --device=cuda:0

# 带界面（可看机器人）
python scripts/rsl_rl/train.py --task=Robotics-Wheelbipe-V14-Flat-v0 \
    --num_envs=4096 --max_iterations=20000 --device=cuda:0
```

- 日志输出到 `logs/rsl_rl/wheelbipe_v14_2_flat_direct/<时间戳>/`，
  每 500 迭代存一个 `model_<iter>.pt`，`params/` 下会 dump `env.yaml` / `agent.yaml`。
- 查看曲线：`tensorboard --logdir logs/rsl_rl/wheelbipe_v14_2_flat_direct`
- 任务默认配置（当前仓库）：
  - 策略：ActorCritic `[256,128,64]` elu；PPO lr=1e-4，entropy=0.005，value_loss=4.0，adaptive schedule
  - 观测 35 维 / 动作 6 维 / 特权观测 78 维；`num_envs` 默认 32（用 `--num_envs` 覆盖）

## 3. 精确复现预训练轻量模型（`pretrained/26_infantry/flat_and_rotation/2026-07-29_23-20-03`）

该预训练跑保存的 `params/agent.yaml` 为：**seed=66、ActorCritic `[128,64,32]`**、max_iterations=20000、
save_interval=500、其余 PPO 超参与当前仓库一致。用 Hydra 覆盖即可逐项对齐：

```bash
python scripts/rsl_rl/train.py --task=Robotics-Wheelbipe-V14-Flat-v0 \
    --num_envs=4096 --max_iterations=20000 --seed=66 --headless --device=cuda:0 \
    agent.policy.actor_hidden_dims=[128,64,32] \
    agent.policy.critic_hidden_dims=[128,64,32]
```

> 注：当前仓库奖励与 2026-07-29 那次运行相比已微调（`track_lin_vel_xy_square`/`track_ang_vel_z_square` 权重 -0.1→-1.0，
> `vel_height_gate_enabled` 等开关），所以曲线不会逐点一致，但训练流程/环境/奖励体系就是仓库自己的。

### 用预训练权重看效果（Play）

play 用当前 runner cfg 构建网络，**默认 256/128/64 与预训练 128/64/32 不匹配会加载失败**，
需临时把 `source/agent_tasks/agent_tasks/direct/wheelbipe/agents/rsl_rl_ppo_cfg.py`
中 `WheelbipeV14FlatPPORunnerCfg` 的 `actor_hidden_dims`/`critic_hidden_dims` 改为 `[128,64,32]`
（文件里已保留注释掉的 `[128,64,32]` 行，取消注释即可），然后：

```bash
# 键盘控制（W/S/A/D 移动，Z/X 高度，Q 跳跃）
python scripts/rsl_rl/play.py --task=Robotics-Wheelbipe-V14-Flat-Play-v0 \
    --num_envs=1 --checkpoint=./pretrained/26_infantry/flat_and_rotation/2026-07-29_23-20-03/model_8000.pt \
    --keyboard --device=cuda:0
```

### 无 GUI 评估（推荐，本机渲染器有驱动兼容问题时用）

```bash
# headless 定量评估：跑 N 个 episode，输出平衡时长/存活率/速度跟踪误差/高度 + CSV
python scripts/eval_checkpoint.py --task=Robotics-Wheelbipe-V14-Flat-Play-v0 \
    --checkpoint=./logs/rsl_rl/wheelbipe_v14_2_flat_direct/<时间戳>/model_1999.pt \
    --num_envs=16 --episodes=200 --device=cuda:0 --headless

# 或直接看训练曲线
tensorboard --logdir logs/rsl_rl/wheelbipe_v14_2_flat_direct
```

## 4. 其它任务 / 算法（可选）

| 任务 | 说明 |
| --- | --- |
| `Robotics-Wheelbipe-V14-Flat-v1` | 平地 PPO + 落地预训练 |
| `Robotics-Wheelbipe-V14-Flat-v2` | 平地 PPO + 小陀螺平移 |
| `Robotics-Wheelbipe-V14-Rough-v0/v1` | 粗糙地形 PPO |
| `Robotics-Wheelbipe-V14-Flat-DreamWaQ-v0` | DreamWaQ（隐式地形想象） |
| `Robotics-Wheelbipe-V14-Flat-HIM-v0` | HIMLoco（历史轨迹估计） |
| `Robotics-Wheelbipe-V14-Flat-NP3OBarlow-v0` | NP3O（BarlowTwins + 安全约束） |

```bash
python scripts/list_envs.py          # 列出全部 Robotics-* 任务
python scripts/view_robot.py --task=Robotics-Wheelbipe-V14-Flat-Play-v0 --num_envs=1 --device=cpu   # 看机器人
```

## 5. 常见问题

0. **GUI 模式崩溃（SIGSEGV in `librtx.scenedb.plugin.so`）**：两个已定位并修复的环境问题——
   - **缺 `libxml2.so.2`**：Isaac Sim 按 Ubuntu 编译，Arch 的 libxml2 只有 `.so.16`，
     导致 GUI 扩展（asset_converter/URDF/MJCF/OGN）加载失败 → 渲染器崩溃、窗口打不开。
     修复：`~/.local/lib/libxml2.so.2 -> /usr/lib/libxml2.so.16` 兼容链接（已建好）。
   - **`LD_LIBRARY_PATH` 残留 isaacgym 旧库**：若 shell 里导出了
     `$HOME/miniconda3/envs/isaacgym/lib`，Isaac Sim 的 X11/GLFW 插件会加载其中旧版
     libX11/libxcb/libstdc++ → 渲染器崩。修复：运行前 `unset LD_LIBRARY_PATH`。
   - **以后开 GUI 一律用** `./run_gui.sh`（自动做上面两件事）：
     ```bash
     ./run_gui.sh python scripts/rsl_rl/play.py --task=Robotics-Wheelbipe-V14-Flat-Play-v0 \
         --num_envs=1 --checkpoint=<model路径> --keyboard
     ```
   - 训练仍推荐 `--headless`；想录视频用 `--headless --video`。
   - 若 `run_gui.sh` 后仍崩在 `librtx.scenedb`：先 `vulkaninfo --summary` 确认 Vulkan 正常，
     再试 `env -u WAYLAND_DISPLAY ./run_gui.sh python ...`（强制 X11/XWayland 路径）。

1. **`nvidia-smi` 报错 / 训练报 CUDA 设备找不到**：检查 `/dev/nvidia*` 设备节点是否存在；
   驱动模块已加载（`lsmod | grep nvidia`）但设备节点缺失时，需重建（需要 root）：
   ```bash
   sudo nvidia-smi   # 或
   # 重新触发 udev：sudo udevadm trigger --subsystem-match=nvidia
   ```
2. **`No module named 'scripts'`**：已修复（入口脚本自动加仓库根目录到 sys.path）；若仍出现，手动
   `export PYTHONPATH=/home/noir/Documents/workspace/wheeled-legged_RL:$PYTHONPATH`。
3. **显存不足**：`--num_envs` 降到 2048 或 1024。
4. **训练过拟合崩塌**：参考仓库 README/预训练 readme，监控曲线并从中途最佳 checkpoint 继续：
   ```bash
   python scripts/rsl_rl/train.py --task=Robotics-Wheelbipe-V14-Flat-v0 --num_envs=4096 \
       --checkpoint=<logs/.../model_XXXX.pt> --resume_training --max_iterations=20000 --headless
   ```
