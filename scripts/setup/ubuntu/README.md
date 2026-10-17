# Ubuntu 24.04 + NVIDIA A10 无头训练环境部署指南

目标：在全新 Ubuntu 24.04 服务器（NVIDIA A10，24GB，Ampere sm_86，无显示器）上，
为 **wheeled-legged_RL** 配置 **无头（headless）训练**环境。全程不需要 GUI。

## 架构

```
Isaac Sim 5.1.0 (pip) + Isaac Lab 2.3.2 (pip) + RSL-RL 3.0.1
    + agent_world / agent_tasks / agent_rl  (wheeled-legged_RL 自己的包)
    + 根目录统一安装，env 接口与入口路径均随源码提供
```

## 版本（与 A10 / CUDA 12.8 匹配）

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| NVIDIA 驱动 | 580（server） | CUDA 12.8 需 ≥570.85；580 也是 Isaac Sim 5.1 验证分支 |
| Python | 3.11 | |
| PyTorch | 2.10.0+cu128 | 支持 A10 (sm_86) |
| Isaac Sim | 5.1.0.0 | pip 元包 |
| Isaac Lab | 2.3.2 | pip 元包 + 源码装 isaaclab_rl/isaaclab_tasks |
| rsl-rl-lib | 3.0.1 | |

## 步骤总览

```bash
# ① 系统 + 驱动（需 root，装完重启）
sudo bash ~/wheeled-legged_RL/scripts/setup/ubuntu/00_system.sh
sudo reboot
nvidia-smi   # 重启后确认 Driver 580.xx + A10

# ② conda 环境 + 依赖（约 30GB 下载，耐心等）
cd ~/wheeled-legged_RL
bash scripts/setup/ubuntu/01_conda.sh

# ③ 只读安装检查（不改写仓库源码）
conda activate isaaclab
bash scripts/setup/ubuntu/02_fixes.sh

# ④ 冒烟验证
bash scripts/setup/ubuntu/03_smoke.sh

# ⑤ 正式训练
bash scripts/setup/ubuntu/04_train.sh
```

## 说明

- **安装入口**：三个包统一通过根目录 `pyproject.toml` 安装，不再分别安装 `source/` 子目录。
- **环境接口**：`src/agent_rl/rsl_rl/env/` 现随 Git 跟踪，`02_fixes.sh` 保留旧入口名但只检查，不生成或覆盖源码。
- **路径**：运行入口会优先加载当前 checkout 的 `src/`，共享辅助模块随根目录安装提供。
- **`--headless`**：A10 无显示器，训练必须带 `--headless`；无需任何 GUI。
- **显存**：A10 24GB，`--num_envs=4096` 够用；不够就降到 2048。
- **日志**：`logs/rsl_rl/wheelbipe_v14_2_flat_direct/<时间戳>/`，每 500 迭代存 checkpoint。
- **看效果（无 GUI）**：用 `eval_checkpoint.py` 定量评估，或 `tensorboard` 看曲线。

## 训练 / 评估命令

```bash
conda activate isaaclab
cd ~/wheeled-legged_RL

# 训练（无头）
PYTHONPATH=$PWD python scripts/rsl_rl/train.py \
    --task=Robotics-Wheelbipe-V14-Flat-v0 --num_envs=4096 --max_iterations=20000 --headless

# 评估（无头，不需渲染器）
PYTHONPATH=$PWD python scripts/eval_checkpoint.py \
    --task=Robotics-Wheelbipe-V14-Flat-Play-v0 \
    --checkpoint=logs/rsl_rl/wheelbipe_v14_2_flat_direct/<时间戳>/model_XXXX.pt \
    --num_envs=16 --episodes=200 --device=cuda:0 --headless

# 看曲线（另开终端）
conda activate isaaclab && tensorboard --logdir logs/rsl_rl/wheelbipe_v14_2_flat_direct
```
