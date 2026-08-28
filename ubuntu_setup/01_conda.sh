#!/usr/bin/env bash
# =============================================================================
# 01 - Miniconda + isaaclab conda 环境 + 依赖（在 wheeled-legged_RL 仓库根目录执行）
# 用法：cd ~/wheeled-legged_RL && bash 01_conda.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(pwd)"
echo ">>> 仓库根目录: $REPO_ROOT"

# ---- [1/6] Miniconda（若未装） ----
if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
    echo "[1/6] 安装 Miniconda"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    rm -f Miniconda3-latest-Linux-x86_64.sh
fi
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# ---- [2/6] 创建 isaaclab 环境 ----
if ! conda env list | grep -q "^isaaclab "; then
    echo "[2/6] 创建 conda 环境 isaaclab (python 3.11)"
    conda create -y -n isaaclab python=3.11
fi
conda activate isaaclab

# ---- [3/6] PyTorch (CUDA 12.8, sm_86 支持 A10) ----
echo "[3/6] 安装 PyTorch 2.10.0+cu128"
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# ---- [4/6] Isaac Sim 5.1 + Isaac Lab 2.3.2（pip 元包） ----
echo "[4/6] 安装 isaacsim 5.1.0 + isaaclab 2.3.2（约 30GB 下载，请耐心）"
pip install isaacsim==5.1.0.0
pip install isaaclab==2.3.2

# isaaclab_rl / isaaclab_tasks 不在 pip 元包里，需从 IsaacLab 源码装
echo "     安装 isaaclab_rl / isaaclab_tasks (IsaacLab v2.3.2 源码)"
if [ ! -d "$HOME/IsaacLab" ]; then
    git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$HOME/IsaacLab"
fi
pip install --no-deps -e "$HOME/IsaacLab/source/isaaclab_rl"
pip install --no-deps -e "$HOME/IsaacLab/source/isaaclab_tasks"
pip install rsl-rl-lib==3.0.1

# ---- [5/6] wheeled-legged_RL 自己的三个包 ----
echo "[5/6] 安装 agent_world / agent_tasks / agent_rl"
pip install -e "$REPO_ROOT/source/agent_world"
pip install -e "$REPO_ROOT/source/agent_tasks"
pip install -e "$REPO_ROOT/source/agent_rl"
pip install tensorboard matplotlib omegaconf prettytable 2>/dev/null || true

# ---- [6/6] 校验关键导入 ----
echo "[6/6] 校验导入"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import rsl_rl, isaaclab, isaaclab_rl, isaaclab_tasks, agent_world, agent_tasks, agent_rl; print('imports OK')"

echo ""
echo "============================================================"
echo "  环境装好。下一步运行: bash 02_fixes.sh   （打仓库必需补丁）"
echo "============================================================"
