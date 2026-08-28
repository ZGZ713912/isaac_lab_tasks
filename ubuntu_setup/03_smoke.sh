#!/usr/bin/env bash
# =============================================================================
# 03 - 无头冒烟验证（确认环境 + 仓库补丁可用）
# 用法：cd ~/wheeled-legged_RL && conda activate isaaclab && bash 03_smoke.sh
# =============================================================================
set -euo pipefail
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

echo "=== [1/3] 确认 GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "=== [2/3] 列出已注册任务（headless 启动 Isaac Sim）==="
timeout 300 python scripts/list_envs.py 2>&1 | grep -E "Robotics-Wheelbipe-V14-Flat-v0|Available Environments" | head

echo "=== [3/3] 小规模冒烟训练（64 envs x 2 iter，headless）==="
timeout 900 python scripts/rsl_rl/train.py \
    --task=Robotics-Wheelbipe-V14-Flat-v0 \
    --num_envs=64 --max_iterations=2 --headless --device=cuda:0 2>&1 | grep -E "Learning iteration|Computation|Total timesteps|Error|Traceback" | head -20

echo ""
echo "============================================================"
echo "  冒烟通过。正式训练见 04_train.sh"
echo "============================================================"
