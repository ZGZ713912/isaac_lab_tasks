#!/usr/bin/env bash
# =============================================================================
# 04 - 正式无头训练 + 评估
# 用法：cd ~/wheeled-legged_RL && conda activate isaaclab && bash 04_train.sh
# =============================================================================
set -euo pipefail
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"

TASK="Robotics-Wheelbipe-V14-Flat-v0"
NUM_ENVS=4096
MAX_ITER=20000

echo ">>> 平地 PPO 无头训练  ($TASK, $NUM_ENVS envs, $MAX_ITER iter)"
echo ">>> 日志: logs/rsl_rl/wheelbipe_v14_2_flat_direct/<时间戳>/"
python scripts/rsl_rl/train.py --task=$TASK --num_envs=$NUM_ENVS --max_iterations=$MAX_ITER --headless --device=cuda:0

echo ""
echo "============================================================"
echo "  训练结束。用 tensorboard 看曲线:"
echo "  conda activate isaaclab && tensorboard --logdir logs/rsl_rl/wheelbipe_v14_2_flat_direct"
echo ""
echo "  用 headless 定量评估看效果（不需要 GUI/渲染器）:"
echo "  python scripts/eval_checkpoint.py --task=Robotics-Wheelbipe-V14-Flat-Play-v0 \\"
echo "      --checkpoint=logs/rsl_rl/wheelbipe_v14_2_flat_direct/<时间戳>/model_XXXX.pt \\"
echo "      --num_envs=16 --episodes=200 --device=cuda:0 --headless"
echo "============================================================"
