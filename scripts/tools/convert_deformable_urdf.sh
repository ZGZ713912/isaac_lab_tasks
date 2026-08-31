#!/bin/bash
# =============================================================================
# 将 deformable_infantry URDF 转换为 USD（Isaac Lab 资产）。
#
# 前置：
#   1) GPU 可用（isaacsim 需要）
#   2) isaaclab conda 环境就绪（REPRODUCE.md 第 0 节）
#   3) 源 URDF：~/legged_gym/resources/robots/deformable_infantry/
#      （deformable_infantry.urdf 使用相对网格路径，urdf/ 与 meshes/ 同级）
#
# 产物：source/agent_world/agent_world/assets/usd_files/deformable_suspension/
#       deformable_suspension.usd
# 运行：
#   bash scripts/tools/convert_deformable_urdf.sh
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
URDF_SRC="${URDF_SRC:-$HOME/legged_gym/resources/robots/deformable_infantry/urdf/deformable_infantry.urdf}"
OUT_DIR="$REPO_ROOT/source/agent_world/agent_world/assets/usd_files/deformable_suspension"

if [ ! -f "$URDF_SRC" ]; then
    echo "ERROR: URDF not found: $URDF_SRC"
    exit 1
fi

mkdir -p "$OUT_DIR"
# 复制 URDF 与网格（保持相对路径结构）
cp -r "$(dirname "$URDF_SRC")" "$OUT_DIR/urdf"
cp -r "$(dirname "$(dirname "$URDF_SRC")")/meshes" "$OUT_DIR/meshes"

cd "$HOME/IsaacLab-v2.3.2" || exit 1
echo ">>> Converting $URDF_SRC -> $OUT_DIR/deformable_suspension.usd"
# 所有关节转为 effort 模式（stiffness/damping 由 ArticulationCfg 执行器配置覆盖，
# 环境内手工 set_joint_effort_target 驱动腿 PD 与平四耦合）
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    "$OUT_DIR/urdf/deformable_infantry.urdf" \
    "$OUT_DIR/deformable_suspension.usd" \
    --joint-target-type none

echo ">>> Done: $OUT_DIR/deformable_suspension.usd"
