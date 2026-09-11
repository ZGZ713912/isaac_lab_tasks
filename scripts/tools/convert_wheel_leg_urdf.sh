#!/bin/bash
# =============================================================================
# 将 Wheel_leg_V1 (urdf_V4.0.urdf) 转换为 USD（Isaac Lab 资产）。
#
# 背景：
#   URDF 原本是 SolidWorks 导出的 `package://urdf_V4.0/meshes/*.STL` 网格路径，
#   在非 ROS 环境下 Isaac 的 URDF importer 解析不到该 scheme，且不会硬报错，
#   于是 2026-09-09 22:38 那次转换只写出了骨架：
#     configuration/Wheel_leg_V1_base.usd 仅 3.5 KB（无任何网格数据）
#   → Isaac 视口里看不到车身/base_link（所有 link 的 visual/collision 都是空壳）。
#
#   现已把 URDF 的 14 处网格路径改为相对路径 ../meshes/*.STL
#   （与能正常转换的 deformable_infantry.urdf 一致）。
#
# 前置：
#   1) GPU 可用（isaacsim 需要）
#   2) isaaclab conda 环境就绪；IsaacLab 检出目录可用 ISAACLAB_DIR 覆盖
#
# 产物：usd_files/Wheel_leg_V1/Wheel_leg_V1.usd (+ configuration/*)
# 校验：configuration/Wheel_leg_V1_base.usd 应达到几十 MB；
#       若仍是几 KB，说明网格依旧没被加载（转换无效）。
#
# 运行：bash scripts/tools/convert_wheel_leg_urdf.sh [传给 convert_urdf.py 的额外参数]
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ASSET_DIR="$REPO_ROOT/source/agent_world/agent_world/assets/usd_files/Wheel_leg_V1"
URDF="$ASSET_DIR/urdf/urdf_V4.0.urdf"
USD="$ASSET_DIR/Wheel_leg_V1.usd"

# ---- 定位 IsaacLab 检出目录（含 isaaclab.sh）----
if [ -z "$ISAACLAB_DIR" ]; then
    for cand in \
        "$HOME/Documents/workspace/IsaacLab-v2.3.2" \
        "$HOME/IsaacLab-v2.3.2" \
        "$HOME/Documents/workspace/IsaacLab"
    do
        if [ -x "$cand/isaaclab.sh" ]; then ISAACLAB_DIR="$cand"; break; fi
    done
fi
if [ -z "$ISAACLAB_DIR" ] || [ ! -x "$ISAACLAB_DIR/isaaclab.sh" ]; then
    echo "ERROR: 找不到 isaaclab.sh，请显式指定：ISAACLAB_DIR=/path/to/IsaacLab bash $0"
    exit 1
fi

# ---- 前置检查：URDF 与网格 ----
[ -f "$URDF" ] || { echo "ERROR: URDF not found: $URDF"; exit 1; }
missing=0
for m in base_link L_link1 L_link2 L_link3 R_link1 R_link2 R_link3; do
    [ -f "$ASSET_DIR/meshes/$m.STL" ] || { echo "ERROR: 缺少网格 $ASSET_DIR/meshes/$m.STL"; missing=1; }
done
[ "$missing" -eq 0 ] || exit 1

echo ">>> IsaacLab : $ISAACLAB_DIR"
echo ">>> URDF     : $URDF"
echo ">>> USD 输出 : $USD"
echo ">>> 转换中（force_usd_conversion=True，会覆盖旧产物）..."

cd "$ISAACLAB_DIR"
# --joint-target-type none：与 config.yaml 里 joint_drive(force/none) 一致
./isaaclab.sh -p scripts/tools/convert_urdf.py "$URDF" "$USD" --joint-target-type none "$@"

# ---- 结果校验：base.usd 是否真的含网格 ----
BASE_USD="$ASSET_DIR/configuration/Wheel_leg_V1_base.usd"
if [ -f "$BASE_USD" ]; then
    SIZE=$(stat -c%s "$BASE_USD")
    SIZE_MB=$((SIZE / 1024 / 1024))
    echo ">>> configuration/Wheel_leg_V1_base.usd = ${SIZE} 字节 (~${SIZE_MB} MB)"
    if [ "$SIZE_MB" -lt 5 ]; then
        echo ">>> [FAIL] base.usd 仍然过小 → 网格没有加载成功，base_link 依旧不可见"
        exit 2
    fi
    echo ">>> [OK] 已包含网格几何（对比：deformable ~137MB / wheelbipe ~27MB）"
else
    echo ">>> [FAIL] 未生成 $BASE_USD"
    exit 2
fi
