#!/bin/bash
# =============================================================================
# Wheel_leg_V1「轮子解析圆柱碰撞体」变体 → USD（A/B 对比用，独立资产目录）
#
# 为什么：
#   urdf_V4.0.urdf 里轮子（L/R_link3）的 <collision> 是 STL mesh，Isaac 的
#   URDF importer 只能把它 cook 成 convex hull（config.yaml: collider_type=convex_hull），
#   即"约 90 边的多边形盘"，不是解析圆柱：
#     • 滚动的接地点是面片棱边，理论上会有微幅多边形效应；
#     • PhysX 对解析圆柱（UsdGeom.Cylinder + CollisionAPI）有专门支持，
#       与三角网格/平面接触更平滑（IsaacLab docs: simulation_performance.rst）。
#   本脚本把轮子碰撞体换成真正的解析圆柱，其余（视觉网格、惯性、关节）完全不变。
#
# 前置：
#   1) GPU 可用（isaacsim 需要）
#   2) isaaclab 检出目录就绪（ISAACLAB_DIR 覆盖，默认与 convert_wheel_leg_urdf.sh 同）
#   3) 先跑：
#      python scripts/tools/prepare_wheel_leg_v1_urdf.py --wheel-collision cylinder
#      → urdf/urdf_V4.0_cylwheel.urdf
#
# 产物：usd_files/Wheel_leg_V1_cylwheel/Wheel_leg_V1_cylwheel.usd (+ configuration/*)
# 校验：configuration/Wheel_leg_V1_cylwheel_physics.usd 内应出现
#       def Cylinder + apiSchemas = ["PhysicsCollisionAPI"]（见脚本末尾检查）
#
# 运行：bash scripts/tools/convert_wheel_leg_urdf_cylinder.sh [额外参数]
#       ASSET_DIR=... ISAACLAB_DIR=... bash scripts/tools/convert_wheel_leg_urdf_cylinder.sh
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/source/agent_world/agent_world/assets/usd_files/Wheel_leg_V1"
ASSET_DIR="${ASSET_DIR:-$REPO_ROOT/source/agent_world/agent_world/assets/usd_files/Wheel_leg_V1_cylwheel}"
URDF_SRC="$SRC_DIR/urdf/urdf_V4.0_cylwheel.urdf"

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

# ---- 前置检查 ----
[ -f "$URDF_SRC" ] || {
    echo "ERROR: $URDF_SRC 不存在"
    echo "先运行：python scripts/tools/prepare_wheel_leg_v1_urdf.py --wheel-collision cylinder"
    exit 1
}
for m in base_link L_link1 L_link2 L_link3 R_link1 R_link2 R_link3; do
    [ -f "$SRC_DIR/meshes/$m.STL" ] || { echo "ERROR: 缺少网格 $SRC_DIR/meshes/$m.STL"; exit 1; }
done
grep -q "<cylinder" "$URDF_SRC" || { echo "ERROR: $URDF_SRC 内没有 <cylinder>，轮子碰撞体没被替换"; exit 1; }

# ---- 组装独立资产目录（URDF + 网格保持相对路径）----
mkdir -p "$ASSET_DIR/urdf"
cp -f "$URDF_SRC" "$ASSET_DIR/urdf/"
[ -d "$ASSET_DIR/meshes" ] || cp -r "$SRC_DIR/meshes" "$ASSET_DIR/meshes"

echo ">>> IsaacLab : $ISAACLAB_DIR"
echo ">>> URDF     : $ASSET_DIR/urdf/urdf_V4.0_cylwheel.urdf"
echo ">>> USD 输出 : $ASSET_DIR/Wheel_leg_V1_cylwheel.usd"
echo ">>> 转换中（force_usd_conversion=True，会覆盖旧产物）..."

cd "$ISAACLAB_DIR"
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    "$ASSET_DIR/urdf/urdf_V4.0_cylwheel.urdf" \
    "$ASSET_DIR/Wheel_leg_V1_cylwheel.usd" \
    --joint-target-type none "$@"

# ---- 结果校验 1：base.usd 含网格 ----
BASE_USD="$ASSET_DIR/configuration/Wheel_leg_V1_cylwheel_base.usd"
PHYS_USD="$ASSET_DIR/configuration/Wheel_leg_V1_cylwheel_physics.usd"
if [ -f "$BASE_USD" ]; then
    SIZE=$(stat -c%s "$BASE_USD")
    echo ">>> configuration/Wheel_leg_V1_cylwheel_base.usd = ${SIZE} 字节 (~$((SIZE / 1024 / 1024)) MB)"
    [ "$SIZE" -gt 5000000 ] || { echo ">>> [FAIL] base.usd 过小 → 网格没加载"; exit 2; }
else
    echo ">>> [FAIL] 未生成 $BASE_USD"; exit 2
fi

# ---- 结果校验 2：物理层里是解析圆柱而不是 convexHull ----
# 用可用的 USD 工具读取（/usr/bin/usdcat 若缺库，可改用带 pxr 的 python3.14）
if [ -f "$PHYS_USD" ]; then
    if grep -qa "Cylinder" "$PHYS_USD"; then
        echo ">>> [OK] 物理层出现 Cylinder（解析圆柱碰撞体已生成）"
    else
        echo ">>> [WARN] 物理层未找到 Cylinder：请在 Isaac 里用 usdcat 确认 ——"
        echo "         /Wheel_leg_V1/L_link3/collisions 下应有 def Cylinder + PhysicsCollisionAPI，"
        echo "         而不是 node_STL_BINARY_ 上的 physics:approximation=\"convexHull\"。"
    fi
fi
echo ">>> Done: $ASSET_DIR/Wheel_leg_V1_cylwheel.usd"
echo ">>> A/B：把 source/agent_world/agent_world/assets/wheel_leg_V1.py 内"
echo "         usd_path 指向 .../Wheel_leg_V1_cylwheel/Wheel_leg_V1_cylwheel.usd 后跑同一任务对比。"
