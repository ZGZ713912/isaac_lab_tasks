#!/bin/bash
# =============================================================================
# deformable_V2 → USD（Isaac Lab 资产）。
#
# 前置：
#   1) GPU 可用（isaacsim 需要）
#   2) 先跑归一化脚本：
#        python scripts/tools/prepare_deformable_v2_urdf.py
#      （把 Y-up 原始 狗v3.urdf 转成 Z-up 的 deformable_V2.urdf，改写 mesh 路径、
#        归一化 effort/velocity、轮子 continuous + 球体碰撞体、写平四 <mimic> 闭链）
#   3) IsaacLab 检出目录可用（ISAACLAB_DIR 覆盖，默认探测同 convert_wheel_leg_urdf.sh）
#
# 产物：usd_files/deformable_V2/deformable_V2.usd (+ configuration/*)
#      usd_files/deformable_V2/config.yaml（由 UrdfConverter 自动写入）
# 校验：
#   - configuration/deformable_V2_base.usd 应达几十 MB；若仍是几 KB，说明网格
#     没被加载（对照 convert_wheel_leg_urdf.sh:5-13 记录的 package:// 事故）
#   - 平四 <mimic> 已转为 PhysxMimicJointAPI（由 convert_urdf_mimic.py 断言）
#   - 物理层应出现解析 Sphere（轮子碰撞体）
#   - 产物中不应残留 package://
#
# 运行：bash scripts/tools/convert_deformable_v2_urdf.sh [额外参数]
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ASSET_DIR="$REPO_ROOT/source/agent_world/agent_world/assets/usd_files/deformable_V2"
URDF="$ASSET_DIR/urdf/deformable_V2.urdf"
USD="$ASSET_DIR/deformable_V2.usd"

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
[ -f "$URDF" ] || {
    echo "ERROR: URDF not found: $URDF"
    echo "先运行：python scripts/tools/prepare_deformable_v2_urdf.py"
    exit 1
}
if grep -q "package://" "$URDF"; then
    echo "ERROR: $URDF 仍含 package:// 路径，请先跑 prepare_deformable_v2_urdf.py"
    exit 1
fi
missing=0
for m in base_link leg_1 leg_2 leg_3 leg_4 upper_leg_1 upper_leg_2 upper_leg_3 upper_leg_4 \
         wheel_set_1 wheel_set_2 wheel_set_3 wheel_set_4 wheel_1 wheel_2 wheel_3 wheel_4; do
    [ -f "$ASSET_DIR/meshes/$m.STL" ] || { echo "ERROR: 缺少网格 $ASSET_DIR/meshes/$m.STL"; missing=1; }
done
[ "$missing" -eq 0 ] || exit 1

echo ">>> IsaacLab : $ISAACLAB_DIR"
echo ">>> URDF     : $URDF"
echo ">>> USD 输出 : $USD"
echo ">>> 转换中（force_usd_conversion=True，会覆盖旧产物）..."

cd "$ISAACLAB_DIR"
# 用本仓库的 convert_urdf_mimic.py（而非 IsaacLab 的 convert_urdf.py）：
# 只有它会把 importer 的 parse_mimic 打开，从而保留平四 <mimic> 闭链。
# --joint-target-type none：与 assets/deformable_V2.py 的执行器配置（effort 模式）一致。
./isaaclab.sh -p "$REPO_ROOT/scripts/tools/convert_urdf_mimic.py" \
    "$URDF" "$USD" --joint-target-type none "$@"

# ---- 结果校验 ----
# 注意：USD crate 是编译编码，明文 grep 不可靠（会假阴性）。唯一可靠的失败信号是
# base 层体积：网格没加载时只有几 KB。碰撞体/关节上限的核查请用 pxr，例如：
#   PXR=<.../extscache/omni.usd.libs-*/pxr>; LD_LIBRARY_PATH=$PXR/bin:$CONDA/lib
#   $CONDA/bin/python -c "from pxr import Usd,UsdGeom; ..."
BASE_USD="$ASSET_DIR/configuration/deformable_V2_base.usd"
if [ -f "$BASE_USD" ]; then
    SIZE=$(stat -c%s "$BASE_USD")
    SIZE_MB=$((SIZE / 1024 / 1024))
    echo ">>> configuration/deformable_V2_base.usd = ${SIZE} 字节 (~${SIZE_MB} MB)"
    if [ "$SIZE_MB" -lt 5 ]; then
        echo ">>> [FAIL] base.usd 仍然过小 → 网格没有加载成功，base_link 依旧不可见"
        exit 2
    fi
    echo ">>> [OK] 已包含网格几何（对照：deformable V1 ~137MB / wheelbipe ~27MB）"
else
    echo ">>> [FAIL] 未生成 $BASE_USD"
    exit 2
fi

echo ">>> 轮子碰撞体应为解析球体 r=0.0769（在 Isaac 或 pxr 中确认 wheel_N/collisions/mesh_0/sphere）"
echo ">>> 产物: $USD"
echo ">>> config.yaml（自动生成）: $ASSET_DIR/config.yaml"
echo ">>> 下一步: python scripts/list_envs.py   # 确认 Robotics-Deformable-Suspension-* 注册"
