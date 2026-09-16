#!/bin/bash
# =============================================================================
# [已废弃] 本脚本原用于把 ~/legged_gym 下的 deformable_infantry(V1) 转成 USD，
# 输出到 usd_files/deformable_suspension/。
#
# 但：
#   - 源 URDF 位于 $HOME/legged_gym/...，不在本仓库内，无法复现；
#   - 资产模块 assets/deformable_infantry.py 与 deformable_V1 目标已不存在；
#   - 现役资产为 deformable_V2（狗v3）。
#
# 请改用：
#   python scripts/tools/prepare_deformable_v2_urdf.py
#   bash   scripts/tools/convert_deformable_v2_urdf.sh
#
# 本脚本保留为转发入口，避免旧文档/命令直接失效。
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo ">>> [DEPRECATED] convert_deformable_urdf.sh 已废弃，转发到 deformable_V2 链路。"
echo ">>> 若需要 V1/deformable_suspension 资产，请从 git 历史恢复对应 URDF 与模块。"

# 默认只做提示；设置 FORWARD=1 时真正执行 V2 转换。
if [ "${FORWARD:-0}" = "1" ]; then
    echo ">>> FORWARD=1 → 先归一化 URDF，再转换 USD"
    python "$REPO_ROOT/scripts/tools/prepare_deformable_v2_urdf.py"
    bash "$REPO_ROOT/scripts/tools/convert_deformable_v2_urdf.sh" "$@"
else
    echo ">>> 提示：bash scripts/tools/convert_deformable_v2_urdf.sh"
    echo ">>> （如需本脚本代跑：FORWARD=1 bash $0）"
fi
