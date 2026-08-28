#!/usr/bin/env bash
# =============================================================================
# GUI 运行辅助脚本（wheeled-legged_RL）
#
# 两个已知问题，本脚本一并解决：
#
# 1) 缺 libxml2.so.2（Isaac Sim 按 Ubuntu 编译，Arch 只有 libxml2.so.16）
#    → 用 ~/.local/lib/libxml2.so.2 兼容链接补上，否则 asset_converter /
#      URDF / MJCF importer / OGN 扩展全部加载失败。
#
# 2) LD_LIBRARY_PATH 残留 isaacgym 环境的旧库（练 legged_gym 时导出的
#    $HOME/miniconda3/envs/isaacgym/lib），Isaac Sim 的 X11/GLFW 窗口插件会
#    加载其中的旧 libX11/libxcb/libstdc++，导致 RTX 渲染器在
#    librtx.scenedb 里 SIGSEGV（窗口打不开）。
#    → 这里把 LD_LIBRARY_PATH 整个清掉重设，只留 shim 目录。
#
# 用法（在仓库根目录执行）：
#   ./run_gui.sh python scripts/rsl_rl/play.py --task=Robotics-Wheelbipe-V14-Flat-Play-v0 \
#       --num_envs=1 --checkpoint=./logs/rsl_rl/wheelbipe_v14_2_flat_direct/<时间戳>/model_1999.pt --keyboard
#   ./run_gui.sh python scripts/rsl_rl/train.py --task=Robotics-Wheelbipe-V14-Flat-v0 --num_envs=64
# =============================================================================

set -e

# 1. 确保 libxml2 兼容链接存在（无需 root）
if [ ! -e "$HOME/.local/lib/libxml2.so.2" ]; then
    mkdir -p "$HOME/.local/lib"
    ln -s /usr/lib/libxml2.so.16 "$HOME/.local/lib/libxml2.so.2"
    echo "[run_gui] 已创建 $HOME/.local/lib/libxml2.so.2 -> /usr/lib/libxml2.so.16"
fi

# 2. 清空被污染的 LD_LIBRARY_PATH，只保留 shim 目录（关键！）
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$HOME/.local/lib"

echo "[run_gui] LD_LIBRARY_PATH=$LD_LIBRARY_PATH (已清空 isaacgym 残留)"
exec "$@"
