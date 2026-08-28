#!/usr/bin/env bash
# =============================================================================
# 00 - 系统依赖 + NVIDIA 驱动（A10 服务器，无头）
# 用法：sudo bash 00_system.sh
# =============================================================================
set -euo pipefail

echo "[1/4] apt 更新 + 基础工具"
apt-get update
apt-get install -y --no-install-recommends \
    build-essential gcc g++ make cmake pkg-config \
    git wget curl ca-certificates \
    python3-dev \
    libgl1 libegl1 libglib2.0-0 libgomp1 \
    libvulkan1 libxkbcommon0 libfontconfig1 libfreetype6 \
    libx11-6 libxcb1 libxau6 libxi6 libxcursor1 libxrender1 \
    libssl3 libsasl2-2

echo "[2/4] 安装 NVIDIA 官方 apt 源"
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

echo "[3/4] 安装驱动 580（>= CUDA 12.8 要求 570.85；580 也是 Isaac Sim 5.1 验证的驱动）"
# 无头服务器用 -server 变体（不装 X 桌面那套）；想要完整驱动就换 cuda-drivers-580
apt-get install -y nvidia-driver-580-server

echo "[4/4] 验证"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi
    echo ">>> 驱动已装。请运行: sudo reboot  重启后再继续下一步。"
    echo ">>> 重启后执行: nvidia-smi  应看到 Driver 580.xx 和 A10 (24GB)"
else
    echo ">>> nvidia-smi 未找到，重启后应可用。"
fi
