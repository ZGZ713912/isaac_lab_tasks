#!/usr/bin/env python3
# =============================================================================
# Wheel_leg_V1 URDF 规范化脚本（SolidWorks 原始导出 → RL 训练约定）
#
# 背景：
#   SolidWorks 导出的 urdf_V4.0_solidworks.urdf 的 base 坐标系为：
#     +X = 左（L 腿在 +X，R 腿在 -X），+Y = 车头反方向，+Z = 上；
#     轮轴沿 X → 轮子只能沿 Y 滚动，而整个 RL 代码（奖励/指令/观测）
#     按 "+X 前进、+Y 左、+Z 上" 的 wheelbipe 约定编写，导致：
#       - lin_vel_x 追踪奖励实际要求机器人"横移"（物理不可实现）；
#       - no_fork 把左右轮距当劈叉惩罚（恒罚）；
#       - 左右腿同号动作产生镜像运动（不对称）。
#   另外 SolidWorks 给所有关节写了 effort=100 / velocity=1，velocity=1 会被
#   USD 转换烘成 PhysX 硬限速（robot.data.joint_vel_limits=1），腿/轮几乎转不动。
#
# 本脚本对 URDF 做三件事（不动任何几何尺寸）：
#   1) 根坐标系整体绕 Z 旋转 +90°：新 +X = 旧 -Y（车头）、新 +Y = 旧 +X（左）；
#      L 腿在 +Y、R 腿在 -Y，轮轴沿 Y，前进方向 +X。
#   2) 统一左右腿关节正方向（翻转 L_joint2 / R_joint1 / R_joint3 的 axis 符号）：
#      两侧同号 = 镜像对称；joint2 减小 = 伸腿蹬地；joint3 正转 = 前进。
#   3) 归一化关节限幅（effort/velocity），解除 1 rad/s 硬限速。
#
# 用法（仓库根目录）：
#   python scripts/tools/prepare_wheel_leg_v1_urdf.py                       # → urdf_V4.0.urdf
#   python scripts/tools/prepare_wheel_leg_v1_urdf.py --wheel-collision cylinder  # → urdf_V4.0_cylwheel.urdf
#   bash scripts/tools/convert_wheel_leg_urdf.sh                            # mesh 轮转 USD
#   bash scripts/tools/convert_wheel_leg_urdf_cylinder.sh                   # 圆柱轮转 USD（A/B）
# =============================================================================
"""Normalize the Wheel_leg_V1 SolidWorks URDF to the repo RL frame conventions."""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ASSET_URDF_DIR = os.path.join(
    _REPO_ROOT, "source", "agent_world", "agent_world", "assets", "usd_files", "Wheel_leg_V1", "urdf"
)
_RAW = os.path.join(_ASSET_URDF_DIR, "urdf_V4.0_solidworks.urdf")
_OUT_MESH = os.path.join(_ASSET_URDF_DIR, "urdf_V4.0.urdf")
_OUT_CYL = os.path.join(_ASSET_URDF_DIR, "urdf_V4.0_cylwheel.urdf")

# 绕 Z 旋转 +90°：p_new = A @ p_old（新 +X = 旧 -Y，新 +Y = 旧 +X）
_A = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# 需要翻转 axis 符号的关节：统一左右正方向
_FLIP_AXIS = {"L_joint2", "R_joint1", "R_joint3"}

# 关节限幅归一化：(effort [N·m], velocity [rad/s])
# 与 wheel_leg_V1.py 执行器配置一致；velocity=1 是 SolidWorks 占位值，必须修正。
_LIMITS = {
    "L_joint1": (40.0, 17.0),
    "R_joint1": (40.0, 17.0),
    "L_joint2": (40.0, 17.0),
    "R_joint2": (40.0, 17.0),
    "L_joint3": (5.0, 60.0),
    "R_joint3": (5.0, 60.0),
}

# 解析圆柱轮碰撞体参数（与 meshes/L_link3.STL 实测尺寸一致）
_CYL = {
    "L_link3": (0.0600000032506, 0.0250000003725, 0.0203500008211),
    "R_link3": (0.0600000031546, 0.0250000003725, -0.0203500008211),
}


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _R_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if abs(np.cos(pitch)) < 1e-9:
        roll = 0.0
        yaw = np.arctan2(R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw)


def _fmt(values) -> str:
    return " ".join(f"{float(v):.12g}" for v in values)


def _transform_origin(origin: ET.Element) -> None:
    xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()], dtype=float)
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    R = _rpy_to_R(*rpy)
    origin.set("xyz", _fmt(_A @ xyz))
    origin.set("rpy", _fmt(_R_to_rpy(_A @ R)))


def _replace_wheel_collision(link: ET.Element) -> None:
    name = link.get("name")
    radius, length, z_offset = _CYL[name]
    for collision in link.findall("collision"):
        link.remove(collision)
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", {"xyz": _fmt((0.0, 0.0, z_offset)), "rpy": "0 0 0"})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "cylinder", {"radius": f"{radius:.12g}", "length": f"{length:.12g}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel-collision",
        choices=("mesh", "cylinder"),
        default="mesh",
        help="轮子碰撞体：mesh=STL 网格（默认，导入为 convex hull）/ cylinder=解析圆柱（A/B 变体）",
    )
    args = parser.parse_args()

    if not os.path.isfile(_RAW):
        raise SystemExit(f"raw URDF not found: {_RAW}\n（可从 git 恢复：git show HEAD:<path>/urdf_V4.0.urdf）")

    tree = ET.parse(_RAW)
    root = tree.getroot()

    # 1) 根连杆（base_link）的全部 origin 旋转到新坐标系
    base_link = None
    for link in root.findall("link"):
        if link.get("name") == "base_link":
            base_link = link
            break
    if base_link is None:
        raise SystemExit("base_link not found in URDF")
    for tag in ("inertial", "visual", "collision"):
        for elem in base_link.findall(tag):
            origin = elem.find("origin")
            if origin is not None:
                _transform_origin(origin)

    # 2) 父节点为 base_link 的关节（L/R_joint1）origin 同样旋转；子关节局部坐标不变
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if parent is not None and parent.get("link") == "base_link":
            origin = joint.find("origin")
            if origin is not None:
                _transform_origin(origin)

    # 3) 统一轴线符号
    for joint in root.findall("joint"):
        if joint.get("name") in _FLIP_AXIS:
            axis = joint.find("axis")
            if axis is None:
                continue
            vec = [float(v) for v in axis.get("xyz", "0 0 1").split()]
            axis.set("xyz", _fmt([-v for v in vec]))

    # 4) 归一化关节 effort/velocity 限幅（原为 100/1，velocity=1 会锁死关节）
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        name = joint.get("name")
        if name not in _LIMITS:
            continue
        effort, velocity = _LIMITS[name]
        limit.set("effort", f"{effort:g}")
        limit.set("velocity", f"{velocity:g}")

    # 5) 可选：轮子碰撞体换成解析圆柱（A/B 对比）
    if args.wheel_collision == "cylinder":
        for link in root.findall("link"):
            if link.get("name") in _CYL:
                _replace_wheel_collision(link)

    out_path = _OUT_CYL if args.wheel_collision == "cylinder" else _OUT_MESH
    root.set("name", "Wheel_leg_V1")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f">>> wrote {out_path}")
    print(f">>> wheel collision: {args.wheel_collision}")
    print(f">>> axis flipped: {sorted(_FLIP_AXIS)}")
    if args.wheel_collision == "cylinder":
        print(">>> next: bash scripts/tools/convert_wheel_leg_urdf_cylinder.sh")
    else:
        print(">>> next: bash scripts/tools/convert_wheel_leg_urdf.sh")


if __name__ == "__main__":
    main()
