#!/usr/bin/env python3
# =============================================================================
# deformable_V2 (SolidWorks 原始导出 狗v3.urdf) → RL 训练约定 URDF
#
# 背景（2026-09-16 核查）：
#   SolidWorks 导出的 狗v3.urdf 是 Y-up 坐标系（SolidWorks 默认）：
#     +X = 车头，+Y = 上，+Z = 左（四轮心在 q=0 时全部 Y=-0.055，共面于 X-Z）；
#   而本仓库 RL 栈（奖励/指令/观测/部署 rmcs_rl）统一按 "+X 前进、+Y 左、+Z 上"，
#   所以必须整体旋转一次。另有三处 SolidWorks 占位值/路径问题会破坏 Isaac 转换：
#
#   1) mesh 路径是 `package://狗v3/meshes/*.STL`：非 ROS 环境下 Isaac 的 URDF
#      importer 解析不到该 scheme，且**不报错**，只写出几 KB 的空壳 USD
#      （见 scripts/tools/convert_wheel_leg_urdf.sh:5-13 记录的同类事故）。
#   2) 所有关节 effort=0 / velocity=0 是 SolidWorks 占位值：会被烘成 PhysX 的
#      maxForce/maxVelocity 硬上限=0 → set_joint_effort_target 被截为零、关节
#      几乎不动（对照 prepare_wheel_leg_v1_urdf.py:13-14 的 velocity=1 事故）。
#   3) 轮关节写成了 revolute + limit[0,1000]，只能单向转；轮子应为 continuous。
#
# 本脚本对 URDF 做七件事（**不改任何几何尺寸，也不动 lower/upper 位置限位**）：
#   1) 根坐标系整体绕 X 旋转 +90°：旧 +Y(上)→新 +Z(上)，旧 +X(车头)→新 +X。
#      只作用于 base_link 自身的 inertial/visual/collision origin，以及父节点为
#      base_link 的根关节（joint_leg_* / joint_upper_leg_*）origin；深层关节
#      origin、<axis>、子链接网格坐标均不变（子帧随整体旋转同步，轴数值不变）。
#   2) 34 处 `package://狗v3/meshes/X.STL` → 相对路径 `../meshes/X.STL`。
#   3) robot name：狗v3 → deformable_V2。
#   4) effort/velocity 归一化：joint_leg_*/joint_upper_leg_* = (40, 17)，
#      joint_wheel_* = (5, 60)；lower/upper 位置限位原样保留。
#   5) joint_wheel_*：revolute → continuous（去掉 lower/upper，保留 effort/velocity）。
#   6) wheel_N 碰撞体：mesh → 解析球体 r=0.0769（全向轮简化，visual 不变）；
#      base_link 碰撞体保留 mesh（convex hull），名义站姿下不会戳地。
#   7) 平四闭链：给 joint_wheel_set_* 写 <mimic joint=joint_leg_* multiplier=+1>、
#      joint_upper_leg_* 写 <mimic multiplier=−1>（offset 0）。URDF 是树，缺的
#      第 4 条边（upper_leg↔wheel_set）无法直接表达，用 mimic 的线性关系等价替代；
#      isaacsim URDF importer 会转成 PhysxMimicJointAPI(gearing/offset) 硬约束。
#
# 用法（仓库根目录）：
#   python scripts/tools/prepare_deformable_v2_urdf.py
#   python scripts/tools/prepare_deformable_v2_urdf.py --raw <path> --out <path>
#
# 产物：source/agent_world/agent_world/assets/usd_files/deformable_V2/urdf/deformable_V2.urdf
# 下一步：bash scripts/tools/convert_deformable_v2_urdf.sh
# =============================================================================
"""Normalize the deformable_V2 (狗v3) SolidWorks URDF to repo RL conventions."""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_URDF_DIR = os.path.join(
    _REPO_ROOT,
    "source",
    "agent_world",
    "agent_world",
    "assets",
    "usd_files",
    "deformable_V2",
    "urdf",
)
_RAW = os.path.join(_URDF_DIR, "狗v3.urdf")
_OUT = os.path.join(_URDF_DIR, "deformable_V2.urdf")

# 绕 X 旋转 +90°：p_new = A @ p_old。旧 +Y(上)→新 +Z，旧 +X(车头)→新 +X，
# 旧 +Z→新 -Y（车体 Z 对称，左右无歧义）。
_A = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

_ROBOT_NAME = "deformable_V2"
_MESH_PREFIX_OLD = "package://狗v3/meshes/"
_MESH_PREFIX_NEW = "../meshes/"

# 关节 effort/velocity 归一化 (effort [N·m], velocity [rad/s])
# wheel_set / upper_leg 是平四从动边，由 mimic 硬约束跟随 leg（见 _MIMIC）。
_LIMITS = {
    "leg": (40.0, 17.0),
    "wheel_set": (40.0, 17.0),
    "upper_leg": (40.0, 17.0),
    "wheel": (5.0, 60.0),
}

# 平四闭链：URDF 是树，缺的第 4 条边（upper_leg↔wheel_set）用标准 <mimic> 表达。
# 几何实测（2026-09-16）：wheel_set 相对 base 姿态恒 45°（严格平行四边形），
# 故 θ_wheel_set = +1·θ_leg、θ_upper_leg = −1·θ_leg，offset 均为 0；
# 与 URDF 限位自洽（leg [0,1.36]、ws [0,1.36] 同号、upper [−1.36,0] 反号）。
# isaacsim URDF importer 会把 <mimic> 转成 PhysxMimicJointAPI(gearing/offset)。
_MIMIC = {
    "wheel_set": 1.0,   # joint_wheel_set_N ← joint_leg_N, multiplier=+1
    "upper_leg": -1.0,  # joint_upper_leg_N ← joint_leg_N, multiplier=-1
}

# 全向轮碰撞体球：半径 = 轮半径（实测 wheel_N.STL 直径 0.1539），
# 球心在轮 link 系 x=-0.0055（轮盘 x∈[-0.025,0.014] 的几何中心）。
_WHEEL_SPHERE_RADIUS = 0.0769
_WHEEL_SPHERE_XYZ = (-0.0055, 0.0, 0.0)


def _classify_joint(name: str) -> str | None:
    """把关节名归类为 leg / wheel_set / wheel / upper_leg（wheel_set 先于 wheel 判断）。"""
    if name.startswith("joint_upper_leg_"):
        return "upper_leg"
    if name.startswith("joint_wheel_set_"):
        return "wheel_set"
    if name.startswith("joint_wheel_"):
        return "wheel"
    if name.startswith("joint_leg_"):
        return "leg"
    return None


def _is_wheel_link(name: str) -> bool:
    return name.startswith("wheel_") and not name.startswith("wheel_set_")


def _mimic_reference(name: str) -> str:
    """joint_wheel_set_1 / joint_upper_leg_1 → joint_leg_1（同角的主动关节）。"""
    return "joint_leg_" + name.rsplit("_", 1)[1]


def _add_mimic(joint: ET.Element, multiplier: float) -> None:
    """写入 URDF 标准 <mimic>：该关节 = multiplier·θ_reference + offset。"""
    mimic = ET.SubElement(joint, "mimic")
    mimic.set("joint", _mimic_reference(joint.get("name")))
    mimic.set("multiplier", f"{multiplier:g}")
    mimic.set("offset", "0")


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
    """把 origin 的 xyz/rpy 从旧根坐标系重表达到新根坐标系（新 = A @ 旧）。"""
    xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()], dtype=float)
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    R = _rpy_to_R(*rpy)
    origin.set("xyz", _fmt(_A @ xyz))
    origin.set("rpy", _fmt(_R_to_rpy(_A @ R)))


def _rotate_inertia(inertia: ET.Element) -> None:
    """惯性张量在新惯性系下重表达：I_new = Aᵀ I_old A。

    仅 base_link 需要：它的 inertial origin 的 rpy 由 0 变为 A，惯性系相对
    link 系转了 A，分量必须同步旋转；子链接的惯性系随整体同步旋转，分量不变。
    """
    keys = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    ixx, ixy, ixz, iyy, iyz, izz = (float(inertia.get(k, "0")) for k in keys)
    I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    J = _A.T @ I @ _A
    for key, value in (
        ("ixx", J[0, 0]), ("ixy", J[0, 1]), ("ixz", J[0, 2]),
        ("iyy", J[1, 1]), ("iyz", J[1, 2]), ("izz", J[2, 2]),
    ):
        inertia.set(key, f"{value:.12g}")


def _replace_wheel_collision(link: ET.Element) -> None:
    for collision in link.findall("collision"):
        link.remove(collision)
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", {"xyz": _fmt(_WHEEL_SPHERE_XYZ), "rpy": "0 0 0"})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "sphere", {"radius": f"{_WHEEL_SPHERE_RADIUS:.12g}"})


def _fk_wheel_positions(root: ET.Element) -> dict[str, np.ndarray]:
    """在 base 系下按 URDF 零位（所有关节 q=0）正运动学求 wheel_N link 原点。"""
    joints = {}
    children: set[str] = set()
    adj: dict[str, list[str]] = {}
    for j in root.findall("joint"):
        name = j.get("name")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        joints[name] = j
        children.add(child)
        adj.setdefault(parent, []).append(name)

    all_links = [l.get("name") for l in root.findall("link")]
    roots = [l for l in all_links if l not in children]
    if len(roots) != 1:
        raise SystemExit(f"expected a single root link, got {roots}")

    T = {roots[0]: np.eye(4)}
    stack = [roots[0]]
    while stack:
        parent = stack.pop()
        for name in adj.get(parent, []):
            j = joints[name]
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            Tj = np.eye(4)
            Tj[:3, :3] = _rpy_to_R(*rpy)
            Tj[:3, 3] = xyz
            child = j.find("child").get("link")
            T[child] = T[parent] @ Tj
            stack.append(child)
    return T


def _report(root: ET.Element) -> None:
    T = _fk_wheel_positions(root)
    wheels = [name for name in T if _is_wheel_link(name)]
    print(">>> 自检（base 系，URDF 零位 q=0）:")
    zs = []
    for name in sorted(wheels):
        p = T[name][:3, 3]
        zs.append(p[2])
        print(f"    {name}: center=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
    if zs:
        coplanar = max(zs) - min(zs) < 1e-6
        clearance = _WHEEL_SPHERE_RADIUS - float(np.mean(zs))
        print(f"    四轮共面(新 Z): {'OK' if coplanar else 'FAIL'}  (z spread {max(zs)-min(zs):.2e})")
        print(f"    base 原点离地 ≈ {clearance:+.4f} m  (期望 ≈ 0.132)")
    if "base_link" in T:
        print(f"    base_link 原点(旋转后)=({T['base_link'][0,3]:+.4f}, "
              f"{T['base_link'][1,3]:+.4f}, {T['base_link'][2,3]:+.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=_RAW, help="SolidWorks 原始 URDF（保持不动）")
    parser.add_argument("--out", default=_OUT, help="归一化后 URDF 输出路径")
    parser.add_argument(
        "--no-mimic",
        dest="mimic",
        action="store_false",
        help="不写平四 <mimic> 闭链标签（默认写入：ws=+1·leg, upper=−1·leg）",
    )
    parser.set_defaults(mimic=True)
    args = parser.parse_args()

    if not os.path.isfile(args.raw):
        raise SystemExit(f"raw SolidWorks URDF not found: {args.raw}")

    tree = ET.parse(args.raw)
    root = tree.getroot()

    # 0) robot name
    root.set("name", _ROBOT_NAME)

    # 1) mesh 路径 package:// → 相对路径；2) 轮子碰撞体 → 球体；3) base_link 自身 origin 旋转
    mesh_fixed = 0
    for link in root.findall("link"):
        name = link.get("name")
        for tag in ("inertial", "visual", "collision"):
            for elem in link.findall(tag):
                for mesh in elem.iter("mesh"):
                    fn = mesh.get("filename", "")
                    if fn.startswith(_MESH_PREFIX_OLD):
                        mesh.set("filename", _MESH_PREFIX_NEW + fn[len(_MESH_PREFIX_OLD):])
                        mesh_fixed += 1
                origin = elem.find("origin")
                if origin is not None and name == "base_link":
                    _transform_origin(origin)
        if name == "base_link":
            inertia = link.find("inertial/inertia")
            if inertia is not None:
                _rotate_inertia(inertia)
        if _is_wheel_link(name):
            _replace_wheel_collision(link)

    # 4) 根关节（parent=base_link）origin 旋转；5) effort/velocity；6) wheel → continuous
    wheel_set_to_continuous = 0
    limits_normalized = 0
    mimic_added = 0
    for joint in root.findall("joint"):
        name = joint.get("name")
        kind = _classify_joint(name)

        parent = joint.find("parent")
        if parent is not None and parent.get("link") == "base_link":
            origin = joint.find("origin")
            if origin is not None:
                _transform_origin(origin)

        if kind is None:
            continue

        limit = joint.find("limit")
        if kind == "wheel":
            joint.set("type", "continuous")
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            for attr in ("lower", "upper"):
                if attr in limit.attrib:
                    del limit.attrib[attr]
            limit.set("effort", f"{_LIMITS['wheel'][0]:g}")
            limit.set("velocity", f"{_LIMITS['wheel'][1]:g}")
            wheel_set_to_continuous += 1
        else:
            if limit is None:
                raise SystemExit(f"joint {name} missing <limit>")
            limit.set("effort", f"{_LIMITS[kind][0]:g}")
            limit.set("velocity", f"{_LIMITS[kind][1]:g}")
            limits_normalized += 1

        # 7) 平四闭链：从动边写 <mimic>（放在 <limit> 之后）
        if args.mimic and kind in _MIMIC:
            _add_mimic(joint, _MIMIC[kind])
            mimic_added += 1

    ET.indent(tree, space="  ")
    tree.write(args.out, encoding="utf-8", xml_declaration=True)

    print(f">>> wrote {args.out}")
    print(f">>> mesh 路径改写: {mesh_fixed} 处 ({_MESH_PREFIX_OLD} → {_MESH_PREFIX_NEW})")
    print(f">>> 根坐标系旋转: 绕 X +90° (旧 +Y(上)→新 +Z, 旧 +X(车头)→新 +X)")
    print(f">>> effort/velocity 归一化: {limits_normalized} 个关节")
    print(f">>> wheel revolute→continuous: {wheel_set_to_continuous} 个关节")
    print(f">>> 平四 <mimic> 闭链: {mimic_added} 处 (ws=+1·leg, upper=−1·leg)"
          if args.mimic else ">>> 平四 <mimic> 闭链: 已禁用")
    print(f">>> robot name: {_ROBOT_NAME}")
    _report(tree.getroot())
    print(f">>> next: bash scripts/tools/convert_deformable_v2_urdf.sh")


if __name__ == "__main__":
    main()
