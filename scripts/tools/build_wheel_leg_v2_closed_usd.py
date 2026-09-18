#!/usr/bin/env python3
"""Author the Wheel_leg_V2 closed-chain USD from scratch (reference-style).

Follows ``闭链参考/urdf/tools/build_chassis_closedchain.py::export_usd``:

* ordinary rigid bodies with mass properties and mesh visuals
* tree joints as ``UsdPhysics.RevoluteJoint`` (axis = local Z)
* loop closures as ``UsdPhysics.SphericalJoint`` with unit local rotations
* gas spring as ``UsdPhysics.PrismaticJoint``
* every closure carries ``physics:excludeFromArticulation = true``

Closure points were derived from the V2 meshes (bore detection) and cross
checked against the reference hole layout, and validated so that both link
frames map to the same world point at q=0.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

from isaaclab.app import AppLauncher

import numpy as np

ROBOT_NAME = "Wheel_leg_V2"
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "source/agent_world/agent_world/assets/usd_files/Wheel_leg_V2"
URDF = ASSET_DIR / "urdf/urdf_v5.0.urdf"
MESH_DIR = ASSET_DIR / "meshes"
DEFAULT_OUTPUT = ASSET_DIR / f"{ROBOT_NAME}.usd"
DEFAULT_CONSTRAINTS = ASSET_DIR / "constraints.json"

STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])

# Closure holes expressed in each link's local frame (x, y); the pin axis is
# local Z.  See constraints.json for the derivation evidence.
FOUR_BAR_CLOSURES = (
    {
        "name": "L_four_bar_P",
        "body0": "L_link1",
        "body1": "LL_link3",
        "hole0": (0.0, 0.1135),
        "hole1": (0.0114, 0.1348),
    },
    {
        "name": "R_four_bar_P",
        "body0": "R_link1",
        "body1": "RR_link3",
        "hole0": (0.0, 0.1135),
        "hole1": (0.0114, 0.1348),
    },
    {
        "name": "L_four_bar_E",
        "body0": "LL_link4",
        "body1": "L_link2",
        "hole0": (0.0, 0.0965),
        "hole1": (0.01711, -0.06478),
    },
    {
        "name": "R_four_bar_E",
        "body0": "RR_link4",
        "body1": "R_link2",
        "hole0": (0.0, 0.0965),
        "hole1": (0.01711, -0.06478),
    },
)

GAS_SPRINGS = (
    {"name": "L_gas_spring", "body0": "LLL_link1", "body1": "LLL_link2"},
    {"name": "R_gas_spring", "body0": "RRR_link1", "body1": "RRR_link2"},
)

GAS_MIN_LENGTH = 0.109
GAS_MAX_LENGTH = 0.172

LINK_COLORS = {
    "base_link": (0.60, 0.65, 0.74),
    "L_link1": (0.48, 0.54, 0.57), "R_link1": (0.48, 0.54, 0.57),
    "L_link2": (0.10, 0.40, 0.90), "R_link2": (0.10, 0.40, 0.90),
    "L_link3": (0.08, 0.10, 0.14), "R_link3": (0.08, 0.10, 0.14),
    "LL_link1": (0.57, 0.23, 0.78), "RR_link1": (0.57, 0.23, 0.78),
    "LL_link2": (0.85, 0.60, 0.10), "RR_link2": (0.85, 0.60, 0.10),
    "LL_link3": (0.08, 0.66, 0.38), "RR_link3": (0.08, 0.66, 0.38),
    "LL_link4": (0.95, 0.30, 0.06), "RR_link4": (0.95, 0.30, 0.06),
    "LLL_link1": (0.62, 0.62, 0.62), "RRR_link1": (0.62, 0.62, 0.62),
    "LLL_link2": (0.34, 0.34, 0.34), "RRR_link2": (0.34, 0.34, 0.34),
}


# --------------------------------------------------------------------------
# URDF parsing and kinematics
# --------------------------------------------------------------------------
def rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def transform(xyz, rpy):
    m = np.eye(4)
    m[:3, :3] = rpy_to_matrix(rpy)
    m[:3, 3] = xyz
    return m


def numbers(values):
    return " ".join(format(float(v), ".12g") for v in np.asarray(values).ravel())


def quaternion(matrix):
    m = matrix[:3, :3]
    t = np.trace(m)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


def read_stl(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    assert len(raw) == 84 + count * 50, f"binary STL expected: {path}"
    return np.frombuffer(raw, dtype=STL_DTYPE, offset=84, count=count)


def parse_urdf(path):
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        origin = inertial.find("origin")
        inertia = inertial.find("inertia")
        keys = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        tensor = np.array(
            (
                (float(inertia.get("ixx")), float(inertia.get("ixy")), float(inertia.get("ixz"))),
                (float(inertia.get("ixy")), float(inertia.get("iyy")), float(inertia.get("iyz"))),
                (float(inertia.get("ixz")), float(inertia.get("iyz")), float(inertia.get("izz"))),
            )
        )
        links[link.get("name")] = {
            "name": link.get("name"),
            "mass": float(inertial.find("mass").get("value")),
            "com": np.array([float(v) for v in origin.get("xyz", "0 0 0").split()]),
            "inertia": tensor,
            "mesh": None,
        }
        visual = link.find("visual")
        if visual is not None and visual.find("geometry/mesh") is not None:
            links[link.get("name")]["mesh"] = visual.find("geometry/mesh").get("filename")

    joints = []
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        joints.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "origin": transform(
                    [float(v) for v in origin.get("xyz", "0 0 0").split()],
                    [float(v) for v in origin.get("rpy", "0 0 0").split()],
                ),
                "axis": np.array([float(v) for v in joint.find("axis").get("xyz").split()]),
            }
        )
    return links, joints


def forward_kinematics(joints):
    frames = {"base_link": np.eye(4)}
    remaining = list(joints)
    while remaining:
        progress = False
        for joint in list(remaining):
            if joint["parent"] in frames:
                frames[joint["child"]] = frames[joint["parent"]] @ joint["origin"]
                remaining.remove(joint)
                progress = True
        if not progress:
            raise RuntimeError(f"unresolved joints: {[j['name'] for j in remaining]}")
    return frames


def world_point(frames, link, local_xyz):
    m = frames[link]
    return m[:3, :3] @ np.asarray(local_xyz, dtype=float) + m[:3, 3]


# --------------------------------------------------------------------------
# closure / gas spring derivation
# --------------------------------------------------------------------------
def derive_four_bar(frames):
    derived = []
    for item in FOUR_BAR_CLOSURES:
        h0 = np.array([item["hole0"][0], item["hole0"][1], 0.0])
        h1 = np.array([item["hole1"][0], item["hole1"][1], 0.0])
        w0 = world_point(frames, item["body0"], h0)
        w1 = world_point(frames, item["body1"], h1)
        # common world point on the shared pin axis: keep in-plane values, use
        # the lateral midpoint so both local frames map to one point.
        common = np.array([0.5 * (w0[0] + w1[0]), 0.5 * (w0[1] + w1[1]), 0.5 * (w0[2] + w1[2])])
        f0, f1 = frames[item["body0"]], frames[item["body1"]]
        local0 = f0[:3, :3].T @ (common - f0[:3, 3])
        local1 = f1[:3, :3].T @ (common - f1[:3, 3])
        xz_error = math.hypot(w0[0] - w1[0], w0[2] - w1[2])
        derived.append(
            {
                "name": item["name"],
                "type": "spherical",
                "body0": item["body0"],
                "body1": item["body1"],
                "local_pos0_m": local0.tolist(),
                "local_pos1_m": local1.tolist(),
                "hole0_local_xy_m": list(item["hole0"]),
                "hole1_local_xy_m": list(item["hole1"]),
                "world_point0_m": w0.tolist(),
                "world_point1_m": w1.tolist(),
                "world_xz_error_m": xz_error,
                "world_lateral_delta_m": float(w0[1] - w1[1]),
            }
        )
    return derived


def derive_gas_springs(frames):
    derived = []
    for item in GAS_SPRINGS:
        f0, f1 = frames[item["body0"]], frames[item["body1"]]
        p0 = f0[:3, 3]
        p1 = f1[:3, 3]
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        z = delta / length
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(z, helper))) > 0.99:
            helper = np.array([1.0, 0.0, 0.0])
        x = np.cross(helper, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        world_rot = np.column_stack([x, y, z])
        local_pos0 = np.zeros(3)
        local_pos1 = f1[:3, :3].T @ (p0 - p1)
        derived.append(
            {
                "name": item["name"],
                "type": "prismatic",
                "body0": item["body0"],
                "body1": item["body1"],
                "local_pos0_m": local_pos0.tolist(),
                "local_pos1_m": local_pos1.tolist(),
                "local_rot0_wxyz": quaternion(np.column_stack([f0[:3, :3].T @ world_rot[:, i] for i in range(3)])).tolist(),
                "local_rot1_wxyz": quaternion(np.column_stack([f1[:3, :3].T @ world_rot[:, i] for i in range(3)])).tolist(),
                "nominal_length_m": length,
                "lower_limit_m": GAS_MIN_LENGTH - length,
                "upper_limit_m": GAS_MAX_LENGTH - length,
                "axis_world": z.tolist(),
            }
        )
    return derived


# --------------------------------------------------------------------------
# USD authoring
# --------------------------------------------------------------------------
def build_usd(output, links, joints, frames, four_bar, gas_springs):
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(root)

    def pose(prim, matrix):
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*matrix[:3, 3]))
        w, x, y, z = quaternion(matrix)
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))

    def quatf(matrix):
        w, x, y, z = quaternion(matrix)
        return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))

    from pxr import Vt

    def mesh_prim(path, link):
        geometry = link["mesh"]
        if geometry is None:
            return None
        relative = geometry.split("meshes/")[-1]
        records = read_stl(MESH_DIR / relative)
        triangle_count = len(records)
        vertices = np.ascontiguousarray(records["vertices"].reshape(-1, 3), dtype=np.float32)
        # one faceVertexCount entry per triangle; three indices per triangle
        counts = np.full(triangle_count, 3, dtype=np.int32)
        indices = np.arange(triangle_count * 3, dtype=np.int32)
        assert int(counts.sum()) == len(indices), "invalid triangle topology"
        normals = np.ascontiguousarray(records["normal"].repeat(3, axis=0), dtype=np.float32)

        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([vertices.min(0), vertices.max(0)])))
        return mesh

    for name, link in links.items():
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim()
        pose(body, frames[name])
        UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
        mass_api = UsdPhysics.MassAPI.Apply(body)
        mass_api.CreateMassAttr(link["mass"])
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*link["com"]))
        values, axes = np.linalg.eigh(link["inertia"])
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        values = np.maximum(values, 1e-8)
        principal = np.eye(4)
        principal[:3, :3] = axes
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
        mass_api.CreatePrincipalAxesAttr(quatf(principal))

        visual = mesh_prim(f"/Robot/{name}/Visual", link)
        if visual is not None:
            visual.CreateDoubleSidedAttr(True)
            visual.CreateDisplayColorAttr([Gf.Vec3f(*LINK_COLORS.get(name, (0.78, 0.80, 0.86)))])

        collision = mesh_prim(f"/Robot/{name}/Collision", link)
        if collision is not None:
            UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr("convexHull")
            UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdGeom.Imageable(collision.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        others = [f"/Robot/{other}" for other in links if other != name]
        UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets(others)

    base = stage.GetPrimAtPath("/Robot/base_link")
    UsdPhysics.ArticulationRootAPI.Apply(base)
    base.AddAppliedSchema("PhysxArticulationAPI")
    base.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)

    for joint in joints:
        prim = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/tree_joints/{joint['name']}")
        prim.CreateBody0Rel().SetTargets([f"/Robot/{joint['parent']}"])
        prim.CreateBody1Rel().SetTargets([f"/Robot/{joint['child']}"])
        origin = joint["origin"]
        alignment = transform([0, 0, 0], [math.pi, 0, 0]) if joint["axis"][2] < 0 else np.eye(4)
        prim.CreateAxisAttr("Z")
        prim.CreateLocalPos0Attr(Gf.Vec3f(*origin[:3, 3]))
        prim.CreateLocalPos1Attr(Gf.Vec3f(0))
        prim.CreateLocalRot0Attr(quatf(origin @ alignment))
        prim.CreateLocalRot1Attr(quatf(alignment))
        prim.CreateExcludeFromArticulationAttr(False)

    for closure in four_bar:
        prim = UsdPhysics.SphericalJoint.Define(stage, f"/Robot/loop_joints/{closure['name']}")
        prim.CreateBody0Rel().SetTargets([f"/Robot/{closure['body0']}"])
        prim.CreateBody1Rel().SetTargets([f"/Robot/{closure['body1']}"])
        prim.CreateLocalPos0Attr(Gf.Vec3f(*closure["local_pos0_m"]))
        prim.CreateLocalPos1Attr(Gf.Vec3f(*closure["local_pos1_m"]))
        prim.CreateLocalRot0Attr(Gf.Quatf(1))
        prim.CreateLocalRot1Attr(Gf.Quatf(1))
        prim.CreateExcludeFromArticulationAttr(True)

    for spring in gas_springs:
        prim = UsdPhysics.PrismaticJoint.Define(stage, f"/Robot/loop_joints/{spring['name']}")
        prim.CreateBody0Rel().SetTargets([f"/Robot/{spring['body0']}"])
        prim.CreateBody1Rel().SetTargets([f"/Robot/{spring['body1']}"])
        prim.CreateLocalPos0Attr(Gf.Vec3f(*spring["local_pos0_m"]))
        prim.CreateLocalPos1Attr(Gf.Vec3f(*spring["local_pos1_m"]))
        prim.CreateLocalRot0Attr(Gf.Quatf(*spring["local_rot0_wxyz"]))
        prim.CreateLocalRot1Attr(Gf.Quatf(*spring["local_rot1_wxyz"]))
        prim.CreateAxisAttr("Z")
        prim.CreateLowerLimitAttr(float(spring["lower_limit_m"]))
        prim.CreateUpperLimitAttr(float(spring["upper_limit_m"]))
        prim.CreateExcludeFromArticulationAttr(True)

    stage.GetRootLayer().Save()
    return stage


def validate(output):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(output))
    rigid = revolute = spherical = prismatic = 0
    excluded_ok = True
    meshes = 0
    invalid_meshes = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid += 1
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute += 1
        if prim.IsA(UsdPhysics.SphericalJoint):
            spherical += 1
            excluded_ok &= bool(UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get())
        if prim.IsA(UsdPhysics.PrismaticJoint):
            prismatic += 1
            excluded_ok &= bool(UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get())
        if prim.IsA(UsdGeom.Mesh):
            meshes += 1
            counts = prim.GetAttribute("faceVertexCounts").Get() or []
            indices = prim.GetAttribute("faceVertexIndices").Get() or []
            points = prim.GetAttribute("points").Get() or []
            if int(sum(counts)) != len(indices) or len(indices) % 3 != 0 or max(indices, default=-1) >= len(points):
                invalid_meshes.append(str(prim.GetPath()))
    return {
        "rigid_bodies": rigid,
        "revolute_joints": revolute,
        "spherical_joints": spherical,
        "prismatic_joints": prismatic,
        "closures_excluded": excluded_ok,
        "meshes": meshes,
        "invalid_meshes": invalid_meshes,
    }


def main(args):
    links, joints = parse_urdf(URDF)
    frames = forward_kinematics(joints)
    four_bar = derive_four_bar(frames)
    gas_springs = derive_gas_springs(frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_usd(args.output, links, joints, frames, four_bar, gas_springs)
    summary = validate(args.output)

    constraints = {
        "robot": ROBOT_NAME,
        "urdf": str(URDF.relative_to(REPO_ROOT)),
        "usd": str(args.output.relative_to(REPO_ROOT)),
        "method": "reference-style: SphericalJoint closures + excludeFromArticulation",
        "tree_joint_count": len(joints),
        "closures": [
            {
                "name": c["name"],
                "type": c["type"],
                "body0": c["body0"],
                "body1": c["body1"],
                "local_pos0_m": c["local_pos0_m"],
                "local_pos1_m": c["local_pos1_m"],
                "hole0_local_xy_m": c["hole0_local_xy_m"],
                "hole1_local_xy_m": c["hole1_local_xy_m"],
                "world_point0_m": c["world_point0_m"],
                "world_point1_m": c["world_point1_m"],
                "world_xz_error_m": c["world_xz_error_m"],
                "world_lateral_delta_m": c["world_lateral_delta_m"],
            }
            for c in four_bar
        ],
        "gas_springs": gas_springs,
        "catalog": {
            "model": "BKB0.45-063-172",
            "pressure_mpa": 10.0,
            "min_length_m": GAS_MIN_LENGTH,
            "max_length_m": GAS_MAX_LENGTH,
            "force_note": "260-380 N curve not modelled (closure only)",
        },
        "summary": summary,
    }
    args.constraints.write_text(json.dumps(constraints, indent=2), encoding="utf-8")

    print(f">>> generated USD: {args.output}")
    print(f">>> constraints:   {args.constraints}")
    print(f">>> {summary}")
    for c in four_bar:
        print(
            f"    {c['name']:14s} XZ err={c['world_xz_error_m']:.6f} m  "
            f"lateral={c['world_lateral_delta_m']:+.6f} m"
        )
    for s in gas_springs:
        print(
            f"    {s['name']:14s} nominal={s['nominal_length_m']:.6f} m  "
            f"limits=[{s['lower_limit_m']:.5f}, {s['upper_limit_m']:.5f}] m"
        )


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _parser.add_argument("--constraints", type=Path, default=DEFAULT_CONSTRAINTS)
    AppLauncher.add_app_launcher_args(_parser)
    _args = _parser.parse_args()
    _launcher = AppLauncher(_args)
    _app = _launcher.app
    main(_args)
    _app.close()
