#!/usr/bin/env python3
"""Export the confirmed chassis geometry as a portable USD asset and animation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

OUT = Path(__file__).resolve().parent
WORKSPACE = OUT.parents[1]
SOURCE = WORKSPACE / "isaac_wheeled_rl_train/reports/v40_linkage_preview_20260914"
sys.path.insert(0, str(SOURCE))
from geometry import Assembly, COLORS, snapshot  # noqa: E402

PARTS = {"C0": "crank", "C1": "lower_coupler", "C2": "thigh", "C3": "rocker",
         "C4": "upper_coupler", "shank": "shank", "wheel": "wheel"}
NAMES = {"base": "base_link", **{f"{s}_{part}": f"{side}_{name}"
         for s, side in (("L", "left"), ("R", "right")) for part, name in PARTS.items()}}
FPS, DURATION = 60, 12


def record(path):
    data = path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def configure(stage, default):
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    prim = UsdGeom.Xform.Define(stage, default).GetPrim()
    stage.SetDefaultPrim(prim)
    return prim


def set_pose(ops, transform, height, time=Usd.TimeCode.Default()):
    position = transform[:3, 3] - np.array([0., 0., height])
    ops[0].Set(Gf.Vec3d(*position), time)
    ops[1].Set(Gf.Matrix4d(transform.T.tolist()).ExtractRotationQuat(), time)


def main():
    names = ("chassis.usdc", "preview.usdc", "kinematics.json", "manifest.json")
    if any((OUT / name).exists() for name in names):
        raise FileExistsError("refusing to overwrite an existing chassis export")
    assembly = Assembly()
    assert assembly.verify()["unchanged"]
    reference, measured = assembly.pose(68., repaired=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=OUT) as staging:
        staging = Path(staging)
        model = Usd.Stage.CreateNew(str(staging / names[0]))
        root = configure(model, "/Chassis")
        root.SetDisplayName("纯底盘")
        root.SetCustomData({"asset_kind": "kinematic_geometry_only", "physics_ready": False,
                            "reference_knee_inner_deg": 68.})
        triangle_count = 0
        for source_name, triangles in assembly.triangles.items():
            name = NAMES[source_name]
            key = source_name.split("_", 1)[-1] if source_name != "base" else "base"
            node = UsdGeom.Xform.Define(model, "/Chassis/" + name)
            node.GetPrim().SetCustomData({"source_component": source_name})
            ops = (node.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble),
                   node.AddOrientOp(UsdGeom.XformOp.PrecisionDouble))
            set_pose(ops, reference[source_name], measured["root_height_m"])
            mesh = UsdGeom.Mesh.Define(model, str(node.GetPath()) + "/Mesh")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(triangles.reshape(-1, 3).astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(triangles) * 3, dtype=np.int32)))
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*COLORS[key])])
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
            mesh.SetNormalsInterpolation("uniform")
            material = UsdShade.Material.Define(model, "/Chassis/Materials/" + key)
            shader = UsdShade.Shader.Define(model, str(material.GetPath()) + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[key]))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.48)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            triangle_count += len(triangles)
        model.GetRootLayer().Save()

        preview = Usd.Stage.CreateNew(str(staging / names[1]))
        configure(preview, "/World")
        preview.SetStartTimeCode(0)
        preview.SetEndTimeCode(FPS * DURATION)
        preview.SetTimeCodesPerSecond(FPS)
        preview.SetFramesPerSecond(FPS)
        chassis = UsdGeom.Xform.Define(preview, "/World/Chassis")
        chassis.GetPrim().GetReferences().AddReference("./chassis.usdc")
        lift = chassis.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        animation_ops = {name: UsdGeom.Xformable(preview.GetPrimAtPath("/World/Chassis/" + NAMES[name]))
                         .GetOrderedXformOps() for name in NAMES}
        max_key_gap = 0.
        for frame in range(FPS * DURATION + 1):
            angle = 57.5 + 22.5 * math.sin(frame / FPS * math.tau / DURATION)
            transforms, sample = assembly.pose(angle, repaired=True)
            lift.Set(Gf.Vec3d(0., 0., sample["root_height_m"]), frame)
            for name, matrix in transforms.items():
                set_pose(animation_ops[name], matrix, sample["root_height_m"], frame)
            max_key_gap = max(max_key_gap, *(s["max_axis_gap_m"] for s in sample["axes"].values()))
        for major, label, width, color in ((True, "Major", .0015, (.55, .61, .68)),
                                            (False, "Minor", .0008, (.25, .29, .34))):
            points = []
            for index in range(-20, 21):
                if (index % 10 == 0) != major:
                    continue
                value = index / 10
                points.extend(((-2, value, 0), (2, value, 0), (value, -2, 0), (value, 2, 0)))
            grid = UsdGeom.BasisCurves.Define(preview, "/World/Grid/" + label)
            grid.CreateTypeAttr("linear")
            grid.CreateCurveVertexCountsAttr([2] * (len(points) // 2))
            grid.CreatePointsAttr(points)
            grid.CreateWidthsAttr([width])
            grid.SetWidthsInterpolation("constant")
            grid.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdLux.DomeLight.Define(preview, "/World/Light").CreateIntensityAttr(250.)
        sun = UsdLux.DistantLight.Define(preview, "/World/Sun")
        sun.CreateIntensityAttr(700.)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(-35., -20., 20.))
        camera = UsdGeom.Camera.Define(preview, "/World/PreviewCamera")
        camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 50.))
        camera.CreateFocalLengthAttr(45.)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.05, 1.9, .95), Gf.Vec3d(-.03, 0., .24), Gf.Vec3d(0, 0, 1))
        camera.AddTransformOp().Set(view.GetInverse())
        preview.GetRootLayer().Save()

        # Re-open composed files and validate both exact poses and interpolation.
        model = Usd.Stage.Open(str(staging / names[0]))
        preview = Usd.Stage.Open(str(staging / names[1]))
        assert model.GetDefaultPrim().GetPath() == Sdf.Path("/Chassis")
        assert sum(p.IsA(UsdGeom.Mesh) for p in model.Traverse()) == 15
        for stage in (model, preview):
            assert not any("Physics" in str(api) for p in stage.Traverse() for api in p.GetAppliedSchemas())
        for source_name, triangles in assembly.triangles.items():
            mesh = UsdGeom.Mesh(model.GetPrimAtPath("/Chassis/" + NAMES[source_name] + "/Mesh"))
            assert np.array_equal(np.asarray(mesh.GetPointsAttr().Get()), triangles.reshape(-1, 3).astype(np.float32))
        max_transform_error, max_midpoint_gap = 0., 0.
        pairs = (("C0", "C0_A", "C1", "C1_A"), ("C1", "C1_D", "C3", "C3_D"),
                 ("C3", "C3_B", "C4", "C4_B"), ("C2", "C2_P", "C3", "C3_P"),
                 ("C4", "C4_E", "shank", "shank_E"))
        for frame in range(FPS * DURATION):
            exact, _ = assembly.pose(57.5 + 22.5 * math.sin(frame / FPS * math.tau / DURATION), True)
            cache = UsdGeom.XformCache(frame)
            for name in NAMES:
                actual = np.asarray(cache.GetLocalToWorldTransform(preview.GetPrimAtPath("/World/Chassis/" + NAMES[name]))).T
                max_transform_error = max(max_transform_error, float(np.abs(actual - exact[name]).max()))
            cache.SetTime(frame + .5)
            for side, linkage in assembly.models.items():
                def pin(part, feature):
                    transform = np.asarray(cache.GetLocalToWorldTransform(
                        preview.GetPrimAtPath("/World/Chassis/" + NAMES[side + "_" + part]))).T
                    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)
                    return (transform @ np.r_[linkage.xy[feature], 0., 1.])[:3]
                for p0, a0, p1, a1 in pairs:
                    max_midpoint_gap = max(max_midpoint_gap, float(np.linalg.norm(pin(p0, a0) - pin(p1, a1))))
        assert max_transform_error < 1e-10 and max_midpoint_gap < 1e-5
        assert snapshot() == assembly.protected
        joints = {joint.get("name"): {key: joint.find(key).attrib
                  for key in ("origin", "axis", "parent", "child")} for joint in assembly.urdf.findall("joint")}
        kinematics = {"schema_version": 1, "asset_name": "chassis", "display_name": "纯底盘",
                      "component_names": NAMES, "source_joint_frames": joints,
                      "source_to_control_rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                      "preview_hip_q_rad": {k: v for k, v in measured["raw_q"].items() if k.endswith("joint1")},
                      "sides": {s: {"axes_local_xy_m": {k: v.tolist() for k, v in m.xy.items()},
                                    "upper_branch": m.sign_upper, "lower_branch": m.sign_lower,
                                    "source_knee_origin_z_rad": float(m.theta0), "source_knee_axis_sign": float(m.axis_sign)}
                                for s, m in assembly.models.items()},
                      "physics_ready": False, "mass_and_actuator_mapping_verified": False}
        (staging / names[2]).write_text(json.dumps(kinematics, indent=2, ensure_ascii=False) + "\n")
        manifest = {"schema_version": 1, "name": "chassis", "display_name": "纯底盘",
                    "status": "user_confirmed_kinematic_geometry_not_dynamics", "physics_ready": False,
                    "visual_bodies": 15, "triangles": triangle_count, "units": "m", "up_axis": "Z",
                    "reference_knee_inner_deg": 68., "preview_range_deg": [35, 80],
                    "animation": {"seconds": DURATION, "samples_per_second": FPS,
                                  "frames": FPS * DURATION + 1, "interpolation": "translate + quaternion orient"},
                    "validation": {"source_vertices_equal": True, "source_files_unchanged": True,
                                   "max_keyframe_pin_gap_m": max_key_gap,
                                   "max_keyframe_transform_error": max_transform_error,
                                   "max_half_frame_pin_gap_m": max_midpoint_gap},
                    "source_sha256": {str(p): record(p)["sha256"] for p in
                                      (SOURCE / "geometry.py", SOURCE / "geometry_audit.json",
                                       SOURCE.parent / "v40_linkage_repair_20260914/independent_geometry_mass/metrology.json")},
                    "files": {name: record(staging / name) for name in names[:3]},
                    "tools": {"export.py": record(Path(__file__)), "README.md": record(OUT / "README.md")}}
        (staging / names[3]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        for name in names:
            os.link(staging / name, OUT / name)
        print(json.dumps({"output": str(OUT), "triangles": triangle_count, **manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
