#!/usr/bin/env python3
"""Convert Wheel_leg_V2 to USD and author the four-bar RevoluteJoint closures.

Isaac Sim 5.1's URDF importer only accepts spherical ``loop_joint`` entries;
it skips revolute loop tags.  Therefore the URDF is imported as a tree first,
then the two closed-chain edges are authored directly as USD/PhysX revolute
joints with explicit body-local frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
root = Path(__file__).resolve().parents[2]
default_input = root / "source/agent_world/agent_world/assets/usd_files/Wheel_leg_V2/urdf/urdf_v5.0.urdf"
default_output = root / "source/agent_world/agent_world/assets/usd_files/Wheel_leg_V2/Wheel_leg_V2_closed.usd"
parser.add_argument("--input", type=Path, default=default_input)
parser.add_argument("--output", type=Path, default=default_output)
parser.add_argument("--fix-base", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path


LOOP_JOINTS = (
    {
        "name": "L_four_bar_closure",
        "body0": "LL_link4",
        "body1": "L_link3",
        "pos0": (0.00827712, -0.00724487, -0.36980174),
        "pos1": (0.23542012, 0.08457446, -0.37359826),
        "rot": (1.0, 0.0, 0.0, 0.0),
    },
    {
        "name": "R_four_bar_closure",
        "body0": "RR_link4",
        "body1": "R_link3",
        "pos0": (0.00565100, -0.00943867, 0.01919846),
        "pos1": (-0.08800131, 0.23572013, 0.02300155),
        # 180 degrees around X flips the local +Z axis to the right-side -Y.
        "rot": (0.0, 1.0, 0.0, 0.0),
    },
)


def _find_body_paths(stage):
    from pxr import UsdPhysics

    wanted = {item["body0"] for item in LOOP_JOINTS} | {item["body1"] for item in LOOP_JOINTS}
    found = {}
    for prim in stage.Traverse():
        if prim.GetName() in wanted and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            found.setdefault(prim.GetName(), prim.GetPath())
    missing = sorted(wanted - found.keys())
    if missing:
        raise RuntimeError(f"cannot find imported rigid bodies: {missing}; found={found}")
    return found


def _author_revolute_closures(usd_path: str) -> None:
    from pxr import Gf, Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"failed to open generated USD: {usd_path}")
    body_paths = _find_body_paths(stage)

    robot_root = stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else Sdf.Path("/urdf_v5_0")
    closure_root = robot_root.AppendChild("WheelLegV2Closures")
    for item in LOOP_JOINTS:
        joint = UsdPhysics.RevoluteJoint.Define(stage, closure_root.AppendChild(item["name"]))
        joint.GetBody0Rel().SetTargets([body_paths[item["body0"]]])
        joint.GetBody1Rel().SetTargets([body_paths[item["body1"]]])
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*item["pos0"]))
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(*item["pos1"]))
        joint.GetLocalRot0Attr().Set(Gf.Quatf(*item["rot"]))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(*item["rot"]))
        joint.GetAxisAttr().Set(UsdPhysics.Tokens.z)
        joint.GetExcludeFromArticulationAttr().Set(True)

    stage.GetRootLayer().Save()
    print(">>> authored PhysX revolute closures:")
    for item in LOOP_JOINTS:
        print(f"    {item['name']}: {body_paths[item['body0']]} <-> {body_paths[item['body1']]}")


def _validate_usd(usd_path: str) -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"failed to open generated USD: {usd_path}")

    found = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            found[prim.GetName()] = prim

    missing = [item["name"] for item in LOOP_JOINTS if item["name"] not in found]
    if missing:
        available = sorted(found)
        raise RuntimeError(f"missing loop revolute joint(s): {missing}; available={available}")

    for item in LOOP_JOINTS:
        name = item["name"]
        prim = found[name]
        body0 = prim.GetRelationship("physics:body0").GetTargets()
        body1 = prim.GetRelationship("physics:body1").GetTargets()
        axis = prim.GetAttribute("physics:axis").Get()
        if len(body0) != 1 or len(body1) != 1:
            raise RuntimeError(f"{name} must have exactly one body0/body1 target")
        print(f"{name}: body0={body0[0]} body1={body1[0]} axis={axis}")

    print(">>> [OK] both Wheel_leg_V2 four-bar loop joints are PhysX revolute joints")


def main() -> None:
    input_path = args_cli.input.resolve()
    output_path = args_cli.output.resolve()
    if not check_file_path(str(input_path)):
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=str(input_path),
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        fix_base=args_cli.fix_base,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
            target_type="none",
        ),
    )
    converter = UrdfConverter(cfg)
    print(f">>> generated: {converter.usd_path}")
    _author_revolute_closures(str(converter.usd_path))
    _validate_usd(str(converter.usd_path))


if __name__ == "__main__":
    main()
    simulation_app.close()
