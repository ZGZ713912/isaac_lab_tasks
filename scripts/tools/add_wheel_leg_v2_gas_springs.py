#!/usr/bin/env python3
"""Add prismatic gas-spring closures and curve metadata to Wheel_leg_V2 USD.

The four-bar closures are authored as RevoluteJoint objects by
``convert_wheel_leg_v2_closed_urdf.py``.  This script adds the two gas-spring
PrismaticJoint objects separately, so the spring force model can be changed
without rebuilding the robot geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
root = Path(__file__).resolve().parents[2]
default_input = root / "source/agent_world/agent_world/assets/usd_files/Wheel_leg_V2/Wheel_leg_V2_closed.usd"
default_output = root / "source/agent_world/agent_world/assets/usd_files/Wheel_leg_V2/Wheel_leg_V2_closed_gas_spring.usd"
parser.add_argument("--input", type=Path, default=default_input)
parser.add_argument("--output", type=Path, default=default_output)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdPhysics


SPRINGS = (
    ("L_gas_spring", "LLL_link1", "LLL_link2"),
    ("R_gas_spring", "RRR_link1", "RRR_link2"),
)

# BKB0.45-063-172 from the supplied catalogue image.
MIN_LENGTH = 0.109
MAX_LENGTH = 0.172
NOMINAL_LENGTH = 0.1622728199088745
FORCE_AT_MIN = 260.0
FORCE_AT_MAX = 380.0
PRESSURE_MPA = 10.0

# Both body-local joint frames are defined from the current zero-pose spring
# axis.  The frame's local +Z is the prismatic axis.
LOCAL_ROT_0 = (0.00162597, -0.00162426, 0.70710492, 0.70710491)
LOCAL_ROT_1 = (0.70710489, 0.70710494, 0.00161519, -0.00163504)
LOCAL_POS_1 = (0.000004556, 0.1622728199088745, 0.0)


def _find_bodies(stage):
    wanted = {name for _, a, b in SPRINGS for name in (a, b)}
    found = {}
    for prim in stage.Traverse():
        if prim.GetName() in wanted and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            found.setdefault(prim.GetName(), prim.GetPath())
    missing = sorted(wanted - found.keys())
    if missing:
        raise RuntimeError(f"missing gas-spring rigid bodies: {missing}")
    return found


def _set_attr(prim, name, type_name, value):
    prim.CreateAttribute(name, type_name).Set(value)


def _author(input_path: Path, output_path: Path) -> None:
    stage = Usd.Stage.Open(str(input_path))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {input_path}")
    bodies = _find_bodies(stage)
    robot_root = stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else Sdf.Path("/urdf_v5_0")
    spring_root = robot_root.AppendChild("WheelLegV2GasSprings")

    for name, body0, body1 in SPRINGS:
        joint = UsdPhysics.PrismaticJoint.Define(stage, spring_root.AppendChild(name))
        joint.GetBody0Rel().SetTargets([bodies[body0]])
        joint.GetBody1Rel().SetTargets([bodies[body1]])
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        # The two gas-spring component origins are separated by the nominal
        # length.  The joint frames themselves must still coincide at q=0.
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(*LOCAL_POS_1))
        joint.GetLocalRot0Attr().Set(Gf.Quatf(*LOCAL_ROT_0))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(*LOCAL_ROT_1))
        joint.GetAxisAttr().Set(UsdPhysics.Tokens.z)
        joint.GetExcludeFromArticulationAttr().Set(True)
        # Isaac Sim USD uses meters for linear limits.
        joint.GetLowerLimitAttr().Set(MIN_LENGTH - NOMINAL_LENGTH)
        joint.GetUpperLimitAttr().Set(MAX_LENGTH - NOMINAL_LENGTH)

        prim = joint.GetPrim()
        _set_attr(prim, "wheelLegV2:gasSpring:pressureMpa", Sdf.ValueTypeNames.Float, PRESSURE_MPA)
        _set_attr(prim, "wheelLegV2:gasSpring:minLengthM", Sdf.ValueTypeNames.Float, MIN_LENGTH)
        _set_attr(prim, "wheelLegV2:gasSpring:maxLengthM", Sdf.ValueTypeNames.Float, MAX_LENGTH)
        _set_attr(prim, "wheelLegV2:gasSpring:nominalLengthM", Sdf.ValueTypeNames.Float, NOMINAL_LENGTH)
        _set_attr(prim, "wheelLegV2:gasSpring:forceAtMinN", Sdf.ValueTypeNames.Float, FORCE_AT_MIN)
        _set_attr(prim, "wheelLegV2:gasSpring:forceAtMaxN", Sdf.ValueTypeNames.Float, FORCE_AT_MAX)

    stage.GetRootLayer().Export(str(output_path))
    print(f">>> generated: {output_path}")
    print(">>> gas-spring joints: L_gas_spring, R_gas_spring")
    print(">>> curve: 260 N at 109 mm -> 380 N at 172 mm, pressure ~= 10 MPa")


if __name__ == "__main__":
    _author(args_cli.input.resolve(), args_cli.output.resolve())
    simulation_app.close()
