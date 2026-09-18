#!/usr/bin/env python3
"""Remove the articulation root from the closed-loop Wheel_leg_V2 USD.

PhysX articulations are tree-structured.  The closed-loop validation asset
must therefore be simulated as ordinary rigid bodies connected by USD joints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
root = Path(__file__).resolve().parents[2]
default_input = root / "source/agent_world/agent_world/assets/usd_files/Wheel_leg_V2/Wheel_leg_V2_closed_gas_spring.usd"
default_output = root / "source/agent_world/agent_world/assets/usd_files/Wheel_leg_V2/Wheel_leg_V2_closed_gas_spring_rigid.usd"
parser.add_argument("--input", type=Path, default=default_input)
parser.add_argument("--output", type=Path, default=default_output)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import PhysxSchema, Usd, UsdPhysics


def main() -> None:
    stage = Usd.Stage.Open(str(args_cli.input.resolve()))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {args_cli.input}")

    removed = []
    for prim in stage.Traverse():
        for api in (UsdPhysics.ArticulationRootAPI, PhysxSchema.PhysxArticulationAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
                removed.append((str(prim.GetPath()), api.__name__))

    stage.GetRootLayer().Export(str(args_cli.output.resolve()))
    print(f">>> generated non-articulation closed-loop USD: {args_cli.output.resolve()}")
    print(f">>> removed articulation APIs: {len(removed)}")
    print(">>> constraints remain as ordinary PhysX RevoluteJoint/PrismaticJoint bodies")


if __name__ == "__main__":
    main()
    simulation_app.close()
