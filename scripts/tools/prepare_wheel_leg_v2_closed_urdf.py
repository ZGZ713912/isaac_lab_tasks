#!/usr/bin/env python3
"""Add the two four-bar closure joints to the prepared Wheel_leg_V2 URDF.

Isaac Sim's URDF importer supports the non-standard ``loop_joint`` tag and
imports a ``type=\"revolute\"`` loop as a PhysX revolute joint.  The source
URDF remains unchanged; this script writes a closed-chain variant next to it.

The closure points are the CAD pivot locations where LL_link4/RR_link4 meet
L_link3/R_link3.  The values below are expressed in each link's local frame.
They were obtained from the matching STL pivot surfaces after the V2 global
coordinate conversion.  The small residual mesh gap is below 0.03 mm.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "source/agent_world/agent_world/assets/usd_files/Wheel_leg_V2/urdf/urdf_v5.0.urdf"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name("urdf_v5.0_closed.urdf")

# Each tuple is (body/link, local pivot xyz, local joint-frame rpy).
# The local joint-frame z axis is the revolute axis.
LOOP_JOINTS = (
    {
        "name": "L_four_bar_closure",
        "link1": ("LL_link4", (0.00827712, -0.00724487, -0.36980174)),
        "link2": ("L_link3", (0.23542012, 0.08457446, -0.37359826)),
        "frame_rpy": (0.0, 0.0, 0.0),
        "axis": "+Y",
    },
    {
        "name": "R_four_bar_closure",
        "link1": ("RR_link4", (0.00565100, -0.00943867, 0.01919846)),
        "link2": ("R_link3", (-0.08800131, 0.23572013, 0.02300155)),
        "frame_rpy": (math.pi, 0.0, 0.0),
        "axis": "-Y",
    },
)


def fmt(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def ensure_link_names(root: ET.Element) -> None:
    links = {link.get("name") for link in root.findall("link")}
    expected = {link for loop in LOOP_JOINTS for link, _ in (loop["link1"], loop["link2"])}
    missing = sorted(expected - links)
    if missing:
        raise ValueError(f"closure link(s) missing from URDF: {missing}")


def add_loop_joints(root: ET.Element) -> None:
    old = {joint.get("name"): joint for joint in root.findall("loop_joint")}
    for loop in LOOP_JOINTS:
        if loop["name"] in old:
            continue
        elem = ET.SubElement(root, "loop_joint", {"name": loop["name"], "type": "revolute"})
        for tag in ("link1", "link2"):
            link_name, xyz = loop[tag]
            # The importer uses the local frame's z axis as the revolute axis.
            ET.SubElement(
                elem,
                tag,
                {
                    "link": link_name,
                    "xyz": fmt(xyz),
                    "rpy": fmt(loop["frame_rpy"]),
                },
            )


def validate(root: ET.Element) -> None:
    loops = {loop.get("name"): loop for loop in root.findall("loop_joint")}
    for expected in LOOP_JOINTS:
        elem = loops.get(expected["name"])
        if elem is None or elem.get("type") != "revolute":
            raise ValueError(f"invalid loop joint: {expected['name']}")
        for tag in ("link1", "link2"):
            child = elem.find(tag)
            if child is None or child.get("link") != expected[tag][0]:
                raise ValueError(f"invalid {tag} in {expected['name']}")
            xyz = tuple(float(value) for value in child.get("xyz").split())
            if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
                raise ValueError(f"invalid pivot xyz in {expected['name']}/{tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tree = ET.parse(args.input)
    root = tree.getroot()
    ensure_link_names(root)
    add_loop_joints(root)
    validate(root)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)

    print(f"wrote: {args.output}")
    print("loop joints: L_four_bar_closure, R_four_bar_closure")
    print("type: revolute (Isaac Sim URDF importer -> PhysX RevoluteJoint)")
    print("spring model: not included yet")


if __name__ == "__main__":
    main()
