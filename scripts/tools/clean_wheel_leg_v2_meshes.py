#!/usr/bin/env python3
"""Remove exporter-duplicated sub-solids from Wheel_leg_V2 right-leg STLs.

The SolidWorks URDF exporter baked one 324-triangle solid (nominal world
position ~= ``(-0.195, -0.179, 0.059)``) into several right-leg link meshes in
addition to its rightful owner ``RR_link4.STL``.  Commit f0d3200 cleaned the
left-leg meshes but the right ones were missed, so every contaminated right
link renders a phantom copy of that part (and its ``convexHull`` collision
spans the main body plus the far-away duplicate).

This tool rebuilds the contaminated STLs without the duplicated sub-solid,
leaving the remaining triangle records (normals, vertices, attribute bytes)
byte-for-byte unchanged.  It also runs a cross-link duplicate check as a
regression guard.

Run with any Python that has numpy::

    python3 scripts/tools/clean_wheel_leg_v2_meshes.py
"""

from __future__ import annotations

import argparse
import math
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

ROBOT_NAME = "Wheel_leg_V2"
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "source/agent_world/agent_world/assets/usd_files" / ROBOT_NAME
MESH_DIR = ASSET_DIR / "meshes"
URDF = ASSET_DIR / "urdf/urdf_v5.0.urdf"

STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])

# link -> the file whose legitimate part defines the duplicated solid
DUPLICATE_SOURCE = "RR_link4"
# right-leg files that wrongly contain a copy of DUPLICATE_SOURCE's solid
CONTAMINATED = ("R_link1", "R_link2", "R_link3", "RRR_link1", "RRR_link2")
# expected triangle count of the duplicated solid (sanity guard)
DUPLICATE_FACES = 324


# --------------------------------------------------------------------------
# URDF kinematics (kept independent of the USD build script)
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


def parse_urdf_frames(path):
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    joints = []
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        joints.append(
            (
                joint.find("parent").get("link"),
                joint.find("child").get("link"),
                transform(
                    [float(v) for v in origin.get("xyz", "0 0 0").split()],
                    [float(v) for v in origin.get("rpy", "0 0 0").split()],
                ),
            )
        )
    frames = {"base_link": np.eye(4)}
    remaining = list(joints)
    while remaining:
        progress = False
        for joint in list(remaining):
            if joint[0] in frames:
                frames[joint[1]] = frames[joint[0]] @ joint[2]
                remaining.remove(joint)
                progress = True
        if not progress:
            raise RuntimeError(f"unresolved joints: {[j for j in remaining]}")
    return frames


# --------------------------------------------------------------------------
# binary STL IO / connectivity
# --------------------------------------------------------------------------
def read_stl(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    assert len(raw) == 84 + count * 50, f"binary STL expected: {path}"
    return np.frombuffer(raw, dtype=STL_DTYPE, offset=84, count=count).copy()


def write_stl(path, records):
    header = f"cleaned by clean_wheel_leg_v2_meshes.py ({ROBOT_NAME})".encode("ascii")
    header = header[:80].ljust(80, b"\x00")
    with Path(path).open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", len(records)))
        handle.write(records.tobytes())


def triangle_components(records):
    """Group triangles into connected sub-solids via shared (welded) vertices."""
    tris = records["vertices"].astype(np.float64)
    n = len(tris)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    key_to_tri = {}
    for i, tri in enumerate(tris):
        for vertex in tri:
            key = (round(float(vertex[0]), 6), round(float(vertex[1]), 6), round(float(vertex[2]), 6))
            other = key_to_tri.setdefault(key, i)
            if other != i:
                union(i, other)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [np.array(indices) for indices in groups.values()]


def component_center_world(records, indices, frame):
    vertices = records["vertices"].reshape(-1, 3)[indices].reshape(-1, 3)
    local = 0.5 * (vertices.min(0) + vertices.max(0))
    return frame[:3, :3] @ local + frame[:3, 3], len(indices)


# --------------------------------------------------------------------------
# cleanup / regression check
# --------------------------------------------------------------------------
def duplicated_part_target(frames):
    records = read_stl(MESH_DIR / f"{DUPLICATE_SOURCE}.STL")
    groups = triangle_components(records)
    assert len(groups) == 1, f"{DUPLICATE_SOURCE} should hold exactly the duplicated solid"
    target, faces = component_center_world(records, groups[0], frames[DUPLICATE_SOURCE])
    assert faces == DUPLICATE_FACES, f"expected {DUPLICATE_FACES} faces, got {faces}"
    return target


def cross_link_duplicates(frames, tolerance):
    occurrences = defaultdict(list)
    for path in sorted(MESH_DIR.glob("*.STL")):
        link = path.stem
        records = read_stl(path)
        for indices in triangle_components(records):
            center, _ = component_center_world(records, indices, frames[link])
            occurrences[tuple(np.round(center, 4))].append(link)
    return {point: links for point, links in occurrences.items() if len(set(links)) > 1}


def clean(frames, tolerance, apply_changes):
    target = duplicated_part_target(frames)
    print(f">>> duplicated solid target world point: {np.round(target, 5).tolist()}")

    removed_total = 0
    for link in CONTAMINATED:
        path = MESH_DIR / f"{link}.STL"
        records = read_stl(path)
        groups = triangle_components(records)
        drop = []
        for indices in groups:
            center, faces = component_center_world(records, indices, frames[link])
            if faces == DUPLICATE_FACES and float(np.linalg.norm(center - target)) <= tolerance:
                drop.append(indices)
        if not drop:
            print(f"    {link:10s} already clean (faces={len(records)})")
            continue
        assert len(drop) == 1, f"{link}: expected one duplicated solid, found {len(drop)}"
        keep = np.ones(len(records), dtype=bool)
        keep[drop[0]] = False
        cleaned = records[keep]
        removed_total += len(drop[0])
        print(f"    {link:10s} faces {len(records)} -> {len(cleaned)} (removed {len(drop[0])})")
        if apply_changes:
            write_stl(path, cleaned)

    if apply_changes:
        print(f">>> removed {removed_total} duplicated triangles total")
        duplicates = cross_link_duplicates(frames, tolerance)
        # R_link2 legitimately overlaps its own captive screws, so only flag
        # sub-solids that appear across *different* links.
        if duplicates:
            raise SystemExit(f"[FAIL] cross-link duplicated sub-solids remain: {duplicates}")
        print(">>> cross-link duplicate check: clean")
    return removed_total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=5.0e-3, help="world-space match tolerance in metres")
    parser.add_argument("--check", action="store_true", help="report only, do not rewrite meshes")
    args = parser.parse_args()

    frames = parse_urdf_frames(URDF)
    clean(frames, args.tolerance, apply_changes=not args.check)


if __name__ == "__main__":
    main()
