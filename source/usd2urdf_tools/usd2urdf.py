#!/usr/bin/env python
"""USD -> URDF converter for Isaac Sim 5.1 pip (uses nvidia.srl UsdToUrdf directly).

Works in PLAIN python (no Isaac Sim GUI boot) by pointing sys.path at the
bundled pxr (omni.usd.libs) and the exporter's pip_prebundle (nvidia.srl).

Usage:
  python usd2urdf.py diagnose --usd <file.usd>
  python usd2urdf.py convert  --usd <file.usd> --out <dir> [--name N]
                              [--remove-node NAME]... [--remove-edge NAME]...
                              [--prefix PREFIX]
"""
import argparse
import os
import sys

SP = os.path.expanduser(
    "~/miniconda3/envs/isaaclab/lib/python3.11/site-packages"
)


def bootstrap():
    """Make pxr + nvidia.srl importable from plain python."""
    usd_libs = None
    cands = sorted(
        d for d in os.listdir(os.path.join(SP, "isaacsim", "extscache"))
        if d.startswith("omni.usd.libs-")
    )
    if cands:
        usd_libs = os.path.join(SP, "isaacsim", "extscache", cands[-1])
    pb = os.path.join(
        SP,
        "isaacsim",
        "exts",
        "isaacsim.asset.exporter.urdf",
        "pip_prebundle",
    )
    if not usd_libs or not os.path.isdir(pb):
        sys.exit("bootstrap: could not locate pxr libs or pip_prebundle")
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        os.path.join(usd_libs, "bin")
        + os.pathsep
        + os.path.expanduser("~/miniconda3/envs/isaaclab/lib"),
    )
    sys.path.insert(0, usd_libs)
    sys.path.insert(0, pb)
    import nvidia  # regular empty __init__ package -> extend its __path__

    nvidia.__path__.append(os.path.join(pb, "nvidia"))


def open_stage(usd_path):
    from pxr import Usd

    return Usd.Stage.Open(usd_path)


def build_graph(stage):
    from nvidia.srl.tools.logger import level_from_name
    from nvidia.srl.from_usd.transform_graph import TransformGraph

    return TransformGraph.init_from_stage(
        stage, parent_link_is_body_1=None, log_level=level_from_name("ERROR")
    )


def _link_cycle_report(graph):
    """Report kinematic loops in the joint-link structure."""
    from nvidia.srl.from_usd._from_usd_helper import NodeType

    nodes = graph.nodes if isinstance(graph.nodes, list) else graph.nodes()
    joints = [n for n in nodes if n.type == NodeType.JOINT]
    print(f"total nodes={len(nodes)}  joints={len(joints)}")

    j_conn = {}
    for j in joints:
        nb = [n for n in j.neighbors if n.type != NodeType.JOINT]
        j_conn[j.name] = nb
        print(
            f"  JOINT {j.name!r:32s} type={j.type} prim={j.prim.GetPath() if j.prim else None}"
        )
        for n in nb:
            print(f"      connects {n.name!r} ({n.type}) "
                  f"{n.prim.GetPath() if n.prim else None}")

    def dfs(v, visited, adj, parent_edge, path):
        visited.add(v)
        for (u, e) in adj.get(v, []):
            if e == parent_edge:
                continue
            if u in visited:
                return True, path + [e]
            r = dfs(u, visited, adj, e, path + [e])
            if r[0]:
                return r
        return False, None

    adj = {}
    for jname, nb in j_conn.items():
        if len(nb) >= 2:
            for i in range(len(nb)):
                for k in range(i + 1, len(nb)):
                    a, b = nb[i].name, nb[k].name
                    adj.setdefault(a, []).append((b, jname))
                    adj.setdefault(b, []).append((a, jname))
    visited = set()
    loops = []
    for v in adj:
        if v not in visited:
            has, path = dfs(v, visited, adj, None, [])
            if has and path:
                loops.append(path)
    if loops:
        print("KINEMATIC LOOPS DETECTED (joint chains closing a cycle):")
        for p in loops:
            print("   ", " -> ".join(p))
    else:
        print("no obvious joint-link cycle found")
    return loops


def cmd_diagnose(args):
    stage = open_stage(args.usd)
    print("defaultPrim:", stage.GetDefaultPrim().GetPath())
    print("top-level prims:")
    for p in stage.GetPseudoRoot().GetChildren():
        print("   ", p.GetPath(), "|", p.GetTypeName())
    graph = build_graph(stage)
    _link_cycle_report(graph)
    from pxr import UsdPhysics

    print("stage joints (UsdPhysics):")
    for prim in stage.Traverse():
        if UsdPhysics.Joint(prim):
            print("   ", prim.GetPath())


def cmd_convert(args):
    from nvidia.srl.tools.logger import level_from_name
    from nvidia.srl.from_usd.to_urdf import UsdToUrdf

    stage = open_stage(args.usd)
    root = args.root or str(stage.GetDefaultPrim().GetPath())
    removals = list(args.remove_node or [])
    if args.mode == "mjcf":
        graph = build_graph(stage)
        nodes = graph.nodes if isinstance(graph.nodes, list) else graph.nodes()
        for n in nodes:
            p = str(n.prim.GetPath()) if n.prim else ""
            if "/loop_joints/" in p or "_guide" in n.name:
                removals.append(n.name)
                print("auto-remove:", n.name, "|", p)
    kwargs = dict(
        root=root,
        node_names_to_remove=removals or None,
        edge_names_to_remove=args.remove_edge or None,
        log_level=level_from_name("ERROR"),
    )
    print("converting root:", root, "removals:", kwargs)
    conv = UsdToUrdf(stage, **kwargs)
    os.makedirs(args.out, exist_ok=True)
    name = args.name or os.path.splitext(os.path.basename(args.usd))[0]
    mesh_dir = os.path.join(args.out, "meshes")
    urdf_path = os.path.join(args.out, name + ".urdf")
    conv.save_to_file(
        urdf_output_path=urdf_path,
        visualize_collision_meshes=True,
        mesh_dir=mesh_dir,
        mesh_path_prefix=args.prefix,
        use_uri_file_prefix=(args.prefix == "file://"),
    )
    print("saved:", urdf_path)
    if args.relativize:
        _relativize(urdf_path, mesh_dir)


def _relativize(urdf_path, mesh_dir):
    import re

    with open(urdf_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(mesh_dir, "meshes")
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("relativized mesh refs -> meshes/")


def main():
    ap = argparse.ArgumentParser(description="USD->URDF via nvidia.srl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--usd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--usd", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--name", default=None)
    c.add_argument("--root", default=None)
    c.add_argument("--prefix", default="./")
    c.add_argument("--relativize", action="store_true")
    c.add_argument("--mode", choices=["plain", "mjcf"], default="plain")
    c.add_argument("--remove-node", action="append", default=[])
    c.add_argument("--remove-edge", action="append", default=[])
    args = ap.parse_args()
    bootstrap()
    if args.cmd == "diagnose":
        cmd_diagnose(args)
    else:
        cmd_convert(args)


if __name__ == "__main__":
    main()
