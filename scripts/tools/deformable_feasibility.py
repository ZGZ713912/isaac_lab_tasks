"""CPU-only horizontal-body/four-sphere contact feasibility on z = x*tan(slope).

This is geometry, not a stability, traction, torque or learned-policy guarantee.
Run directly for a JSON scan; no Isaac Sim imports or output files are needed.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
SLOPES = (0, 2, 5, 8, 10, 17, 20)
YAWS = tuple(range(0, 360, 15))


def utilities():
    # Only the unrelated periodic-terrain import is stubbed; geometry is real.
    terrains = types.ModuleType("agent_world.terrains")
    terrains.periodic_slope_angle = lambda *args: 0.0
    spec = importlib.util.spec_from_file_location(
        "deformable_geometry_utils",
        ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension/cfg_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"agent_world.terrains": terrains}):
        spec.loader.exec_module(module)
    return module


def scan_pose(du, slope_deg, yaw_deg, bottom_bounds, safety=0.006):
    """Return an exact contact-height interval and, if safe, a numerical witness.

    bottom_bounds are base-frame (xmin, xmax, ymin, ymax, zbottom).
    Safety and wheel gaps are signed normal distances in metres. The bottom
    rectangle is conservative, matching the dynamic environment's envelope.
    """
    if not -90 < slope_deg < 90 or safety < 0:
        raise ValueError("Slope must be inside (-90, 90) and safety nonnegative")
    slope, yaw = math.radians(slope_deg), math.radians(yaw_deg)
    c, s = math.cos(slope), math.sin(slope)
    cy, sy = math.cos(yaw), math.sin(yaw)

    def heights(q):
        centers, _ = du.wheel_geometry(q)
        world_x = cy * centers[..., 0] - sy * centers[..., 1]
        return du.WHEEL_RADIUS / c - centers[..., 2] + math.tan(slope) * world_x

    # Each corner has H(q) = C + A*cos(q) + B*sin(q). Recover A/B
    # from the actual utility, then include stationary points, not just limits.
    samples = heights(torch.tensor([[0.0] * 4, [math.pi / 2] * 4,
                                    [math.pi] * 4], dtype=torch.float64))
    a = (samples[0] - samples[2]) / 2
    b = samples[1] - (samples[0] + samples[2]) / 2
    knots, ranges = [], []
    for i in range(4):
        stationary = math.atan2(b[i].item(), a[i].item())
        points = sorted({0.0, du.Q_LOW} | {
            stationary + k * math.pi for k in range(-2, 3)
            if 0 < stationary + k * math.pi < du.Q_LOW})
        values = [heights(torch.full((4,), q, dtype=torch.float64))[i].item() for q in points]
        knots.append((points, values))
        ranges.append((min(values), max(values)))
    contact_lo = max(lo for lo, _ in ranges)
    contact_hi = min(hi for _, hi in ranges)
    xmin, xmax, ymin, ymax, zbottom = bottom_bounds
    corners_x = [cy * x - sy * y for x in (xmin, xmax) for y in (ymin, ymax)]
    body_lo = max(math.tan(slope) * x for x in corners_x) - zbottom + safety / c
    height = max(contact_lo, body_lo)
    feasible = height <= contact_hi + 1e-12
    result = dict(slope_deg=slope_deg, yaw_deg=yaw_deg, feasible=feasible,
                  contact_height_interval_m=[contact_lo, contact_hi],
                  body_min_height_m=body_lo, safety_normal_m=safety,
                  reason="feasible" if feasible else (
                      "no_four_wheel_contact_height" if contact_lo > contact_hi + 1e-12
                      else "body_clearance"))
    if not feasible:
        return result
    roots = []
    for i, (points, values) in enumerate(knots):
        candidates = [q for q, value in zip(points, values) if abs(value - height) <= 1e-12]
        for lo, hi, flo, fhi in zip(points, points[1:], values, values[1:]):
            if (flo - height) * (fhi - height) >= 0:
                continue
            for _ in range(50):
                mid = (lo + hi) / 2
                fmid = heights(torch.full((4,), mid, dtype=torch.float64))[i].item()
                if (flo - height) * (fmid - height) <= 0:
                    hi = mid
                else:
                    lo, flo = mid, fmid
            candidates.append((lo + hi) / 2)
        roots.append(max(candidates))
    q = torch.tensor(roots, dtype=torch.float64)
    centers, _ = du.wheel_geometry(q)
    world_x = cy * centers[:, 0] - sy * centers[:, 1]
    gaps = c * (height + centers[:, 2]) - s * world_x - du.WHEEL_RADIUS
    clearance = min(c * (height + zbottom) - s * x for x in corners_x)
    result.update(base_height_m=height, q_urdf=roots,
                  wheel_normal_gap_m=gaps.tolist(), body_normal_clearance_m=clearance)
    return result


def height_comparison(du, samples=1001):
    q = torch.linspace(0, du.Q_LOW, samples, dtype=torch.float64)
    centers, _ = du.wheel_geometry(q[:, None].expand(-1, 4))
    analytic = du.WHEEL_RADIUS - centers[:, 0, 2]
    polynomial = du.q_to_base_height(q)
    error = polynomial - analytic
    index = error.abs().argmax().item()
    return dict(samples=samples, max_abs_error_m=error[index].abs().item(),
                max_error_q_urdf=q[index].item(),
                polynomial_minus_analytic_at_endpoints_m=[error[0].item(), error[-1].item()],
                analytic_endpoint_heights_m=[analytic[0].item(), analytic[-1].item()],
                polynomial_endpoint_heights_m=[polynomial[0].item(), polynomial[-1].item()])


def best_effort_pose(du, slope_deg, yaw_deg, bottom_bounds, safety=0.006):
    """Find a feasible minimum-tilt witness; not a certified global/dynamic optimum."""
    import numpy as np
    from scipy.optimize import minimize

    horizontal = scan_pose(du, slope_deg, yaw_deg, bottom_bounds, safety)
    if horizontal["feasible"]:
        return dict(slope_deg=slope_deg, yaw_deg=yaw_deg, best_feasible_tilt_deg=0.0,
                    roll_deg=0.0, pitch_deg=0.0, base_height_m=horizontal["base_height_m"],
                    q_urdf=horizontal["q_urdf"], body_normal_clearance_m=horizontal["body_normal_clearance_m"],
                    max_abs_wheel_normal_gap_m=max(abs(x) for x in horizontal["wheel_normal_gap_m"]),
                    horizontal_feasible=True, globally_horizontal_optimal=True, certified_global_optimum=True)
    theta, yaw = math.radians(slope_deg), math.radians(yaw_deg)
    normal = np.array([-math.sin(theta), 0., math.cos(theta)])
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    xmin, xmax, ymin, ymax, zb = bottom_bounds
    corners = np.array([[x, y, zb] for x in (xmin, xmax) for y in (ymin, ymax)])

    def rotation(v):
        roll, pitch = v[:2]
        cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
        return rz @ np.array([[cp, sp * sr, sp * cr], [0., cr, -sr], [-sp, cp * sr, cp * cr]])

    def contacts(v):
        centers, _ = du.wheel_geometry(torch.tensor(v[3:], dtype=torch.float64))
        return (centers.numpy() @ rotation(v).T) @ normal + v[2] * normal[2] - du.WHEEL_RADIUS

    def clearance(v):
        return (corners @ rotation(v).T) @ normal + v[2] * normal[2]

    def objective(v):
        return 1.0 - math.cos(v[0]) * math.cos(v[1])

    # Tangent-to-plane starts are feasible and cover different common extensions.
    local_normal = rz.T @ normal
    initial_roll = -math.asin(local_normal[1])
    initial_pitch = math.atan2(local_normal[0], local_normal[2])
    witnesses = []
    for q0 in (.15, .5, .8, du.Q_MINANGLE):
        initial = np.array([initial_roll, initial_pitch, 0., q0, q0, q0, q0])
        initial[2] = -contacts(initial).mean() / normal[2]
        solved = minimize(objective, initial, method="SLSQP",
                          bounds=[(-math.pi / 4, math.pi / 4)] * 2 + [(0., 1.)] + [(0., du.Q_LOW)] * 4,
                          constraints=[{"type": "eq", "fun": lambda v: 100. * contacts(v)},
                                       {"type": "ineq", "fun": lambda v: 100. * (clearance(v) - safety)}],
                          options={"ftol": 1.e-12, "maxiter": 300})
        v = solved.x
        if (np.isfinite(v).all() and np.abs(contacts(v)).max() < 2.e-5
                and clearance(v).min() >= safety - 2.e-5
                and v[3:].min() >= -1.e-6 and v[3:].max() <= du.Q_LOW + 1.e-6):
            witnesses.append((objective(v), v, bool(solved.success)))
    if not witnesses:
        return dict(slope_deg=slope_deg, yaw_deg=yaw_deg, horizontal_feasible=False,
                    best_feasible_tilt_deg=None, certified_global_optimum=False,
                    reason="no_valid_numerical_witness")
    _, v, converged = min(witnesses, key=lambda item: item[0])
    return dict(slope_deg=slope_deg, yaw_deg=yaw_deg, horizontal_feasible=False,
                best_feasible_tilt_deg=math.degrees(math.acos(max(-1., min(1., 1. - objective(v))))),
                roll_deg=math.degrees(v[0]), pitch_deg=math.degrees(v[1]), base_height_m=float(v[2]),
                q_urdf=v[3:].tolist(), body_normal_clearance_m=float(clearance(v).min()),
                max_abs_wheel_normal_gap_m=float(np.abs(contacts(v)).max()),
                solver_converged=converged, valid_multistart_witnesses=len(witnesses),
                certified_global_optimum=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety", type=float, default=0.006, help="Normal body clearance, metres")
    parser.add_argument("--slopes", nargs="+", type=float, default=list(SLOPES))
    parser.add_argument("--best-effort", action="store_true", help="Also compute feasible partial-leveling witnesses.")
    args = parser.parse_args()
    import trimesh

    mesh = trimesh.load(ROOT / "source/agent_world/agent_world/assets/usd_files/deformable_V2/meshes/base_link.STL",
                        force="mesh")
    vertices = torch.as_tensor(mesh.vertices.copy(), dtype=torch.float64)
    vertices = torch.stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]), dim=-1)
    lo, hi = vertices.amin(0), vertices.amax(0)
    bounds = [lo[0].item(), hi[0].item(), lo[1].item(), hi[1].item(), lo[2].item()]
    du = utilities()
    cases = [scan_pose(du, slope, yaw, bounds, args.safety) for slope in args.slopes for yaw in YAWS]
    report = dict(q_range_urdf=[0, du.Q_LOW], q_minangle_urdf=du.Q_MINANGLE,
                          bottom_bounds_m=bounds, height_comparison=height_comparison(du),
                          summary=[dict(slope_deg=slope,
                                        feasible=sum(case["feasible"] for case in cases if case["slope_deg"] == slope),
                                        total=len(YAWS)) for slope in args.slopes], cases=cases)
    if args.best_effort:
        report["reference_limits"] = "Conservative geometry only; no torque, friction, speed or certified nonzero global optimum."
        report["best_effort_cases"] = [best_effort_pose(du, slope, yaw, bounds, args.safety)
                                       for slope in args.slopes for yaw in YAWS]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
