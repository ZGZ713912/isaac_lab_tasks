"""Continuous-heading contact bound for the task's symmetric rotating-link legs.

This relaxes body clearance, force, traction and transient motion. It gives a
necessary rigid-geometry condition, not a learned-policy or hardware guarantee.
"""

import argparse
import hashlib
import json
import math

import torch

from deformable_feasibility import ROOT, joint_range, utilities


def _trig_min(a, b, low, high):
    """Minimum of a*cos(q) + b*sin(q), including interior stationary points."""
    stationary = math.atan2(b, a)
    candidates = [low, high] + [stationary + k * math.pi for k in range(-2, 3)
                               if low < stationary + k * math.pi < high]
    return min(a * math.cos(q) + b * math.sin(q) for q in candidates)


def contact_limit(du, q_range=None, normal_gap_tolerance_m=0.):
    """Bound the terrain/body relative angle for *any* heading and leg angles.

    For this task, r(q)=r0+a*cos(q)+b*sin(q), z(q)=z0+a*sin(q)-b*cos(q).
    The proof requires r'>0, z'>0 on the stroke. Of the two opposite wheel
    pairs, at least one has horizontal normal projection t>=1/sqrt(2).
    Its uphill projection has its minimum at q_low. The downhill projection
    is convex (second derivative cos(lambda)*r' + t*sin(lambda)*z'>0),
    hence has its maximum at an endpoint. For exact contact only q_high can
    reach the uphill minimum. With symmetric normal gap tolerance g, either
    endpoint can overlap, giving necessary conditions

      sin(lambda)*sqrt(2)*r_low <= 2*g, or
      cos(lambda)*(z_high-z_low) - sin(lambda)*(r_low+r_high)/sqrt(2) + 2*g >= 0.

    The terrain/body relative angle differs from the terrain/world angle by
    at most the body/world tilt (spherical triangle inequality). That yields
    the lower bound on body tilt, without equating Euler angles to tilt.
    An obtuse terrain/body normal angle is treated by reversing the plane
    normal. Its world tilt bound is 180-lambda_limit-slope, which is no lower
    than slope-lambda_limit for terrain slopes below 90 degrees.
    """
    low, high = joint_range(du, q_range)
    if not 0 <= normal_gap_tolerance_m < math.inf:
        raise ValueError("Normal gap tolerance must be finite and nonnegative")
    samples = torch.tensor([0., math.pi / 2, math.pi], dtype=torch.float64)
    centers, _ = du.wheel_geometry(samples[:, None].expand(-1, 4))
    radial = centers[..., :2].norm(dim=-1)[:, 0]
    z = centers[:, 0, 2]
    r0, z0 = (radial[0] + radial[2]) / 2, (z[0] + z[2]) / 2
    a, b = ((radial[0] - radial[2]) / 2).item(), (radial[1] - r0).item()
    if abs((z[1] - z0).item() - a) > 1.e-12 or abs((z[0] - z0).item() + b) > 1.e-12:
        raise ValueError("This bound requires the task's rotating-link geometry")

    # Regression check against the actual utility. The analytic profile and
    # its symmetry are explicit assumptions, rather than a sampled proof.
    q = torch.linspace(low, high, 19, dtype=torch.float64)
    actual, _ = du.wheel_geometry(q[:, None].expand(-1, 4))
    r = r0 + a * q.cos() + b * q.sin()
    zz = z0 + a * q.sin() - b * q.cos()
    signs = q.new_tensor(((1, -1), (1, 1), (-1, 1), (-1, -1))) / math.sqrt(2.)
    expected = torch.cat((r[:, None, None] * signs[None],
                          zz[:, None, None].expand(-1, 4, 1)), dim=-1)
    if not torch.allclose(actual, expected, rtol=0., atol=1.e-12):
        raise ValueError("Wheel geometry does not match the symmetric analytic profile")
    r_prime_min = _trig_min(b, -a, low, high)
    z_prime_min = _trig_min(a, b, low, high)
    if min(r_prime_min, z_prime_min) <= 0 or r.min().item() <= 0:
        raise ValueError("The proof requires strictly increasing positive radius and height")

    r_low, r_high, z_low, z_high = r[0].item(), r[-1].item(), zz[0].item(), zz[-1].item()
    vertical_stroke = z_high - z_low
    opposite_pair_lever = (r_low + r_high) / math.sqrt(2.)
    if 2 * normal_gap_tolerance_m >= opposite_pair_lever:
        relative_limit = math.pi / 2
    else:
        relative_limit = (math.atan2(vertical_stroke, opposite_pair_lever)
                          + math.asin(2 * normal_gap_tolerance_m
                                      / math.hypot(vertical_stroke, opposite_pair_lever)))
    same_endpoint_limit = math.asin(min(1., math.sqrt(2.) * normal_gap_tolerance_m / r_low))
    relative_limit = max(relative_limit, same_endpoint_limit)
    return dict(q_range_urdf_rad=[low, high], radial_endpoints_m=[r_low, r_high],
                height_endpoints_m=[z_low, z_high], vertical_stroke_m=vertical_stroke,
                radial_derivative_min_m=r_prime_min, height_derivative_min_m=z_prime_min,
                normal_gap_tolerance_m=normal_gap_tolerance_m,
                maximum_contact_relative_grade_deg=math.degrees(relative_limit),
                relative_grade_is_acute_plane_normal_angle=True,
                scope="Necessary contact condition for the explicit symmetric analytic wheel geometry; "
                      "all continuous headings and all joint positions in the specified stroke.",
                relaxed_constraints=["body collision", "force and contact compliance",
                                     "torque", "traction", "transient motion"])


def tilt_bound(limit, slope_deg):
    if not 0 <= slope_deg < 90:
        raise ValueError("Slope must be finite in [0, 90) degrees")
    lower = max(0., slope_deg - limit["maximum_contact_relative_grade_deg"])
    return dict(slope_deg=slope_deg, minimum_body_tilt_lower_bound_deg=lower,
                simultaneous_rigid_contact_and_tilt_at_most_3deg_not_possible=lower > 3.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slopes", nargs="+", type=float, default=[0, 5, 10, 17, 20])
    parser.add_argument("--normal-gap-tolerance", type=float, default=0.,
                        help="Symmetric normal distance allowance per wheel; not a PhysX setting.")
    args = parser.parse_args()
    du = utilities()
    limits = (du.URDF_ZERO_PHYSICAL_ANGLE - math.radians(75),
              du.URDF_ZERO_PHYSICAL_ANGLE - math.radians(16))
    result = contact_limit(du, limits, args.normal_gap_tolerance)
    source = ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension/cfg_utils.py"
    result.update(geometry_source=str(source),
                  geometry_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  cases=[tilt_bound(result, slope) for slope in args.slopes])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
