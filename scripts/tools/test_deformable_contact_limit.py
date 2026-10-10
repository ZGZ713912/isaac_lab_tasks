"""Independent contact witnesses and counterexamples for the analytic bound."""

import math
from types import SimpleNamespace

import pytest
import torch

from deformable_contact_limit import contact_limit, tilt_bound
from deformable_feasibility import scan_pose, utilities


def fitted_range(du):
    return (du.URDF_ZERO_PHYSICAL_ANGLE - math.radians(75),
            du.URDF_ZERO_PHYSICAL_ANGLE - math.radians(16))


def test_contact_angle_limit_has_exact_four_wheel_endpoint_witness():
    du = utilities()
    low, high = fitted_range(du)
    result = contact_limit(du, (low, high))
    grade = result["maximum_contact_relative_grade_deg"]
    assert grade == pytest.approx(10.939164719576297, abs=1.e-10)
    # At the most favorable normal direction, uphill/downhill wheels use
    # opposite endpoints and share exactly the same plane projection.
    # With a negative-x normal, the forward wheels are on the downhill side.
    q = torch.tensor([high, high, low, low], dtype=torch.float64)
    normal = q.new_tensor([-math.sin(math.radians(grade)), 0., math.cos(math.radians(grade))])
    centers, _ = du.wheel_geometry(q)
    projection = centers @ normal
    assert (projection.max() - projection.min()).item() < 1.e-12
    assert tilt_bound(result, 20)["minimum_body_tilt_lower_bound_deg"] == pytest.approx(9.060835280423703)
    assert tilt_bound(result, 17)["simultaneous_rigid_contact_and_tilt_at_most_3deg_not_possible"]
    assert not tilt_bound(result, 10)["simultaneous_rigid_contact_and_tilt_at_most_3deg_not_possible"]


def test_just_above_limit_has_empty_contact_intersection_across_headings():
    du = utilities()
    limits = fitted_range(du)
    grade = contact_limit(du, limits)["maximum_contact_relative_grade_deg"] + .0001
    # Independent exact sinusoidal extrema in scan_pose, rather than the
    # bound's endpoint formula; body clearance does not affect this check.
    for yaw in range(0, 360, 3):
        result = scan_pose(du, grade, yaw, [-.001, .001, -.001, .001, -.001],
                           safety=0, q_range=limits)
        lo, hi = result["contact_height_interval_m"]
        assert lo > hi


def test_small_gap_allowance_relaxes_bound_and_admits_endpoint_gap_witness():
    du = utilities()
    low, high = fitted_range(du)
    result = contact_limit(du, (low, high), .001)
    assert 10.939 < result["maximum_contact_relative_grade_deg"] < 11.17
    q = torch.tensor([high, high, low, low], dtype=torch.float64)
    centers, _ = du.wheel_geometry(q)
    grade = math.radians(result["maximum_contact_relative_grade_deg"])
    normal = q.new_tensor([-math.sin(grade), 0., math.cos(grade)])
    projection = centers @ normal
    assert (projection.max() - projection.min()).item() == pytest.approx(.002, abs=1.e-12)
    assert tilt_bound(result, 17)["minimum_body_tilt_lower_bound_deg"] > 5.8


def test_rejects_geometry_outside_proof_assumptions():
    du = utilities()
    with pytest.raises(ValueError, match="strictly increasing"):
        contact_limit(du, (0., 1.6))
    def changed(q):
        centers, roll = du.wheel_geometry(q)
        centers[..., 0] *= 1.01
        return centers, roll
    altered = SimpleNamespace(Q_LOW=du.Q_LOW, wheel_geometry=changed)
    with pytest.raises(ValueError):
        contact_limit(altered, fitted_range(du))
    for value in [-.01, math.nan, math.inf]:
        with pytest.raises(ValueError):
            contact_limit(du, fitted_range(du), value)
    for slope in [-1., 90., math.nan, math.inf]:
        with pytest.raises(ValueError):
            tilt_bound(contact_limit(du, fitted_range(du)), slope)


def test_large_gap_allowance_includes_same_endpoint_contact_case():
    du = utilities()
    result = contact_limit(du, fitted_range(du), .22)
    assert result["maximum_contact_relative_grade_deg"] == 90.
    assert tilt_bound(result, 20)["minimum_body_tilt_lower_bound_deg"] == 0.
