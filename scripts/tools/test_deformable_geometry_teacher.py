"""Contact and clearance witnesses for the training-only geometric guide."""

import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deformable_feasibility import utilities
from deformable_geometry_teacher import GeometryTeacher


def teacher():
    fallback = torch.nn.Linear(160, 4)
    torch.nn.init.zeros_(fallback.weight)
    torch.nn.init.constant_(fallback.bias, .1)
    return GeometryTeacher(fallback, utilities(), lower=.05235635310555198, upper=1.0821006117822065,
                           baseline=1.0646473192622632,
                           bottom_bounds=(-.29002431, .37709999, -.29002431, .29002431, -.027000001))


def observations(grades):
    x = torch.zeros(len(grades), 5, 32)
    angle = torch.tensor(grades) * math.pi/180
    x[:, :, 7] = angle.sin()[:, None]
    x[:, :, 9] = -angle.cos()[:, None]
    x[:, :, 10:14] = 1.0646473192622632
    x[:, :, 18:22] = .1
    return x.flatten(1)


def test_flat_uses_the_already_verified_feedback_without_height_drift():
    m = teacher()
    x = observations([0.])
    torch.testing.assert_close(m(x), m.fallback(x))
    goal, valid, _ = m.targets(x)
    assert valid.all()
    torch.testing.assert_close(goal, torch.full_like(goal, m.baseline), atol=5.e-6, rtol=0)


def test_grade_targets_retain_four_contact_witnesses_stroke_and_cad_clearance():
    m = teacher()
    x = observations([-20., -17., -10., -5., 5., 10., 17., 20.])
    goal, valid, details = m.targets(x)
    assert valid.all()
    assert ((goal >= m.lower) & (goal <= m.upper)).all()
    centers, _ = m.utilities.wheel_geometry(goal)
    n, h = details["goal_normal"], details["normal_height"]
    sphere_gap = h[:, None] + (centers * n[:, None]).sum(-1) - m.utilities.WHEEL_RADIUS
    assert sphere_gap.abs().max() < 5.e-6
    lo, _, _ = m._interval(n)
    assert (h >= lo - 2.e-6).all()
    fraction = details["correction_fraction"]
    assert (fraction > .25).all()
    assert (fraction[[0, 1, 6, 7]] < 1).all()  # Keep steep residual tilt explicit.
    actions = m(x)
    q = x.reshape(-1, 5, 32)[:, -1, 10:14]
    target = m.utilities.suspension_target(actions, torch.full((len(x), 1), m.baseline), m.upper, m.lower)
    assert ((target >= m.lower-1.e-6) & (target <= m.upper+1.e-6)).all()
    baseline = q.new_full((len(x), 1), m.baseline)
    fallback_target = m.utilities.suspension_target(m.fallback(x), baseline, m.upper, m.lower)
    assert (target-fallback_target).abs().max() <= m.correction_limit+1.e-6
    assert torch.isfinite(actions).all()


def test_steep_grade_feedback_is_preserved_even_when_geometry_has_a_witness():
    m = teacher()
    x = observations([-20., -17., 17., 20.])
    goal, valid, details = m.targets(x)
    assert valid.all()
    assert (details["estimated_grade_deg"] > 12).all()
    torch.testing.assert_close(m(x), m.fallback(x), atol=2.e-6, rtol=2.e-5)


@pytest.mark.parametrize("grade", [5., 11.])
def test_slow_correction_preserves_fast_base_and_uses_existing_action_memory(grade):
    m = teacher()
    x = observations([grade])
    x.reshape(1, 5, 32)[:, :, 22:26] = .1  # previously applied fallback output
    direct = m(x)
    m.correction_smoothing = .05
    slow = m(x)
    baseline = torch.tensor([[m.baseline]])
    base = m.utilities.suspension_target(m.fallback(x), baseline, m.upper, m.lower)
    q_direct = m.utilities.suspension_target(direct, baseline, m.upper, m.lower)
    q_slow = m.utilities.suspension_target(slow, baseline, m.upper, m.lower)
    torch.testing.assert_close(q_slow-base, .05*(q_direct-base), atol=2.e-6, rtol=1.e-4)
    # A previous correction is retained; filtering is independent of the
    # measured joint's tracking error and never caps the base's PID support.
    x.reshape(1, 5, 32)[:, -1, 22:26] = direct
    repeated = m.utilities.suspension_target(m(x), baseline, m.upper, m.lower)
    torch.testing.assert_close(repeated, q_direct, atol=2.e-6, rtol=1.e-4)


def test_new_sensor_response_of_base_is_not_low_pass_filtered():
    m = teacher()
    with torch.no_grad():
        m.fallback.weight[:, 4*32+4] = -2.0
    x = observations([5.])
    frames = x.reshape(1, 5, 32)
    frames[:, :, 22:26] = .1
    frames[:, -1, 4] = .1  # a new gyro observation, absent from the previous history
    baseline = torch.tensor([[m.baseline]])
    base = m.utilities.suspension_target(m.fallback(x), baseline, m.upper, m.lower)
    direct = m.utilities.suspension_target(m(x), baseline, m.upper, m.lower)
    m.correction_smoothing = .05
    slow = m.utilities.suspension_target(m(x), baseline, m.upper, m.lower)
    torch.testing.assert_close(slow, base + .05*(direct-base), atol=2.e-6, rtol=1.e-4)
    previous_base = m.utilities.suspension_target(torch.full((1, 4), .1), baseline, m.upper, m.lower)
    assert (base-previous_base).abs().max() > .05
    assert (slow-previous_base).abs().max() > .9*(base-previous_base).abs().max()


@pytest.mark.parametrize("yaw", [-.375, -.125, .125, .375])
def test_yaw_reserve_keeps_base_support_even_with_a_previous_correction(yaw):
    m = teacher()
    m.yaw_support_gate = True
    m.correction_smoothing = .01
    x = observations([5., 10., 17.])
    frames = x.reshape(3, 5, 32)
    frames[:, :, 3] = yaw
    frames[:, :, 22:26] = -.05
    actual, base = m(x), m.fallback(x)
    torch.testing.assert_close(actual[1:], base[1:], atol=2.e-6, rtol=2.e-5)
    assert (actual[0]-base[0]).abs().max() > .001
