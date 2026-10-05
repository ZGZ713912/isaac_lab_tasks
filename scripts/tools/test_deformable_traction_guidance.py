"""Check physical force accounting that binary wheel contact cannot verify."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "traction_guidance", Path(__file__).resolve().parents[2]
    / "source/agent_tasks/agent_tasks/direct/deformable_suspension/traction_guidance.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plane(degrees=20.0):
    a = math.radians(degrees)
    normal = torch.tensor([[-math.sin(a), 0.0, math.cos(a)]])
    uphill = torch.tensor([math.cos(a), 0.0, math.sin(a)])
    cross = torch.tensor([0.0, 1.0, 0.0])
    return normal, torch.stack((uphill, uphill, cross, cross))[None]


def margin(loads, normal=None, directions=None, friction=.9):
    if normal is None:
        normal, directions = plane()
    return MODULE.uphill_drive_margin(
        torch.tensor([loads]), directions, normal, torch.tensor([[friction]]),
        mass_kg=26.3365312, wheel_torque_limit_nm=5.0, wheel_radius_m=.0769)


def test_four_contacting_wheels_can_still_lack_uphill_drive_force():
    sufficient = margin([65., 65., 65., 65.])
    # All loads exceed the 3 N contact threshold, but the driven uphill pair
    # carries too little load. Cross-slope wheels cannot supply this force.
    insufficient = margin([20., 20., 110., 110.])
    assert sufficient[0].item() == 0.0
    assert insufficient[0].item() > 0.0
    assert insufficient[1].item() == pytest.approx(36.0)
    assert insufficient[2].item() == pytest.approx(88.364793, rel=1.e-6)


def test_force_capacity_respects_torque_cap_and_roll_direction_sign():
    normal, directions = plane()
    positive = margin([1000., 1000., 1000., 1000.], normal, directions)
    negative = margin([1000., 1000., 1000., 1000.], normal, -directions)
    torch.testing.assert_close(positive[1], negative[1])
    assert positive[1].item() == pytest.approx(2.0 * 5.0 / .0769)
    assert margin([65., 65., 65., 65.], friction=0.)[1].item() == 0.0


def test_flat_ground_has_no_static_uphill_demand_or_deficit():
    normal, directions = plane(0.)
    for result in margin([0., 0., 0., 0.], normal, directions):
        assert result.item() == 0.0


def test_capacity_is_invariant_to_joint_order_and_negative_loads_add_no_force():
    normal, directions = plane()
    loads = torch.tensor([[65., 20., 110., 45.]])
    kwargs = dict(mass_kg=26.3365312, wheel_torque_limit_nm=5., wheel_radius_m=.0769)
    original = MODULE.uphill_drive_margin(loads, directions, normal, torch.tensor([[.9]]), **kwargs)
    order = [2, 0, 3, 1]
    shuffled = MODULE.uphill_drive_margin(loads[:, order], directions[:, order], normal, torch.tensor([[.9]]), **kwargs)
    for a, b in zip(original, shuffled):
        torch.testing.assert_close(a, b)
    assert margin([-10., -10., 65., 65.])[1].item() == 0.0
