"""Behavioral checks for support-constrained leveling, without Isaac imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict

import pytest
import torch

from test_deformable_dynamic import utilities

ROOT = Path(__file__).resolve().parents[2]


def gravity(degrees):
    angle = torch.deg2rad(torch.as_tensor(degrees))
    return torch.stack((angle.sin(), torch.zeros_like(angle), -angle.cos()), -1)


def test_horizontal_unloaded_corner_cannot_beat_supported_twenty_degree_body():
    du = utilities()
    gaps = torch.zeros(3, 4)
    loads = torch.tensor([[40., 40., 40., 40.], [3., 40., 40., 40.],
                          [40., 40., 40., 40.]])
    gaps[2, 0] = .004  # tolerance + one gap scale
    score, support, _ = du.suspension_supported_leveling_score(gravity([20., 0., 0.]), gaps, loads)
    assert score[0] > 0 and score[1] <= 0 and score[2] <= 0
    assert support[0] == 1 and support[1] == support[2] == .5


def test_unequal_supported_loads_and_leg_permutations_are_free():
    du = utilities()
    loads = torch.tensor([[20., 60., 110., 220.], [100., 100., 100., 100.]])
    gaps = torch.tensor([[0., -.001, .001, 0.], [0., 0., 0., 0.]])
    score, support, level = du.suspension_supported_leveling_score(gravity([5., 5.]), gaps, loads)
    torch.testing.assert_close(score[0], score[1])
    torch.testing.assert_close(support, torch.ones(2))
    permuted = du.suspension_supported_leveling_score(gravity([5., 5.]), gaps[:, [3, 0, 2, 1]],
                                                     loads[:, [3, 0, 2, 1]])
    for expected, actual in zip((score, support, level), permuted):
        torch.testing.assert_close(expected, actual)


def test_airborne_recovery_has_dense_gap_signal_and_leveling_remains_monotone():
    du = utilities()
    gaps = torch.zeros(4, 4)
    gaps[:, 0] = torch.tensor([.03, .01, .004, .001])
    gaps.requires_grad_()
    loads = torch.full((4, 4), 40.)
    loads[:, 0] = 0.
    score, _, _ = du.suspension_supported_leveling_score(gravity([10.] * 4), gaps, loads)
    assert (score.diff() > 0).all()  # approach terrain despite zero measured load
    score.sum().backward()
    assert (gaps.grad[:3, 0] < 0).all() and torch.isfinite(gaps.grad).all()
    supported, _, _ = du.suspension_supported_leveling_score(
        gravity([0., 3., 5., 10., 20., 45.]), torch.zeros(6, 4), torch.full((6, 4), 40.))
    assert (supported.diff() < 0).all() and supported[-1] > 0


def test_reloading_weakest_wheel_does_not_require_equal_loads():
    du = utilities()
    loads = torch.tensor([[0., 40., 100., 200.], [5., 40., 100., 200.],
                          [10., 40., 100., 200.], [20., 40., 100., 200.]], requires_grad=True)
    score, _, _ = du.suspension_supported_leveling_score(gravity([10.] * 4), torch.zeros(4, 4), loads)
    assert (score.diff() > 0).all()
    score.sum().backward()
    assert (loads.grad[:3, 0] > 0).all()
    assert loads.grad[:, 1:].count_nonzero() == 0


def test_clearance_reduces_joint_bonus_without_releasing_support_or_tilt():
    du = utilities()
    loads = torch.full((4, 4), 40.)
    loads[-1, 0] = 0.
    cost = torch.tensor([0., .5, 2., 0.], requires_grad=True)
    score, _, quality = du.suspension_supported_leveling_score(gravity([10., 10., 10., 0.]),
                                                             torch.zeros(4, 4), loads, clearance_cost=cost)
    assert score[0] > score[1] > score[2] > 0 and score[3] < 0
    assert (quality[:3].diff() < 0).all()
    score.sum().backward()
    assert (cost.grad[:3] < 0).all() and torch.isfinite(cost.grad).all()


@pytest.mark.parametrize('kwargs', [dict(tilt_scale_deg=0.), dict(gap_scale_m=0.),
                                    dict(min_load_n=-1.), dict(tilt_scale_deg=float('nan')),
                                    dict(gap_tolerance_m=-.001), dict(contact_force_threshold_n=20.),
                                    dict(contact_force_threshold_n=-1.)])
def test_invalid_joint_scales_rejected(kwargs):
    with pytest.raises(ValueError):
        utilities().suspension_supported_leveling_score(gravity([5.]), torch.zeros(1, 4),
                                                       torch.ones(1, 4), **kwargs)


def test_joint_reward_bounded_and_finite_through_large_gaps_and_inversion():
    scores, _, _ = utilities().suspension_supported_leveling_score(
        gravity([0., 20., 90., 180.]), torch.tensor([[0.] * 4, [.2] * 4, [1.e6] * 4, [.01] * 4]),
        torch.tensor([[50.] * 4, [0.] * 4, [0.] * 4, [-1.] * 4]))
    assert torch.isfinite(scores).all() and (scores >= -1.).all() and (scores <= 1.).all()


def test_joint_task_drops_competing_rewards_but_retains_physical_reset_fallback():
    path = ROOT / 'source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_cfg.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'DeformableFittedSupportLevelingJointEnvCfg')
    node.decorator_list = []

    class Parent:
        best_effort_leveling = True
        rewards = OrderedDict(all_wheel_contact=40., wheel_load_balance=4., tilt_quadratic=-40.,
                              flat_orientation_x_exp=2., flat_orientation_y_exp=2., termination=-200.)

    scope = dict(DeformableFittedSupportLevelingMixedEnvCfg=Parent, OrderedDict=OrderedDict)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    cfg = scope[node.name]()
    for name in ('all_wheel_contact', 'wheel_load_balance', 'tilt_quadratic',
                 'flat_orientation_x_exp', 'flat_orientation_y_exp'):
        assert cfg.rewards[name] == 0.
    assert cfg.rewards['termination'] == Parent.rewards['termination']
    assert Parent.rewards['wheel_load_balance'] == 4.
    assert cfg.best_effort_leveling  # reset's feasible tangent fallback stays enabled
    assert cfg.support_gap_weight == cfg.support_load_weight == cfg.best_effort_tilt_weight == 0.
    assert cfg.clearance_margin_weight == 0. and cfg.joint_supported_leveling_weight > 0.


def test_real_environment_uses_joint_score_once_and_keeps_support_metrics():
    path = ROOT / 'source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_env.py'
    env_node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'DeformableDynamicEnv')
    method = next(n for n in env_node.body if isinstance(n, ast.FunctionDef) and n.name == '_get_rewards')

    class Base:
        def _get_rewards(self):
            return torch.zeros(2)

    wrapper = ast.ClassDef(name='TelemetryEnv', bases=[ast.Name(id='Base', ctx=ast.Load())],
                           keywords=[], body=[method], decorator_list=[])
    scope = dict(Base=Base, torch=torch, du=utilities())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), 'exec'), scope)
    e = scope['TelemetryEnv']()
    e.cfg = SimpleNamespace(
        baseline_reward_weight=0., enforce_tunnel_height=False, best_effort_leveling=True,
        best_effort_tilt_weight=999., support_gap_weight=0., support_load_weight=0.,
        support_gap_tolerance_m=.001, support_gap_scale_m=.01, support_min_load_n=20.,
        joint_supported_leveling_weight=80., joint_leveling_tilt_scale_deg=10., joint_support_gap_scale_m=.003,
        clearance_margin_weight=999., clearance_margin_m=.006,
        commands_world_frame=False, max_leg_torque=44., horizontal_tolerance_deg=3.,
        wheel_contact_force_threshold=3., leg_target_upper_limit=1.08,
        max_body_top_height=.255, height_settle_steps=50, reward_scale=1., reward_total_clip=1000.)
    e._legs_idx = list(range(4))
    e.q_cmd = torch.full((2,), 1.)
    e.leg_target = torch.ones(2, 4)
    e.body_top_height = torch.full((2,), .25)
    e.chassis_clearance = torch.full((2,), .006)
    e.wheel_normal_forces = torch.tensor([[40., 40., 40., 40.], [0., 40., 40., 40.]])
    e._drive_command = torch.zeros(2, 3)
    e._drive_cmd_b = lambda: e._drive_command
    e._friction = torch.ones(2, 1)
    e._last_wheel_slip = torch.zeros(2, 4)
    e._leg_actuator = None
    e._support_metrics_enabled = True
    e._metrics = torch.zeros(2, 27)
    e._metric_steps = torch.zeros(2)
    e.episode_length_buf = torch.full((2,), 100)
    e._normals_at = lambda x: x.new_tensor([0., 0., 1.]).expand_as(x)
    e._wheel_geometry_w = lambda: (None, torch.zeros(2, 4, 3), None, None, None)
    e._ground_height = lambda x: torch.zeros(2, 4)
    e.robot = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.ones(2, 4), applied_torque=torch.zeros(2, 4), root_link_pos_w=torch.zeros(2, 3),
        root_link_lin_vel_b=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3),
        projected_gravity_b=gravity([20., 0.])))
    torch.testing.assert_close(e._get_rewards(), torch.tensor([16., -80. * 3. / 37.]), atol=1.e-5, rtol=1.e-5)
    assert e._metric_steps.tolist() == [1., 1.]
    torch.testing.assert_close(e._metrics[:, -3], torch.tensor([.2, -3. / 37.]))
    torch.testing.assert_close(e._metrics[:, -2], torch.tensor([1., 17. / 37.]))
