"""Sensor-only flat admission, acceleration rejection and temporal contracts."""

import math

import pytest
import torch

from test_deformable_dynamic import load_file

guard_module = load_file("deformable_runtime_guard_test",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/runtime_guard.py")


@pytest.mark.parametrize("grade,expected", [(0., True), (2., True), (2.9, True), (5., False), (10., False), (20., False)])
def test_sensor_grade_admission(grade, expected):
    guard = guard_module.SuspensionRuntimeGuard(1, "cpu")
    angle = math.radians(grade)
    gravity = torch.tensor([[math.sin(angle), 0., -math.cos(angle)]])
    q = torch.full((1, 4), math.atan2(.13694, .029108)-math.radians(19.))
    for _ in range(25):
        guard.update(q, torch.zeros(1, 3), -guard_module.GRAVITY*gravity, torch.zeros(1, 3), gravity, .005)
    assert guard.flat_hold.item() is expected
    assert guard.grade_deg.item() == pytest.approx(grade, abs=.03)


def test_horizontal_body_on_unequal_legs_is_not_flat_ground():
    guard = guard_module.SuspensionRuntimeGuard(1, "cpu")
    q = torch.tensor([[.3, 1.04, 1.04, .3]])
    gravity = torch.tensor([[0., 0., -1.]])
    for _ in range(30):
        guard.update(q, torch.zeros(1, 3), -guard_module.GRAVITY*gravity, torch.zeros(1, 3), gravity, .005)
    assert guard.grade_deg.item() > 3.5
    assert not guard.flat_hold.item()


def test_flat_hold_survives_acceleration_braking_and_turning():
    guard = guard_module.SuspensionRuntimeGuard(2, "cpu")
    q = torch.full((2, 4), 1.0646473192622632)
    gravity = torch.tensor([[0., 0., -1.]]).expand(2, -1)
    velocity = torch.zeros(2, 3)
    for tick in range(800):
        accel = 0. if tick < 40 else (2. if tick < 240 else (-2. if tick < 640 else 0.))
        velocity[:, 0] += accel * .005
        gyro = torch.tensor([[0., 0., 0.], [0., 0., 1.]])
        force = torch.cross(gyro, velocity, dim=-1) - guard_module.GRAVITY*gravity
        force[:, 0] += accel
        compensated, _ = guard.update(q, gyro, force, velocity, gravity, .005)
        if tick >= 20:
            assert guard.flat_hold.all()
            assert compensated[:, :2].abs().max().item() < .0005
    guard.update(q, gyro, torch.zeros_like(force), velocity, gravity, .005)
    assert not guard.flat_hold.any()
    assert not guard.valid.any()


def test_nonplanar_and_bad_sensor_frames_release_then_reconfirm():
    guard = guard_module.SuspensionRuntimeGuard(2, "cpu")
    q = torch.full((2, 4), 1.0646473192622632)
    g = torch.tensor([[0., 0., -1.]]).expand(2, -1)
    zeros = torch.zeros(2, 3)
    for _ in range(20):
        guard.update(q, zeros, -guard_module.GRAVITY*g, zeros, g, .005)
    assert guard.flat_hold.all()
    q[0] = torch.tensor([.2, 1.04, .2, 1.04])
    q[1, 0] = float("nan")
    _, changed = guard.update(q, zeros, -guard_module.GRAVITY*g, zeros, g, .005)
    assert changed.all()
    assert not guard.flat_hold.any()
    assert guard.plane_error_m[0] > .006
    assert not guard.valid[1]


def test_history_retains_40ms_span_and_10ms_action_feedback():
    history = guard_module.CausalPolicyHistory(2, 5, .005, .01, "cpu")
    for tick in range(12):
        frame = torch.full((2, 32), float(tick))
        result = history.push(frame, torch.full((2, 4), float(tick)))
    torch.testing.assert_close(result[0, :, 10], torch.tensor([3., 5., 7., 9., 11.]))
    torch.testing.assert_close(result[0, -1, 22:26], torch.full((4,), 10.))
    history.reset(torch.tensor([0]))
    result = history.push(torch.full((2, 32), 12.), torch.full((2, 4), 12.))
    torch.testing.assert_close(result[0, :, 10], torch.full((5,), 12.))
    assert result[0, :, 22:26].count_nonzero() == 0
    torch.testing.assert_close(result[1, :, 10], torch.tensor([4., 6., 8., 10., 12.]))
    with pytest.raises(ValueError):
        guard_module.CausalPolicyHistory(1, 5, .006, .01, "cpu")


def test_flat_learning_supervises_each_leg_and_excludes_reference_conflict():
    from test_deformable_policy_preservation import module, Policy
    algorithm = module.DiagnosticPPO(Policy(), device="cpu", flat_hold_weight=1.)
    obs = {"training_flat_hold": torch.tensor([[1.], [0.]])}
    actions = torch.tensor([[-.03, .05, .2, -.1], [.2, -.2, .3, -.4]], requires_grad=True)
    loss, fraction = algorithm._flat_hold_loss(obs, actions)
    loss.backward()
    assert fraction.item() == .5
    assert (actions.grad[0] * actions[0] > 0).all()
    assert actions.grad[1].count_nonzero() == 0
    with pytest.raises(ValueError):
        algorithm._flat_hold_loss({}, actions)


def test_flat_selector_cannot_preserve_the_old_raised_reference():
    from test_deformable_policy_preservation import module, Policy, histories
    policy = Policy()
    algorithm = module.DiagnosticPPO(policy, device="cpu", steep_preservation_weight=1.,
                                     reference_all_postures=True, flat_hold_weight=1.)
    algorithm.initialize_steep_reference()
    obs = histories([0., 0.])
    obs["training_flat_hold"] = torch.tensor([[1.], [0.]])
    reference = policy.actor(obs["policy"]).detach()
    actions = (reference+.1).requires_grad_()
    loss, fraction = algorithm._steep_preservation_loss(obs, actions)
    loss.backward()
    assert fraction.item() == .5
    assert actions.grad[0].count_nonzero() == 0
    assert (actions.grad[1] > 0).all()


def test_acceptance_cannot_hide_one_raised_leg_in_a_mean(tmp_path):
    import json
    import numpy as np
    from deformable_guarded_assess import flat_assessment
    (tmp_path/"traces").mkdir()
    checkpoint = tmp_path/"model.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    row = dict(failure_adjusted_all_contact_rate=1., tilt_deg=dict(abs_p95=0.),
               terminated_resets=0, physical_terminated_resets=0, terrain_boundary_violations=0,
               mean_abs_slip_m_s=0.)
    report = dict(checkpoint=str(checkpoint), seed=1, runtime_guard=dict(policy_rate_hz=200.),
                  terrain=dict(grade_deg=0.), results=dict(POLICY=dict(static=row)))
    (tmp_path/"run.log").write_text(json.dumps(report)+'\n')
    q = np.full((300, 1, 4), math.atan2(.13694,.029108)-math.radians(17.))
    q[..., 0] -= math.radians(2.)
    guard = np.zeros((300, 1, 9))
    guard[..., 0] = 1.
    np.savez(tmp_path/"traces/policy_static.npz", episode_age_s=np.full((300,1),2.), runtime_guard=guard,
             leg_target=q, joint_q=q, raw_policy_actions=np.zeros_like(q),
             time_s=np.linspace(1.,2.5,300), columns=np.array(("cmd_vx_body", "cmd_vy_body", "cmd_wz")),
             metrics=np.zeros((300,1,3)))
    assessment = flat_assessment(tmp_path)
    assert not assessment["passed"]
    assert not assessment["cases"]["static"]["checks"]["strict_minangle_target"]
    assert assessment["cases"]["static"]["target_error_max_deg"] == pytest.approx(2.)
