"""CPU contract tests, without launching Isaac Sim. Run with pytest."""

import importlib.util
from pathlib import Path
import sys
import types
import math
from unittest.mock import patch

import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utilities():
    terrains = types.ModuleType("agent_world.terrains")
    terrains.periodic_slope_angle = lambda *args: 0.0
    with patch.dict(sys.modules, {"agent_world.terrains": terrains}):
        return load_file("deformable_utils_test", "source/agent_tasks/agent_tasks/direct/deformable_suspension/cfg_utils.py")


def test_encoder_twist_roundtrip_and_track_width():
    du = utilities()
    q = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.05, 0.8, 0.3, 1.0]])
    twist = torch.tensor([[0.8, -0.4, 6.283185], [-1.0, 0.6, -6.283185]])
    speed = (du.omni_matrix(q) @ twist.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(du.estimate_twist(q, speed), twist, atol=1e-5, rtol=1e-5)
    centers, _ = du.wheel_geometry(q)
    assert centers[1, 0, 0] > centers[0, 0, 0]
    assert (du.omni_matrix(q)[..., 2] < 0).all()  # positive axle rotation drives negative yaw
    assert speed.abs().max() < 60.0


def test_suspension_action_full_stroke_and_roundtrip():
    du = utilities()
    baseline = torch.full((2, 1), du.Q_MINANGLE)
    actions = torch.tensor([[-1.0, -0.5, 0.0, 1.0], [0.1, 0.5, -0.2, -0.9]])
    targets = du.suspension_target(actions, baseline, du.Q_LOW)
    torch.testing.assert_close(du.suspension_action(targets, baseline, du.Q_LOW), actions)
    assert targets[0, 0] == 0.0
    assert targets[0, 2] == du.Q_MINANGLE
    assert targets[0, 3] == du.Q_LOW


def test_traction_airborne_friction_circle_and_slip_sign():
    du = utilities()
    slip = torch.tensor([[100.0, -100.0, 0.01, 1.0]])
    load = torch.tensor([[50.0, 50.0, 50.0, 0.0]])
    fx, fy = du.wheel_traction(slip, torch.ones_like(slip), load, torch.tensor([[0.6]]), 120.0, 2.0)
    assert fx[0, 0] > 0 and fx[0, 1] < 0
    assert fx[0, 3] == 0 and fy[0, 3] == 0
    assert (torch.sqrt(fx.square() + fy.square()) <= 0.6 * load + 1e-5).all()


def wheel_controller():
    module = load_file("wheel_drive_test", "source/agent_tasks/agent_tasks/direct/deformable_suspension/wheel_drive.py")
    cfg = types.SimpleNamespace(adrc_dt=.001, wheel_velocity_kp=.8, wheel_velocity_ki=2.,
        wheel_torque_limit=5., wheel_speed_limit=60., wheel_acceleration_limit=20.,
        wheel_axial_inertia=.002092387)
    return module.WheelVelocityPI((2, 4), "cpu", cfg, dtype=torch.float64)


def test_wheel_pi_tracks_speed_under_load_with_bounded_acceleration():
    controller = wheel_controller()
    speed = torch.zeros_like(controller.target)
    request = speed.new_tensor([[10., -10., 5., -5.]]).expand_as(speed)
    contact = torch.ones_like(speed, dtype=torch.bool)
    load = request.sign() * .6  # constant motor load, e.g. climbing a slope
    for _ in range(10000):
        previous = controller.target.clone()
        torque = controller.update(request, speed, contact)
        assert (controller.target - previous).abs().max() <= .0200000001
        speed += .001 * (torque - load) / .002092387  # URDF axial wheel inertia
    torch.testing.assert_close(speed, request, atol=1e-4, rtol=0)
    torch.testing.assert_close(controller.torque, load, atol=1e-4, rtol=0)


def test_wheel_pi_stall_airborne_and_selective_reset():
    controller = wheel_controller()
    speed = torch.zeros_like(controller.target)
    contact = torch.ones_like(speed, dtype=torch.bool)
    for _ in range(1000):
        controller.update(torch.full_like(speed, 60.), speed, contact)
    assert controller.torque.abs().max() <= 5.
    # Saturating P effort freezes the integral instead of accumulating stall error.
    integral = controller.integral.clone()
    for _ in range(20):
        controller.update(torch.full_like(speed, 60.), speed, contact)
    torch.testing.assert_close(controller.integral, integral)
    controller.update(torch.full_like(speed, 60.), speed, torch.zeros_like(contact))
    assert controller.integral.count_nonzero() == 0
    retained = controller.target[1].clone()
    controller.reset(torch.tensor([0]))
    assert controller.target[0].count_nonzero() == 0 and controller.torque[0].count_nonzero() == 0
    torch.testing.assert_close(controller.target[1], retained)


def test_temporal_transformer_uses_old_frames_and_exports():
    transformer = load_file("deformable_transformer_test", "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_transformer.py")
    layout = {"global": list(range(10)) + [30, 31],
              "legs": [[10+i, 14+i, 18+i, 22+i, 26+i] for i in range(4)]}
    model = transformer.LegTokenTransformer(256, 4, layout, history_length=8, head="per_leg")
    obs = torch.randn(2, 256, requires_grad=True)
    result = model(obs)
    assert result.shape == (2, 4)
    result.sum().backward()
    assert obs.grad[:, :32].abs().sum() > 0
    traced = torch.jit.trace(model.eval(), obs.detach())
    torch.testing.assert_close(traced(obs.detach()), result.detach())
    critic_layout = {"global": list(range(10)) + [30, 31, 32, 33, 34, 35],
                     "legs": [[10+i, 14+i, 18+i, 22+i, 26+i, 36+i] for i in range(4)]}
    policy = transformer.ActorCriticTransformer(
        {"policy": obs.detach(), "critic": torch.randn(2, 40)},
        {"policy": ["policy"], "critic": ["critic"]}, 4,
        history_length=8, actor_layout=layout, critic_layout=critic_layout)
    assert policy.act_inference({"policy": obs.detach()}).shape == (2, 4)
    assert policy.evaluate({"critic": torch.randn(2, 40)}).shape == (2, 1)


def test_adrc_matches_rmcs_scalar_updates_and_resets():
    adrc = load_file("adrc_test", "source/agent_tasks/agent_tasks/direct/deformable_suspension/adrc.py")
    cfg = types.SimpleNamespace(
        adrc_dt=0.001, leg_max_physical_angle=math.radians(75), adrc_b0=-1.0, adrc_kt=1.0,
        adrc_td_h=0.001, adrc_td_r=50.0, adrc_td_max_acc=math.inf, adrc_td_max_vel=math.inf,
        adrc_eso_w0=250.0, adrc_z3_limit=1e9, adrc_k1=30.0, adrc_k2=17.0,
        adrc_alpha1=0.75, adrc_alpha2=0.7, adrc_delta=0.02, adrc_u_min=-200.0,
        adrc_u_max=200.0, adrc_output_min=-200.0, adrc_output_max=200.0, max_leg_torque=25.0)
    controller = adrc.LegADRC((2, 4), "cpu", cfg, dtype=torch.float64)
    ids = torch.arange(2)
    q = torch.full((2, 4), 1.0563, dtype=torch.float64)
    controller.reset(ids, q, q)
    x1 = z1 = cfg.leg_max_physical_angle - 1.0563
    x2 = z2 = z3 = last_u = 0.0
    def fal(e, alpha):
        return e / cfg.adrc_delta**(1-alpha) if abs(e) <= cfg.adrc_delta else math.copysign(abs(e)**alpha, e)
    saturated = False
    for step in range(300):
        measured_q = 1.0563 + 0.01 * math.sin(step * 0.05)
        target_q = 0.9 if step < 150 else 1.0563
        e = z1 - (cfg.leg_max_physical_angle - measured_q)
        z1 += 0.001 * (z2 - 750.0 * e)
        z2 += 0.001 * (z3 - last_u - 187500.0 * e)
        z3 = max(-1e9, min(1e9, z3 - 0.001 * 15625000.0 * e))
        d = 50.0 * 0.001**2
        a0 = 0.001 * x2
        y = x1 - (cfg.leg_max_physical_angle - target_q) + a0
        sign = lambda value: (value > 0) - (value < 0)
        a = a0 + sign(y) * (math.sqrt(d * (d + 8 * abs(y))) - d) / 2 if abs(y) > d else a0 + y
        fh = -50 * a / d if abs(a) <= d else -50 * sign(a)
        x1 += 0.001 * x2
        x2 += 0.001 * fh
        last_u = max(-200, min(200, -(30 * fal(x1-z1, .75) + 17 * fal(x2-z2, .7) - z3)))
        actual = controller.update(q.new_full(q.shape, measured_q), q.new_full(q.shape, target_q))
        expected = q.new_tensor([x1, x2, z1, z2, z3, last_u])
        states = torch.stack([state[0, 0] for state in (controller.x1, controller.x2, controller.z1,
                                                     controller.z2, controller.z3, controller.last_u)])
        torch.testing.assert_close(states, expected, atol=1e-8, rtol=1e-10)
        assert actual.abs().max() <= 25
        saturated |= abs(last_u) > 25
    assert saturated
    retained = controller.z3[1].clone()
    controller.reset(torch.tensor([0]), q[:1], q[:1])
    assert controller.last_u[0].count_nonzero() == 0 and controller.z3[0].count_nonzero() == 0
    torch.testing.assert_close(controller.z3[1], retained)


def test_adrc_saturated_feedback_uses_bounded_motor_command():
    adrc = load_file("adrc_feedback_test", "source/agent_tasks/agent_tasks/direct/deformable_suspension/adrc.py")
    # Reuse only the update-independent observer state to isolate its feedback input.
    cfg = types.SimpleNamespace(adrc_dt=.001, adrc_b0=-10., adrc_delta=.02,
        leg_max_physical_angle=1.3, adrc_eso_w0=250., adrc_z3_limit=1e9,
        adrc_td_r=50., adrc_td_h=.001, adrc_td_max_acc=math.inf, adrc_td_max_vel=math.inf,
        adrc_k1=30., adrc_k2=17., adrc_alpha1=.75, adrc_alpha2=.7,
        adrc_u_min=-200., adrc_u_max=200., adrc_kt=1., adrc_output_min=-200.,
        adrc_output_max=200., max_leg_torque=25., adrc_feedback_applied_torque=True)
    controller = adrc.LegADRC((1, 4), "cpu", cfg, dtype=torch.float64)
    q = torch.ones(1, 4, dtype=torch.float64)
    controller.reset(torch.tensor([0]), q, q)
    controller.last_u.fill_(200.)
    controller.applied_u.fill_(25.)
    controller.update(q, q)
    torch.testing.assert_close(controller.z2, torch.full_like(q, -.25))


def test_physical_urdf_and_deployment_angle_roundtrips():
    du = utilities()
    angles = torch.linspace(du.PHYSICAL_MIN_ANGLE, du.PHYSICAL_MAX_ANGLE, 101, dtype=torch.float64)
    for forward, inverse in ((du.physical_angle_to_urdf_q, du.urdf_q_to_physical_angle),
                             (du.physical_angle_to_deployment_q, du.deployment_q_to_physical_angle)):
        torch.testing.assert_close(inverse(forward(angles)), angles, atol=1e-15, rtol=0)
        for angle in (du.PHYSICAL_MIN_ANGLE, math.radians(45), du.PHYSICAL_MAX_ANGLE):
            assert inverse(forward(angle)) == pytest.approx(angle, abs=1e-15)
    assert du.Q_MINANGLE == pytest.approx(math.radians(75 - 17))
    assert du.physical_angle_to_deployment_q(du.PHYSICAL_MIN_ANGLE) == pytest.approx(du.LEG_UPPER_LIMIT)
    assert du.physical_angle_to_urdf_q(du.PHYSICAL_MAX_ANGLE) == 0
    assert du.physical_angle_to_deployment_q(du.PHYSICAL_MAX_ANGLE) == 0
    assert du.Q_MINANGLE != pytest.approx(du.LEG_UPPER_LIMIT)


@pytest.mark.parametrize("noise_std_type", ["scalar", "log"])
def test_transformer_inherited_act_applies_distribution_floor(noise_std_type):
    transformer = load_file("deformable_floor_test", "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_transformer.py")
    obs = {"policy": torch.randn(3, 26), "critic": torch.randn(3, 34)}
    policy = transformer.ActorCriticTransformer(
        obs, {"policy": ["policy"], "critic": ["critic"]}, 4,
        noise_std_type=noise_std_type, min_noise_std=0.15)
    assert "act" not in transformer.ActorCriticTransformer.__dict__
    with torch.no_grad():
        if noise_std_type == "scalar":
            policy.std.copy_(torch.tensor([-1.0, 0.0, 0.1, 0.3]))
        else:
            policy.log_std.copy_(torch.tensor([-100.0, -10.0, math.log(0.1), math.log(0.3)]))
    with patch.object(policy, "update_distribution", wraps=policy.update_distribution) as update:
        actions = policy.act(obs)
        assert isinstance(update.call_args.args[0], torch.Tensor)
        torch.testing.assert_close(update.call_args.args[0], obs["policy"])
    expected = torch.tensor([0.15, 0.15, 0.15, 0.3]).expand(3, -1)
    torch.testing.assert_close(policy.distribution.stddev, expected)
    assert actions.shape == (3, 4) and torch.isfinite(actions).all()
    assert torch.isfinite(policy.get_actions_log_prob(actions)).all()
    assert torch.isfinite(policy.entropy).all()


def test_geometry_scan_contact_clearance_and_infeasible_cases():
    scan = load_file("deformable_feasibility_test", "scripts/tools/deformable_feasibility.py")
    du = utilities()
    bounds = (-0.13, 0.13, -0.13, 0.13, du.BODY_BOTTOM_OFFSET)
    outcomes = []
    for slope in scan.SLOPES:
        for yaw in scan.YAWS:
            result = scan.scan_pose(du, slope, yaw, bounds)
            outcomes.append(result["feasible"])
            if result["feasible"]:
                q = torch.tensor(result["q_urdf"], dtype=torch.float64)
                assert ((q >= 0) & (q <= du.Q_LOW)).all()
                centers, _ = du.wheel_geometry(q)
                theta, psi = math.radians(slope), math.radians(yaw)
                x = centers[:, 0] * math.cos(psi) - centers[:, 1] * math.sin(psi)
                normal_gap = ((result["base_height_m"] + centers[:, 2]) * math.cos(theta)
                              - x * math.sin(theta) - du.WHEEL_RADIUS)
                torch.testing.assert_close(normal_gap, torch.zeros(4, dtype=torch.float64), atol=1e-11, rtol=0)
                assert result["body_normal_clearance_m"] >= 0.006 - 1e-11
                if slope:
                    vertical_gap = result["base_height_m"] + centers[:, 2] - x * math.tan(theta) - du.WHEEL_RADIUS
                    assert vertical_gap.abs().min() > 1e-5
            else:
                assert "q_urdf" not in result
                assert result["reason"] in ("no_four_wheel_contact_height", "body_clearance")
    assert any(outcomes) and not all(outcomes)
    assert not scan.scan_pose(du, 0, 0, bounds, safety=1.0)["feasible"]
    flat = scan.scan_pose(du, 0, 0, bounds)
    assert flat["feasible"]
    assert flat["q_urdf"] == pytest.approx([du.Q_LOW] * 4, abs=1e-10)


def test_height_polynomial_comparison_and_mismatch_not_used_for_contact():
    scan = load_file("deformable_height_test", "scripts/tools/deformable_feasibility.py")
    du = utilities()
    report = scan.height_comparison(du)
    assert report["max_abs_error_m"] < 1e-6
    bounds = (-0.13, 0.13, -0.13, 0.13, du.BODY_BOTTOM_OFFSET)
    original = scan.scan_pose(du, 5, 30, bounds)
    polynomial = du.q_to_base_height
    with patch.object(du, "q_to_base_height", side_effect=lambda q: polynomial(q) + 0.01):
        assert scan.height_comparison(du)["max_abs_error_m"] > 0.009
        assert scan.scan_pose(du, 5, 30, bounds) == original


def test_geometry_scan_actual_chassis_envelope_outcomes():
    import trimesh

    scan = load_file("deformable_mesh_scan_test", "scripts/tools/deformable_feasibility.py")
    du = utilities()
    mesh = trimesh.load(ROOT / "source/agent_world/agent_world/assets/usd_files/deformable_V2/meshes/base_link.STL",
                        force="mesh")
    vertices = torch.as_tensor(mesh.vertices.copy(), dtype=torch.float64)
    vertices = torch.stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]), dim=-1)
    lo, hi = vertices.amin(0), vertices.amax(0)
    bounds = [lo[0].item(), hi[0].item(), lo[1].item(), hi[1].item(), lo[2].item()]
    counts = []
    for slope in scan.SLOPES:
        results = [scan.scan_pose(du, slope, yaw, bounds) for yaw in scan.YAWS]
        counts.append(sum(result["feasible"] for result in results))
        for result in results:
            if result["feasible"]:
                assert max(abs(gap) for gap in result["wheel_normal_gap_m"]) < 1e-11
                assert result["body_normal_clearance_m"] >= 0.006 - 1e-11
        if slope == 10:
            assert [result["yaw_deg"] for result in results if result["feasible"]] == [90, 180, 270]
        if slope == 8:
            assert sum(result["reason"] == "body_clearance" for result in results) == 4
            assert sum(result["reason"] == "no_four_wheel_contact_height" for result in results) == 4
    assert counts == [24, 24, 24, 16, 3, 0]
