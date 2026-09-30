"""CPU contract tests, without launching Isaac Sim. Run with pytest."""

import importlib.util
from pathlib import Path
import sys
import types
import math
from unittest.mock import patch

import torch

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


def test_traction_airborne_friction_circle_and_slip_sign():
    du = utilities()
    slip = torch.tensor([[100.0, -100.0, 0.01, 1.0]])
    load = torch.tensor([[50.0, 50.0, 50.0, 0.0]])
    fx, fy = du.wheel_traction(slip, torch.ones_like(slip), load, torch.tensor([[0.6]]), 120.0, 2.0)
    assert fx[0, 0] > 0 and fx[0, 1] < 0
    assert fx[0, 3] == 0 and fy[0, 3] == 0
    assert (torch.sqrt(fx.square() + fy.square()) <= 0.6 * load + 1e-5).all()


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
