"""CPU contract tests, without launching Isaac Sim. Run with pytest."""

import importlib.util
from pathlib import Path
import sys
import types
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
