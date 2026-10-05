"""Frozen-policy preservation must constrain steep histories without fixing flats."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "deformable_preservation_test", ROOT / "source/agent_rl/agent_rl/rsl_rl/algorithms/ppo_diagnostics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Policy(nn.Module):
    is_recurrent = False
    actor_obs_normalization = False

    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(256, 4, bias=False)
        self.actor.frame_size = 32
        self.critic = nn.Linear(40, 1)

    def get_actor_obs(self, obs):
        return obs["policy"]


def histories(grades):
    obs = torch.zeros(len(grades), 8, 32)
    angle = torch.deg2rad(torch.tensor(grades))
    obs[:, -1, 7] = angle.sin()
    obs[:, -1, 9] = -angle.cos()
    return {"policy": obs.flatten(1)}


def test_reference_is_frozen_and_flat_actor_gradients_are_unconstrained():
    policy = Policy()
    algorithm = module.DiagnosticPPO(policy, device="cpu", steep_preservation_weight=1.)
    algorithm.initialize_steep_reference()
    reference = {k: v.clone() for k, v in algorithm._steep_reference_actor.state_dict().items()}
    obs = histories([0., 2., 5.5, 20.])
    mean = (policy.actor(obs["policy"]).detach() + .03).requires_grad_()
    loss, gate_fraction = algorithm._steep_preservation_loss(obs, mean)
    loss.backward()
    assert loss.item() == pytest.approx(.375)
    assert gate_fraction.item() == pytest.approx(.375)
    assert mean.grad[:2].count_nonzero() == 0
    assert (mean.grad[2:] > 0).all()
    assert all(p.grad is None and not p.requires_grad for p in algorithm._steep_reference_actor.parameters())
    with torch.no_grad():
        policy.actor.weight.add_(1.)
    for name, value in algorithm._steep_reference_actor.state_dict().items():
        torch.testing.assert_close(value, reference[name])
    assert all(p.grad is None for p in policy.critic.parameters())


def test_reference_uses_latest_imu_frame_and_requires_initialization():
    policy = Policy()
    algorithm = module.DiagnosticPPO(policy, device="cpu", steep_preservation_weight=1.)
    obs = histories([0.])
    with pytest.raises(RuntimeError, match="Initialize"):
        algorithm._steep_preservation_loss(obs, torch.ones(1, 4))
    algorithm.initialize_steep_reference()
    old_frame = obs["policy"].reshape(1, 8, 32)
    old_frame[:, :-1, 7] = math.sin(math.radians(20))
    old_frame[:, :-1, 9] = -math.cos(math.radians(20))
    mean = policy.actor(obs["policy"]) + .1
    loss, fraction = algorithm._steep_preservation_loss(obs, mean)
    assert loss.item() == fraction.item() == 0
    policy.actor_obs_normalization = True
    with pytest.raises(ValueError, match="unnormalized"):
        algorithm.initialize_steep_reference()


def test_default_diagnostics_do_not_enable_a_reference():
    algorithm = module.DiagnosticPPO(Policy(), device="cpu")
    assert algorithm.steep_preservation_weight == 0 and algorithm._steep_reference_actor is None
    with pytest.raises(ValueError, match="Invalid"):
        module.DiagnosticPPO(Policy(), device="cpu", steep_preservation_weight=-1.)


def test_flat_common_posture_cost_preserves_differences_and_steep_actions():
    algorithm = module.DiagnosticPPO(Policy(), device="cpu", flat_posture_weight=1.)
    action = torch.tensor([[-.2]*4, [-2./58.]*4, [-.1, 0., -.02, -.01], [-.2]*4], requires_grad=True)
    loss, fraction = algorithm._flat_posture_loss(histories([0., 0., 0., 20.]), action)
    loss.backward()
    assert fraction.item() == pytest.approx(.75)
    assert loss.item() == pytest.approx((9.6/10.)**2/4)
    assert (action.grad[0] < 0).all()  # increase native q to lower the body
    assert action.grad[1:].count_nonzero() == 0
    # A low target gets no further lowering incentive from this soft cost.
    loss, _ = algorithm._flat_posture_loss(histories([0.]), torch.ones(1, 4))
    assert loss.item() == 0


def test_lowest_corner_posture_allows_unequal_targets_needed_on_gentle_grades():
    algorithm = module.DiagnosticPPO(Policy(), device="cpu", flat_posture_weight=1., flat_posture_anchor="lowest")
    action = torch.tensor([[-2./58., -.5, -.4, -.3], [-.2]*4], requires_grad=True)
    loss, _ = algorithm._flat_posture_loss(histories([0., 0.]), action)
    loss.backward()
    assert action.grad[0].count_nonzero() == 0  # one low corner is sufficient
    assert (action.grad[1] < 0).any()  # retain a lowering signal when every corner is high
    assert loss.item() == pytest.approx((9.6/10.)**2/2)


def test_flat_posture_spread_gate_allows_a_level_body_with_different_leg_angles():
    algorithm = module.DiagnosticPPO(Policy(), device="cpu", flat_posture_weight=1.,
                                     flat_posture_max_spread_deg=6.)
    obs = histories([0., 0., 0.])
    frames = obs["policy"].reshape(3, 8, 32)
    frames[:, -1, 10:14] = torch.deg2rad(torch.tensor([[45., 45., 45., 45.],
                                                    [39., 39., 17., 17.],
                                                    [30., 30., 27., 27.]]))
    actions = torch.full((3, 4), -.2, requires_grad=True)
    loss, fraction = algorithm._flat_posture_loss(obs, actions)
    loss.backward()
    assert fraction.item() == pytest.approx(.5, abs=1.e-6)
    assert loss.item() == pytest.approx(.96**2 * .5, abs=1.e-6)
    assert (actions.grad[0] < 0).all()
    assert actions.grad[1].count_nonzero() == 0
    torch.testing.assert_close(actions.grad[2], actions.grad[0] * .5)


def test_posture_spread_gate_reads_latest_encoder_frame_and_defaults_off():
    algorithm = module.DiagnosticPPO(Policy(), device="cpu", flat_posture_weight=1.,
                                     flat_posture_max_spread_deg=6.)
    obs = histories([0.]);frames = obs["policy"].reshape(1, 8, 32)
    frames[:, :-1, 10:14] = torch.tensor([0., 1., 0., 1.])
    loss, fraction = algorithm._flat_posture_loss(obs, torch.full((1, 4), -.2))
    assert fraction.item() == 1 and loss.item() == pytest.approx(.96**2)
    assert module.DiagnosticPPO(Policy(), device="cpu").flat_posture_max_spread_deg == 0
    with pytest.raises(ValueError, match="leg spread"):
        module.DiagnosticPPO(Policy(), device="cpu", flat_posture_max_spread_deg=-1.)


def test_second_reference_blends_only_training_targets_and_is_frozen():
    policy = Policy()
    algorithm = module.DiagnosticPPO(policy, device="cpu", steep_preservation_weight=1.)
    with pytest.raises(RuntimeError, match="primary"):
        algorithm.initialize_steep_grade_reference(policy.actor.state_dict())
    algorithm.initialize_steep_reference()
    other = nn.Linear(256, 4, bias=False)
    with torch.no_grad():
        other.weight.copy_(policy.actor.weight)
        other.weight[:, -32].add_(.03)
    algorithm.initialize_steep_grade_reference(other.state_dict())
    obs = histories([20., 20., 20., 0.]);obs["policy"][:, -32] = 1.
    obs["training_steep_reference_mix"] = torch.tensor([[0.], [.5], [1.], [1.]])
    actions = policy.actor(obs["policy"]).detach().requires_grad_()
    loss, fraction = algorithm._steep_preservation_loss(obs, actions)
    loss.backward()
    assert loss.item() == pytest.approx((.25 + 1.)/4, abs=1.e-6)
    assert fraction.item() == pytest.approx(.75)
    assert actions.grad[0].count_nonzero() == actions.grad[3].count_nonzero() == 0
    assert (actions.grad[1:3] < 0).all()
    assert all(p.grad is None and not p.requires_grad for p in algorithm._steep_grade_reference_actor.parameters())
    del obs["training_steep_reference_mix"]
    with pytest.raises(ValueError, match="privileged"):
        algorithm._steep_preservation_loss(obs, actions)


def test_reference_selector_survives_rollout_batches_and_never_enters_actor():
    from tensordict import TensorDict
    from rsl_rl.storage.rollout_storage import RolloutStorage
    actor_spec = importlib.util.spec_from_file_location(
        "selector_actor", ROOT / "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_transformer.py")
    actor_module = importlib.util.module_from_spec(actor_spec);actor_spec.loader.exec_module(actor_module)
    obs = TensorDict({"policy": torch.zeros(4,256), "critic": torch.zeros(4,40),
                      "training_steep_reference_mix": torch.arange(4.)[:,None]/3.}, batch_size=[4])
    obs["policy"][:,0] = obs["training_steep_reference_mix"][:,0]
    actor = actor_module.ActorCriticTransformer(obs, {"policy":["policy"],"critic":["critic"]},
        4, history_length=8, d_model=8, nhead=2, num_layers=1, dim_ff=16, head_hidden=8)
    original = actor.actor(actor.get_actor_obs(obs))
    changed = obs.clone();changed["training_steep_reference_mix"] = 1 - changed["training_steep_reference_mix"]
    torch.testing.assert_close(actor.actor(actor.get_actor_obs(changed)), original, rtol=0, atol=0)
    assert actor.get_actor_obs(obs).shape == (4,256)
    storage = RolloutStorage("rl",4,2,obs,[4],device="cpu")
    storage.observations[0].copy_(obs);storage.observations[1].copy_(obs)
    batches=list(storage.mini_batch_generator(2,1));assert len(batches)==2
    for batch, *_ in batches:
        torch.testing.assert_close(batch["training_steep_reference_mix"][:,0], batch["policy"][:,0])


def test_all_posture_reference_retains_an_already_good_flat_policy():
    policy = Policy()
    algorithm = module.DiagnosticPPO(policy, device="cpu", steep_preservation_weight=1.,
                                     reference_all_postures=True)
    algorithm.initialize_steep_reference()
    obs = histories([0.,20.]);mean=(policy.actor(obs["policy"]).detach()+.03).requires_grad_()
    loss,fraction=algorithm._steep_preservation_loss(obs,mean);loss.backward()
    assert loss.item() == pytest.approx(1.,abs=1.e-6) and fraction.item()==1.
    assert (mean.grad>0).all()
    assert not module.DiagnosticPPO(Policy(),device="cpu").reference_all_postures
