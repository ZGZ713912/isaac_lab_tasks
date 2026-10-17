"""Reference admission protects startup while freeing supported slip recovery."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('motion_admission',ROOT/'src/agent_rl/rsl_rl/algorithms/deformable_motion_admission.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def states():
    actor=torch.zeros(4,5,32);actor[:,:,9]=-1.
    critic=torch.zeros(4,40);critic[:,36:40]=1.  # 50 N per wheel.
    return actor.flatten(1),critic


def test_supported_parking_slip_relaxes_but_stable_parking_remains_protected():
    actor,critic=states();critic[1,32]=.7
    factor=module.motion_preservation_factor(actor,critic)
    torch.testing.assert_close(factor,torch.tensor([1.,.1,1.,1.]))


def test_contact_loss_encoder_ramp_and_command_change_keep_reference_protection():
    actor,critic=states();critic[:,32]=.7
    critic[0,36]=.04  # 2 N, not supported.
    actor[1,-2]=.5  # Encoder estimates substantial motion rather than commanded zero.
    actor[2,1]=.8  # Command recently changed inside the H5 window.
    factor=module.motion_preservation_factor(actor,critic)
    torch.testing.assert_close(factor,torch.tensor([1.,1.,1.,.1]))


def test_forward_motion_admission_does_not_weaken_reference_during_drive_slew():
    actor,critic=states();frames=actor.reshape(-1,5,32);frames[:,:,1]=.8
    critic[:,32]=.4;frames[1:,:,30]=.8
    factor=module.motion_preservation_factor(actor,critic)
    torch.testing.assert_close(factor,torch.tensor([1.,.1,.1,.1]))
    assert not factor.requires_grad


def test_wrong_or_nonfinite_training_contract_fails_instead_of_weakening_reference():
    actor,critic=states()
    with pytest.raises(ValueError,match='contracts'):
        module.motion_preservation_factor(actor[:,:128],critic)
    actor[0,0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        module.motion_preservation_factor(actor,critic)


def test_ppo_preservation_uses_training_gate_without_critic_gradients():
    from test_deformable_policy_preservation import module as diagnostics
    from test_deformable_mlp_replay import ReplayPolicy
    policy=ReplayPolicy()
    policy.get_actor_obs=lambda obs:obs['policy']
    policy.get_critic_obs=lambda obs:obs['critic']
    algorithm=diagnostics.DiagnosticPPO(policy,device='cpu',steep_preservation_weight=1.,
                                      steep_reference_action_scale=.03,reference_all_postures=True,
                                      reference_motion_admission=True)
    algorithm.initialize_steep_reference()
    actor,critic=states();critic[1,32]=.7;critic.requires_grad_(True)
    mean=(policy.actor(actor).detach()+.03).requires_grad_()
    loss,_=algorithm._steep_preservation_loss({'policy':actor,'critic':critic},mean)
    loss.backward()
    assert mean.grad[1].norm()==pytest.approx(.1*mean.grad[0].norm().item())
    assert critic.grad is None and all(p.grad is None for p in algorithm._steep_reference_actor.parameters())


def test_zero_reference_fraction_frees_slip_but_preserves_ramp_and_weak_contact():
    from test_deformable_policy_preservation import module as diagnostics
    from test_deformable_mlp_replay import ReplayPolicy
    policy=ReplayPolicy()
    policy.get_actor_obs=lambda obs:obs['policy']
    policy.get_critic_obs=lambda obs:obs['critic']
    algorithm=diagnostics.DiagnosticPPO(policy,device='cpu',steep_preservation_weight=1.,
                                      steep_reference_action_scale=.03,reference_all_postures=True,
                                      reference_motion_admission=True,reference_motion_relaxed_fraction=0.)
    algorithm.initialize_steep_reference()
    actor,critic=states();critic[:,32]=.7
    actor[0,1]=.8  # Command changed inside history: preserve startup response.
    critic[2,36]=.04  # 2 N: preserve the contact-restoring response.
    actor[3,-2]=.5  # Encoder has not caught up: preserve slew response.
    critic.requires_grad_(True)
    mean=(policy.actor(actor).detach()+.03).requires_grad_()
    loss,_=algorithm._steep_preservation_loss({'policy':actor,'critic':critic},mean)
    loss.backward()
    assert mean.grad[1].count_nonzero()==0
    assert (mean.grad[[0,2,3]]>0).all()
    assert critic.grad is None and all(p.grad is None for p in algorithm._steep_reference_actor.parameters())


def test_zero_reference_fraction_also_frees_slipping_neighborhood_feedback():
    from test_deformable_policy_preservation import module as diagnostics
    from test_deformable_mlp_replay import ReplayPolicy
    policy=ReplayPolicy()
    policy.get_actor_obs=lambda obs:obs['policy']
    policy.get_critic_obs=lambda obs:obs['critic']
    algorithm=diagnostics.DiagnosticPPO(policy,device='cpu',steep_preservation_weight=1.,
                                      reference_all_postures=True,reference_neighborhood_jitter=.5,
                                      reference_motion_admission=True,reference_motion_relaxed_fraction=0.)
    algorithm.initialize_steep_reference()
    actor,critic=states();critic[:,32]=.7
    with torch.no_grad():
        policy.actor.weight[:,4].add_(1.)
    loss,_=algorithm._steep_preservation_loss({'policy':actor,'critic':critic},policy.actor(actor))
    loss.backward()
    assert loss.item()==0. and policy.actor.weight.grad.count_nonzero()==0


@pytest.mark.parametrize('fraction',[-.1,1.,float('nan'),float('inf')])
def test_ppo_rejects_invalid_motion_fraction(fraction):
    from test_deformable_policy_preservation import module as diagnostics
    from test_deformable_mlp_replay import ReplayPolicy
    with pytest.raises(ValueError,match='Motion.*fraction'):
        diagnostics.DiagnosticPPO(ReplayPolicy(),device='cpu',reference_motion_relaxed_fraction=fraction)
