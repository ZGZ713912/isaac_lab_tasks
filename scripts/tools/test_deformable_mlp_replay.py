"""Replay must preserve whole-vehicle partitions and useful local feedback."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from deformable_mlp_replay import SCENARIOS, failed_vehicles, select_training_samples
from test_deformable_policy_preservation import module


def test_balanced_replay_excludes_holdouts_and_failed_vehicles():
    scenarios = np.repeat(sorted(SCENARIOS), 4)
    obs = np.zeros((6, 24, 160), np.float32)
    obs[:, :, 0] = np.arange(6)[:, None]
    obs[:, :, 1] = np.arange(24)[None]
    a, pairs = select_training_samples(obs, scenarios, 1., np.random.default_rng(7), 6, [0, 1])
    b, again = select_training_samples(obs, scenarios, 1., np.random.default_rng(7), 6, [0, 1])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(pairs, again)
    assert len(a) == 72 and not np.any(pairs[:, 1] % 4 == 3)
    assert not np.any(np.isin(pairs[:, 1], [0, 1]))
    for scenario in SCENARIOS:
        selected = a[scenarios[pairs[:, 1]] == scenario]
        assert len(selected) == 12 and np.sum(selected[:, 0] < 2) == 6


def test_boundary_episode_ids_exclude_the_entire_correct_vehicle():
    scenarios = np.repeat(sorted(SCENARIOS), 4)
    rows = {scenario: dict(physical_terminated_resets=0, failure_adjusted_all_contact_rate=1.,
                           episodes=[dict(env_id=2, terminated=False, timeout=False, terrain_boundary=False)])
            for scenario in SCENARIOS}
    rows['spin_negative']['episodes'][0]['terrain_boundary'] = True
    report = {'results': {'POLICY': rows}}
    expected = int(np.flatnonzero(scenarios == 'spin_negative')[2])
    assert failed_vehicles(report, scenarios) == [expected]
    rows['spin_negative']['episodes'][0]['env_id'] = 4
    with pytest.raises(ValueError, match='episode IDs'):
        failed_vehicles(report, scenarios)


class ReplayPolicy(nn.Module):
    is_recurrent = False
    actor_obs_normalization = False

    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(160, 4, bias=False)
        self.actor.frame_size = 32
        self.critic = nn.Linear(40, 1)


def dataset(tmp_path, holdout=False, seed=2419, failed=False):
    path = tmp_path / 'policy_obs.npz'
    observations = np.zeros((5, 160), np.float32)
    observations[:, -23] = -1.
    ids = np.full(5, 3 if holdout else 0, dtype=np.int64)
    np.savez(path, policy_obs=observations, env_id=ids,
             seed=np.full(5, seed, dtype=np.int64), grade_deg=np.array([0, 5, 10, 17, 20], np.float32))
    manifest = dict(schema='deformable_sensor_replay_v1', partition='training_vehicles_env_id_mod4_ne3',
                    samples=5, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    sources=[dict(seed=seed, grade_deg=g, physics_sha256='physical model',
                                  excluded_failed_vehicles=[0] if failed else []) for g in [0, 5, 10, 17, 20]])
    path.with_suffix('.json').write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize('invalid', ['holdout', 'evaluation_seed', 'hash', 'physics', 'failed_vehicle'])
def test_replay_rejects_data_leakage_wrong_identity_and_failed_vehicles(tmp_path, invalid):
    policy = ReplayPolicy()
    alg = module.DiagnosticPPO(policy, device='cpu', steep_preservation_weight=1., reference_replay_weight=10.)
    alg.initialize_steep_reference()
    path = dataset(tmp_path, holdout=invalid == 'holdout', seed=1234 if invalid == 'evaluation_seed' else 2419,
                   failed=invalid == 'failed_vehicle')
    if invalid == 'hash':
        path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='replay'):
        alg.initialize_reference_replay(path, 'wrong model' if invalid == 'physics' else 'physical model')


def test_replay_constrains_gain_changes_without_critic_or_reference_gradients(tmp_path):
    policy = ReplayPolicy()
    alg = module.DiagnosticPPO(policy, device='cpu', steep_preservation_weight=1., reference_replay_weight=10.,
                               reference_neighborhood_jitter=.5, reference_replay_batch_size=32)
    alg.initialize_steep_reference()
    alg.initialize_reference_replay(dataset(tmp_path), 'physical model')
    assert alg._reference_replay_loss().item() == 0
    with torch.no_grad():
        policy.actor.weight[:, -28].add_(.1)  # Zero on stored states; altered gyro feedback nearby.
    loss = alg._reference_replay_loss()
    assert loss.item() > 0
    loss.backward()
    assert policy.actor.weight.grad[:, -28].abs().sum() > 0
    assert all(p.grad is None for p in policy.critic.parameters())
    assert all(p.grad is None and not p.requires_grad for p in alg._steep_reference_actor.parameters())


def test_replay_requires_a_frozen_reference_and_valid_parameters():
    with pytest.raises(ValueError, match='frozen'):
        module.DiagnosticPPO(ReplayPolicy(), device='cpu', reference_replay_weight=1.)
    for value in [-1., float('nan')]:
        with pytest.raises(ValueError, match='replay'):
            module.DiagnosticPPO(ReplayPolicy(), device='cpu', reference_replay_weight=value)


def test_maximum_projection_preserves_raw_means_without_changing_the_critic(tmp_path):
    policy=ReplayPolicy()
    alg=module.DiagnosticPPO(policy,device='cpu',steep_preservation_weight=1.,reference_replay_weight=10.,
                             reference_replay_max_delta=2.e-5)
    alg.initialize_steep_reference();alg.initialize_reference_replay(dataset(tmp_path),'physical model')
    critic={key:value.clone() for key,value in policy.critic.state_dict().items()}
    with torch.no_grad():
        policy.actor.weight[:, -23].add_(.002)
    delta,retained=alg._project_replay_drift()
    assert 0 < retained < 1 and delta <= 2.e-5
    for key,value in policy.critic.state_dict().items():
        torch.testing.assert_close(value,critic[key],rtol=0,atol=0)
    # A permitted correction survives; projection must not unconditionally freeze the actor.
    assert not torch.equal(policy.actor.weight,alg._steep_reference_actor.weight)
