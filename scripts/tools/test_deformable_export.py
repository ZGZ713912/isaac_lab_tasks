"""Rebuild the saved deployment actor without Isaac or vehicle motion."""

import sys
import pytest
import torch
import yaml

from test_deformable_dynamic import ROOT, load_file


@pytest.mark.parametrize('geometry,pair_filter', [(False, False), (True, False), (True, True)])
def test_export_rebuild_preserves_saved_actor_behavior(tmp_path, geometry, pair_filter):
    exporter = load_file('deformable_export_test', 'scripts/rsl_rl/export_onnx.py')
    sys.path.insert(0, str(ROOT / 'source/agent_rl'))
    from agent_rl.rsl_rl.modules.actor_critic_transformer import LegTokenTransformer

    layout = dict(global_=list(range(10)) + [30, 31],
                  legs=[[10+i, 14+i, 18+i, 22+i, 26+i] for i in range(4)])
    layout['global'] = layout.pop('global_')
    reference = LegTokenTransformer(256, 4, layout, d_model=16,
        nhead=4, num_layers=1, dim_ff=32, head_hidden=16, head='per_leg',
        history_length=8, use_leg_geometry_features=geometry,
        previous_action_pair_filter=pair_filter).eval()
    if geometry:
        # A learned directional cue must survive export, not just its shape.
        with torch.no_grad():
            reference.leg_embed.weight[:, -1].fill_(.2)
    state = {'actor.' + k: v for k, v in reference.state_dict().items()}
    state['log_std'] = torch.zeros(4)
    policy = dict(actor_layout=layout, history_length=8, d_model=16, nhead=4, num_layers=1,
                  dim_ff=32, head_hidden=16, actor_head='per_leg')
    if geometry or pair_filter:
        policy.update(use_leg_geometry_features=geometry, previous_action_pair_filter=pair_filter)
    params = tmp_path / 'params'
    params.mkdir()
    (params / 'agent.yaml').write_text(yaml.safe_dump({'policy': policy}))
    actor, obs_dim, act_dim, history = exporter._build_transformer_actor(state, str(tmp_path / 'model.pt'))
    assert (obs_dim, act_dim, history) == (256, 4, 8)
    x = torch.randn(3, 8, 32, generator=torch.Generator().manual_seed(2027))
    with torch.no_grad():
        torch.testing.assert_close(actor(x), reference(x), atol=1.e-6, rtol=1.e-6)
    if geometry:
        policy['use_leg_geometry_features'] = False
        (params / 'agent.yaml').write_text(yaml.safe_dump({'policy': policy}))
        with pytest.raises(RuntimeError, match='size mismatch'):
            exporter._build_transformer_actor(state, str(tmp_path / 'model.pt'))
