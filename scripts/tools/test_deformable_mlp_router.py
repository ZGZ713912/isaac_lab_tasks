"""Routed MLP inference must preserve the selected raw mean and sensor-only ABI."""

import importlib.util
import copy
from pathlib import Path
import sys

import torch
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "source/agent_rl/agent_rl/rsl_rl/modules/sensor_routed_mlp.py"
spec = importlib.util.spec_from_file_location("sensor_routed_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def policy():
    torch.manual_seed(42)
    actor = module.SensorRoutedMLP([[12, 8], [16, 8], [20, 8]], [8])
    with torch.no_grad():
        for parameter in actor.router.parameters():
            parameter.zero_()
        actor.router[-1].bias.copy_(torch.tensor([-20., 20., -20.]))
        actor.experts[1][-1].bias.fill_(2.)
    return actor


def test_selected_expert_returns_bit_identical_unclipped_mean():
    actor = policy()
    x = torch.randn(12, 160)
    x[:, -14:-10] = .1
    assert torch.equal(actor(x), actor.experts[1](x))
    assert actor(x).max() > 1
    assert torch.equal(actor(x), actor(x.clone()))


def test_uncertain_and_unloaded_sensors_retain_the_default_expert():
    actor = policy()
    x = torch.zeros(4, 160)
    assert torch.equal(actor(x), actor.experts[0](x))
    x[:, -14:-10] = .1
    with torch.no_grad():
        actor.router[-1].bias.zero_()
    assert torch.equal(actor(x), actor.experts[0](x))


def test_ppo_updates_only_the_selected_expert_and_keeps_router_frozen():
    actor = policy()
    x = torch.randn(4, 160)
    x[:, -14:-10] = .1
    actor(x).square().mean().backward()
    assert all(parameter.grad is None for parameter in actor.router.parameters())
    assert sum(float(parameter.grad.abs().sum()) for parameter in actor.experts[1].parameters()) > 0
    for index in (0, 2):
        assert all(parameter.grad is None or not parameter.grad.any() for parameter in actor.experts[index].parameters())


def test_native_exporter_and_torchscript_preserve_routing_and_flat_history():
    actor = policy()
    spec = importlib.util.spec_from_file_location("routed_native_export", ROOT/"scripts/rsl_rl/export_onnx.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    rebuilt, inputs, outputs = exporter._build_actor({"actor."+k: v for k, v in actor.state_dict().items()})
    assert (inputs, outputs) == (160, 4)
    scripted = torch.jit.script(rebuilt)
    x = torch.randn(6, 160)
    x[:3, -14:-10] = .1
    x[3:, -14:-10] = 0
    assert torch.equal(rebuilt(x), actor(x))
    assert torch.equal(scripted(x), actor(x))


def test_resuming_cannot_silently_change_routing_or_expert_contract(tmp_path):
    spec = importlib.util.spec_from_file_location("routed_checkpoint_contract", ROOT/"scripts/utils/deformable_checkpoint.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    config = dict(class_name="ActorCriticSuspensionRoutedMLP",history_length=5,
                  actor_hidden_dims=[12,8],critic_hidden_dims=[16,8],activation="elu",
                  actor_obs_normalization=False,critic_obs_normalization=False,
                  expert_hidden_dims=[[12,8],[16,8],[20,8]],router_hidden_dims=[8],
                  routing_confidence=.995,routing_load_threshold=.02)
    params = tmp_path/"params"
    params.mkdir()
    (params/"agent.yaml").write_text(yaml.safe_dump(dict(policy=config,algorithm=dict(class_name="DiagnosticPPO"))))
    checker.validate_deformable_policy(tmp_path/"model_0.pt",config)
    for key,value in (("class_name","ActorCriticSuspensionMLP"),("history_length",8),
                      ("expert_hidden_dims",[[12,8],[16,9],[20,8]]),("router_hidden_dims",[9]),
                      ("routing_confidence",.98),("routing_load_threshold",0.)):
        changed=copy.deepcopy(config);changed[key]=value
        with pytest.raises(ValueError,match=key):
            checker.validate_deformable_policy(tmp_path/"model_0.pt",changed)
