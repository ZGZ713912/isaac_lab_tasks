"""Exercise real ORT inference, history ordering and rejected policy contracts without Kit."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
from onnx_policy import OnnxPolicy, load_policy_config, resolve_policy_config, validate_saved_fields


def model_file(tmp_path, shape=(1, 6), metadata=None, extra=False, nonfinite=False):
    obs = helper.make_tensor_value_info("sensor_history", TensorProto.FLOAT, list(shape))
    actions = helper.make_tensor_value_info("motor_actions", TensorProto.FLOAT, [1, 2])
    weights = np.zeros((6, 2), dtype=np.float32)
    weights[0, 0], weights[5, 0] = 1., -1.
    weights[1, 1], weights[4, 1] = 1., -1.
    if nonfinite:
        weights[:] = np.nan
    nodes = [helper.make_node("Flatten", ["sensor_history"], ["flat"], axis=1),
             helper.make_node("MatMul", ["flat", "weights"], ["motor_actions"])]
    inputs = [obs]
    if extra:
        inputs.append(helper.make_tensor_value_info("hidden_state", TensorProto.FLOAT, [1, 6]))
    graph = helper.make_graph(nodes, "history-order", inputs, [actions], [numpy_helper.from_array(weights, "weights")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    if metadata:
        helper.set_model_props(model, metadata)
    p = tmp_path / "model.onnx"
    onnx.save(model, p)
    return p


def policy(path, **kwargs):
    return OnnxPolicy(path, observation_dim=6, action_dim=2, history_length=3, frame_dim=2, **kwargs)


@pytest.mark.parametrize("shape", [(1, 6), ("batch", 6), (1, 3, 2), ("batch", "history", 2)])
def test_flat_and_rank3_preserve_oldest_first_history_and_replacement_on_reset(tmp_path, shape):
    actor = policy(model_file(tmp_path, shape))
    torch.testing.assert_close(actor(torch.arange(6, dtype=torch.float32)[None]), torch.tensor([[-5., -3.]]))
    # A reset supplies replacement observations; no stale history exists in ORT.
    torch.testing.assert_close(actor(torch.zeros(1, 6)), torch.zeros(1, 2))
    assert actor.audit()["policy_calls"] == 2


@pytest.mark.parametrize("metadata", [dict(rmcs_history_length="8"), dict(rmcs_obs_frame_size="32"),
    dict(training_obs_history_order="newest_to_oldest"), dict(rmcs_obs_layout="v3-history=8|sensors@2"),
    dict(rmcs_obs_layout="v3-history=3|sensors@4")])
def test_metadata_mismatches_are_rejected(tmp_path, metadata):
    with pytest.raises(ValueError, match="metadata|history|frame"):
        policy(model_file(tmp_path, metadata=metadata))


def test_extra_state_inputs_and_nonfinite_outputs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="extra inputs"):
        policy(model_file(tmp_path, extra=True))
    with pytest.raises(ValueError, match="nonfinite actions"):
        policy(model_file(tmp_path, nonfinite=True))


def test_wrong_robot_shapes_and_nonfinite_observations_are_rejected(tmp_path):
    path = model_file(tmp_path)
    with pytest.raises(ValueError, match="input shape"):
        OnnxPolicy(path, observation_dim=230, action_dim=6, history_length=5, frame_dim=46)
    actor = policy(path)
    with pytest.raises(ValueError, match="Nonfinite ONNX observations"):
        actor(torch.full((1, 6), float("nan")))
    with pytest.raises(ValueError, match="Expected float32"):
        actor(torch.zeros(1, 6, dtype=torch.float64))


def test_isaac_wrapper_tensordict_passes_only_actor_group(tmp_path):
    from tensordict import TensorDict

    actor = policy(model_file(tmp_path))
    observations = TensorDict(dict(policy=torch.arange(6, dtype=torch.float32)[None],
                                   critic=torch.full((1, 40), float("nan"))), batch_size=[1])
    torch.testing.assert_close(actor(observations), torch.tensor([[-5., -3.]]))


def test_metadata_action_period_and_observation_semantics_are_rejected(tmp_path):
    saved = dict(sim=dict(dt="0.001"), decimation="5", action_contract_version="v3",
                 real2sim_observation_version="current_fraction_v2")
    with pytest.raises(ValueError, match="training_observation_version"):
        policy(model_file(tmp_path, metadata=dict(training_observation_version="torque_v1")), saved_env=saved)
    contract = json.dumps(dict(version="v3", policy_period_s=.01))
    with pytest.raises(ValueError, match="policy period"):
        policy(model_file(tmp_path, metadata=dict(training_action_contract=contract)), saved_env=saved)


def test_params_discovery_has_no_checkpoint_dependency_and_rejects_normalization(tmp_path):
    run = tmp_path / "run"
    exported, params = run / "exported", run / "params"
    exported.mkdir(parents=True); params.mkdir()
    path = model_file(exported)
    (params / "env.yaml").write_text("observation_space: 6\n")
    (params / "agent.yaml").write_text("policy: {actor_obs_normalization: false}\n")
    assert resolve_policy_config(path) == params
    assert resolve_policy_config(path, run) == params
    assert resolve_policy_config(path, params) == params
    assert load_policy_config(params)[0]["observation_space"] == "6"
    (params / "agent.yaml").write_text("policy: {actor_obs_normalization: true}\n")
    with pytest.raises(ValueError, match="Normalized"):
        load_policy_config(params)
    with pytest.raises(ValueError, match="policy-config"):
        resolve_policy_config(path, tmp_path / "missing")


def test_same_width_but_changed_scale_or_physics_clock_is_rejected():
    cfg = NS(cmd_scale=1., sim=NS(dt=.001))
    with pytest.raises(ValueError, match="cmd_scale"):
        validate_saved_fields(cfg, dict(cmd_scale="2.0", sim=dict(dt=".001")), ["cmd_scale"])
    with pytest.raises(ValueError, match="timestep"):
        validate_saved_fields(cfg, dict(cmd_scale="1.0", sim=dict(dt=".002")), ["cmd_scale"])


def test_deformable_configuration_contract_can_be_checked_without_a_checkpoint(tmp_path):
    spec = importlib.util.spec_from_file_location("onnx_deformable_config", ROOT / "scripts/utils/deformable_checkpoint.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    policy = dict(class_name="ActorCriticSuspensionMLP", history_length=5)
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(dict(policy=policy, algorithm=dict(class_name="PPO"))))
    (tmp_path / "env.yaml").write_text("action_contract_version: current_v3\n")
    cfg = NS(action_contract_version="current_v3", real2sim_enabled=False, runtime_guard_enabled=False)
    config.validate_deformable_checkpoint(cfg, None, policy, params_dir=tmp_path)
    (tmp_path / "env.yaml").write_text("action_contract_version: legacy_v1\n")
    with pytest.raises(ValueError, match="action contract"):
        config.validate_deformable_checkpoint(cfg, None, policy, params_dir=tmp_path)


@pytest.mark.parametrize("robot", ["deformable", "wheel-leg"])
@pytest.mark.parametrize("flag", ["--onnx", "--checkpoint"])
def test_unified_entry_selects_only_requested_onnx_without_default_checkpoint(tmp_path, robot, flag):
    path = model_file(tmp_path)
    # This checks launch arguments only; no workstation map or simulator is needed.
    field = tmp_path / "field"
    field.mkdir()
    (field / "collision.usda").write_text("#usda 1.0\n")
    (field / "asset_package.json").write_text(json.dumps({
        "entries": {"isaac_static_collision_usd": "collision.usda"},
    }))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/rsl_rl/play_rmuc.py"), robot,
        flag, str(path), "--policy-config", str(tmp_path), "--field-root", str(field),
        "--python", sys.executable, "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"--onnx {path}" in result.stdout and "--checkpoint" not in result.stdout
    assert f"--policy-config {tmp_path}" in result.stdout
    assert f"--field-root {field}" in result.stdout
