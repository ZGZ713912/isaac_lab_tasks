"""Five-frame MLP learning, history lifecycle and checkpoint contracts."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MLP = load("mlp5_policy", "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_suspension_mlp.py").ActorCriticSuspensionMLP
checkpoint = load("mlp5_checkpoint", "scripts/utils/deformable_checkpoint.py")
evaluator = load("mlp5_eval", "scripts/tools/deformable_suspension_eval.py")
sys.path.insert(0, str(ROOT / "scripts/tools"))
pipeline = load("mlp5_pipeline", "scripts/tools/deformable_mlp_train.py")


def agent_config():
    return dict(class_name="OnPolicyRunner", num_steps_per_env=24,
                obs_groups={"policy": ["policy"], "critic": ["critic"]},
                policy=dict(class_name="ActorCriticSuspensionMLP", history_length=5,
                            actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64],
                            activation="elu", actor_obs_normalization=False, critic_obs_normalization=False),
                algorithm=dict(class_name="DiagnosticPPO"))


def test_mlp_learns_from_every_frame_and_reloads_exactly():
    torch.manual_seed(47)
    obs = {"policy": torch.randn(8, 160, requires_grad=True), "critic": torch.randn(8, 40)}
    cfg = agent_config()
    options = {key: value for key, value in cfg["policy"].items() if key != "class_name"}
    policy = MLP(obs, cfg["obs_groups"], 4, **options, noise_std_type="log")
    actions = policy.act_inference(obs)
    actions.square().mean().backward()
    assert (obs["policy"].grad.reshape(8, 5, 32).abs().sum((0, 2)) > 0).all()
    before = actions.detach().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.e-4)
    optimizer.step()
    assert not torch.equal(policy.act_inference(obs), before)
    restored = MLP(obs, cfg["obs_groups"], 4, **options, noise_std_type="log")
    restored.load_state_dict(policy.state_dict(), strict=True)
    torch.testing.assert_close(restored.act_inference(obs), policy.act_inference(obs))
    assert not any(isinstance(m, torch.nn.MultiheadAttention) for m in policy.modules())
    assert policy.actor[0].in_features == 160 and policy.critic[0].in_features == 40


def test_contact_recovery_matches_eval_precision_without_changing_other_tasks():
    old_cuda=torch.backends.cuda.matmul.allow_tf32
    old_cudnn=torch.backends.cudnn.allow_tf32
    old_precision=torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        assert checkpoint.configure_recovery_precision(dict(class_name="ActorCriticTransformer"),
                                                       dict(steep_preservation_weight=1.)) is None
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        cfg=dict(class_name="ActorCriticSuspensionRoutedMLP",history_length=5)
        assert checkpoint.configure_recovery_precision(cfg,dict(steep_preservation_weight=0.)) is None
        result=checkpoint.configure_recovery_precision(cfg,dict(steep_preservation_weight=10.))
        assert result["actor_math"] == "float32" and not result["allow_tf32"]
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32=old_cuda
        torch.backends.cudnn.allow_tf32=old_cudnn


def test_existing_exporter_preserves_flat_mlp_history_actions():
    exporter = load("mlp5_export", "scripts/rsl_rl/export_onnx.py")
    cfg = agent_config()
    obs = {"policy": torch.randn(3, 160), "critic": torch.randn(3, 40)}
    options = {key: value for key, value in cfg["policy"].items() if key != "class_name"}
    policy = MLP(obs, cfg["obs_groups"], 4, **options)
    state = policy.state_dict()
    assert not exporter._is_transformer(state)
    actor, width, actions = exporter._build_actor(state)
    assert (width, actions) == (160, 4)
    torch.testing.assert_close(actor(obs["policy"]), policy.act_inference(obs))
    scripted = torch.jit.script(actor)
    torch.testing.assert_close(scripted(obs["policy"]), policy.act_inference(obs))


@pytest.mark.parametrize("width,history", [(159, 5), (165, 5), (160, True), (160, 5.0)])
def test_malformed_history_cannot_silently_change_frame_size(width, history):
    obs = {"policy": torch.zeros(2, width), "critic": torch.zeros(2, 40)}
    with pytest.raises(ValueError, match="32D frames"):
        MLP(obs, agent_config()["obs_groups"], 4, history_length=history)


def test_five_frame_play_and_evaluation_use_saved_architecture(tmp_path):
    (tmp_path / "params").mkdir()
    cfg = agent_config()
    path = tmp_path / "params/agent.yaml"
    path.write_text(yaml.safe_dump(cfg))
    before = path.read_bytes()
    loaded = checkpoint.load_deformable_agent_config(tmp_path / "model_49.pt", "cpu")
    assert loaded["policy"] == cfg["policy"]
    assert evaluator.validate_agent(loaded, 5) == 5
    checkpoint.validate_deformable_policy(tmp_path / "model_49.pt", cfg["policy"])
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="--history"):
        evaluator.validate_agent(loaded, 8)


@pytest.mark.parametrize("key,value", [("class_name", "ActorCriticTransformer"), ("history_length", 8),
                                      ("actor_hidden_dims", [128, 64]), ("activation", "relu")])
def test_incompatible_warm_start_is_rejected_before_partial_weight_loading(tmp_path, key, value):
    (tmp_path / "params").mkdir()
    cfg = agent_config()
    cfg["policy"][key] = value
    (tmp_path / "params/agent.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="without --checkpoint"):
        checkpoint.validate_deformable_policy(tmp_path / "model_49.pt", agent_config()["policy"])


def test_production_history_is_chronological_cached_and_resets_per_environment():
    path = ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_env.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_observations")
    scope = dict(torch=torch, du=SimpleNamespace(estimate_twist=lambda q, w: torch.zeros(q.shape[0], 3)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
    data = SimpleNamespace(joint_pos=torch.zeros(2, 8), joint_vel=torch.zeros(2, 8),
                           applied_torque=torch.zeros(2, 8), root_ang_vel_b=torch.zeros(2, 3),
                           projected_gravity_b=torch.tensor([[0., 0., -1.]]).expand(2, -1),
                           root_lin_vel_b=torch.zeros(2, 3))
    env = SimpleNamespace(robot=SimpleNamespace(data=data), _legs_idx=list(range(4)), _wheels_idx=list(range(4, 8)),
                          _leg_actuator=None, _encoder_bias=torch.zeros(2, 4), _gyro_bias=torch.zeros(2, 3),
                          cfg=SimpleNamespace(encoder_noise_std=0., gyro_noise_std=0., gravity_noise_std=0.,
                                              max_sensor_delay_steps=0, action_contract_version="minangle_physical_v3"),
                          num_envs=2, device="cpu", q_cmd=torch.ones(2), actions=torch.zeros(2, 4),
                          _drive_cmd_b=lambda: torch.zeros(2, 3), body_top_height=torch.ones(2),
                          wheel_normal_forces=torch.ones(2, 4), _obs_tick=-1, _obs_cache=None,
                          _history=torch.zeros(2, 5, 32), _history_valid=torch.zeros(2, dtype=torch.bool),
                          _sensor_fifo=torch.zeros(2, 1, 32), _delay=torch.zeros(2, dtype=torch.long),
                          common_step_counter=0)
    read = scope["_get_observations"]
    read(env)
    for tick in range(1, 7):
        env.common_step_counter = tick
        data.joint_pos[:, :4] = tick
        obs = read(env)
        assert read(env) is obs  # Multiple reads cannot advance the history clock.
    torch.testing.assert_close(obs["policy"].reshape(2, 5, 32)[:, :, 10], torch.tensor([[2., 3., 4., 5., 6.]]).expand(2, -1))
    env._history_valid[0] = False
    env.common_step_counter = 7
    data.joint_pos[:, :4] = 7
    frames = read(env)["policy"].reshape(2, 5, 32)
    torch.testing.assert_close(frames[0, :, 10], torch.full((5,), 7.))
    torch.testing.assert_close(frames[1, :, 10], torch.arange(3., 8.))


def test_mlp_environment_only_changes_history_and_has_matching_tasks():
    directory = ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension"
    tree = ast.parse((directory / "dynamic_cfg.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeformableFittedSupportLevelingMLPEnvCfg")
    assert cls.bases[0].id == "DeformableFittedSupportLevelingJointEnvCfg"
    fields = {target.id for node in cls.body if isinstance(node, ast.Assign) for target in node.targets}
    assert fields == {"policy_history_length", "observation_space"}
    registry = (directory / "__init__.py").read_text()
    assert '("Support-Leveling-Joint-History-MLP", "DeformableFittedSupportLevelingMLPEnvCfg", "DeformableFittedSupportLevelingMLPPPORunnerCfg")' in registry
    assert '("Rough-Keyboard-Play-History-MLP", "DeformableFittedMLPKeyboardPlayEnvCfg", "DeformableFittedSupportLevelingMLPPPORunnerCfg")' in registry


def test_long_pipeline_dry_run_keeps_both_budgets_and_all_grades(tmp_path, capsys):
    output = tmp_path / "long"
    with patch.object(pipeline, "run_child") as run:
        assert pipeline.main(["--checkpoint", "model_49.pt", "--output-dir", str(output)]) == 0
        run.assert_not_called()
    commands = capsys.readouterr().out
    assert "--max_iterations 500" in commands and "--max_iterations 10000" in commands
    assert "--num_envs 128" in commands and "--num_envs 512" in commands
    assert commands.count("--resume_training") == 2
    assert commands.count("--grades 0 5 10 17 20") == 2
    assert not output.exists()


@pytest.mark.parametrize("args", [["--iterations", "0"], ["--short-iterations", "0"],
                                 ["--num-envs", "0"], ["--timeout-hours", "nan"], ["--seed", "-1"]])
def test_long_pipeline_invalid_budgets_do_not_launch(tmp_path, args):
    with pytest.raises(SystemExit):
        pipeline.main(["--checkpoint", "model.pt", "--output-dir", str(tmp_path / "new"), *args])


def test_pipeline_rejects_nonfinite_checkpoint_and_duplicate_reward(tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    (params / "agent.yaml").write_text(yaml.safe_dump(agent_config()))
    model = params / "real2sim_model.json"
    model.write_text("{}")
    env = dict(policy_history_length=5, observation_space=160, state_space=40,
               action_contract_version="minangle_physical_v3", real2sim_observation_version="current_fraction_v2",
               real2sim_enabled=True, real2sim_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
               joint_supported_leveling_weight=80., best_effort_tilt_weight=0., support_gap_weight=0.,
               support_load_weight=0., clearance_margin_weight=0., rewards=dict(all_wheel_contact=0.,
               wheel_load_balance=0., tilt_quadratic=0., flat_orientation_x_exp=0., flat_orientation_y_exp=0.))
    (params / "env.yaml").write_text(yaml.safe_dump(env))
    checkpoint_path = tmp_path / "model_49.pt"
    payload = dict(iter=49, model_state_dict={"actor": torch.ones(4)}, optimizer_state_dict={})
    torch.save(payload, checkpoint_path)
    assert pipeline.checkpoint_info(checkpoint_path)["iteration"] == 49
    payload["model_state_dict"]["actor"][0] = float("nan")
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="nonfinite"):
        pipeline.checkpoint_info(checkpoint_path)
    env["rewards"]["wheel_load_balance"] = 4.
    (params / "env.yaml").write_text(yaml.safe_dump(env))
    with pytest.raises(ValueError, match="Duplicate Joint reward"):
        pipeline.checkpoint_info(checkpoint_path)


def test_long_pipeline_completion_summary_does_not_waive_large_grade_failure(tmp_path):
    directory = tmp_path / "evaluation"
    directory.mkdir()
    for grade in (0, 5, 10, 17, 20):
        row = dict(failure_adjusted_all_contact_rate=1., failure_adjusted_contact_and_horizontal_rate=.8,
                   tilt_deg=dict(abs_p95=grade / 2), terminated_resets=1,
                   physical_terminated_resets=0, terrain_boundary_violations=1)
        report = dict(history=5, passed=grade < 10,
                      results=dict(POLICY={str(i): row for i in range(6)}))
        (directory / f"grade_{grade}.json").write_text(json.dumps(report))
    result = pipeline.summarize_benchmark(directory)
    assert not result["strict_horizontal_goal_achieved"]
    assert result["grades"]["20"]["worst_tilt_p95_deg"] == 10.
    assert result["grades"]["20"]["physical_terminations"] == 0
    assert result["grades"]["20"]["terrain_boundary_violations"] == 6
    assert result["grades"]["20"]["total_terminated_resets"] == 6
    assert (tmp_path / "evaluation_assessment.json").is_file()


def test_short_regression_rejects_before_any_long_training(tmp_path):
    output = tmp_path / "rejected"
    with patch.object(pipeline, "checkpoint_info", side_effect=[dict(iteration=49), dict(iteration=549)]), \
            patch.object(pipeline, "latest_checkpoint", return_value=tmp_path / "model_549.pt"), \
            patch.object(pipeline, "summarize_benchmark", return_value=dict(strict_horizontal_goal_achieved=False)), \
            patch.object(pipeline, "promotion_assessment", return_value=dict(transformer_level_recovered=False)), \
            patch.object(pipeline, "run_child") as run:
        result = pipeline.main(["--checkpoint", "model_49.pt", "--output-dir", str(output), "--run"])
    assert result == 2
    assert [call.args[1] for call in run.call_args_list] == ["train_joint500", "metrics_joint500", "benchmark_joint500"]
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "rejected"
    assert status["current_job"] is None
    assert not status["strict_horizontal_goal_achieved"]


def test_incomplete_baseline_cannot_promote_long_training(tmp_path):
    with patch.object(pipeline, "load_reports", return_value={0: {}}):
        with pytest.raises(ValueError, match="complete five-grade"):
            pipeline.promotion_assessment(tmp_path / "baseline", tmp_path / "candidate")
