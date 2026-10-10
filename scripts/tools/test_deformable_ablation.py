"""CPU-only policy and subprocess orchestration tests; no simulator imports."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ablation = load("ablation_test", "scripts/tools/deformable_ablation.py")
MLP = load("mlp_test", "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_suspension_mlp.py").ActorCriticSuspensionMLP


@pytest.mark.parametrize("history", [1, 4, 5, 8])
@pytest.mark.parametrize("noise_type", ["scalar", "log"])
@pytest.mark.parametrize("floor", [0.0, 0.03])
def test_policy_interface_and_floor(history, noise_type, floor):
    obs = {"policy": torch.randn(3, 32 * history), "critic": torch.randn(3, 40)}
    groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = MLP(obs, groups, 4, history_length=history, min_noise_std=floor,
                 noise_std_type=noise_type, actor_hidden_dims=[8], critic_hidden_dims=[8],
                 actor_obs_normalization=True, critic_obs_normalization=True,
                 d_model=64, nhead=4, num_layers=2, dim_ff=128, head_hidden=64,
                 actor_head="per_leg", actor_layout={}, critic_layout={})
    policy.update_normalization(obs)
    with torch.no_grad():
        (policy.log_std if noise_type == "log" else policy.std).fill_(-100)
    actions = policy.act(obs)
    torch.testing.assert_close(policy.action_std, torch.full((3, 4), max(floor, 1.e-6)))
    assert actions.shape == policy.act_inference(obs).shape == (3, 4)
    assert policy.evaluate(obs).shape == (3, 1)
    assert torch.isfinite(policy.entropy).all()
    assert torch.isfinite(policy.get_actions_log_prob(actions)).all()
    assert policy.load_state_dict(policy.state_dict()) is True
    policy.reset(torch.ones(3, dtype=torch.bool))


@pytest.mark.parametrize("kwargs", [{"init_noise_std": 0}, {"min_noise_std": -1},
                                     {"history_length": 2}, {"state_dependent_std": True}])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        MLP({"policy": torch.zeros(2, 32)}, {"policy": ["policy"], "critic": ["policy"]}, 4, **kwargs)


def test_default_dry_run_matrix(capsys):
    with patch.object(ablation.subprocess, "run") as run:
        assert ablation.main([]) == 0
        run.assert_not_called()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 18
    for model in ablation.MODELS:
        for history in (1, 4, 8):
            for seed in (42, 123, 2026):
                command = next(line for line in lines if f"_{model}_h{history}_s{seed}" in line)
                assert sys.executable in command and "scripts/rsl_rl/train.py" in command
                for arg in (f"agent.policy.class_name={ablation.MODELS[model]}",
                            f"agent.policy.history_length={history}", f"env.policy_history_length={history}",
                            f"env.observation_space={32 * history}", "--max_iterations 100", "--num_envs 128",
                            "agent.policy.noise_std_type=log", "agent.policy.min_noise_std=0.03"):
                    assert arg in command


@pytest.mark.parametrize("args", [["--num-envs", "129"], ["--iterations", "0"],
                                   ["--timeout", "0"], ["--seeds", "-1"],
                                   ["--seeds", "42", "42"]])
def test_invalid_cli(args):
    with pytest.raises(SystemExit):
        ablation.main(args)


def test_checkpoint_selection(tmp_path):
    run = tmp_path / "2026-01-01_unique"
    (run / "params").mkdir(parents=True)
    (run / "params/agent.yaml").touch()
    for name in ("model_9.pt", "model_100.pt", "model_bad.pt"):
        (run / name).touch()
    assert ablation.latest_checkpoint(tmp_path, "unique") == run / "model_100.pt"
    with pytest.raises(FileNotFoundError):
        ablation.latest_checkpoint(tmp_path, "different")


def test_sequential_evaluation_and_failure_reporting(capsys):
    checkpoint = ROOT / "fake/model_99.pt"
    with patch.object(ablation, "latest_checkpoint", return_value=checkpoint), patch.object(
        ablation.subprocess, "run", side_effect=[None, None, subprocess.TimeoutExpired("train", 5),
                                               subprocess.CalledProcessError(2, "train")]
    ) as run:
        assert ablation.main(["--run", "--evaluate", "--models", "mlp", "--seeds", "42", "--timeout", "5"]) == 1
    assert run.call_count == 4
    assert run.call_args_list[0].args[0][1] == "scripts/rsl_rl/train.py"
    eval_command = run.call_args_list[1].args[0]
    assert eval_command[1] == "scripts/tools/deformable_suspension_eval.py"
    assert str(checkpoint) in eval_command
    assert eval_command[eval_command.index("--history") + 1] == "1"
    for call in run.call_args_list:
        assert call.kwargs == {"cwd": ROOT, "timeout": 5, "check": True}
    output = capsys.readouterr()
    assert '"status": "failed"' in output.out and "FAILED" in output.err
