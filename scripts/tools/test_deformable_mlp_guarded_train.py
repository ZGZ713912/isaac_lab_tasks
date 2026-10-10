"""A long recovery run must never bypass failed physics or checkpoint identity."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deformable_mlp_guarded_train as pipeline
from test_deformable_mlp_repair_assess import reports


def test_completed_reports_require_exact_checkpoint_identity(tmp_path):
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"reviewed checkpoint")
    candidate, baseline = tmp_path / "candidate", tmp_path / "baseline"
    candidate.mkdir(); baseline.mkdir()
    data = reports()
    for grade, report in data.items():
        report["checkpoint"] = str(checkpoint)
        for directory in (candidate, baseline):
            (directory / f"grade_{grade}.json").write_text(json.dumps(report))
    status = dict(status="completed", checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (candidate / "status.json").write_text(json.dumps(status))
    assert pipeline.qualify(checkpoint, candidate, baseline)["transformer_level_recovered"]
    checkpoint.write_bytes(b"later unreviewed checkpoint")
    with pytest.raises(ValueError, match="exact checkpoint"):
        pipeline.qualify(checkpoint, candidate, baseline)


def test_failed_admission_never_starts_any_training(tmp_path):
    output = tmp_path / "long"
    with patch.object(pipeline, "checkpoint_info", return_value={}), \
            patch.object(pipeline, "qualify", return_value=dict(transformer_level_recovered=False)), \
            patch.object(pipeline, "run_child") as run:
        result = pipeline.main(["--checkpoint", "model.pt", "--qualified-evaluation-dir", "evaluation",
                                "--output-dir", str(output), "--run"])
    assert result == 2 and not output.exists()
    run.assert_not_called()


def test_failed_preflight_retains_the_admitted_checkpoint_and_skips_long_training(tmp_path):
    output = tmp_path / "long"
    source = tmp_path / "source.pt"
    candidate = tmp_path / "candidate.pt"
    info = dict(iteration=99, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[256, 128, 64])
    with patch.object(pipeline, "checkpoint_info", return_value=info), \
            patch.object(pipeline, "qualify", side_effect=[dict(transformer_level_recovered=True),
                                                           dict(transformer_level_recovered=False)]), \
            patch.object(pipeline, "latest_checkpoint", return_value=candidate), \
            patch.object(pipeline, "actor_update_info", return_value=dict(policy_updated=True)), \
            patch.object(pipeline, "run_child") as run:
        result = pipeline.main(["--checkpoint", str(source), "--qualified-evaluation-dir", "evaluation",
                                "--output-dir", str(output), "--run"])
    assert result == 2
    assert [call.args[1] for call in run.call_args_list] == ["train_preflight", "benchmark_preflight"]
    state = json.loads((output / "status.json").read_text())
    assert state["status"] == "rejected" and state["last_accepted_checkpoint"] == str(source)
    assert state["rejected_checkpoint"] == str(candidate)


def test_fresh_stage_uses_saved_actor_and_critic_widths_without_optimizer_resume():
    args = SimpleNamespace(seed=47, device="cuda:0")
    info = dict(actor_hidden_dims=[576, 288, 144], critic_hidden_dims=[512, 256, 128])
    command = pipeline.training_command(args, Path("model.pt"), "preflight", 100, 128, info)
    assert "agent.policy.actor_hidden_dims=[576,288,144]" in command
    assert "agent.policy.critic_hidden_dims=[512,256,128]" in command
    assert "--resume_training" not in command
    assert command[command.index("--max_iterations")+1] == "100"


def test_routed_stage_uses_its_registered_task_and_all_saved_routing_fields():
    args = SimpleNamespace(seed=47, device="cuda:0")
    info = dict(policy_class=pipeline.ROUTED_CLASS, actor_hidden_dims=[12,8], critic_hidden_dims=[16,8],
                routed_parameters=dict(expert_hidden_dims=[[12,8],[16,8],[20,8]],router_hidden_dims=[8],
                                       routing_confidence=.995,routing_load_threshold=.02))
    command = pipeline.training_command(args, Path("model.pt"), "routed_preflight", 100, 128, info)
    assert command[command.index("--task")+1] == pipeline.ROUTED_TASK
    assert "agent.policy.expert_hidden_dims=[[12,8],[16,8],[20,8]]" in command
    assert "agent.policy.routing_confidence=0.995" in command
    assert "agent.policy.routing_load_threshold=0.02" in command


def test_training_replay_is_training_only_and_requires_the_routed_task():
    args = SimpleNamespace(seed=47, device="cuda:0", reference_replay=Path("training/policy_obs.npz"))
    info = dict(policy_class=pipeline.ROUTED_CLASS, actor_hidden_dims=[12,8], critic_hidden_dims=[16,8],
                routed_parameters={})
    command = pipeline.training_command(args, Path("model.pt"), "replay", 100, 128, info)
    assert command[command.index("--reference_replay")+1] == str(args.reference_replay)
    assert "agent.algorithm.reference_replay_weight=10.0" in command
    info["policy_class"] = "ActorCriticSuspensionMLP"
    with pytest.raises(ValueError, match="routed"):
        pipeline.training_command(args, Path("model.pt"), "replay", 100, 128, info)


def test_critic_noise_only_changes_do_not_count_as_policy_optimization(tmp_path):
    import torch
    a=tmp_path/"before.pt";b=tmp_path/"after.pt"
    state={"actor.experts.0.0.weight":torch.ones(4,160),"actor.experts.0.0.bias":torch.zeros(4),
           "actor.router.0.weight":torch.ones(3,160),"critic.0.weight":torch.zeros(1,40),
           "log_std":torch.zeros(4)}
    torch.save(dict(model_state_dict=state),a)
    state["critic.0.weight"].add_(1);state["log_std"].add_(1)
    torch.save(dict(model_state_dict=state),b)
    assert not pipeline.actor_update_info(a,b)["policy_updated"]
    state["actor.experts.0.0.bias"].add_(1.e-5)
    torch.save(dict(model_state_dict=state),b)
    result=pipeline.actor_update_info(a,b)
    assert result["policy_updated"] and result["changed_tensors"]==1
    state["actor.router.0.weight"].add_(1.e-5)
    torch.save(dict(model_state_dict=state),b)
    with pytest.raises(ValueError,match="router changed"):
        pipeline.actor_update_info(a,b)
