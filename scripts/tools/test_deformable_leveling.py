"""CPU tests for leveling objectives, matched acceptance and manual launch."""

import ast
import copy
import importlib.util
import json
import math
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_deformable_dynamic import utilities

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load("deformable_leveling_check")
launcher = load("deformable_leveling_short_train")


def test_slope_height_preference_does_not_block_available_leg_extension():
    du = utilities()
    grades = torch.deg2rad(torch.tensor([0., 2.5, 5., 20., -20.]))
    normals = torch.stack((-grades.sin(), torch.zeros_like(grades), grades.cos()), -1)
    height = torch.full((5,), .30)
    unchanged = du.suspension_body_height_cost(height, .255, normals)
    flat_only = du.suspension_body_height_cost(height, .255, normals, 5.)
    torch.testing.assert_close(unchanged, torch.full((5,), 20.25))
    torch.testing.assert_close(flat_only, unchanged * torch.tensor([1., .5, 0., 0., 0.]), atol=1.e-5, rtol=1.e-5)
    assert du.suspension_body_height_cost(torch.full((5,), .24), .255, normals).count_nonzero() == 0
    # Releasing the height preference must not release the actual tilt cost.
    gravity = torch.stack((grades.sin(), torch.zeros_like(grades), -grades.cos()), -1)
    tilt = du.suspension_tilt_cost(gravity, torch.zeros(5), gate_by_contact=False)
    assert tilt[3].item() == pytest.approx(math.radians(20))


def test_leveling_runner_drops_inherited_reference_and_retains_actor_contract():
    # Execute the actual new class with its parent hook. This isolates config
    # mutation from Isaac imports while testing inherited nonzero constraints.
    path = ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension/agents/rsl_rl_ppo_cfg.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "DeformableFittedLevelingPPORunnerCfg")
    node.decorator_list = []

    class Parent:
        def __post_init__(self):
            self.algorithm = SimpleNamespace(steep_preservation_weight=2., reference_all_postures=True,
                                            flat_posture_weight=1., entropy_coef=.0003)

    scope = {"DeformableFittedMixedCornerPPORunnerCfg": Parent,
             "DeformableHistoryTransformerPolicyCfg": lambda **kw: SimpleNamespace(**kw)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    cfg = scope[node.name]();cfg.__post_init__()
    assert cfg.algorithm.steep_preservation_weight == cfg.algorithm.flat_posture_weight == 0
    assert not cfg.algorithm.reference_all_postures
    assert cfg.policy.init_noise_std >= .05 and cfg.policy.min_noise_std >= .02
    assert cfg.policy.use_leg_geometry_features
    assert not cfg.policy.actor_obs_normalization and not cfg.policy.critic_obs_normalization


def test_leveling_reward_distinguishes_large_tilt_and_retains_contact_constraints():
    path = ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_cfg.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "DeformableFittedLevelingEnvCfg")
    node.decorator_list = []

    class Parent:
        rewards = dict(all_wheel_contact=40., tilt_quadratic=-12.,
                       flat_orientation_x_exp=1.5, flat_orientation_y_exp=1.5,
                       action_rate=-2., termination=-200.)

    scope = {"DeformableFittedMixedCornerEnvCfg": Parent, "OrderedDict": OrderedDict}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    cfg = scope[node.name]()
    # The large-error orientation bonus should have useful resolution before
    # the policy reaches 3 degrees; unloaded wheels must not erase tilt cost.
    bonus15 = math.exp(-math.sin(math.radians(15))**2 / cfg.orientation_y_exp_sigma)
    bonus10 = math.exp(-math.sin(math.radians(10))**2 / cfg.orientation_y_exp_sigma)
    assert bonus10 - bonus15 > .1
    assert cfg.best_effort_tilt_weight > 2 * 12.
    assert not cfg.best_effort_contact_gating
    assert cfg.height_penalty_flat_only and cfg.baseline_extension_penalty_weight == 0
    assert cfg.rewards['all_wheel_contact'] == Parent.rewards['all_wheel_contact']
    assert cfg.rewards['termination'] == Parent.rewards['termination']
    assert cfg.clearance_margin_m >= .006
    assert Parent.rewards['tilt_quadratic'] == -12.  # parent reward table not mutated


def test_leveling_task_registration_selects_the_new_environment_and_runner():
    path = ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension/__init__.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.For) and ast.unparse(n.target) == '(_suffix, _cfg, _runner)')
    registrations = {}
    def register(id, **kwargs):
        registrations[id] = kwargs
    scope = dict(gym=SimpleNamespace(register=register),
                 __name__='agent_tasks.direct.deformable_suspension',
                 agents=SimpleNamespace(__name__='agent_tasks.direct.deformable_suspension.agents'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    entry = registrations[launcher.TASK]['kwargs']
    assert entry['env_cfg_entry_point'].endswith(':DeformableFittedLevelingEnvCfg')
    assert entry['rsl_rl_cfg_entry_point'].endswith(':DeformableFittedLevelingPPORunnerCfg')


def report(grade, p95=15.):
    row = dict(settled_samples=500, failure_adjusted_all_contact_rate=1.,
               failure_adjusted_contact_and_horizontal_rate=1., episode_success_rate=1.,
               tilt_deg={"abs_p95": p95, "rms": p95}, terminated_resets=0,
               physical_terminated_resets=0, terrain_boundary_violations=0,
               initial_state_sha256="paired")
    return dict(seed=1234, steps=600, step_dt_s=.01, num_envs=16, total_envs=96, history=8,
                command_profile="play", command_frame="body", friction=.9,
                policy_preprocessing={"previous_action_pair_filter": False},
                terrain={"grade_deg": grade}, real2sim={"model_sha256": "fitted", "mode": "nominal"},
                results={"POLICY": {name: copy.deepcopy(row) for name in checker.SCENARIOS}})


def test_contact_and_parking_alone_cannot_accept_the_previous_policy():
    old = report(20, 17.)
    result = checker.compare_reports(old, old)
    assert all(r["safe_contact"] for r in result["cases"].values())
    assert not result["leveling_stage_accepted"]
    assert not result["strict_horizontal_passed"]


def test_gentle_grade_requires_strict_horizontal_and_steep_requires_measured_improvement():
    assert checker.compare_reports(report(5, 2.5), report(5, 5.))['leveling_stage_accepted']
    assert not checker.compare_reports(report(5, 3.1), report(5, 5.))['leveling_stage_accepted']
    result = checker.compare_reports(report(20, 12.), report(20, 17.))
    assert result['leveling_stage_accepted'] and not result['strict_horizontal_passed']


@pytest.mark.parametrize("failure", ["contact", "physical", "boundary", "no_settled_samples"])
def test_better_tilt_cannot_hide_a_physical_failure(failure):
    candidate, old = report(20, 12.), report(20, 17.)
    row = candidate['results']['POLICY']['static']
    if failure == "contact":
        row['failure_adjusted_all_contact_rate'] = .97
    elif failure == "physical":
        row['physical_terminated_resets'] = row['terminated_resets'] = 1
    elif failure == "boundary":
        row['terrain_boundary_violations'] = row['terminated_resets'] = 1
    else:
        row['settled_samples'] = 0;row['tilt_deg'] = dict(abs_p95=None, rms=None)
    assert not checker.compare_reports(candidate, old)['leveling_stage_accepted']


@pytest.mark.parametrize("field", ["seed", "command_frame", "steps", "initial_state"])
def test_leveling_comparison_rejects_unpaired_measurements(field):
    candidate, old = report(20, 12.), report(20, 17.)
    if field == "initial_state":
        candidate['results']['POLICY']['static']['initial_state_sha256'] = 'different'
    else:
        candidate[field] = dict(seed=2027, command_frame='world', steps=300)[field]
    with pytest.raises(ValueError, match="Unmatched"):
        checker.compare_reports(candidate, old)


def test_short_training_dry_run_never_launches_or_creates_artifacts(tmp_path, capsys):
    output = tmp_path / 'manual_short'
    with patch.object(launcher.subprocess, 'run') as run:
        assert launcher.main(['--output-dir', str(output)]) == 0
        run.assert_not_called()
    text = capsys.readouterr().out
    assert launcher.TASK in text and '--max_iterations 101' in text
    assert '--finetune_noise_std 0.08' in text
    assert '--resume_training' not in text and '--steep_teacher_checkpoint' not in text
    assert not output.exists()


def test_training_completion_exports_only_this_run_and_still_requires_leveling(tmp_path):
    checkpoint = tmp_path / 'warm/model_100.pt'
    checkpoint.parent.mkdir();checkpoint.write_bytes(b'warm fixture')
    params = checkpoint.parent / 'params';params.mkdir()
    for name in ('agent.yaml', 'env.yaml', 'real2sim_model.json'):
        (params / name).write_text('{}')
    runtime = tmp_path / 'python';runtime.touch()
    output = tmp_path / 'result'
    source_files = [
        'scripts/rsl_rl/train.py', 'scripts/tools/deformable_leveling_short_train.py',
        'source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_cfg.py',
        'source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_env.py',
        'source/agent_tasks/agent_tasks/direct/deformable_suspension/cfg_utils.py',
        'source/agent_tasks/agent_tasks/direct/deformable_suspension/agents/rsl_rl_ppo_cfg.py',
        'source/agent_rl/agent_rl/rsl_rl/algorithms/ppo_diagnostics.py',
    ]
    for relative in source_files:
        file = tmp_path / relative;file.parent.mkdir(parents=True, exist_ok=True);file.write_text('# fixture\n')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if '--run_name' in command:
            suffix = command[command.index('--run_name') + 1]
            folder = tmp_path / 'logs/rsl_rl' / launcher.EXPERIMENT / ('exact_' + suffix)
            folder.mkdir(parents=True);(folder / 'model_100.pt').write_bytes(b'trained fixture')
        return SimpleNamespace(returncode=0)
    with patch.object(launcher, 'ROOT', tmp_path), patch.object(launcher.subprocess, 'run', side_effect=run):
        assert launcher.main(['--run', '--checkpoint', str(checkpoint), '--python', str(runtime),
                              '--output-dir', str(output)]) == 0
    state = json.loads((output / 'status.json').read_text())
    assert state['status'] == 'training_completed_pending_leveling_evaluation'
    assert not state['achieved_leveling'] and len(commands) == 2  # train + CPU plots; no auto GPU evaluation
    assert all(Path(state['followup_commands'][name][2]).is_absolute() for name in ('benchmark', 'leveling_check'))
    assert 'model_100.pt' in (output / 'next_commands.txt').read_text()
