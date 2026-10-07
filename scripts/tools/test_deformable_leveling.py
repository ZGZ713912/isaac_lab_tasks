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


def test_dense_support_cost_distinguishes_lifted_corners_without_equalizing_load():
    du = utilities()
    gaps = torch.tensor([[0., -.001, 0., .001], [.004, 0., .004, 0.],
                         [.010, 0., .010, 0.]], requires_grad=True)
    loads = torch.tensor([[8., 20., 100., 200.], [0., 125., 0., 125.], [0., 125., 0., 125.]])
    gap_cost, load_cost = du.suspension_support_costs(gaps, loads)
    assert gap_cost[0] == load_cost[0] == 0  # unequal but supported is free
    assert 0 < gap_cost[1] < gap_cost[2]
    assert load_cost[1] == load_cost[2] == .5
    gap_cost.sum().backward()
    assert (gaps.grad[1:, [0, 2]] > 0).all()  # nearer terrain always costs less
    assert torch.isfinite(gaps.grad).all()
    # A small tilt improvement cannot repay the added weak-support cost alone.
    tilt_gain = 60 * math.radians(3)
    assert 40 * gap_cost[1] + 12 * load_cost[1] > tilt_gain


def test_dense_support_cost_handles_touching_unloaded_wheels_and_large_gaps():
    du = utilities()
    gap, load = du.suspension_support_costs(torch.tensor([[0., 0., 0., 0.], [100., 100., 100., 100.]]),
                                          torch.tensor([[0., 8., 80., 100.], [0., 0., 0., 0.]]))
    assert gap[0] == 0 and load[0] == .25
    assert gap[1] == 5 and load[1] == 1
    for kwargs in (dict(gap_scale_m=0), dict(min_load_n=0), dict(gap_tolerance_m=-1)):
        with pytest.raises(ValueError):
            du.suspension_support_costs(torch.zeros(1, 4), torch.ones(1, 4), **kwargs)


def test_motion_sampling_covers_pure_play_commands_and_both_spin_signs():
    du = utilities()
    draws = torch.full((6, 7), .9)
    draws[:, 0] = torch.tensor([.1, .3, .5, .7, .7, .9])
    draws[4, 3] = .1
    commands = du.suspension_motion_commands(draws, (.8, .5, 1.5), 1., 1., .2)
    torch.testing.assert_close(commands, torch.tensor([
        [0., 0., 0.], [.8, 0., 0.], [0., .5, 0.],
        [0., 0., 1.5], [0., 0., -1.5], [.8, .5, 1.5]]))
    # Early curriculum genuinely parks; translation can start before spinning.
    assert du.suspension_motion_commands(draws, (.8, .5, 1.5), 0., 0., .2).count_nonzero() == 0
    scaled = du.suspension_motion_commands(draws, (.8, .5, 1.5), .5, 0., .2)
    assert scaled[:, 2].count_nonzero() == 0 and scaled[:, 0].max() == .4


def test_motion_sampling_has_bounded_magnitudes_and_explicit_standing_coverage():
    du = utilities()
    generator = torch.Generator().manual_seed(2027)
    commands = du.suspension_motion_commands(torch.rand(20000, 7, generator=generator), (.8, .5, 1.5), 1., 1., .2)
    assert (commands.abs() <= torch.tensor([.8, .5, 1.5])).all()
    assert .18 < (commands == 0).all(-1).float().mean() < .22
    for axis in range(3):
        pure = (commands[:, axis] != 0) & (commands[:, [i for i in range(3) if i != axis]] == 0).all(-1)
        assert .18 < pure.float().mean() < .22
        assert (commands[pure, axis] > 0).any() and (commands[pure, axis] < 0).any()
    with pytest.raises(ValueError):
        du.suspension_motion_commands(torch.rand(2, 6), (.8, .5, 1.5), 1., 1., .2)


def test_shared_terrain_covers_larger_training_grids_and_keeps_benchmarks_unchanged():
    du=utilities()
    assert du.suspension_training_terrain_size(96,8.,(150.,150.)) == (150.,150.)
    assert du.suspension_training_terrain_size(256,8.,(150.,150.)) == (150.,150.)
    assert du.suspension_training_terrain_size(512,8.,(150.,150.)) == (194.,194.)
    assert du.suspension_training_terrain_size(1024,8.,(150.,300.)) == (258.,300.)
    with pytest.raises(ValueError):
        du.suspension_training_terrain_size(0,8.,(150.,150.))


def test_clearance_reserve_does_not_consume_the_steep_leveling_stroke():
    du=utilities()
    grades=torch.deg2rad(torch.tensor([0.,5.,7.5,10.,20.]))
    normal=torch.stack((-grades.sin(),torch.zeros_like(grades),grades.cos()),-1)
    clearance=torch.full((5,),.006,requires_grad=True)
    cost=du.suspension_grade_clearance_cost(clearance,normal,.018,.006)
    torch.testing.assert_close(cost,torch.tensor([4/9,4/9,.25,0.,0.]),atol=1.e-6,rtol=1.e-6)
    cost.sum().backward()
    assert (clearance.grad[:3]<0).all() and torch.isfinite(clearance.grad).all()
    assert du.suspension_grade_clearance_cost(torch.full((5,),.018),normal,.018,.006).count_nonzero()==0
    with pytest.raises(ValueError):
        du.suspension_grade_clearance_cost(clearance,normal,.018,0.)


def test_gentle_precision_uses_ground_grade_and_preserves_steep_support_priority():
    du = utilities()
    grades = torch.deg2rad(torch.tensor([0., 5., 10., 13.5, 17., 20., -20.]))
    normal = torch.stack((-grades.sin(), torch.zeros_like(grades), grades.cos()), -1)
    weight = du.suspension_grade_precision_multiplier(normal, 2.)
    torch.testing.assert_close(weight, torch.tensor([2., 2., 2., 1.5, 1., 1., 1.]), atol=1.e-6, rtol=1.e-6)
    torch.testing.assert_close(du.suspension_grade_precision_multiplier(normal), torch.ones(7))
    # Equal body tilt receives greater precision pressure only on gentle ground.
    tilt = torch.full((7,), math.radians(4), requires_grad=True)
    gravity = torch.stack((tilt.sin(), torch.zeros_like(tilt), -tilt.cos()), -1)
    cost = 120 * weight * du.suspension_tilt_cost(gravity, torch.ones(7), gate_by_contact=False)
    cost.sum().backward()
    torch.testing.assert_close(tilt.grad[2], 2 * tilt.grad[5])
    torch.testing.assert_close(tilt.grad[5], tilt.grad[6])
    assert torch.isfinite(tilt.grad).all()
    for multiplier in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            du.suspension_grade_precision_multiplier(normal, multiplier)


@pytest.mark.parametrize('command', [(0.,0.,0.),(.4,0.,0.),(0.,0.,-1.5)])
def test_steep_motion_reserve_preserves_parking_and_gentle_grade_stroke(command):
    du = utilities()
    grades = torch.deg2rad(torch.tensor([5.,10.,17.,20.]))
    normal = torch.stack((-grades.sin(),torch.zeros_like(grades),grades.cos()),-1)
    clearance = torch.full((4,),.006,requires_grad=True)
    body_command = torch.tensor(command).expand(4,3)
    cost = du.suspension_grade_clearance_cost(clearance,normal,.018,.006,
                                            motion_command=body_command,steep_motion_margin_m=.012)
    moving = any(command)
    torch.testing.assert_close(cost,torch.tensor([4/9,0.,.25 if moving else 0.,.25 if moving else 0.]),atol=1.e-6,rtol=1.e-6)
    cost.sum().backward()
    assert torch.isfinite(clearance.grad).all()
    if moving:
        assert (clearance.grad[2:] < 0).all()


def test_steep_parking_attitude_transient_gets_reserve_without_changing_10deg():
    du = utilities()
    grades = torch.deg2rad(torch.tensor([10.,20.]))
    normal = torch.stack((-grades.sin(),torch.zeros_like(grades),grades.cos()),-1)
    clearance = torch.full((2,),.006)
    command = torch.zeros(2,3)
    quiet = du.suspension_grade_clearance_cost(clearance,normal,.018,.006,
        motion_command=command,steep_motion_margin_m=.012,attitude_rate=torch.zeros(2,2))
    transient = du.suspension_grade_clearance_cost(clearance,normal,.018,.006,
        motion_command=command,steep_motion_margin_m=.012,attitude_rate=torch.tensor([[.25,0.],[.25,0.]]))
    torch.testing.assert_close(quiet,torch.zeros(2),atol=1.e-6,rtol=1.e-6)
    torch.testing.assert_close(transient,torch.tensor([0.,.25]),atol=1.e-6,rtol=1.e-6)


def test_flat_leg_alignment_releases_on_a_grade_or_a_single_sloping_footprint():
    du=utilities()
    q=torch.tensor([[.8,.8,.8,.8],[.8,.9,.8,.9],[.8,.9,.8,.9]],requires_grad=True)
    normals=torch.zeros(3,4,3);normals[...,2]=1.
    theta=math.radians(5.)
    normals[2,0]=torch.tensor([-math.sin(theta),0.,math.cos(theta)])
    cost=du.suspension_flat_leg_spread_cost(q,normals)
    assert cost[0]==cost[2]==0 and cost[1]>0
    cost.sum().backward()
    assert q.grad[1,0]<0 and q.grad[1,1]>0
    assert q.grad[2].count_nonzero()==0  # needed slope leg difference stays free


def test_support_stages_start_gently_and_preserve_the_original_leveling_recipe():
    path = ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_cfg.py"
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name.startswith("DeformableFittedSupportLeveling")]
    for node in nodes:
        node.decorator_list = []
    def terrain(**kwargs):
        return SimpleNamespace(terrain_generator=SimpleNamespace(
            sub_terrains={"periodic_slope": SimpleNamespace(segment_length=3., angle_choices=None)}))
    class Parent:
        best_effort_tilt_weight = 60.
        support_min_load_n = 8.
        rewards = dict(all_wheel_contact=40., tilt_quadratic=-40., termination=-200.)
        leg_max_physical_angle = math.radians(75.)
        max_leg_torque = 44.37
    scope = dict(DeformableFittedLevelingEnvCfg=Parent, _make_periodic_slope_terrain=terrain)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    five, ten, mixed = (scope[f"DeformableFittedSupportLeveling{name}EnvCfg"]() for name in ("Five", "Ten", "Mixed"))
    assert five.terrain.terrain_generator.sub_terrains["periodic_slope"].angle_choices == (5.,)
    assert ten.terrain.terrain_generator.sub_terrains["periodic_slope"].angle_choices == (0., 5., 10.)
    assert mixed.terrain.terrain_generator.sub_terrains["periodic_slope"].angle_choices == (0., 5., 10., 17., 20.)
    assert five.cmd_rel_standing_envs > mixed.cmd_rel_standing_envs
    motion = scope['DeformableFittedSupportLevelingMotionEnvCfg']()
    assert motion.terrain.terrain_generator.sub_terrains['periodic_slope'].angle_choices == (0.,5.)
    assert motion.support_motion_commands and ten.support_motion_commands and mixed.support_motion_commands
    assert motion.cmd_lin_vel_x_range == ten.cmd_lin_vel_x_range == mixed.cmd_lin_vel_x_range == (-.8, .8)
    assert motion.cmd_ang_vel_z_range == ten.cmd_ang_vel_z_range == mixed.cmd_ang_vel_z_range == (-1.5, 1.5)
    assert motion.support_min_load_n == 15.
    assert ten.support_min_load_n == mixed.support_min_load_n > motion.support_min_load_n
    assert ten.support_load_weight == mixed.support_load_weight > motion.support_load_weight
    assert five.best_effort_tilt_weight == motion.best_effort_tilt_weight == Parent.best_effort_tilt_weight
    assert ten.best_effort_tilt_weight == mixed.best_effort_tilt_weight > motion.best_effort_tilt_weight
    assert ten.gentle_precision_multiplier == mixed.gentle_precision_multiplier == 2.
    assert ten.baseline_extension_penalty_weight == mixed.baseline_extension_penalty_weight == 0.
    assert mixed.clearance_steep_motion_margin_m >= .018
    assert mixed.motion_curriculum_iterations <= 200  # full commands before short-run end
    assert motion.clearance_margin_m >= .018
    assert motion.clearance_grade_margin_m == ten.clearance_grade_margin_m == mixed.clearance_grade_margin_m == .006
    assert five.clearance_margin_m >= .012
    for cfg in (five, ten, mixed):
        assert cfg.support_gap_weight > 0 and cfg.support_load_weight > 0
        assert cfg.support_min_load_n > 3.
        assert cfg.best_effort_tilt_weight >= Parent.best_effort_tilt_weight
        assert cfg.rewards == Parent.rewards and cfg.leg_max_physical_angle == Parent.leg_max_physical_angle


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


def test_precision_stages_match_launcher_noise_and_keep_actor_contract():
    path = ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension/agents/rsl_rl_ppo_cfg.py"
    names = {"DeformableFittedLevelingPPORunnerCfg", "DeformableFittedSupportLevelingPPORunnerCfg",
             "DeformableFittedSupportLevelingTenPPORunnerCfg", "DeformableFittedSupportLevelingMixedPPORunnerCfg"}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []

    class Parent:
        def __post_init__(self):
            self.algorithm = SimpleNamespace(steep_preservation_weight=2., reference_all_postures=True,
                                            flat_posture_weight=1., entropy_coef=.001)

    scope = {"DeformableFittedMixedCornerPPORunnerCfg": Parent,
             "DeformableHistoryTransformerPolicyCfg": lambda **kw: SimpleNamespace(**kw)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    for stage in ('ten', 'mixed'):
        cfg = scope[f'DeformableFittedSupportLeveling{stage.title()}PPORunnerCfg']()
        cfg.__post_init__()
        assert cfg.policy.init_noise_std == launcher.NOISE_STD[stage]
        assert 0 < cfg.policy.min_noise_std < cfg.policy.init_noise_std < launcher.NOISE_STD['five']
        assert cfg.policy.use_leg_geometry_features
        assert not cfg.policy.actor_obs_normalization and not cfg.policy.critic_obs_normalization
        assert cfg.algorithm.steep_preservation_weight == cfg.algorithm.flat_posture_weight == 0
        assert not cfg.algorithm.reference_all_postures


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
    assert entry['env_cfg_entry_point'].endswith(':DeformableFittedSupportLevelingFiveEnvCfg')
    assert entry['rsl_rl_cfg_entry_point'].endswith(':DeformableFittedSupportLevelingPPORunnerCfg')
    for stage, task in launcher.TASKS.items():
        assert registrations[task]['kwargs']['env_cfg_entry_point'].endswith(
            f':DeformableFittedSupportLeveling{stage.title()}EnvCfg')
        runner_suffix = '' if stage == 'five' else stage.title()
        assert registrations[task]['kwargs']['rsl_rl_cfg_entry_point'].endswith(
            f':DeformableFittedSupportLeveling{runner_suffix}PPORunnerCfg')


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
    assert launcher.TASK in text and '--max_iterations 201' in text
    assert '--finetune_noise_std 0.05' in text
    assert '--resume_training' not in text and '--steep_teacher_checkpoint' not in text
    assert not output.exists()


@pytest.mark.parametrize('stage,iterations', [('five', 201), ('motion', 401), ('ten', 401), ('mixed', 601)])
def test_stage_budgets_reach_the_configured_motion_curriculum(stage, iterations, tmp_path, capsys):
    with patch.object(launcher.subprocess, 'run') as run:
        assert launcher.main(['--stage', stage, '--output-dir', str(tmp_path / 'dry')]) == 0
        run.assert_not_called()
    text = capsys.readouterr().out
    assert launcher.TASKS[stage] in text and f'--max_iterations {iterations}' in text
    assert launcher.EXPERIMENTS[stage] in text


def test_precision_noise_override_reaches_training_without_launching(tmp_path, capsys):
    with patch.object(launcher.subprocess, 'run') as run:
        assert launcher.main(['--stage', 'ten', '--noise-std', '.01',
                              '--output-dir', str(tmp_path / 'dry')]) == 0
        run.assert_not_called()
    assert '--finetune_noise_std 0.01' in capsys.readouterr().out
    assert not (tmp_path / 'dry').exists()


@pytest.mark.parametrize('value', ['0', '-0.01', 'nan', 'inf'])
def test_precision_noise_rejects_invalid_values_before_launch(value, tmp_path):
    with patch.object(launcher.subprocess, 'run') as run, pytest.raises(SystemExit):
        launcher.main(['--stage', 'ten', '--noise-std', value,
                       '--output-dir', str(tmp_path / 'dry')])
    run.assert_not_called()
    assert not (tmp_path / 'dry').exists()


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
            iteration = int(command[command.index('--max_iterations') + 1]) - 1
            folder.mkdir(parents=True);(folder / f'model_{iteration}.pt').write_bytes(b'trained fixture')
        return SimpleNamespace(returncode=0)
    with patch.object(launcher, 'ROOT', tmp_path), patch.object(launcher.subprocess, 'run', side_effect=run):
        assert launcher.main(['--run', '--checkpoint', str(checkpoint), '--python', str(runtime),
                              '--output-dir', str(output)]) == 0
    state = json.loads((output / 'status.json').read_text())
    assert state['status'] == 'training_completed_pending_leveling_evaluation'
    assert not state['achieved_leveling'] and len(commands) == 2  # train + CPU scalars; no auto GPU evaluation
    assert not state['plots_enabled'] and '--metrics-only' in commands[1]
    assert not (output / 'charts').exists()
    assert all(Path(state['followup_commands'][name][2]).is_absolute() for name in ('benchmark', 'leveling_check'))
    assert 'model_200.pt' in (output / 'next_commands.txt').read_text()


def test_numeric_training_export_creates_no_figures(tmp_path):
    exporter = load('deformable_training_report')
    with patch.object(exporter, 'read_scalars', return_value={'Train/mean_reward': [dict(step=1, value=2.)]}), \
         patch.object(exporter.plt, 'subplots', side_effect=AssertionError('No charts requested')):
        records = exporter.export_training_metrics([('fixture', tmp_path / 'run')], tmp_path)
    assert records['fixture']['scalars']['Train/mean_reward'][0]['value'] == 2.
    assert {p.name for p in tmp_path.iterdir()} == {'training_metrics.json'}
