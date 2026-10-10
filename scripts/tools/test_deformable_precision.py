"""CPU contracts for explicit fine-tune noise, grade tests and batch selection."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


noise = load("precision_noise", "scripts/rsl_rl/finetune_utils.py")
pipeline = load("precision_pipeline", "scripts/tools/deformable_precision_pipeline.py")
evaluate = load("precision_eval", "scripts/tools/deformable_suspension_eval.py")
geometry = load("precision_geometry", "scripts/tools/deformable_feasibility.py")


def policy(kind="log"):
    result = torch.nn.Module()
    result.noise_std_type = kind
    result.min_noise_std = .03
    result.actor = torch.nn.Linear(3, 4)
    value = math.log(.49) if kind == "log" else .49
    result.register_parameter("log_std" if kind == "log" else "std", torch.nn.Parameter(torch.full((4,), value)))
    return result


@pytest.mark.parametrize("kind", ["log", "scalar"])
def test_explicit_noise_reset_changes_only_noise(kind):
    p = policy(kind)
    before = p.actor.weight.clone()
    assert noise.reset_policy_noise(p, .15) == pytest.approx(.49)
    actual = p.log_std.exp() if kind == "log" else p.std
    torch.testing.assert_close(actual, torch.full((4,), .15))
    torch.testing.assert_close(p.actor.weight, before)


@pytest.mark.parametrize("std", [0., -.1, .01, math.nan, math.inf])
def test_invalid_noise_does_not_mutate_loaded_parameters(std):
    p = policy()
    before = p.log_std.clone()
    with pytest.raises(ValueError):
        noise.reset_policy_noise(p, std)
    torch.testing.assert_close(p.log_std, before)


def test_state_dependent_noise_is_rejected():
    p = policy()
    p.state_dependent_std = True
    with pytest.raises(ValueError):
        noise.reset_policy_noise(p, .15)


def test_grade_configuration_and_invalid_bounds():
    sub = SimpleNamespace(angle_range=(0, 5), segment_length=1, angle_choices=(5, 10, 17, 20))
    cfg = SimpleNamespace(terrain=SimpleNamespace(terrain_generator=SimpleNamespace(sub_terrains={"periodic_slope": sub})),
                          boundary_reset_enabled=True, spawn_dir_jitter=True, scene=SimpleNamespace(env_spacing=8.))
    evaluate.configure_grade(cfg, 17)
    assert sub.angle_range == (17, 17) and sub.segment_length == 20
    assert sub.angle_choices is None
    assert not cfg.boundary_reset_enabled and not cfg.spawn_dir_jitter
    assert cfg.scene.env_spacing == 6.
    with pytest.raises(ValueError):
        evaluate.configure_grade(cfg, 21)


def test_dry_run_is_read_only_and_grades_are_not_dropped(tmp_path, capsys):
    output = tmp_path / "new_run"
    with patch.object(pipeline.subprocess, "Popen") as launch:
        assert pipeline.main(["--checkpoint", "model.pt", "--output-dir", str(output)]) == 0
        launch.assert_not_called()
    commands = capsys.readouterr().out
    assert "--finetune_noise_std 0.15" in commands and "--max_iterations 1000" in commands
    for grade in (10, 17, 20):
        assert f"--grade-deg {grade}.0" in commands
    assert not output.exists()


def test_selection_rejects_leveling_by_loss_of_contact():
    def report(contact, joint, p95, terminated=0):
        return {"results": {"POLICY": {"static": {
            "terminated_resets": terminated, "failure_adjusted_all_contact_rate": contact,
            "failure_adjusted_contact_and_horizontal_rate": joint, "tilt_deg": {"abs_p95": p95}}}}}
    baseline = report(.99, .7, 4)
    assert pipeline.score(baseline) > pipeline.score(report(.8, .79, 2))
    assert pipeline.score(report(.99, .9, 3.5)) > pipeline.score(baseline)
    assert pipeline.score(baseline) > pipeline.score(report(1, 1, 1, terminated=1))


def test_extra_grade_diagnostics_preserve_contact_acceptance_data():
    sample = [1, 2, 3, 1, 0, 20, .01, 0, 0, .25, .5, .02, .004, .9, 0, .6, 1, 0, .6, .3, .75]
    result = evaluate.summarize([{"samples": [sample], "terminated": False, "timeout": False,
                                  "settled_samples": 1, "success": False}])
    assert result["all_contact_rate"] == 1 and result["contact_and_horizontal_rate"] == 0
    assert result["mean_abs_slip_m_s"] == .004 and result["target_tracking_error_rad"] == .02
    assert result["mean_velocity_world"] == [.9, 0, .6]
    assert result["leg_target_rate_saturation_rate"] == .75


def test_geometric_reference_checks_all_contact_and_clearance():
    du = geometry.utilities()
    bounds = [-.13, .13, -.13, .13, -.027]
    flat = geometry.best_effort_pose(du, 0, 45, bounds)
    assert flat["best_feasible_tilt_deg"] == 0 and flat["certified_global_optimum"]
    steep = geometry.best_effort_pose(du, 20, 45, bounds)
    assert steep["best_feasible_tilt_deg"] is not None and 0 < steep["best_feasible_tilt_deg"] < 20
    assert not steep["certified_global_optimum"]
    assert steep["max_abs_wheel_normal_gap_m"] < 2.e-5
    assert steep["body_normal_clearance_m"] >= .006 - 2.e-5
    assert min(steep["q_urdf"]) >= -1.e-6 and max(steep["q_urdf"]) <= du.Q_LOW + 1.e-6


def test_v3_geometry_respects_actual_stroke_and_clearance():
    du = geometry.utilities()
    bounds = [-.13, .13, -.13, .13, -.027]
    limits = (du.URDF_ZERO_PHYSICAL_ANGLE - math.radians(75),
              du.URDF_ZERO_PHYSICAL_ANGLE - math.radians(16))
    steep = geometry.best_effort_pose(du, 20, 45, bounds, q_range=limits)
    assert steep["best_feasible_tilt_deg"] is not None
    assert min(steep["q_urdf"]) >= limits[0] - 1.e-6
    assert max(steep["q_urdf"]) <= limits[1] + 1.e-6
    assert steep["max_abs_wheel_normal_gap_m"] < 2.e-5
    assert steep["body_normal_clearance_m"] >= .006 - 2.e-5
    with pytest.raises(ValueError):
        geometry.scan_pose(du, 0, 0, bounds, q_range=(limits[1], limits[0]))


def test_velocity_error_metrics_use_individual_errors_not_error_of_mean():
    base = [0, 0, 0, 1, 1, 20, .01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .25, 0]
    result = evaluate.summarize([{
        "samples": [base + [1., 2.], base + [3., 4.]], "terminated": False, "timeout": False,
        "settled_samples": 2, "success": False,
    }])
    assert result["linear_speed_error_m_s"]["rms"] == pytest.approx(math.sqrt(5))
    assert result["yaw_error_rad_s"]["rms"] == pytest.approx(math.sqrt(10))


def test_play_command_profile_matches_gui_amplitudes_and_preserves_stress_commands():
    assert evaluate.command_at('forward', 0, 600) == (1., 0., 0.)
    assert evaluate.command_at('spin_positive', 0, 600) == (0., 0., 2 * math.pi)
    assert evaluate.command_at('forward', 0, 600, 'play') == (.8, 0., 0.)
    assert evaluate.command_at('lateral', 0, 600, 'play') == (0., .5, 0.)
    assert evaluate.command_at('spin_negative', 0, 600, 'play') == (0., 0., -1.5)
    assert evaluate.command_at('dynamic', 150, 600, 'play') == (.4, 0., 1.5)
    assert evaluate.command_at('dynamic', 300, 600, 'play') == (-.4, 0., -1.5)
    with pytest.raises(ValueError, match='profile'):
        evaluate.command_at('static', 0, 600, 'unknown')


def test_velocity_tracking_uses_command_frame_at_a_rotated_heading():
    # A robot facing +world-Y correctly follows its body-forward command.
    velocity_world = torch.tensor([[0., .8, 0.]])
    velocity_body = torch.tensor([[.8, 0., 0.]])
    command = torch.tensor([[.8, 0., 0.]])
    assert evaluate.linear_velocity_error(velocity_world, velocity_body, command, 'body').item() == 0
    assert evaluate.linear_velocity_error(velocity_world, velocity_body, command, 'world').item() == pytest.approx(math.sqrt(1.28))
    with pytest.raises(ValueError, match='frame'):
        evaluate.linear_velocity_error(velocity_world, velocity_body, command, 'invalid')


def test_body_commands_are_not_reported_as_world_commands():
    sample = [0, 0, 0, 1, 1, 20, .01, 0, 0, 0, 0, 0, 0, 0, .8, 0, .8, 0, 0, .25, 0, 0, 0]
    result = evaluate.summarize([dict(samples=[sample], terminated=False, timeout=False,
                                     settled_samples=1, success=True)], 'body')
    assert 'mean_command_world' not in result
    assert result['mean_command_body'] == [.8, 0, 0]
    assert result['mean_velocity_world'] == [0, .8, 0]


def test_pre_settle_failure_retains_missing_tilt_and_fails_acceptance():
    result = evaluate.summarize([dict(samples=[], terminated=True, timeout=False,
                                     settled_samples=0, success=False)])
    assert result['tilt_deg']['abs_p95'] is None
    assert result['failure_adjusted_all_contact_rate'] == 0
    assert result['short_failed_episodes'] == 1
    assert not evaluate.acceptance(result)['passed']


def test_ramp_boundary_exit_is_counted_separately_and_cannot_pass():
    row = [0, 0, 0, 1, 1, 20, .01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .25, 0]
    result = evaluate.summarize([dict(samples=[row], terminated=True, physical_terminated=False,
                                     terrain_boundary=True, timeout=False, settled_samples=1, success=False)])
    assert result['terminated_resets'] == 1
    assert result['physical_terminated_resets'] == 0
    assert result['terrain_boundary_violations'] == 1
    assert not evaluate.acceptance(result)['passed']
