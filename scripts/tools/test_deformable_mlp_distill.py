"""The trained sensor-only MLP must preserve normalization at raw-input export."""

import importlib.util
import copy
from pathlib import Path
import sys

import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tools"))
from deformable_mlp_distill import (
    fold_input_normalization, join_mlp_actors, validate_geometry_training_report,
    startup_sample_weights, weighted_mse,
    selected_feedback_targets,
    reference_residual_targets, validate_anchor_reference,
    validate_training_rollout_safety,
)


def test_startup_objective_prioritizes_early_feedback_without_splitting_vehicles():
    import numpy as np
    weights = startup_sample_weights(6, 8, .5, 2., 6.).reshape(6, 8)
    assert np.all(weights[:4] == 6.) and np.all(weights[4:] == 1.)
    held = np.broadcast_to(np.arange(8)[None] % 4 == 3, (6, 8))
    assert np.all(weights[held].reshape(6, 2) == weights[:, :2])
    predicted = torch.ones(2, 4, requires_grad=True)
    target = torch.zeros_like(predicted)
    loss = weighted_mse(predicted, target, torch.tensor([6., 1.]))
    loss.backward()
    torch.testing.assert_close(predicted.grad[0], 6*predicted.grad[1])
    early_error, late_error = predicted.detach().clone(), predicted.detach().clone()
    early_error[1] = 0
    late_error[0] = 0
    assert weighted_mse(early_error, target, torch.tensor([6., 1.])) > weighted_mse(
        late_error, target, torch.tensor([6., 1.]))


@pytest.mark.parametrize("step_dt,duration,weight", [(0., 2., 6.), (.01, float('nan'), 6.), (.01, 2., .5)])
def test_bad_startup_weighting_contract_is_rejected(step_dt, duration, weight):
    with pytest.raises(ValueError):
        startup_sample_weights(6, 8, step_dt, duration, weight)


def test_unselected_feedback_keeps_raw_saturated_means_and_active_teacher_labels():
    frozen = torch.tensor([[-1.3, 1.2, .4, -.2], [-1.4, 1.1, .3, -.5]])
    teacher = torch.tensor([[-.8, .9, .2, -.4], [-1.4, 1.3, .1, -.3]])
    selected = torch.tensor([False, True])
    actual = selected_feedback_targets(teacher, frozen, selected)
    torch.testing.assert_close(actual[0], frozen[0])
    torch.testing.assert_close(actual[1], teacher[1])
    assert actual[0, 0] < -1 and actual[0, 1] > 1
    assert actual[1, 0] < -1 and actual[1, 1] > 1


def test_regional_mentor_preserves_raw_means_and_never_trains_reference():
    frozen = torch.tensor([[-1.3, 1.2, .4, -.2], [-1.4, 1.1, .3, -.5]], requires_grad=True)
    reference = torch.tensor([[-.8, .9, .2, -.4], [-1.2, 1.4, .1, -.3]], requires_grad=True)
    target = reference_residual_targets(reference, frozen, torch.tensor([False, True]))
    torch.testing.assert_close(target[0], torch.zeros(4))
    torch.testing.assert_close(target[1], (reference-frozen)[1])
    predicted = torch.zeros_like(target, requires_grad=True)
    (predicted-target).square().mean().backward()
    assert reference.grad is None and frozen.grad is None
    assert predicted.grad[1].abs().sum() > 0 and predicted.grad[0].abs().sum() == 0


def test_mentor_must_pass_the_exact_selected_grades(monkeypatch, tmp_path):
    import deformable_mlp_guarded_train as guarded
    monkeypatch.setattr(guarded, "checkpoint_info", lambda path: dict(sha256="same-model-bytes"))
    monkeypatch.setattr(guarded, "qualify", lambda *args: dict(
        grades={"17": dict(recovered=True), "20": dict(recovered=False)}))
    approved = validate_anchor_reference(tmp_path/"model.pt", tmp_path/"reports", [17])
    assert approved["grades"] == [17] and approved["sha256"] == "same-model-bytes"
    with pytest.raises(ValueError, match="every selected grade"):
        validate_anchor_reference(tmp_path/"model.pt", tmp_path/"reports", [17, 20])
    with pytest.raises(ValueError, match="five-grade protocol"):
        validate_anchor_reference(tmp_path/"model.pt", tmp_path/"reports", [17.5])


def test_folded_raw_actor_and_exporter_preserve_actions():
    torch.manual_seed(17)
    mean = torch.randn(160)
    scale = torch.logspace(-1.3, 1., 160)
    actor = torch.nn.Sequential(torch.nn.Linear(160, 32), torch.nn.ELU(), torch.nn.Linear(32, 4))
    x = mean + torch.randn(24, 160) * scale
    reference = actor((x - mean) / scale).detach()
    fold_input_normalization(actor[0], mean, scale)
    torch.testing.assert_close(actor(x), reference, atol=2.e-6, rtol=2.e-5)
    spec = importlib.util.spec_from_file_location("distill_export", ROOT / "scripts/rsl_rl/export_onnx.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    rebuilt, width, outputs = exporter._build_actor({f"actor.{k}": v for k, v in actor.state_dict().items()})
    assert (width, outputs) == (160, 4)
    torch.testing.assert_close(rebuilt(x), reference, atol=2.e-6, rtol=2.e-5)


def test_normalization_folding_does_not_mutate_statistics():
    layer = torch.nn.Linear(3, 4)
    mean = torch.tensor([1., -3., .4])
    scale = torch.tensor([.05, 2., .1])
    old_mean, old_scale = mean.clone(), scale.clone()
    fold_input_normalization(layer, mean, scale)
    assert torch.equal(mean, old_mean) and torch.equal(scale, old_scale)


def test_residual_sum_folds_into_exportable_raw_mlp_with_identical_feedback():
    torch.manual_seed(29)
    def mlp(widths):
        layers = []
        for i, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
            layers.append(torch.nn.Linear(inputs, outputs))
            if i < len(widths)-2:
                layers.append(torch.nn.ELU())
        return torch.nn.Sequential(*layers)
    base, correction = mlp([160, 48, 24, 12, 4]), mlp([160, 16, 8, 4, 4])
    mean, scale = torch.randn(160), torch.rand(160)+.05
    x = mean + scale*torch.randn(24, 160)
    reference = base((x-mean)/scale) + correction((x-mean)/scale)
    joined = join_mlp_actors(base, correction)
    fold_input_normalization(joined[0], mean, scale)
    torch.testing.assert_close(joined(x), reference, atol=2.e-6, rtol=2.e-5)
    assert [m.out_features for m in joined if isinstance(m, torch.nn.Linear)] == [64, 32, 16, 4]
    old = base.state_dict()
    torch.nn.init.zeros_(correction[-1].weight)
    torch.nn.init.zeros_(correction[-1].bias)
    joined = join_mlp_actors(base, correction)
    torch.testing.assert_close(joined(x), base(x), atol=2.e-6, rtol=2.e-5)
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in old.items())


def test_unsafe_geometry_labels_are_rejected_but_steep_residual_is_retained():
    from deformable_mlp_repair_assess import SCENARIOS
    row = dict(physical_terminated_resets=0, terrain_boundary_violations=0,
               failure_adjusted_all_contact_rate=1., tilt_deg=dict(abs_p95=12.))
    report = dict(training_data_controller="sensor_only_geometry_guide",
                  results=dict(POLICY={name: copy.deepcopy(row) for name in SCENARIOS}))
    validate_geometry_training_report(report)
    for key, value in (("physical_terminated_resets", 1), ("terrain_boundary_violations", 1),
                       ("failure_adjusted_all_contact_rate", .979)):
        bad = copy.deepcopy(report)
        bad["results"]["POLICY"]["static"][key] = value
        with pytest.raises(ValueError, match="Unsafe"):
            validate_geometry_training_report(bad)
    del report["results"]["POLICY"]["dynamic"]
    with pytest.raises(ValueError, match="Unsafe"):
        validate_geometry_training_report(report)


def test_student_visited_raw_labels_require_safe_full_scenario_coverage():
    from deformable_mlp_repair_assess import SCENARIOS
    report = dict(training_data_controller="actual_student_checkpoint.pt", results=dict(POLICY={
        name: dict(physical_terminated_resets=0, terrain_boundary_violations=0,
                   failure_adjusted_all_contact_rate=.999) for name in SCENARIOS}))
    validate_training_rollout_safety(report)
    report["results"]["POLICY"]["spin_negative"]["physical_terminated_resets"] = 1
    with pytest.raises(ValueError, match="Unsafe"):
        validate_training_rollout_safety(report)
