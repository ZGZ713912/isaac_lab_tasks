"""A complete, matched physics result is necessary for policy promotion."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deformable_mlp_repair_assess import GRADES, SCENARIOS, assess


def reports():
    row = dict(failure_adjusted_all_contact_rate=1.,
               failure_adjusted_contact_and_horizontal_rate=1.,
               tilt_deg=dict(abs_p95=.08), physical_terminated_resets=0,
               terrain_boundary_violations=0,
               initial_state_sha256="matched_reset",
               linear_speed_error_m_s=dict(abs_p95=.1), yaw_error_rad_s=dict(abs_p95=.1),
               acceptance=dict(passed=True))
    report = dict(seed=1234, steps=600, num_envs=16, command_profile="play", command_frame="body",
                  step_dt_s=.01, settle_s=.5,
                  real2sim=dict(mode="nominal", observation_version="current_fraction_v2", model_sha256="model"),
                  results=dict(POLICY={name: copy.deepcopy(row) for name in SCENARIOS}))
    return {grade: copy.deepcopy(report) for grade in GRADES}


def test_promotion_requires_every_grade_and_all_safety_metrics():
    baseline = reports()
    assert assess(baseline, copy.deepcopy(baseline))["transformer_level_recovered"]
    missing = copy.deepcopy(baseline)
    del missing[20]
    assert not assess(baseline, missing)["transformer_level_recovered"]
    for key, value in (("failure_adjusted_all_contact_rate", .99),
                       ("physical_terminated_resets", 1), ("terrain_boundary_violations", 1)):
        candidate = copy.deepcopy(baseline)
        candidate[20]["results"]["POLICY"]["spin_negative"][key] = value
        assert not assess(baseline, candidate)["transformer_level_recovered"]
    candidate = copy.deepcopy(baseline)
    candidate[17]["results"]["POLICY"]["dynamic"]["tilt_deg"]["abs_p95"] = .081
    assert not assess(baseline, candidate)["transformer_level_recovered"]


@pytest.mark.parametrize("section,key,value", [(None, "seed", 2401), (None, "steps", 100),
                                              (None, "command_frame", "world"),
                                              ("real2sim", "model_sha256", "other")])
def test_unpaired_protocol_cannot_be_called_restored(section, key, value):
    baseline, candidate = reports(), reports()
    target = candidate[10] if section is None else candidate[10][section]
    target[key] = value
    with pytest.raises(ValueError, match="Unmatched"):
        assess(baseline, candidate)


def test_restoring_baseline_never_waives_strict_horizontal_failure():
    baseline = reports()
    for grade in (10, 17, 20):
        for row in baseline[grade]["results"]["POLICY"].values():
            row["tilt_deg"]["abs_p95"] = grade / 2
            row["acceptance"]["passed"] = False
    result = assess(baseline, copy.deepcopy(baseline))
    assert result["transformer_level_recovered"]
    assert not result["strict_horizontal_goal_achieved"]


def test_geometry_teacher_result_cannot_be_promoted_as_a_pure_mlp():
    baseline, candidate = reports(), reports()
    candidate[10]["training_data_controller"] = "sensor_only_geometry_guide"
    with pytest.raises(ValueError, match="Training controllers"):
        assess(baseline, candidate)


def test_equal_seeds_do_not_hide_different_reset_positions():
    baseline, candidate = reports(), reports()
    candidate[20]["results"]["POLICY"]["forward"]["initial_state_sha256"] = "different_cell_spacing"
    with pytest.raises(ValueError, match="physical reset state"):
        assess(baseline, candidate)
