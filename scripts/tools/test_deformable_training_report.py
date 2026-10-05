"""Check benchmark reports preserve failures and measured baseline posture."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('training_report_test',
    Path(__file__).with_name('deformable_training_report.py'))
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)


def evaluation(tmp_path, *, settled=10, p95=2., steps=600, name='eval.json'):
    row = dict(settled_samples=settled, failure_adjusted_all_contact_rate=1. if settled else 0.,
               failure_adjusted_contact_and_horizontal_rate=1. if settled else 0.,
               tilt_deg=dict(abs_p95=p95, rms=p95), terminated_resets=0 if settled else 1)
    path = tmp_path / name
    path.write_text(json.dumps(dict(seed=1234, steps=steps, step_dt_s=.01,
        terrain=dict(grade_deg=5), results={'POLICY': {'static': row}, 'ZERO': {'static': row}})))
    return path


def test_plot_retains_null_when_episode_fails_before_settling(tmp_path):
    path = evaluation(tmp_path, settled=0, p95=None)
    with patch.object(reporting, 'save_figure'):
        rows = reporting.plot_evaluations([('candidate', path)], tmp_path)
    assert all(row['tilt_p95_max'] is None and row['tilt_rms_max'] is None for row in rows)
    assert all(row['scenarios_without_settled_samples'] == ['static'] for row in rows)
    assert all(row['contact_min'] == 0 and row['terminated_resets'] == 1 for row in rows)
    saved = json.loads((tmp_path / 'evaluation_metrics.json').read_text())
    assert saved[0]['tilt_p95_max'] is None
    reporting.plt.close('all')


def test_same_policy_label_cannot_hide_different_evaluation_duration(tmp_path):
    short = evaluation(tmp_path, steps=400, name='short.json')
    long = evaluation(tmp_path, steps=600, name='long.json')
    with pytest.raises(ValueError, match='Conflicting benchmark conditions'):
        reporting.plot_evaluations([('candidate', short), ('candidate', long)], tmp_path)


def test_same_policy_label_cannot_merge_body_and_world_command_frames(tmp_path):
    world = evaluation(tmp_path, name='world.json')
    body = evaluation(tmp_path, name='body.json')
    data = json.loads(body.read_text())
    data['command_frame'] = 'body'
    body.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Conflicting benchmark conditions'):
        reporting.plot_evaluations([('candidate', world), ('candidate', body)], tmp_path)


def test_baseline_summary_measures_lowest_leg_without_forcing_equal_angles(tmp_path):
    angles = np.array([[[60., 60., 60., 60.]], [[17., 30., 40., 45.]], [[21., 32., 41., 48.]]])
    zero = np.deg2rad(78.)
    path = tmp_path / 'trace.npz'
    np.savez(path, episode_age_s=np.array([[.1], [.6], [.7]]), physical_angle_zero_rad=zero,
             joint_q=zero - np.deg2rad(angles), leg_target=zero - np.deg2rad(angles),
             commanded_current_raw=np.zeros_like(angles))
    summary = reporting.trace_summary(path)
    assert summary['settled_samples'] == 2
    assert summary['mean_lowest_leg_extension_deg'] == pytest.approx(2.)
    assert summary['lowest_leg_within_2deg_of_baseline_rate'] == .5
    assert summary['mean_physical_angle_deg'] == pytest.approx([19., 31., 40.5, 46.5])


def test_action_cycle_diagnostic_excludes_reset_jumps(tmp_path):
    ages = np.array([.60, .61, .62, .10, .20, .60, .61])[:, None]
    actions = np.array([0., .8, 0., 1., 1., 0., .8])[:, None, None]
    actions = np.repeat(actions, 4, axis=-1)
    zero = np.deg2rad(78.)
    path = tmp_path / 'cycle.npz'
    np.savez(path, episode_age_s=ages, physical_angle_zero_rad=zero,
             joint_q=np.full_like(actions, zero - np.deg2rad(17.)),
             leg_target=np.full_like(actions, zero - np.deg2rad(17.)),
             commanded_current_raw=np.zeros_like(actions), raw_policy_actions=actions)
    summary = reporting.trace_summary(path)
    assert summary['mean_abs_action_difference_lag1'] == pytest.approx(.8)
    assert summary['mean_abs_action_difference_lag2'] == pytest.approx(0.)
