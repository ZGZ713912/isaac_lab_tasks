"""CPU contracts for current-domain units, timing, resets and audit statistics."""
import importlib.util
import math
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / 'source/agent_tasks/agent_tasks/direct/deformable_suspension'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


actuator = load('real2sim_test', ENV / 'real2sim.py')
adrc = load('real2sim_adrc_test', ENV / 'adrc.py')
audit = load('real2sim_audit_test', ROOT / 'scripts/tools/deformable_real2sim_identify.py')
checkpoint = load('real2sim_checkpoint_test', ROOT / 'scripts/utils/deformable_checkpoint.py')


def controller_cfg(**kwargs):
    values = dict(adrc_dt=.001, adrc_b0=-1., adrc_delta=.02,
                  leg_max_physical_angle=math.radians(75), adrc_eso_w0=250., adrc_z3_limit=1e9,
                  adrc_td_r=50., adrc_td_h=.001, adrc_td_max_acc=math.inf, adrc_td_max_vel=math.inf,
                  adrc_k1=30., adrc_k2=17., adrc_alpha1=.75, adrc_alpha2=.7,
                  adrc_u_min=-200., adrc_u_max=200., adrc_kt=1., adrc_output_min=-200., adrc_output_max=200.,
                  max_leg_torque=25., adrc_feedback_applied_torque=True,
                  adrc_output_domain='current_raw', adrc_controller_output_to_current_raw=5.74635241301908)
    return SimpleNamespace(**(values | kwargs))


def make_actuator(**kwargs):
    cfg = SimpleNamespace(adrc_dt=.001, **({'real2sim_randomize': False,
                          'real2sim_command_delay_steps_range': (0, 0),
                          'real2sim_feedback_delay_steps_range': (0, 0),
                          'real2sim_command_period_steps_range': (1, 1)} | kwargs))
    a = actuator.Real2SimActuator((2, 4), 'cpu', cfg, actuator.load_real2sim_model(), dtype=torch.float64)
    q = torch.zeros(2, 4, dtype=torch.float64)
    a.reset([0, 1], q, q)
    return a, q


def test_counts_are_not_nm_and_rounding_matches_cpp():
    a, q = make_actuator()
    current = q.new_tensor([[20480., -20480., 1.5, -2.5]]).expand_as(q)
    effort = a.apply(current, q, q)
    torch.testing.assert_close(a.command_current_raw[0], q.new_tensor([2048., -2048., 2., -3.]))
    torch.testing.assert_close(effort[0], q.new_tensor([25., -25., 2.*25/2048, -3.*25/2048]))
    torch.testing.assert_close(a.last_effort, effort)


def test_hold_and_transport_delay_have_distinct_effects():
    a, q = make_actuator(real2sim_command_period_steps_range=(2, 2), real2sim_command_delay_steps_range=(1, 1))
    outputs, prepared = [], []
    for value in [10., 20., 30., 40., 50.]:
        outputs.append(a.apply(q.new_full(q.shape, value), q, q)[0, 0].item() / (25 / 2048))
        prepared.append(a.command_current_raw[0, 0].item())
    assert prepared == [10., 10., 30., 30., 50.]
    assert outputs == pytest.approx([0., 10., 10., 30., 30.])


def test_encoder_delay_and_selective_reset_do_not_leak_previous_episode():
    a, q = make_actuator(real2sim_feedback_delay_steps_range=(2, 2))
    observed = []
    for i in range(5):
        observed.append(a.sensor_measurement(q + i + 1, q)[0][0, 0].item())
    assert observed == [0., 0., 1., 2., 3.]
    a.reset([0], q + 9., q)
    measured, _, current = a.sensor_measurement(q + 10., q)
    assert measured[0, 0].item() == 9.
    assert measured[1, 0].item() == 4.
    assert current[0].count_nonzero() == 0


def test_feedback_current_is_independent_of_unknown_torque_gain_and_friction():
    a, q = make_actuator(real2sim_torque_scale_range=(.5, .5), real2sim_coulomb_friction_range=(.2, .2))
    effort = a.apply(q.new_full(q.shape, 1000.), q, q + 1.)
    _, _, current = a.sensor_measurement(q, q)
    torch.testing.assert_close(current, q.new_full(q.shape, 1000.))
    assert effort[0, 0] == pytest.approx(1000.*25/2048*.5 - .2)


def test_backlash_reversal_through_zero_and_passive_friction():
    a, q = make_actuator(real2sim_backlash_range=(.01, .01), real2sim_coulomb_friction_range=(.2, .2))
    for value in [100., 0., -100.]:
        effort = a.apply(q.new_full(q.shape, value), q, q + 1.)
    assert effort[0, 0] == pytest.approx(-100.*25/2048*.25 - .2)
    effort = a.apply(q.new_full(q.shape, -100.), q + .02, q + 1.)
    assert effort[0, 0] == pytest.approx(-100.*25/2048 - .2)


def test_each_joint_randomizes_and_reset_preserves_other_env():
    torch.manual_seed(2)
    a, q = make_actuator(real2sim_randomize=True, real2sim_torque_scale_range=(.9, 1.1))
    before = a._torque_scale.clone()
    assert before[0].unique().numel() > 1
    a.reset([0], q, q)
    torch.testing.assert_close(a._torque_scale[1], before[1])
    assert not torch.equal(a._torque_scale[0], before[0])


@pytest.mark.parametrize('field,value', [
    ('real2sim_command_delay_steps_range', (-1, 1)),
    ('real2sim_command_period_steps_range', (0, 2)),
    ('real2sim_feedback_delay_steps_range', (1, 1.5)),
    ('real2sim_torque_lag_tau_s', -1.),
    ('real2sim_angle_noise_std_range', (math.nan, 1.)),
    ('real2sim_torque_per_current_raw', 0.),
])
def test_invalid_actuator_parameters_rejected(field, value):
    with pytest.raises(ValueError):
        make_actuator(**{field: value})


def test_adrc_raw_multiplier_and_prepared_command_feedback():
    cfg = controller_cfg()
    c = adrc.LegADRC((1, 4), 'cpu', cfg, dtype=torch.float64)
    q = torch.ones(1, 4, dtype=torch.float64)
    c.reset([0], q, q)
    c.last_u.fill_(999.)
    c.applied_u.fill_(2048.)
    prepared = q.new_full(q.shape, 100.)
    out = c.update(q, q, applied_current_raw=prepared)
    torch.testing.assert_close(c.z2, q.new_full(q.shape, -.001*100./cfg.adrc_controller_output_to_current_raw))
    torch.testing.assert_close(out, (c.last_u*cfg.adrc_controller_output_to_current_raw).clamp(-2048, 2048))


def test_legacy_kt_does_not_change_internal_clamp_order():
    cfg = controller_cfg(adrc_output_domain='torque', adrc_kt=2., adrc_u_min=-1., adrc_u_max=1.)
    c = adrc.LegADRC((1, 4), 'cpu', cfg, dtype=torch.float64)
    q = torch.ones(1, 4, dtype=torch.float64)
    c.reset([0], q, q)
    c.z3.fill_(100.)
    c.update(q, q)
    torch.testing.assert_close(c.last_u, q.new_full(q.shape, 2.))


def test_step_rise_is_90_minus_10_and_recordings_are_not_merged():
    def rows(offset):
        samples = [(0., 17., 17.), (.1, 17., 75.), (.2, 22.8, 75.), (.9, 69.2, 75.), (1., 75., 75.)]
        return [(t, 'left_front', math.radians(v), math.radians(target), 0., 0., 0.) for t,v,target in samples]
    result = audit._step_metrics([{'file':'a', 'step_rows':rows(0)}, {'file':'b', 'step_rows':rows(0)}])
    transitions = result['left_front']['transitions']
    assert len(transitions) == 2
    assert all(t['rise_10_90_s'] == pytest.approx(.7) for t in transitions)


def test_step_window_does_not_use_following_jump():
    samples = [(0.,17.,17.), (.1,17.,75.), (.2,30.,75.), (.3,30.,17.), (.4,80.,17.)]
    rows = [(t,'left_front',math.radians(v),math.radians(target),0.,0.,0.) for t,v,target in samples]
    t = audit._step_metrics([{'file':'a','step_rows':rows}])['left_front']['transitions'][0]
    assert t['rise_10_90_s'] is None
    assert t['overshoot_deg'] == 0


def test_can_period_uses_sequence_increment_and_skips_incomplete_pairs():
    item = {'last_feedback_time_s': None, 'last_feedback_sequence': None,
            'feedback_period_s': audit._Stats(), 'sequence_duplicates': {'feedback': 0}}
    for t, seq in [(1., 100), (1.004, 102), (1.004, 102), (math.nan, 103), (1.01, 105)]:
        audit._record_period(item, 'feedback', t, seq)
    assert item['feedback_period_s'].as_dict()['mean'] == pytest.approx(.002)
    assert item['feedback_period_s'].n == 2
    assert item['sequence_duplicates']['feedback'] == 1


@pytest.mark.parametrize('saved_real2sim,task_real2sim', [(False, True), (True, False)])
def test_current_observation_checkpoint_mismatch_rejected(tmp_path, saved_real2sim, task_real2sim):
    params = tmp_path / 'params'
    params.mkdir()
    (params / 'env.yaml').write_text('action_contract_version: minangle_residual_v2\n'
        f'real2sim_enabled: {str(saved_real2sim).lower()}\nreal2sim_observation_version: current_fraction_v1\n')
    cfg = SimpleNamespace(action_contract_version='minangle_residual_v2', real2sim_enabled=task_real2sim,
                          real2sim_observation_version='current_fraction_v1')
    with pytest.raises(ValueError, match='observation contract mismatch'):
        checkpoint.validate_deformable_checkpoint(cfg, tmp_path / 'model_2.pt')


def test_real2sim_model_snapshot_must_match_saved_hash(tmp_path):
    params = tmp_path / 'params'
    params.mkdir()
    model_path = params / 'real2sim_model.json'
    model_path.write_text('{}')
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (params / 'env.yaml').write_text('action_contract_version: minangle_residual_v2\n'
        'real2sim_enabled: true\nreal2sim_observation_version: current_fraction_v1\n'
        f'real2sim_model_sha256: {digest}\n')
    cfg = SimpleNamespace(action_contract_version='minangle_residual_v2', real2sim_enabled=True,
                          real2sim_observation_version='current_fraction_v1')
    checkpoint.validate_deformable_checkpoint(cfg, tmp_path / 'model_2.pt')
    assert cfg.real2sim_model_path == str(model_path)
    model_path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='matching'):
        checkpoint.validate_deformable_checkpoint(cfg, tmp_path / 'model_2.pt')
