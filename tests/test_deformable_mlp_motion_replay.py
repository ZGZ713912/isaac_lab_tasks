"""Admitted preservation data protects transients without preserving sliding."""

import numpy as np
import pytest

from deformable_mlp_motion_replay import eligible_states, select_admitted, source_control_contract
from deformable_mlp_replay import SCENARIOS


def samples():
    scenarios=np.repeat(sorted(SCENARIOS),4)
    critic=np.zeros((60,24,40),np.float32)
    critic[:,:,9]=-1.
    obs=np.zeros((60,24,160),np.float32)
    obs[:,:,0]=np.arange(60)[:,None];obs[:,:,1]=np.arange(24)[None]
    return obs,critic,scenarios


def test_sliding_steady_states_are_excluded_but_initial_contact_states_remain():
    _,critic,_=samples();critic[:,:,:]=0.;critic[:,:,32]=.7
    eligible,steady=eligible_states(critic,.1)
    assert not steady[:5].any() and eligible[:5].all()
    assert steady[6:].all() and not eligible[6:].any()


def test_all_scenarios_preserve_whole_vehicle_holdouts_and_do_not_sample_sliding_states():
    obs,critic,scenarios=samples();critic[:,1::4,32]=.7
    selected,pairs,eligible,_=select_admitted(obs,critic,scenarios,.1,np.random.default_rng(4),16,[0])
    assert len(selected)==192 and not np.any(pairs[:,1]%4==3) and not np.any(pairs[:,1]==0)
    assert eligible[pairs[:,0],pairs[:,1]].all()
    assert not np.any((pairs[:,1]%4==1)&(pairs[:,0]>5))
    for case in SCENARIOS:
        case_rows=pairs[scenarios[pairs[:,1]]==case]
        assert len(case_rows)==32 and (case_rows[:,0]<20).sum()==16


def test_empty_admitted_stratum_fails_instead_of_falling_back_to_bad_motion():
    obs,critic,scenarios=samples();critic[:,:,32]=.7
    with pytest.raises(ValueError,match='No admitted training states'):
        select_admitted(obs,critic,scenarios,.1,np.random.default_rng(4),16,[])


def test_command_yaw_scale_and_ramp_are_not_mistaken_for_steady_sliding():
    _,critic,_=samples();critic[:,:,3]=.375;critic[:,:,6]=.75
    eligible,steady=eligible_states(critic,.1)
    assert not steady[:8].any() and eligible.all()
    critic[:,:,6]=0.
    eligible,steady=eligible_states(critic,.1)
    assert eligible[:8].all() and not eligible[9:].any()


def test_source_slew_sensor_and_command_frames_must_agree(tmp_path):
    params=tmp_path/'params';params.mkdir();cfg=params/'env.yaml'
    cfg.write_text('drive_linear_acceleration_limit: 2.0\ndrive_yaw_acceleration_limit: 4.0\n')
    report=dict(real2sim=dict(mode='fixed_parameters_zero_temporal_noise',observation_version='current_fraction_v2'),command_frame='body')
    data=dict(controller_checkpoint=str(tmp_path/'model.pt'))
    assert source_control_contract(report,data)['drive_slew_limits']==[2.,4.]
    report['command_frame']='world'
    with pytest.raises(ValueError,match='body-frame'):
        source_control_contract(report,data)
    report['command_frame']='body'
    cfg.write_text('drive_linear_acceleration_limit: 1.0\ndrive_yaw_acceleration_limit: 4.0\n')
    with pytest.raises(ValueError,match='command slew'):
        source_control_contract(report,data)
