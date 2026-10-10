"""Motion admission must not hide sliding, transitions or failing episodes."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from deformable_mlp_motion_audit import DEFAULT_LIMITS, steady_samples, trace_motion, audit_motion
import deformable_mlp_prepare_deployment as prepare_module


def trace_case(tmp_path, linear=.01, yaw=.01, boundary=0, physical=0):
    steps=600
    times=(np.arange(steps)+1)*.01
    names=['cmd_vx_body','cmd_vy_body','cmd_wz','linear_speed_error_m_s','yaw_error_rad_s']
    metrics=np.zeros((steps,1,len(names)))
    metrics[...,3]=linear;metrics[...,4]=yaw
    path=tmp_path/'trace.npz'
    np.savez(path,columns=names,metrics=metrics,time_s=times,episode_age_s=times[:,None],command_frame='body')
    row=dict(failure_adjusted_all_contact_rate=1.,physical_terminated_resets=physical,terrain_boundary_violations=boundary)
    report=dict(steps=steps,num_envs=1,command_frame='body',step_dt_s=.01)
    return path,row,report


def test_stationary_contact_and_tilt_cannot_hide_parking_slide(tmp_path):
    path,row,report=trace_case(tmp_path,linear=.7)
    result=trace_motion(path,row,report,DEFAULT_LIMITS)
    assert not result['passed'] and not result['checks']['parking_speed']
    assert result['parking_speed_p95_m_s']==pytest.approx(.7)


@pytest.mark.parametrize('boundary,physical',[(1,0),(0,1)])
def test_good_tracking_cannot_hide_physical_or_boundary_failure(tmp_path,boundary,physical):
    path,row,report=trace_case(tmp_path,boundary=boundary,physical=physical)
    assert not trace_motion(path,row,report,DEFAULT_LIMITS)['passed']


def test_transition_settling_restarts_for_each_command_and_each_episode():
    time=(np.arange(600)+1)*.01
    age=np.tile(time[:,None],(1,2));age[400:,1]-=4.
    cmd=np.zeros((600,2,3));cmd[150:300,:,0]=.8;cmd[300:450,:,0]=-.8
    mask=steady_samples(time,age,cmd,.5)
    assert mask[149].all() and not mask[150:200].any() and mask[250].all()
    assert not mask[300:350].any() and not mask[400:450,1].any()
    assert mask[599].all()


def test_steady_gate_respects_declared_yaw_slew_and_reset_ramp():
    time=(np.arange(600)+1)*.01
    age=time[:,None].copy()
    cmd=np.zeros((600,1,3));cmd[:300,:,2]=1.5;cmd[300:,:,2]=-1.5
    mask=steady_samples(time,age,cmd,.5,(2.,4.))
    # Initial 0 -> 1.5 rad/s needs .375 s; reversing needs .75 s.
    assert not mask[:87].any() and mask[100].all()
    assert not mask[300:425].any() and mask[450].all()
    age[450:]-=4.5
    mask=steady_samples(time,age,cmd,.5,(2.,4.))
    assert not mask[450:537].any() and mask[550].all()


def test_failed_training_or_missing_grade_cannot_establish_motion_readiness(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps(dict(status='running')))
    with pytest.raises(ValueError,match='completed independent'):
        audit_motion(tmp_path)
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'checkpoint')
    import hashlib
    (tmp_path/'status.json').write_text(json.dumps(dict(status='completed',checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
    params=tmp_path/'params';params.mkdir();(params/'env.yaml').write_text(
        'drive_linear_acceleration_limit: 2.0\ndrive_yaw_acceleration_limit: 4.0\n')
    with pytest.raises(ValueError,match='all five grades'):
        audit_motion(tmp_path)


def test_motion_gate_runs_before_rmcs_contract_work_even_after_long_training_passes(tmp_path):
    training=tmp_path/'train';training.mkdir()
    (training/'status.json').write_text(json.dumps(dict(stages=[dict(name='long_10')])))
    with patch.object(prepare_module,'audit_training',return_value=('model.pt',{},{})), \
         patch.object(prepare_module,'audit_motion',return_value=dict(motion_ready=False)), \
         patch.object(prepare_module,'run_logged') as run:
        with pytest.raises(ValueError,match='Deployment motion gate failed'):
            prepare_module.prepare(training,tmp_path/'output',tmp_path/'rmcs')
        run.assert_not_called()
