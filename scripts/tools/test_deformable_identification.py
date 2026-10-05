"""Identification checks independent of a running Isaac session or private CSVs."""
import importlib.util
import math
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from deformable_urdf_dynamics import ReducedLegDynamics, CSV_IN_URDF_ORDER
from deformable_assembly_fit import fit, grounded_target, matrix
from deformable_fit_data import load_recording, FIELDS, GLOBALS, JOINTS

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('identified_mechanism_test',ROOT/'source/agent_tasks/agent_tasks/direct/deformable_suspension/real2sim.py')
actuator=importlib.util.module_from_spec(spec);spec.loader.exec_module(actuator)


def test_cad_gravity_is_potential_gradient_in_all_four_independent_coordinates():
    m=ReducedLegDynamics();rng=np.random.default_rng(3)
    q=rng.uniform(.06,1.1,(7,4)); gravity=rng.normal(size=(7,3));gravity*=9.81/np.linalg.norm(gravity,axis=1)[:,None]
    mass,g,_,wheels=m.quantities(q,gravity)
    total=sum(link[0] for link in m.links.values());eps=1.e-6
    for j in range(4):
        shift=np.zeros(4);shift[j]=eps
        plus=m.quantities(q+shift)[2];minus=m.quantities(q-shift)[2]
        derivative=-total*np.sum((plus-minus)*gravity,axis=1)/(2*eps)
        np.testing.assert_allclose(g[:,j],derivative,atol=1.e-7,rtol=1.e-6)
    np.testing.assert_allclose(mass,mass.transpose(0,2,1),atol=1.e-12)
    assert np.linalg.eigvalsh(mass).min()>0
    np.testing.assert_array_equal(np.sign(wheels[0,:,:2]),[[1,-1],[1,1],[-1,1],[-1,-1]])


def example_document():
    return {'joint_order':CSV_IN_URDF_ORDER,'absolute_shaft_torque_calibrated':False,
        'urdf_zero_physical_angle_deg':78.,'models':{name:{'torque_per_count':.02+j*.001,
            'extra_inertia':.03+j*.001,'viscous_friction':.1,'coulomb_friction':.5,
            'constant_effort_bias':0.,'assembly_residual_cos_nm':0.,'assembly_residual_sin_nm':0.,
            'upper_stop_physical_angle_deg':75.,'stop_stiffness_nm_rad':1000.,
            'stop_compression_damping_nm_s_rad':4.} for j,name in enumerate(CSV_IN_URDF_ORDER)}}


def test_fitted_passive_losses_and_stop_have_physical_signs():
    a=actuator.FittedLegMechanism(example_document(),'cpu',torch.float64)
    q=torch.full((2,4),.8,dtype=torch.float64)
    v=torch.tensor([[-1.,-.1,.1,1.],[2.,-2.,.01,-.01]],dtype=torch.float64)
    losses=a.effort(torch.zeros_like(q),q,v)
    assert torch.all(losses*v<=0)
    current=torch.full_like(q,100.)
    torch.testing.assert_close(a.effort(current,q,v)-losses,100*a.gain.expand_as(q))
    q=a.stop_q[None].expand_as(q)-.01
    at_rest=a.effort(torch.zeros_like(q),q,torch.zeros_like(q))
    torch.testing.assert_close(at_rest,torch.full_like(q,10.))
    moving_in=a.effort(torch.zeros_like(q),q,-torch.ones_like(q))
    moving_out=a.effort(torch.zeros_like(q),q,torch.ones_like(q))
    assert torch.all(moving_in>at_rest) and torch.all(moving_out<at_rest)


def test_wrong_corner_order_and_nonfinite_fits_are_rejected():
    d=example_document();d['joint_order']=('left_front','left_back','right_back','right_front')
    with pytest.raises(ValueError,match='URDF corners'):actuator.FittedLegMechanism(d,'cpu')
    d=example_document();d['models']['left_front']['extra_inertia']=math.nan
    with pytest.raises(ValueError,match='non-finite'):actuator.FittedLegMechanism(d,'cpu')


def test_inverse_fit_recovers_known_current_scale_using_separate_load_condition():
    rng=np.random.default_rng(8);physics=ReducedLegDynamics();zero=np.arctan2(.13694,.029108)
    mass=np.diag(physics.quantities(np.zeros((1,4)))[0][0]);gain=.024
    truth=np.array([gain]+[.03,.12,.55,0.,0.,0.]*4)
    chunks=[]
    for ground in (False,True):
        n=3000
        q=rng.uniform(.2,1.,(n,1 if ground else 4));q=np.broadcast_to(q,(n,4)).copy()
        v=rng.uniform(.08,.3,(n,1 if ground else 4))*rng.choice([-1,1],(n,1 if ground else 4))
        v=np.broadcast_to(v,(n,4)).copy();acc=rng.uniform(-1,1,(n,4))
        needed=physics.quantities(q)[1]+mass*acc
        chunk={'stage':3,'time':np.linspace(0,50,n),'q':q,'v':v,'a':acc,'needed':needed,
               'current':np.zeros((n,4)),'filtered_current':np.zeros((n,4))}
        passive=.03*acc+.12*v+.55*np.tanh(v/.02)
        target=needed.copy()
        if ground:
            target-=(needed.sum(-1)-grounded_target(chunk,physics))[:,None]/4
        chunk['current']=chunk['filtered_current']=(target+passive)/gain
        chunks.append(chunk)
    estimate=fit([chunks[0]],[chunks[1]],physics,zero)
    assert estimate[0]==pytest.approx(gain,rel=2.e-3)
    for j in range(4):np.testing.assert_allclose(estimate[1+6*j:4+6*j],truth[1+6*j:4+6*j],rtol=.01,atol=.001)


def test_partial_numeric_cache_is_rebuilt_from_unchanged_source(tmp_path):
    import pandas as pd
    path=tmp_path/'record.csv'
    columns=[f'/identification/{f}' for f in GLOBALS]+['/chassis/imu/roll','/chassis/imu/pitch']
    columns += [f'/chassis/{j}_joint/{f}' for j in JOINTS for f in FIELDS]
    pd.DataFrame({column:[1.,2.,3.] for column in columns}).to_csv(path,index=False)
    Path(str(path)+'.json').write_text(json.dumps({'condition':'suspended'}))
    Path(str(path)+'.status.json').write_text(json.dumps({'completed':True}))
    first=load_recording(path,tmp_path/'cache')
    (tmp_path/'cache'/(first['sha256']+'.npz')).write_bytes(b'partial cache')
    recovered=load_recording(path,tmp_path/'cache')
    assert first['sha256']==recovered['sha256']
    np.testing.assert_array_equal(first['arrays']['current_raw'],recovered['arrays']['current_raw'])
