"""Independent primal LPs verify the holding-capacity dual, including degeneracy."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import linprog

sys.path.insert(0,str(Path(__file__).resolve().parent))
import deformable_tire_wrench_audit as audit


def primal(a,b,d,c):
    result=linprog(-a,A_eq=np.stack((b,d)),b_eq=np.zeros(2),bounds=list(zip(-c,c)),method='highs')
    assert result.success
    return -result.fun


def test_two_constraint_dual_matches_independent_primal_programs():
    rng=np.random.default_rng(2424)
    a,b,d=rng.uniform(-1,1,(3,257,4));c=rng.uniform(0,100,(257,4))
    c[:10,::2]=0.;c[10]=0.
    actual=audit.balanced_capacity(a,b,d,c)
    expected=np.array([primal(*row) for row in zip(a,b,d,c)])
    np.testing.assert_allclose(actual,expected,atol=1.e-7,rtol=1.e-9)


def test_rank_one_rank_zero_and_unloaded_limits_match_independent_programs():
    rng=np.random.default_rng(2425)
    a=rng.uniform(-1,1,(18,4));b=rng.uniform(-1,1,(18,4));d=3*b;c=rng.uniform(0,100,(18,4))
    b[:4]=0;d[:4]=0;c[5]=0;c[6,1:]=0
    actual=audit.balanced_capacity(a,b,d,c)
    expected=np.array([primal(*row) for row in zip(a,b,d,c)])
    np.testing.assert_allclose(actual,expected,atol=1.e-7,rtol=1.e-9)


def test_yaw_balance_can_remove_an_optimistic_lateral_only_margin():
    # Two wheels roll uphill/downhill; the transverse pair must counter the
    # yaw moment while preserving zero net transverse force.
    a=np.array([[-1.,0.,1.,0.]])
    b=np.array([[0.,-1.,0.,1.]])
    d=np.full((1,4),-.4)
    c=np.array([[10.,10.,60.,10.]])
    lateral=audit.balanced_capacity(a,b,np.zeros_like(d),c)
    complete=audit.balanced_capacity(a,b,d,c)
    assert complete[0]<lateral[0]
    assert complete[0]==pytest.approx(primal(a[0],b[0],d[0],c[0]))


def geometry():
    n=np.array([[-np.sin(np.deg2rad(20)),0,np.cos(np.deg2rad(20))]])
    up=np.array([[np.cos(np.deg2rad(20)),0,np.sin(np.deg2rad(20))]])
    side=np.array([[0.,1.,0.]])
    signs=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])/np.sqrt(2)
    roll=(signs[:,0,None]*up+signs[:,1,None]*side)[None]
    axle=np.cross(n[:,None],roll)
    com=np.array([[[.2,-.2,0.],[.2,.2,0.],[-.2,.2,0.],[-.2,-.2,0.]]])
    points=com-.0769*n[:,None]
    return roll,points,com,axle,np.array([[0.,0.,.1]]),n,np.full((1,4),1000.),np.array([[.9]])


def test_actual_motor_lever_includes_inclined_wheel_axle():
    values=list(geometry())
    upright=audit.force_coefficients(*values,torque_limit=5.)
    np.testing.assert_allclose(upright[-1],.0769,atol=1.e-12)
    np.testing.assert_allclose(upright[3],5/.0769,atol=1.e-9)
    angle=np.deg2rad(30)
    values[3]=values[3]*np.cos(angle)+values[5][:,None]*np.sin(angle)
    inclined=audit.force_coefficients(*values,torque_limit=5.)
    np.testing.assert_allclose(inclined[-1],.0769*np.cos(angle),atol=1.e-12)
    np.testing.assert_allclose(inclined[3],5/(.0769*np.cos(angle)),atol=1.e-9)
    assert (inclined[3]>upright[3]).all()


def test_nonfinite_limits_and_nonplanar_rolling_axes_are_rejected():
    with pytest.raises(ValueError,match='finite'):
        audit.balanced_capacity(np.zeros((2,4)),np.zeros((2,4)),np.zeros((2,4)),np.full((2,4),np.nan))
    values=list(geometry());values[0]=values[5][:,None].repeat(4,axis=1)
    with pytest.raises(ValueError,match='planar'):
        audit.force_coefficients(*values,torque_limit=5.)


def test_old_traces_cannot_fabricate_missing_actuation_geometry(tmp_path,monkeypatch):
    params=tmp_path/'params';params.mkdir()
    (params/'env.yaml').write_text('wheel_torque_limit: 5.0\nwheel_contact_force_threshold: 3.0\n')
    path=tmp_path/'old_trace.npz';np.savez_compressed(path,metrics=np.zeros((2,1,23)))
    monkeypatch.setattr(audit,'audit_motion',lambda _:dict(checkpoint=str(tmp_path/'model.pt')))
    monkeypatch.setattr(audit,'load_reports',lambda _:{17:dict(results=dict(POLICY=dict(static=dict(trace=str(path)))) )})
    with pytest.raises(ValueError,match='geometry is missing'):
        audit.audit(tmp_path,grades=(17,))


def test_contact_spin_slip_matches_independent_finite_rotation_derivative():
    from scipy.spatial.transform import Rotation
    roll,points,com,axle,_,normal,_,_=geometry()
    radius=.0769
    angle=np.deg2rad([0.,10.,30.,60.])[None,:,None]
    axle=axle*np.cos(angle)+normal[:,None]*np.sin(angle)
    # A wheel-COM offset along its axle does not change spin lever or slip.
    points=points-.0055*axle
    spin=np.array([[2.,-3.,0.,5.]])
    velocity=radius*spin[...,None]*roll
    actual,delta=audit.slip_geometry(roll,points,com,axle,spin,velocity,np.zeros((1,4)),radius)
    eps=1.e-6;r=(points-com).reshape(-1,3)
    omega=(spin[...,None]*axle).reshape(-1,3)
    plus=Rotation.from_rotvec(eps*omega).apply(r)
    minus=Rotation.from_rotvec(-eps*omega).apply(r)
    surface=velocity+(plus-minus).reshape(1,4,3)/(2*eps)
    expected=-(surface*roll).sum(-1)
    np.testing.assert_allclose(actual,expected,atol=1.e-10)
    np.testing.assert_allclose(delta, radius*(1-np.cos(angle[...,0]))*spin,atol=1.e-10)


def test_inconsistent_recorded_slip_phase_is_rejected():
    roll,points,com,axle,*_=geometry()
    with pytest.raises(ValueError,match='actual fixed-radius tire calculation'):
        audit.slip_geometry(roll,points,com,axle,np.ones((1,4)),np.zeros((1,4,3)),np.ones((1,4)),.0769)
