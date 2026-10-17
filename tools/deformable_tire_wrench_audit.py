"""Diagnose quasi-static tire holding capacity from actual actuation snapshots.

This never fits a teacher or changes a policy. Positive capacity is not a
whole-vehicle dynamic guarantee. The wheel-only tangential bound omits pitch/
roll balance, transient inertia, contact compliance and external wrenches.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from deformable_mlp_motion_audit import audit_motion, steady_samples
from deformable_mlp_repair_assess import load_reports


def balanced_capacity(uphill, lateral, yaw, limits):
    """Maximize uphill force with zero lateral force/yaw moment and box limits.

    The LP dual is min_(lambda,eta) sum(c_i |a_i+lambda*b_i+eta*d_i|).
    Enumerate intersections of its zero lines. Rank-one constraints reduce to
    their one-dimensional breakpoints; rank zero gives the unconstrained sum.
    """
    arrays=[np.asarray(x,dtype=np.float64) for x in (uphill,lateral,yaw,limits)]
    if (any(x.shape != arrays[0].shape or x.ndim != 2 or x.shape[1] != 4
            or not np.isfinite(x).all() for x in arrays) or (arrays[3]<0).any()):
        raise ValueError('Capacity requires aligned finite four-wheel coefficients and nonnegative force limits')
    a,b,d,c=arrays
    result=np.sum(c*np.abs(a),axis=1)
    full_rank=np.zeros(len(a),dtype=bool)
    for p,q in itertools.combinations(range(4),2):
        determinant=b[:,p]*d[:,q]-b[:,q]*d[:,p]
        scale=np.maximum(np.linalg.norm(np.stack((b[:,p],d[:,p]),axis=1),axis=1)
                         *np.linalg.norm(np.stack((b[:,q],d[:,q]),axis=1),axis=1),1.e-30)
        valid=np.abs(determinant)>1.e-12*scale
        full_rank|=valid
        denominator=np.where(valid,determinant,1.)
        lam=(-a[:,p]*d[:,q]+a[:,q]*d[:,p])/denominator
        eta=(-b[:,p]*a[:,q]+b[:,q]*a[:,p])/denominator
        cost=np.sum(c*np.abs(a+lam[:,None]*b+eta[:,None]*d),axis=1)
        result=np.minimum(result,np.where(valid,cost,np.inf))
    if (~full_rank).any():
        ids=np.flatnonzero(~full_rank)
        coefficients=np.stack((b[ids],d[ids]),axis=-1)
        norms=np.linalg.norm(coefficients,axis=-1)
        largest=norms.argmax(axis=1)
        direction=coefficients[np.arange(len(ids)),largest]/np.maximum(norms.max(axis=1)[:,None],1.e-30)
        beta=(coefficients*direction[:,None]).sum(-1)
        for wheel in range(4):
            valid=np.abs(beta[:,wheel])>1.e-30
            point=-a[ids,wheel]/np.where(valid,beta[:,wheel],1.)
            cost=np.sum(c[ids]*np.abs(a[ids]+point[:,None]*beta),axis=1)
            result[ids]=np.minimum(result[ids],np.where(valid,cost,np.inf))
    return result


def force_coefficients(roll, points, wheel_com, axle, whole_com, normal, load, friction, torque_limit):
    """Use actual contact/axle lever arms, including axle inclination."""
    roll,points,wheel_com,axle,whole_com,normal,load,friction=[np.asarray(x,dtype=np.float64)
        for x in (roll,points,wheel_com,axle,whole_com,normal,load,friction)]
    n=len(load)
    if (any(x.shape!=(n,4,3) for x in (roll,points,wheel_com,axle))
            or whole_com.shape!=(n,3) or normal.shape!=(n,3) or load.shape!=(n,4)
            or friction.shape!=(n,1) or not np.isfinite(torque_limit) or torque_limit<=0
            or not all(np.isfinite(x).all() for x in (roll,points,wheel_com,axle,whole_com,normal,load,friction))
            or any(np.max(np.abs(np.linalg.norm(x,axis=-1)-1.))>1.e-4 for x in (roll,axle,normal))
            or np.max(np.abs((roll*normal[:,None]).sum(-1)))>1.e-4):
        raise ValueError('Wrench coefficients require finite aligned unit directions and planar rolling axes')
    vertical=np.zeros_like(normal);vertical[:,2]=1.
    tangent=vertical-normal[:,2:3]*normal
    sine=np.linalg.norm(tangent,axis=1)
    uphill=tangent/np.maximum(sine[:,None],1.e-30)
    flat=sine<1.e-12
    if flat.any():
        uphill[flat]=[1.,0.,0.]
    side=np.cross(normal,uphill)
    a=(roll*uphill[:,None]).sum(-1)
    b=(roll*side[:,None]).sum(-1)
    d=(np.cross(points-whole_com[:,None],roll)*normal[:,None]).sum(-1)
    lever=np.abs((np.cross(points-wheel_com,roll)*axle).sum(-1))
    motor=torque_limit/np.maximum(lever,1.e-30)
    limits=np.minimum(np.maximum(friction,0.)*np.maximum(load,0.),motor)
    return a,b,d,limits,sine,lever


def slip_geometry(roll,points,wheel_com,axle,spin,velocity,model_slip,radius):
    """Compare recorded tire slip with the actual contact-point spin velocity.

    Unit positive wheel spin rotates around the recorded local-x axle. The
    physical surface velocity adds omega cross (point - wheel COM), rather
    than assuming a radius independent of axle/ground inclination.
    """
    roll,points,wheel_com,axle,spin,velocity,model_slip=[np.asarray(x,dtype=np.float64)
        for x in (roll,points,wheel_com,axle,spin,velocity,model_slip)]
    n=len(spin)
    if (any(x.shape!=(n,4,3) for x in (roll,points,wheel_com,axle,velocity))
            or spin.shape!=(n,4) or model_slip.shape!=(n,4) or not np.isfinite(radius) or radius<=0
            or not all(np.isfinite(x).all() for x in (roll,points,wheel_com,axle,spin,velocity,model_slip))):
        raise ValueError('Slip geometry needs aligned finite actuation-phase spin and contact velocity')
    along=(velocity*roll).sum(-1)
    if not np.allclose(model_slip,radius*spin-along,atol=2.e-6,rtol=1.e-5):
        raise ValueError('Recorded slip does not match the actual fixed-radius tire calculation')
    surface=velocity+spin[...,None]*np.cross(axle,points-wheel_com)
    actual=-(surface*roll).sum(-1)
    return actual,model_slip-actual


def audit(directory,grades=(17,20)):
    motion=audit_motion(directory)
    reports=load_reports(directory)
    params=Path(motion['checkpoint']).parent/'params/env.yaml'
    cfg=yaml.load(params.read_text(),Loader=yaml.BaseLoader)
    torque=float(cfg['wheel_torque_limit']);contact=float(cfg['wheel_contact_force_threshold'])
    rows=[]
    for grade in grades:
        report=reports[grade];row=report['results']['POLICY']['static'];path=Path(row['trace'])
        with np.load(path,allow_pickle=False) as trace:
            required={'traction_trace_schema','traction_snapshot','tire_roll_direction_w','wheel_contact_points_w',
                'wheel_com_positions_w','wheel_axle_direction_w','articulation_com_position_w',
                'wheel_ground_normals_w','body_masses_kg','tire_active_load_n','tire_sensed_load_n',
                'tire_friction','tire_contact_gap_m','tire_contact_gap_limit_m',
                'tire_wheel_spin_rad_s','tire_contact_velocity_without_spin_w','tire_model_slip_m_s','tire_spin_radius_m'}
            if (not required.issubset(trace.files) or str(trace['traction_trace_schema'])!='wheel_tire_wrench_v1'
                    or str(trace['traction_snapshot'])!='last_physics_actuation_before_integration'):
                raise ValueError('Actual tire-actuation geometry is missing; do not infer yaw lever arms from base pose')
            columns=[str(x) for x in trace['columns']];metrics=trace['metrics']
            command=metrics[..., [columns.index('cmd_vx_body'),columns.index('cmd_vy_body'),columns.index('cmd_wz')]]
            steady=steady_samples(trace['time_s'],trace['episode_age_s'],command,
                                  motion['thresholds']['transition_settle_s'],motion['drive_slew_limits_m_s2_and_rad_s2'])
            if not np.all(np.abs(command)<1.e-6):
                raise ValueError('Static holding audit requires an actual zero command')
            normals=trace['wheel_ground_normals_w'];shape=metrics.shape[:2]
            if normals.shape!=shape+(4,3) or not np.isfinite(normals).all():
                raise ValueError('Tire geometry does not align with actual trace cadence')
            planar=np.max(np.abs(normals-normals[...,:1,:]),axis=(-1,-2))<1.e-5
            mask=steady&planar
            if not mask.any():
                raise ValueError('No settled coplanar tire samples; a holding bound is unproven')
            a,b,d,c,sine,lever=force_coefficients(
                trace['tire_roll_direction_w'][mask],trace['wheel_contact_points_w'][mask],
                trace['wheel_com_positions_w'][mask],trace['wheel_axle_direction_w'][mask],
                trace['articulation_com_position_w'][mask],normals[mask][:,0],
                trace['tire_active_load_n'][mask],trace['tire_friction'][mask],torque)
            radius=float(trace['tire_spin_radius_m'])
            _,slip_delta=slip_geometry(trace['tire_roll_direction_w'][mask],trace['wheel_contact_points_w'][mask],
                trace['wheel_com_positions_w'][mask],trace['wheel_axle_direction_w'][mask],
                trace['tire_wheel_spin_rad_s'][mask],trace['tire_contact_velocity_without_spin_w'][mask],
                trace['tire_model_slip_m_s'][mask],radius)
            masses=trace['body_masses_kg']
            if masses.ndim!=2 or len(masses)!=shape[1] or (masses<0).any() or not np.isfinite(masses).all():
                raise ValueError('Actual articulation mass provenance is invalid')
            mass=np.broadcast_to(masses.sum(-1)[None],shape)[mask]
            gravity=mass*9.81*sine
            naive=np.sum(c*np.abs(a),axis=1)
            lateral=balanced_capacity(a,b,np.zeros_like(d),c)
            capacity=balanced_capacity(a,b,d,c)
            sensed=trace['tire_sensed_load_n'][mask];active=trace['tire_active_load_n'][mask]
            gap=trace['tire_contact_gap_m'][mask];gap_limit=float(trace['tire_contact_gap_limit_m'])
            suppressed=(sensed>contact)&(active<=0)
            if not np.array_equal(active,np.where(gap<=gap_limit,sensed,0.)):
                raise ValueError('Recorded active tire loads disagree with the actual gap guard')
            speed=metrics[...,columns.index('linear_speed_error_m_s')][mask]
            deficits=capacity+1.e-6<gravity
            fast=speed>.3
            rows.append(dict(grade_deg=grade,trace=str(path.resolve()),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                settled_samples=int(steady.sum()),coplanar_samples=int(mask.sum()),
                unconstrained_deficit_rate=float((naive+1.e-6<gravity).mean()),
                lateral_balanced_deficit_rate=float((lateral+1.e-6<gravity).mean()),
                lateral_and_yaw_balanced_deficit_rate=float(deficits.mean()),
                fast_deficit_rate=float(deficits[fast].mean()) if fast.any() else None,
                drive_capacity_p05_n=float(np.quantile(capacity,.05)),gravity_uphill_p95_n=float(np.quantile(gravity,.95)),
                sensed_contact_but_tire_inactive_fraction=float(suppressed.mean()),
                suppressed_wheels_always_above_gap_guard=bool((gap[suppressed]>gap_limit).all()),
                axle_torque_lever_p05_p95_m=np.quantile(lever,[.05,.95]).tolist(),
                spin_lever_over_tire_model_radius_p05_p95=np.quantile(lever/radius,[.05,.95]).tolist(),
                model_minus_contact_geometry_slip_abs_p95_m_s=float(np.quantile(np.abs(slip_delta),.95))))
            if 'policy_routing_index' in trace.files:
                routes=trace['policy_routing_index']
                if routes.shape!=shape or not np.issubdtype(routes.dtype,np.integer) or (routes < -1).any():
                    raise ValueError('Policy route trace does not align with actual inference cadence')
                ids,counts=np.unique(routes[mask],return_counts=True)
                rows[-1]['actual_policy_route_counts']={str(index):int(count) for index,count in zip(ids,counts)}
    return dict(status='completed',purpose='Diagnostic only; no evaluation states used for fitting or training selection',
        checkpoint=motion['checkpoint'],checkpoint_sha256=motion['checkpoint_sha256'],cases=rows,
        scope='Necessary quasi-static four-wheel tangential bound at measured active loads and actual axle lever arms. '
              'Zero lateral force and yaw moment are required; pitch/roll equilibrium, contact compliance, transient inertia '
              'and additional external wrenches are omitted. Positive margin does not prove dynamic motion readiness.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grades',type=int,nargs='+',default=[17,20])
    args=parser.parse_args();result=audit(args.evaluation_dir.resolve(),args.grades)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['cases'],indent=2))


if __name__=='__main__': main()
