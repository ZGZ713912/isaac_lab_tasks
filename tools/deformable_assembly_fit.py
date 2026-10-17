"""Joint fit of suspended dynamics and common-motion grounded load balance.

Grounded data constrain the current scale using the CAD vehicle mass. They do
not identify individual contact forces. The conservative residual accounts for
repeatable assembly/model mismatch without calling it measured wear or torque.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from deformable_contact_fit import integrate_current
from deformable_dynamics_fit import FIT_STAGES, prepare
from deformable_fit_data import load_recording
from deformable_urdf_dynamics import CSV_IN_URDF_ORDER, ReducedLegDynamics, URDF


def matrix(chunk):
    """Per-leg linear effort regressors, with a common current/effort gain."""
    n = len(chunk['q'])
    x = np.zeros((n, 4, 25))
    x[:, :, 0] = chunk['filtered_current']
    for j in range(4):
        q, v = chunk['q'][:,j], chunk['v'][:,j]
        x[:,j,1+6*j:7+6*j] = np.stack((-chunk['a'][:,j], -v, -np.tanh(v/.02),
                                       -np.ones(n), -np.cos(q), -np.sin(q)), axis=-1)
    return x


def grounded_target(chunk, physics):
    """Common-q vertical virtual work; independent of four-wheel load split.

    Only near-synchronous, moving, low-frequency samples are used. This check
    assumes a horizontal supporting plane, no other supports and wheel contact.
    """
    q = chunk['q'].mean(axis=1)
    mass = sum(x[0] for x in physics.links.values())
    zprime = .029108*np.cos(q)+.13694*np.sin(q)
    return chunk['needed'].sum(axis=1)-mass*9.81*zprime


def ground_mask(chunk, zero):
    alpha = np.rad2deg(zero-chunk['q'].mean(axis=1))
    return ((alpha > 18) & (alpha < 72) & (abs(chunk['v']).min(axis=1) > .04)
            & (chunk['q'].std(axis=1) < .02))


def fit(training, grounded, physics, zero):
    xs, ys = [], []
    for c in training:
        if c['stage'] not in FIT_STAGES:
            continue
        alpha = np.rad2deg(zero-c['q'])
        valid = ((alpha > 0) & (alpha < 72) & (abs(c['v']) > .04)
                 & (abs(c['current']) < 1900))
        valid[:9] = valid[-9:] = False
        xs.append(matrix(c)[valid]); ys.append(c['needed'][valid])
    for c in grounded:
        if c['stage'] != 3:
            continue
        # The second half and the complete fast sweep are reserved for checking.
        valid = ground_mask(c,zero) & (c['time'] < 60.)
        xs.append(.5*matrix(c)[valid].sum(axis=1))
        ys.append(.5*grounded_target(c,physics)[valid])
    x, y = np.concatenate(xs), np.concatenate(ys)
    lower = [.001]+[0,0,0,-1,-3,-3]*4
    upper = [.1]+[.5,5,5,1,3,3]*4
    weights = np.ones(len(y))
    regularization = np.diag([0]+[.1,.1,.1,5,5,5]*4)
    for _ in range(12):
        result = lsq_linear(np.concatenate((x*weights[:,None],regularization)),
                            np.concatenate((y*weights,np.zeros(25))),
                            bounds=(lower,upper), lsq_solver='exact', max_iter=100)
        residual = x@result.x-y
        weights = np.sqrt(1/np.maximum(1,abs(residual)/.12))
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def fit_stops(training, j, zero, parameters):
    cs = [c for c in training if c['stage'] in FIT_STAGES]
    q = np.concatenate([c['q'][:,j] for c in cs])
    v = np.concatenate([c['v'][:,j] for c in cs])
    current = np.concatenate([c['current'][:,j] for c in cs])
    residual = np.concatenate([matrix(c)[:,j]@parameters-c['needed'][:,j] for c in cs])
    dry = parameters[3+6*j]
    top = (np.rad2deg(zero-q)>73)&(abs(v)<.025)&(abs(current)<2100)
    def objective(s):
        depth = np.maximum(zero-np.deg2rad(s[0])-q[top],0.)
        # Undo the arbitrary tanh value at standstill: stiction is an interval.
        r = residual[top]+dry*np.tanh(v[top]/.02)+s[1]*depth
        return np.sign(r)*np.maximum(abs(r)-dry,0.)
    stop = least_squares(objective,[74.,1000.],bounds=([70,1],[77,10000]),
                         loss='soft_l1',f_scale=.15,x_scale='jac')
    depth = np.maximum(zero-np.deg2rad(stop.x[0])-q,0.)
    contact_velocity = np.maximum(-v,0.)*(depth>0)
    moving = (depth>0)&(abs(v)>.04)
    damping = least_squares(lambda d:(residual+stop.x[1]*depth+d[0]*contact_velocity)[moving],
                            [1.],bounds=([0],[50]),loss='soft_l1',f_scale=.2)
    extra,b,dry,bias,cos,sin = parameters[1+6*j:7+6*j]
    return {'parameters':[float(x) for x in (parameters[0],extra,b,dry,bias)],
            'torque_per_count':float(parameters[0]), 'extra_inertia':float(extra),
            'viscous_friction':float(b), 'coulomb_friction':float(dry),
            'constant_effort_bias':float(bias), 'assembly_residual_cos_nm':float(cos),
            'assembly_residual_sin_nm':float(sin), 'upper_stop_physical_angle_deg':float(stop.x[0]),
            'stop_stiffness_nm_rad':float(stop.x[1]),
            'stop_compression_damping_nm_s_rad':float(damping.x[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path('/home/noir/Documents/workspace'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    training, grounded, diagnostic, sources = [], [], [], []
    physics = ReducedLegDynamics()
    for path in sorted(args.workspace.glob('deformable*/*.csv')):
        r = load_recording(path,args.output.parent/'cache')
        sources.append({'path':r['path'],'sha256':r['sha256'],'completed':r['status'].get('completed')})
        if not r['status'].get('completed'):
            continue
        chunks,inertia,zero = prepare(r,physics)
        destination = grounded if r['metadata']['condition']=='grounded' else (
            diagnostic if '23-03-11' in str(path) else training)
        destination.extend(chunks)
    p = fit(training,grounded,physics,zero)
    models = {name:fit_stops(training,j,zero,p) for j,name in enumerate(CSV_IN_URDF_ORDER)}
    replay = {}
    for c in diagnostic:
        if c['stage'] in (*FIT_STAGES,10):
            replay[str(c['stage'])] = {name:integrate_current(c,j,models[name],inertia[j],zero)[0]
                                      for j,name in enumerate(CSV_IN_URDF_ORDER)}
    ground_checks = {}
    for c in grounded:
        if c['stage'] not in (3,7):
            continue
        valid = ground_mask(c,zero)&((c['time']>=60.) if c['stage']==3 else True)
        predicted = matrix(c).sum(axis=1)@p
        expected = grounded_target(c,physics)
        residual = predicted[valid]-expected[valid]
        ground_checks[str(c['stage'])] = {'rows':int(valid.sum()),'assumption':'common_q_virtual_work_quasistatic',
            'sum_effort_rmse_nm':float(np.sqrt(np.mean(residual**2))),
            'sum_effort_abs_p95_nm':float(np.quantile(abs(residual),.95)),
            'reference_effort_rms_nm':float(np.sqrt(np.mean(expected[valid]**2))),
            'residual_mean_nm':float(np.mean(residual))}
    result = {'model_version':'assembly_current_v3_candidate','absolute_shaft_torque_calibrated':False,
        'scope':'simulation_equivalent_conditional_on_CAD_mass_and_common_ground_contact_assumptions',
        'diagnostic_is_blind_acceptance':False,'joint_order':CSV_IN_URDF_ORDER,
        'urdf_sha256':hashlib.sha256(URDF.read_bytes()).hexdigest(),'source_files':sources,
        'urdf_zero_physical_angle_deg':float(np.rad2deg(zero)),'cad_inertia_kg_m2':inertia.tolist(),
        'cad_total_mass_kg':sum(x[0] for x in physics.links.values()),'linear_parameters':p.tolist(),
        'training_files':sorted({c['file'] for c in training}),
        'grounded_fit_interval':'stage3_first60seconds',
        'out_of_fit_diagnostic_files':sorted({c['file'] for c in diagnostic}),
        'models':models,'measured_current_rollout_diagnostic':replay,'grounded_out_of_fit_checks':ground_checks}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'models':models,'diagnostic':replay,'ground':ground_checks},indent=2),flush=True)


if __name__=='__main__':
    main()
