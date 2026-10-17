"""Suspended current/effort fit with a unilateral compliant angular stop.

These are simulation-equivalent parameters conditional on CAD inertia/gravity.
The final step recording has already been inspected during model development;
results on it are explicitly out-of-fit diagnostics, not a blind acceptance test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from deformable_dynamics_fit import DT, FIT_STAGES, features, prepare
from deformable_fit_data import load_recording
from deformable_urdf_dynamics import CSV_IN_URDF_ORDER, ReducedLegDynamics


def fit_leg(chunks, j, zero):
    chunks = [c for c in chunks if c['stage'] in FIT_STAGES]
    x = np.concatenate([features(c, j) for c in chunks])
    y = np.concatenate([c['needed'][:, j] for c in chunks])
    q = np.concatenate([c['q'][:, j] for c in chunks])
    v = np.concatenate([c['v'][:, j] for c in chunks])
    alpha = np.rad2deg(zero-q)
    # Moving samples away from the top contact identify the free mechanism.
    valid = ((alpha > 0) & (alpha < 72) & (abs(v) > .04) & (abs(x[:, 0]) < 1900))
    free = least_squares(lambda p: x[valid]@p-y[valid], [.02, .01, .1, .5, -.2],
                         bounds=([1e-5, 0, 0, 0, -3], [.2, .5, 10, 10, 3]),
                         loss='soft_l1', f_scale=.1, x_scale='jac', max_nfev=200)
    p = free.x
    # Near-static top contact is especially informative: hundreds of counts
    # occur with negligible acceleration. Coulomb friction there is an interval,
    # rather than zero because tanh(velocity) happens to be zero.
    top = (alpha > 73) & (abs(v) < .025) & (abs(x[:, 0]) < 2100)
    def stop_residual(s):
        stop_angle, stiffness = s
        penetration = np.maximum(zero-np.deg2rad(stop_angle)-q[top], 0.)
        residual = x[top, :3]@p[:3]-p[4]+stiffness*penetration-y[top]
        return np.sign(residual)*np.maximum(abs(residual)-p[3], 0.)
    stop = least_squares(stop_residual, [74., 1000.], bounds=([70., 1.], [77., 10000.]),
                         loss='soft_l1', f_scale=.15, x_scale='jac', max_nfev=200)
    alpha_stop, stiffness = stop.x
    penetration = np.maximum(zero-np.deg2rad(alpha_stop)-q, 0.)
    # A compression-only damper cannot pull the mechanism into the stop.
    contact_velocity = np.maximum(-v, 0.)*(penetration > 0)
    moving_top = (penetration > 0) & (abs(v) > .04)
    damping = least_squares(lambda d: (x@p+stiffness*penetration+d[0]*contact_velocity-y)[moving_top],
                            [1.], bounds=([0.], [50.]), loss='soft_l1', f_scale=.2).x[0]
    residual = x@p+stiffness*penetration+damping*contact_velocity-y
    return {'parameters':p.tolist(), 'torque_per_count':float(p[0]), 'extra_inertia':float(p[1]),
            'viscous_friction':float(p[2]), 'coulomb_friction':float(p[3]), 'constant_effort_bias':float(p[4]),
            'upper_stop_physical_angle_deg':float(alpha_stop), 'stop_stiffness_nm_rad':float(stiffness),
            'stop_compression_damping_nm_s_rad':float(damping), 'free_rows':int(valid.sum()),
            'static_stop_rows':int(top.sum()), 'free_effort_rmse_nm':float(np.sqrt(np.mean(residual[valid]**2))),
            'free_effort_abs_p95_nm':float(np.quantile(abs(residual[valid]), .95))}


def integrate_current(chunk, j, model, cad_inertia, zero, *, horizon=.5, substeps=5):
    """Replay measured current without feeding future measured states back."""
    gain, extra, viscous, dry, bias = model['parameters']
    stop_q = zero-np.deg2rad(model['upper_stop_physical_angle_deg'])
    stiffness = model['stop_stiffness_nm_rad']
    damping = model['stop_compression_damping_nm_s_rad']
    residual_cos = model.get('assembly_residual_cos_nm', 0.)
    residual_sin = model.get('assembly_residual_sin_nm', 0.)
    count = int(round(horizon/DT))
    starts = np.arange(0, len(chunk['q'])-count, count)
    indices = starts.copy()
    q = chunk['q'][starts, j].copy()
    v = chunk['raw_velocity'][starts, j].copy()
    angle_error, velocity_error, prediction = [], [], []
    dt = DT/substeps
    for k in range(count):
        indices = starts+k
        for _ in range(substeps):
            gravity = chunk['g_cos'][indices,j]*np.cos(q)+chunk['g_sin'][indices,j]*np.sin(q)
            depth = np.maximum(stop_q-q, 0.)
            contact = stiffness*depth + damping*np.maximum(-v, 0.)*(depth > 0)
            drive = (gain*chunk['current'][indices,j]-gravity-viscous*v-bias+contact
                     - residual_cos*np.cos(q)-residual_sin*np.sin(q))
            stuck = (abs(v) < .01) & (abs(drive) <= dry)
            friction = dry*np.sign(np.where(abs(v) >= .01, v, drive))
            v_next = v+(drive-friction)/(cad_inertia+extra)*dt
            v_next = np.where(stuck | ((v*v_next < 0) & (abs(drive) <= dry)), 0., v_next)
            q += .5*(v+v_next)*dt
            v = v_next
        prediction.append(q.copy())
        angle_error.append(q-chunk['q'][indices+1,j])
        velocity_error.append(v-chunk['raw_velocity'][indices+1,j])
    errors = np.rad2deg(np.array(angle_error).T)
    return {'horizon_s':horizon, 'windows':len(starts),
            'angle_rmse_deg':float(np.sqrt(np.mean(errors**2))),
            'angle_abs_p95_deg':float(np.quantile(abs(errors), .95)),
            'final_angle_rmse_deg':float(np.sqrt(np.mean(errors[:,-1]**2))),
            'velocity_rmse_rad_s':float(np.sqrt(np.mean(np.square(velocity_error))))}, np.array(prediction).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/home/noir/Documents/workspace'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    training, diagnostic = [], []
    physics = ReducedLegDynamics()
    for path in sorted(args.workspace.glob('deformable*/*.csv')):
        record = load_recording(path, args.output.parent/'cache')
        if not record['status'].get('completed') or record['metadata']['condition'] != 'suspended':
            continue
        chunks, inertia, zero = prepare(record, physics)
        (diagnostic if '23-03-11' in str(path) else training).extend(chunks)
    models = {name:fit_leg(training,j,zero) for j,name in enumerate(CSV_IN_URDF_ORDER)}
    results = {}
    for chunk in diagnostic:
        if chunk['stage'] in (*FIT_STAGES,10):
            results[str(chunk['stage'])] = {name:integrate_current(chunk,j,models[name],inertia[j],zero)[0]
                                          for j,name in enumerate(CSV_IN_URDF_ORDER)}
    result = {'scope':'simulation_equivalent_conditional_on_CAD_mass_and_inertia',
              'absolute_shaft_torque_calibrated':False, 'model_version':'dry_friction_stop_v2',
              'diagnostic_is_blind_acceptance':False, 'joint_order':CSV_IN_URDF_ORDER,
              'urdf_zero_physical_angle_deg':float(np.rad2deg(zero)), 'cad_inertia_kg_m2':inertia.tolist(),
              'training_files':sorted({c['file'] for c in training}),
              'out_of_fit_diagnostic_files':sorted({c['file'] for c in diagnostic}),
              'models':models, 'measured_current_rollout_diagnostic':results}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'models':models,'diagnostic':results},indent=2),flush=True)


if __name__ == '__main__':
    main()
