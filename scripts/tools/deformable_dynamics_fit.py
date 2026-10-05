"""Fit simulation-equivalent current/effort parameters; keep a whole run held out.

The CAD/Isaac mass model fixes the otherwise ambiguous torque/inertia scale.
The output is conditional on that model, never a measured shaft-torque claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from deformable_fit_data import JOINTS, load_recording
from deformable_urdf_dynamics import ReducedLegDynamics, URDF, CSV_IN_URDF_ORDER

FIT_STAGES = (3, 7, 8)
DT = .005


def gravity_base(imu):
    roll, pitch = imu.T
    return 9.81 * np.stack((np.sin(pitch), -np.sin(roll)*np.cos(pitch), -np.cos(roll)*np.cos(pitch)), axis=-1)


def prepare(recording, model, window=9):
    data = recording['arrays']
    order = [JOINTS.index(name) for name in CSV_IN_URDF_ORDER]
    # The URDF rod vector is (0.029108, 0, -0.13694) at q=0. A command
    # limit of 75 degrees is not its geometrical zero-angle calibration.
    alpha_zero = np.arctan2(.13694, .029108)
    mass = np.diag(model.quantities(np.zeros((1, 4)))[0][0])
    cos_basis = model.quantities(np.zeros((3, 4)), np.eye(3))[1]
    sin_basis = model.quantities(np.full((3, 4), np.pi/2), np.eye(3))[1]
    chunks = []
    for stage in np.unique(data['stage']):
        rows = np.flatnonzero(data['stage'] == stage)
        for ids in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1):
            if len(ids) < 2*window:
                continue
            time = data['feedback_time_s'][ids].mean(axis=1)
            if np.any(np.diff(time) <= 0):
                raise ValueError('Nonmonotone distinct feedback time; split or deduplicate before fitting')
            grid = np.arange(time[0], time[-1], DT)
            def interpolate(value):
                return np.stack([np.interp(grid, time, col) for col in value[ids].T], axis=-1)
            q = alpha_zero - interpolate(data['physical_angle'][:, order])
            velocity = -interpolate(data['physical_velocity'][:, order])
            current = interpolate(data['current_raw'][:, order])
            target = alpha_zero - interpolate(data['target_physical_angle'][:, order])
            imu = interpolate(data['imu'])
            v = savgol_filter(velocity, window, 3, axis=0)
            acceleration = savgol_filter(velocity, window, 3, deriv=1, delta=DT, axis=0)
            filtered_current = savgol_filter(current, window, 3, axis=0)
            grav = gravity_base(imu)
            g_cos, g_sin = grav @ cos_basis, grav @ sin_basis
            gravity = g_cos*np.cos(q) + g_sin*np.sin(q)
            needed_effort = mass * acceleration + gravity
            valid = ((np.abs(current) < .95*2048) & (np.abs(v) > .025)
                     & (q > alpha_zero-np.deg2rad(76)) & (q < alpha_zero-np.deg2rad(15)))
            valid[:window] = valid[-window:] = False
            # Filter windows cannot contain command saturation or angle excursions.
            from scipy.ndimage import minimum_filter1d
            valid = minimum_filter1d(valid.astype(np.uint8), window, axis=0, mode='constant').astype(bool)
            chunks.append(dict(file=recording['path'], stage=int(stage), time=grid-grid[0],
                               q=q, v=v, raw_velocity=velocity, current=current, filtered_current=filtered_current,
                               a=acceleration, g_cos=g_cos, g_sin=g_sin, needed=needed_effort,
                               target=target, valid=valid))
    return chunks, mass, float(alpha_zero)


def features(chunk, j):
    return np.stack((chunk['filtered_current'][:, j], -chunk['a'][:, j], -chunk['v'][:, j],
                     -np.tanh(chunk['v'][:, j]/.02), -np.ones(len(chunk['q']))), axis=-1)


def fit_leg(chunks, j):
    x, y = [], []
    for c in chunks:
        if c['stage'] not in FIT_STAGES:
            continue
        keep = c['valid'][:, j]
        x.append(features(c, j)[keep]); y.append(c['needed'][keep, j])
    x, y = np.concatenate(x), np.concatenate(y)
    result = least_squares(lambda p: x@p-y, [.025, .01, .05, .7, 0.],
                           bounds=([.00001, 0., 0., 0., -10.], [.2, 1., 10., 10., 10.]),
                           loss='soft_l1', f_scale=.1, x_scale='jac', max_nfev=200)
    p = result.x
    residual = x@p-y
    return {'torque_per_count':float(p[0]), 'extra_inertia':float(p[1]),
            'viscous_friction':float(p[2]), 'coulomb_friction':float(p[3]), 'constant_effort_bias':float(p[4]),
            'parameters':p.tolist(), 'rows':len(y), 'success':bool(result.success),
            'fit_effort_rmse_nm':float(np.sqrt(np.mean(residual**2))),
            'fit_effort_abs_p95_nm':float(np.quantile(np.abs(residual),.95))}


def rollout(chunk, j, params, cad_inertia, *, horizon=.5):
    """Open-loop current playback, resetting only at non-overlapping windows.

    The initial stiction force is selected from the initial measured current,
    gravity and velocity. No future angle/velocity/target enters integration.
    """
    gain, extra, viscous, dry, bias = params
    samples = int(round(horizon/DT))
    q_errors, v_errors = [], []
    final_errors = []
    for begin in range(0, len(chunk['q'])-samples, samples):
        q, v = chunk['q'][begin, j], chunk['raw_velocity'][begin, j]
        for k in range(begin, begin+samples):
            # Four substeps improve stiction handling without changing input data.
            for _ in range(4):
                gravity = chunk['g_cos'][k,j]*np.cos(q)+chunk['g_sin'][k,j]*np.sin(q)
                drive = gain*chunk['current'][k,j] - gravity - viscous*v - bias
                if abs(v) < .01 and abs(drive) <= dry:
                    v = 0.
                else:
                    friction = dry*np.sign(v if abs(v) >= .01 else drive)
                    acceleration = (drive-friction)/(cad_inertia+extra)
                    v_next = v + acceleration*DT/4
                    if v*v_next < 0 and abs(drive) <= dry:
                        v_next = 0.
                    q += .5*(v+v_next)*DT/4
                    v = v_next
            q_errors.append(q-chunk['q'][k+1,j])
            v_errors.append(v-chunk['raw_velocity'][k+1,j])
        final_errors.append(q_errors[-1])
    return {'horizon_s':horizon, 'windows':len(final_errors),
            'angle_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(np.square(q_errors))))),
            'angle_abs_p95_deg':float(np.rad2deg(np.quantile(np.abs(q_errors), .95))),
            'final_angle_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(np.square(final_errors))))),
            'velocity_rmse_rad_s':float(np.sqrt(np.mean(np.square(v_errors))))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/home/noir/Documents/workspace'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, default=9)
    args = parser.parse_args()
    if args.window < 5 or args.window % 2 != 1:
        parser.error('window must be odd and at least 5')
    paths = sorted(args.workspace.glob('deformable*/*.csv'))
    records = [load_recording(p, args.output.parent/'cache') for p in paths]
    training, heldout, grounded = [], [], []
    physics = ReducedLegDynamics()
    for record in records:
        if not record['status'].get('completed'):
            continue
        chunks, inertia, alpha_zero = prepare(record, physics, args.window)
        if record['metadata']['condition'] == 'grounded':
            grounded.extend(chunks)
        elif '23-03-11' in record['path']:
            heldout.extend(chunks)
        else:
            training.extend(chunks)
    models = {name:fit_leg(training,j) for j,name in enumerate(CSV_IN_URDF_ORDER)}
    validation = {}
    for chunk in heldout:
        if chunk['stage'] not in (*FIT_STAGES, 10):
            continue
        validation[str(chunk['stage'])] = {name:rollout(chunk, j, models[name]['parameters'], inertia[j])
                                          for j,name in enumerate(CSV_IN_URDF_ORDER)}
    result = {'scope':'simulation_equivalent_torque_conditional_on_CAD_mass_and_inertia',
              'absolute_shaft_torque_calibrated':False, 'urdf_sha256':hashlib.sha256(URDF.read_bytes()).hexdigest(),
              'cad_inertia_kg_m2':inertia.tolist(), 'urdf_zero_physical_angle_deg':float(np.rad2deg(alpha_zero)),
              'joint_order':CSV_IN_URDF_ORDER, 'fit_stages':FIT_STAGES, 'derivative_window':args.window,
              'training_files':sorted({c['file'] for c in training}), 'heldout_files':sorted({c['file'] for c in heldout}),
              'grounded_validation_status':'not_yet_evaluated', 'models':models, 'heldout_measured_current_rollout':validation}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'models':models,'heldout':validation},indent=2),flush=True)


if __name__ == '__main__':
    main()
