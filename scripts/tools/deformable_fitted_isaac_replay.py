"""Replay independent suspended CSV windows in the actual Isaac articulation.

Each cloned robot is initialized once at a window boundary, then driven only
by measured current for 0.5 s. Future measured joint states never drive it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'source/agent_world')]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fit',type=Path,required=True)
parser.add_argument('--csv',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--windows-per-stage',type=int,default=24)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import numpy as np
    import torch
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_from_euler_xyz
    from agent_world.assets.deformable_V2 import DeformableInfantryCFG
    from deformable_fit_data import load_recording
    from deformable_dynamics_fit import prepare
    from deformable_urdf_dynamics import ReducedLegDynamics, CSV_IN_URDF_ORDER

    spec = importlib.util.spec_from_file_location('fitted_actuator',ROOT/'source/agent_tasks/agent_tasks/direct/deformable_suspension/real2sim.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    fit_bytes = args.fit.read_bytes()
    fit = json.loads(fit_bytes)
    mechanism = module.FittedLegMechanism(fit,args.device)
    data = load_recording(args.csv,args.output.parent/'cache')
    if not data['status'].get('completed') or data['metadata']['condition']!='suspended':
        raise ValueError('fixed-root replay requires a complete suspended recording')
    physics = ReducedLegDynamics()
    chunks,_,zero = prepare(data,physics)
    states,velocities,currents,targets,orientations,stages = [],[],[],[],[],[]
    for chunk in chunks:
        if chunk['stage'] not in (3,7,8,10):
            continue
        starts = np.arange(0,len(chunk['q'])-100,100)
        if len(starts)>args.windows_per_stage:
            starts = starts[np.linspace(0,len(starts)-1,args.windows_per_stage,dtype=int)]
        for begin in starts:
            states.append(chunk['q'][begin]); velocities.append(chunk['raw_velocity'][begin])
            currents.append(chunk['current'][begin:begin+100])
            targets.append(chunk['q'][begin+1:begin+101])
            # Recover base gravity from the retained basis and recorded IMU is
            # unnecessary here: interpolate the original IMU at stage-relative t.
            mask = data['arrays']['stage']==chunk['stage']
            time = data['arrays']['feedback_time_s'][mask].mean(axis=1)
            imu = data['arrays']['imu'][mask]
            orientations.append([np.interp(time[0]+chunk['time'][begin],time,imu[:,i]) for i in range(2)])
            stages.append(chunk['stage'])
    count = len(states)
    sim = SimulationContext(SimulationCfg(dt=.001,device=args.device))
    robot_cfg = DeformableInfantryCFG.copy()
    robot_cfg.prim_path = '{ENV_REGEX_NS}/Robot'
    robot_cfg.spawn.articulation_props.fix_root_link = True
    # USD rigid-body angular speed uses degrees/s; use the training setting.
    robot_cfg.spawn.rigid_props.max_angular_velocity = 7200.
    robot_cfg.init_state.pos = (0.,0.,2.)
    for actuator in robot_cfg.actuators.values():
        actuator.effort_limit = None
        actuator.effort_limit_sim = 300.
        actuator.velocity_limit = None
        actuator.velocity_limit_sim = 100.
        actuator.damping = 0.
        actuator.friction = 0.
    @configclass
    class ReplaySceneCfg(InteractiveSceneCfg):
        robot = robot_cfg
    scene = InteractiveScene(ReplaySceneCfg(num_envs=count,env_spacing=2.,replicate_physics=True))
    sim.reset()
    sim._disable_app_control_on_stop_handle = True
    scene.update(.001)
    robot = scene['robot']
    projection = torch.zeros((len(robot.joint_names),4),device=args.device)
    legs=[]
    for j in range(4):
        legs.append(robot.joint_names.index(f'joint_leg_{j+1}'))
        for prefix,multiplier in [('joint_leg',1),('joint_wheel_set',1),('joint_upper_leg',-1)]:
            projection[robot.joint_names.index(f'{prefix}_{j+1}'),j]=multiplier
    q=torch.tensor(np.array(states),device=args.device,dtype=torch.float32)
    v=torch.tensor(np.array(velocities),device=args.device,dtype=torch.float32)
    current=torch.tensor(np.array(currents),device=args.device,dtype=torch.float32)
    reference=np.array(targets)
    angles=torch.tensor(np.array(orientations),device=args.device,dtype=torch.float32)
    root=robot.data.default_root_state.clone(); root[:,:3]+=scene.env_origins
    root[:,3:7]=quat_from_euler_xyz(angles[:,0],angles[:,1],torch.zeros(count,device=args.device))
    robot.write_root_pose_to_sim(root[:,:7])
    robot.write_joint_state_to_sim(q@projection.T,v@projection.T)
    armature=torch.zeros_like(robot.data.joint_pos); armature[:,legs]=mechanism.armature
    robot.write_joint_armature_to_sim(armature)
    # Stress sweeps in the real recording exceed the training target range. A
    # software 0..1.36 stop would conceal the learned mechanical stop response.
    limits=robot.data.joint_pos_limits.clone(); limits[...,0]=-3.; limits[...,1]=3.
    robot.write_joint_position_limit_to_sim(limits)
    scene.reset(); sim.forward(); scene.update(.001)
    predictions=[]
    with torch.inference_mode():
        for step in range(500):
            q=robot.data.joint_pos[:,legs]; v=robot.data.joint_vel[:,legs]
            torque=mechanism.effort(current[:,step//5],q,v)
            robot.set_joint_effort_target(torque,joint_ids=legs)
            scene.write_data_to_sim(); sim.step(render=False); scene.update(.001)
            if step%5==4:
                predictions.append(robot.data.joint_pos[:,legs].cpu().numpy().copy())
    predicted=np.array(predictions).transpose(1,0,2)
    error=np.rad2deg(predicted-reference)
    metrics={}
    for stage in sorted(set(stages)):
        selected=np.array(stages)==stage
        metrics[str(stage)]={name:{'angle_rmse_deg':float(np.sqrt(np.mean(error[selected,:,j]**2))),
                                  'angle_abs_p95_deg':float(np.quantile(abs(error[selected,:,j]),.95)),
                                  'windows':int(selected.sum())}
                             for j,name in enumerate(CSV_IN_URDF_ORDER)}
    artifact=args.output.with_suffix('.npz')
    np.savez_compressed(artifact,predicted_q=predicted,reference_q=reference,stages=stages,
                        current=np.array(currents),initial_q=np.array(states),physical_zero=zero)
    report={'fit':str(args.fit.resolve()),'fit_sha256':hashlib.sha256(fit_bytes).hexdigest(),
            'recording':data['path'],'csv_sha256':data['sha256'],
            'horizon_s':.5,'future_state_feedback':False,'fixed_base':robot.is_fixed_base,
            'windows':count,'all_finite':bool(np.isfinite(predicted).all()),'metrics':metrics,
            'traces':str(artifact.resolve()),'root_orientation':'initial measured IMU per window, held fixed'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('FITTED_REPLAY',json.dumps(report),flush=True)
finally:
    app.close()
