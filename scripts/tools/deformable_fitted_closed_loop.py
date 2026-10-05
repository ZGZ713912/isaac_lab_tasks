"""Reference-only closed-loop Isaac step replay, including ADRC and CAN timing."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT)]+[str(ROOT/'source'/p) for p in ('agent_tasks','agent_world','agent_rl')]
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--csv',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--model',type=Path,help='Optional saved actuator model; defaults to the V3 task model.')
parser.add_argument('--duration',type=float,default=30.)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
try:
    import numpy as np
    import torch
    from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
    from isaaclab.sim import SimulationContext,SimulationCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_from_euler_xyz
    from agent_tasks.direct.deformable_suspension.dynamic_cfg import DeformableFittedPrecisionEnvCfg
    from agent_tasks.direct.deformable_suspension.adrc import LegADRC
    from agent_tasks.direct.deformable_suspension.real2sim import Real2SimActuator
    from deformable_fit_data import load_recording,JOINTS
    from deformable_urdf_dynamics import CSV_IN_URDF_ORDER

    cfg=DeformableFittedPrecisionEnvCfg()
    if args.model:
        cfg.real2sim_model_path=str(args.model.resolve())
    model_bytes=Path(cfg.real2sim_model_path).read_bytes()
    model_snapshot=args.output.with_suffix('.model.json')
    model_snapshot.parent.mkdir(parents=True,exist_ok=True)
    model_snapshot.write_bytes(model_bytes)
    cfg.real2sim_model_path=str(model_snapshot.resolve())
    cfg.real2sim_randomize=False
    cfg.real2sim_torque_scale_range=(1.,1.)
    cfg.real2sim_command_delay_steps_range=(0,0)
    cfg.real2sim_feedback_delay_steps_range=(1,1)
    for field in ('real2sim_current_noise_std_range','real2sim_current_sensor_noise_std_range',
                  'real2sim_angle_noise_std_range','real2sim_velocity_noise_std_range'):
        setattr(cfg,field,(0.,0.))
    recorded=load_recording(args.csv,args.output.parent/'cache')
    a=recorded['arrays'];ids=np.flatnonzero(a['stage']==10)
    if not recorded['status'].get('completed') or len(ids)<100:
        raise ValueError('requires completed step recording')
    order=[JOINTS.index(name) for name in CSV_IN_URDF_ORDER]
    time=a['host_time_s'][ids];time=time-time[0]
    # Stage 10 is a piecewise-constant target; preserve its jump boundaries.
    sim_time=np.arange(0,min(args.duration,time[-1]),.001)
    source_ids=ids[np.searchsorted(time,sim_time,side='right')-1]
    target=torch.tensor(cfg.leg_physical_angle_zero-a['target_physical_angle'][source_ids][:,order],
                        dtype=torch.float32,device=args.device)
    target_velocity=torch.tensor(a['target_physical_velocity'][source_ids][:,order],
                                 dtype=torch.float32,device=args.device)
    initial_q=cfg.leg_physical_angle_zero-a['physical_angle'][ids[0],order]
    initial_v=-a['physical_velocity'][ids[0],order]
    taus=[.0,.002,.0046,.008]
    n=len(taus)
    sim=SimulationContext(SimulationCfg(dt=.001,device=args.device))
    robot_cfg=cfg.robot_cfg.copy();robot_cfg.prim_path='{ENV_REGEX_NS}/Robot'
    robot_cfg.spawn.articulation_props.fix_root_link=True
    robot_cfg.init_state.pos=(0.,0.,2.)
    for actuator in robot_cfg.actuators.values():
        actuator.effort_limit=None;actuator.effort_limit_sim=300.
        actuator.velocity_limit=None;actuator.velocity_limit_sim=100.
        actuator.damping=0.;actuator.friction=0.
    @configclass
    class BenchSceneCfg(InteractiveSceneCfg):
        robot=robot_cfg
    scene=InteractiveScene(BenchSceneCfg(num_envs=n,env_spacing=2.,replicate_physics=True))
    sim.reset();sim._disable_app_control_on_stop_handle=True;scene.update(.001)
    robot=scene['robot'];projection=torch.zeros((len(robot.joint_names),4),device=args.device);legs=[]
    for j in range(4):
        legs.append(robot.joint_names.index(f'joint_leg_{j+1}'))
        for prefix,multiplier in [('joint_leg',1),('joint_wheel_set',1),('joint_upper_leg',-1)]:
            projection[robot.joint_names.index(f'{prefix}_{j+1}'),j]=multiplier
    q=torch.tensor(initial_q,dtype=torch.float32,device=args.device).expand(n,-1).clone()
    v=torch.tensor(initial_v,dtype=torch.float32,device=args.device).expand(n,-1).clone()
    actuator=Real2SimActuator((n,4),args.device,cfg)
    actuator.lag_alpha=torch.tensor([1. if tau==0 else 1.-np.exp(-.001/tau) for tau in taus],device=args.device)[:,None]
    controller=LegADRC((n,4),args.device,cfg)
    controller.reset(torch.arange(n,device=args.device),q,target[0].expand_as(q))
    actuator.reset(torch.arange(n,device=args.device),q,v)
    root=robot.data.default_root_state.clone();root[:,:3]+=scene.env_origins
    imu=torch.tensor(a['imu'][ids[0]],device=args.device,dtype=torch.float32)
    root[:,3:7]=quat_from_euler_xyz(imu[0].expand(n),imu[1].expand(n),torch.zeros(n,device=args.device))
    robot.write_root_pose_to_sim(root[:,:7]);robot.write_joint_state_to_sim(q@projection.T,v@projection.T)
    armature=torch.zeros_like(robot.data.joint_pos);armature[:,legs]=actuator.mechanism.armature
    robot.write_joint_armature_to_sim(armature)
    limits=robot.data.joint_pos_limits.clone();limits[...,0]=-3.;limits[...,1]=3.
    robot.write_joint_position_limit_to_sim(limits)
    scene.reset();sim.forward();scene.update(.001)
    states=[];currents=[];timestamps=[]
    with torch.inference_mode():
        for k,t in enumerate(sim_time):
            q=robot.data.joint_pos[:,legs];v=robot.data.joint_vel[:,legs]
            measured_q,_,_=actuator.sensor_measurement(q,v)
            raw=controller.update(measured_q,target[k].expand_as(q),
                applied_current_raw=actuator.command_current_raw,
                target_physical_velocity=target_velocity[k].expand_as(q))
            effort=actuator.apply(raw,q,v)
            robot.set_joint_effort_target(effort,joint_ids=legs)
            scene.write_data_to_sim();sim.step(render=False);scene.update(.001)
            if k%5==4:
                states.append(robot.data.joint_pos[:,legs].cpu().numpy().copy())
                currents.append(actuator._motor_current.cpu().numpy().copy());timestamps.append(t+.001)
            if k%5000==0:print('CLOSED_LOOP_PROGRESS',round(t,3),flush=True)
    times=np.array(timestamps);predicted=cfg.leg_physical_angle_zero-np.array(states)
    reference=np.stack([np.interp(times,time,a['physical_angle'][ids,j]) for j in order],axis=-1)
    ref_current=np.stack([np.interp(times,time,a['current_raw'][ids,j]) for j in order],axis=-1)
    # Initial ESO/lag states are not logged in these files; exclude the first
    # 5 s from metrics instead of fabricating matching controller hidden states.
    evaluate=times>=5.
    metrics=[]
    for i,tau in enumerate(taus):
        error=np.rad2deg(predicted[evaluate,i]-reference[evaluate])
        current_error=np.array(currents)[evaluate,i]-ref_current[evaluate]
        metrics.append({'tau_s':tau,'angle_rmse_deg':np.sqrt(np.mean(error**2,axis=0)).tolist(),
            'angle_abs_p95_deg':np.quantile(abs(error),.95,axis=0).tolist(),
            'current_rmse_counts':np.sqrt(np.mean(current_error**2,axis=0)).tolist()})
    traces=args.output.with_suffix('.npz')
    np.savez_compressed(traces,time=times,predicted_physical_angle=predicted,reference_physical_angle=reference,
                        predicted_current=np.array(currents),reference_current=ref_current,taus=taus)
    result={'input':'target physical angle and finite target velocity only','measured_state_feedback':False,
            'initial_state':'measured joint q/qd once; ESO reset; first5s excluded',
            'csv_sha256':recorded['sha256'],'joint_order':CSV_IN_URDF_ORDER,
            'model_sha256':hashlib.sha256(model_bytes).hexdigest(),
            'model_snapshot':str(model_snapshot.resolve()),
            'command_period_steps':2,'feedback_period_steps':2,'feedback_delivery_phase_steps':1,
            'duration_s':float(times[-1]),'all_finite':bool(np.isfinite(predicted).all()),
            'metrics':metrics,'traces':str(traces.resolve())}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('CLOSED_LOOP_RESULT',json.dumps(result),flush=True)
finally:
    app.close()
