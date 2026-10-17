"""Train and physically test an explicit default-split MLP specialist.

The source must have complete paired Isaac Sim acceptance. Conversion preserves
its existing action function but is not labeled as independently accepted.
Only the trained candidate's own five-grade reports can promote its bytes.
Run serially after the current simulation owner has ended; dry-run is default.
"""
import argparse
import hashlib
import math
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

from deformable_mlp_default_split import build
from deformable_mlp_guarded_train import (
    ROOT, SPLIT_CLASS, actor_update_info, benchmark_command, checkpoint_info,
    latest_checkpoint, policy_experiment, qualify, run_child, timestamp,
    training_command, write_json,
)
from deformable_mlp_motion_audit import audit_motion
from deformable_mlp_train import SOURCES
from deformable_tire_wrench_audit import audit as audit_tires


def run_trial(args):
    args.output_dir.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',started_at=timestamp(),jobs=[],
               last_accepted_checkpoint=str(args.checkpoint),total_training_updates=0,
               motion_ready=False,deployment_ready=False)
    write_json(args.output_dir/'status.json',state)
    try:
        assembly=build(args.checkpoint,args.qualified_evaluation_dir,args.split_router,
                       args.output_dir/'initial_model',args.baseline_dir)
        initial=args.output_dir/'initial_model/model_0.pt'
        state.update(source_sha256=assembly['source_sha256'],conversion=assembly,
                     initial_model_independently_evaluated=False,source_sha256s={})
        for relative in dict.fromkeys([*SOURCES,str(Path(__file__).relative_to(ROOT)),
            'tools/deformable_mlp_default_split.py',
            'tools/deformable_mlp_guarded_train.py',
            'tools/deformable_mlp_motion_audit.py',
            'tools/deformable_tire_wrench_audit.py',
            'src/agent_rl/rsl_rl/modules/sensor_routed_mlp.py',
            'src/agent_rl/rsl_rl/modules/actor_critic_suspension_default_split_mlp.py',
            'src/agent_rl/rsl_rl/modules/__init__.py',
            'src/agent_rl/rsl_rl/algorithms/deformable_motion_admission.py']):
            content=(ROOT/relative).read_bytes();path=args.output_dir/'source'/relative
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
            state['source_sha256s'][relative]=hashlib.sha256(content).hexdigest()
        name=args.output_dir.name+'_short'
        settings=SimpleNamespace(**vars(args),trainable_experts=[3],motion_aware_preservation=True,
                                 reference_replay_max_delta=0.)
        info=checkpoint_info(initial)
        if info['policy_class']!=SPLIT_CLASS:
            raise ValueError('Specialist trial requires the explicit four-expert conversion')
        command=training_command(settings,initial,name,args.iterations,args.num_envs,info)
        training_grade=getattr(args,'training_grade',None)
        if training_grade is not None:
            # Change training exposure only. The subsequent benchmark still
            # supplies every original grade, command and exact reset protocol.
            if training_grade != 20.:
                raise ValueError('Expert-3 specialization is restricted to the explicit 20-degree training grade')
            command.extend([
                'env.terrain.terrain_generator.sub_terrains.periodic_slope.angle_range=[20.0,20.0]',
                'env.terrain.terrain_generator.sub_terrains.periodic_slope.angle_choices=[20.0]',
            ])
            state['training_exposure']=dict(grade_deg=training_grade,
                scope='Training periodic ramps only; flat intervals and crests remain physical. Independent five-grade benchmark unchanged.')
        state['optimization_command']=command
        write_json(args.output_dir/'status.json',state)
        run_child(command,'train_short',args,state)
        candidate=latest_checkpoint(policy_experiment(SPLIT_CLASS),name)
        info=checkpoint_info(candidate)
        if info['iteration']!=args.iterations-1:
            raise ValueError('Specialist did not complete the declared fresh PPO update budget')
        update=actor_update_info(initial,candidate)
        if not update['policy_updated']:
            raise ValueError('Specialist actor did not update; critic/noise-only training is insufficient')
        state.update(total_training_updates=args.iterations,actor_update=update,candidate=str(candidate),
                     candidate_sha256=info['sha256'])
        evaluation=args.output_dir/'evaluation'
        run_child(benchmark_command(args,candidate,evaluation),'benchmark_short',args,state)
        acceptance=qualify(candidate,evaluation,args.baseline_dir)
        write_json(args.output_dir/'assessment.json',acceptance)
        motion=audit_motion(evaluation);write_json(args.output_dir/'motion_assessment.json',motion)
        tires=audit_tires(evaluation);write_json(args.output_dir/'tire_wrench_assessment.json',tires)
        state.update(transformer_level_recovered=acceptance['transformer_level_recovered'],
                     strict_horizontal_goal_achieved=acceptance['strict_horizontal_goal_achieved'],
                     motion_ready=motion['motion_ready'],trace_instrumentation_verified=True,
                     ended_at=timestamp())
        if not acceptance['transformer_level_recovered']:
            state.update(status='rejected',rejected_checkpoint=str(candidate))
            return 2
        state.update(status='completed',last_accepted_checkpoint=str(candidate),
                     promotion_scope='Full paired baseline only; final training and deployment require useful motion and leveling')
        return 0
    except BaseException as error:
        state.update(status='failed',ended_at=timestamp(),error=str(error))
        raise
    finally:
        write_json(args.output_dir/'status.json',state)


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--qualified-evaluation-dir',type=Path,required=True)
    parser.add_argument('--split-router',type=Path,required=True)
    parser.add_argument('--reference-replay',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,default=ROOT/'outputs/deformable_long2499_joint_reward_review_20261008/evaluation')
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--num-envs',type=int,default=128)
    parser.add_argument('--training-grade',type=float,choices=[20.],default=None,
                        help='Explicit expert-3 training exposure; does not change independent evaluation grades.')
    parser.add_argument('--seed',type=int,default=53)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--timeout-hours',type=float,default=4.)
    parser.add_argument('--learning-rate',type=float,default=5.e-6)
    parser.add_argument('--preservation-weight',type=float,default=1.)
    parser.add_argument('--reference-action-scale',type=float,default=.03)
    parser.add_argument('--reference-replay-weight',type=float,default=.1)
    parser.add_argument('--reference-replay-action-scale',type=float,default=.03)
    parser.add_argument('--motion-relaxed-fraction',type=float,default=0.)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args(argv)
    if (not 1<=args.iterations<=1000 or not 1<=args.num_envs<=1024 or args.seed<0 or args.seed in (1234,4321)
            or not math.isfinite(args.motion_relaxed_fraction) or not 0<=args.motion_relaxed_fraction<1
            or any(not math.isfinite(v) or v<=0 for v in [args.timeout_hours,args.learning_rate,
                args.preservation_weight,args.reference_action_scale,args.reference_replay_weight,
                args.reference_replay_action_scale])):
        parser.error('Finite positive trial settings, a valid motion fraction and separate training seed required')
    for key in ['checkpoint','qualified_evaluation_dir','split_router','reference_replay','output_dir','baseline_dir']:
        setattr(args,key,getattr(args,key).resolve())
    if not args.run:
        command=[str(ROOT/'run_gui.sh'),sys.executable,'-B','-u',str(Path(__file__).relative_to(ROOT)),*argv]
        print('Required: source acceptance, exact clone, fresh expert-3 update, independent five-grade benchmark, motion and tire audit')
        print('Dry-run: no source acceptance, conversion or physics execution claimed')
        print(shlex.join(command+['--run']))
        return 0
    return run_trial(args)


if __name__=='__main__': raise SystemExit(main())
