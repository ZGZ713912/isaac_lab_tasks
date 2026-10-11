"""Explicitly clone a qualified default MLP into a separate trainable specialist.

The original three-way router and all original expert tensors are preserved.
The fourth expert initially equals expert zero, so sensor-based subdivision
does not change actions. Conversion is not independent physics acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from deformable_mlp_guarded_train import (
    ROOT, ROUTED_CLASS, SPLIT_CLASS, SPLIT_EXPERIMENT, checkpoint_info, qualify,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_synthetic_augmentation(routing,checkpoint):
    """Bind optional nominal-pose training to its producer and source physics."""
    augmentation=routing.get('synthetic_training_augmentation')
    if augmentation is None:
        return None
    path=Path(augmentation['path']);document=json.loads(path.read_text())
    model_path=checkpoint.parent/'params/real2sim_model.json'
    model=json.loads(model_path.read_text())
    held=augmentation['held_noise_results']
    recall=next((row.get('candidate_recall') for row in held if row['grade_deg']==20),None)
    if (digest(path)!=augmentation['sha256'] or digest(path.with_suffix('.py'))!=document['source_sha256']
            or document['status']!='completed' or document['real2sim_model_sha256']!=digest(model_path)
            or document['urdf_sha256']!=model['fitted_mechanism']['urdf_sha256']
            or document['absolute_shaft_torque_calibrated'] is not False
            or not math.isfinite(document['geometry_parity_max_m']) or document['geometry_parity_max_m']>1.e-5
            or augmentation['held_noise_seed'] in (1234,4321)
            or len(held)!=2 or {row['grade_deg'] for row in held}!={17,20}
            or any(row['wrong_override_states'] for row in held)
            or not isinstance(recall,(float,int)) or not math.isfinite(recall) or not .98<=recall<=1.):
        raise ValueError('Synthetic router augmentation lacks matching source physics, producer or coverage')
    for row in document['cases']:
        holding=row['holding']
        currents=holding['required_nominal_static_motor_current_raw']
        torques=holding['required_wheel_joint_torque_nm']
        if (not holding['holding_feasible'] or not holding['nominal_motor_command_within_limit']
                or not holding['wheel_torque_within_limit'] or len(currents)!=4 or len(torques)!=4
                or not all(math.isfinite(x) for x in [*currents,*torques])
                or max(map(abs,currents))>model['simulation_effort_mapping']['motor_command_limit_current_raw']
                or max(map(abs,torques))>5.):
            raise ValueError('Synthetic router pose exceeds its nominal actuation limits')
    return dict(path=str(path.resolve()),sha256=digest(path),producer_sha256=document['source_sha256'],
                source_physics_sha256=digest(model_path),coverage=held,
                scope='Nominal static synthetic training and independent observation perturbations; not physical or mechanical-pose holdout acceptance')


def validate_recorded_pose_augmentation(routing, checkpoint):
    """Bind optional posture sensors to supported, disjoint training vehicles."""
    augmentation = routing.get('recorded_pose_training_augmentation')
    if augmentation is None:
        return None
    if (augmentation.get('evaluation_data_used') is not False
            or augmentation.get('action_targets_used') is not False
            or augmentation.get('privileged_actor_inputs_used') is not False
            or digest(Path(augmentation['data_loader_source'])) != augmentation['data_loader_sha256']):
        raise ValueError('Recorded pose router must use byte-checked training sensors only')
    datasets, held = augmentation['datasets'], augmentation['held_out_results']
    if not datasets or len(datasets) != len(held):
        raise ValueError('Recorded pose router lacks disjoint whole-vehicle coverage')
    physics = digest(checkpoint.parent / 'params/real2sim_model.json')
    checked = []
    for source, coverage in zip(datasets, held):
        for path_key, hash_key in (
                ('path', 'sha256'), ('report', 'report_sha256'),
                ('collection_status', 'status_sha256'), ('teacher_source', 'teacher_source_sha256'),
                ('evaluator_source', 'evaluator_source_sha256')):
            if digest(Path(source[path_key])) != source[hash_key]:
                raise ValueError('Recorded pose source bytes changed')
        report = json.loads(Path(source['report']).read_text())
        status = json.loads(Path(source['collection_status']).read_text())
        if (source['seed'] in (1234, 4321) or source['seed'] != report['seed']
                or source['seed'] != status['seed'] or type(source['seed']) is not int
                or status['status'] != 'completed' or 'training' not in status['purpose']
                or report['history'] != 5 or report['command_frame'] != 'body'
                or report['terrain']['grade_deg'] != 20. or source['grade_deg'] != 20.
                or report['real2sim']['model_sha256'] != physics or source['physics_sha256'] != physics
                or report.get('training_data_controller') != 'privileged_wheel_load_training_geometry_guide'
                or Path(report['checkpoint']).resolve() != checkpoint.resolve()
                or source['checkpoint_sha256'] != digest(checkpoint)
                or source.get('action_targets_used') is not False
                or source.get('privileged_critic_inputs_used') is not False
                or source['teacher_source_sha256'] != status['experiment_teacher_sha256']
                or source['evaluator_source_sha256'] != status['experiment_evaluator_sha256']):
            raise ValueError('Recorded pose collection contract, seed or physics differs')
        train, validation = source['train_vehicles'], source['held_vehicles']
        excluded = set(source['excluded_failed_vehicles'])
        if (not train or not validation or set(train) & set(validation)
                or any(type(i) is not int or i < 0 or i % 4 == 3 or i in excluded for i in train)
                or any(type(i) is not int or i < 0 or i % 4 != 3 or i in excluded for i in validation)):
            raise ValueError('Recorded pose train and hold vehicles overlap or include failures')
        for case in source['training_cases']:
            row = report['results']['POLICY'][case]
            if (row['failure_adjusted_all_contact_rate'] < .98
                    or row['physical_terminated_resets'] or row['terrain_boundary_violations']):
                raise ValueError('Recorded pose sensors include unsupported or failed command cases')
        with np.load(source['path'], allow_pickle=False) as data:
            raw, scenarios = data['policy_obs'], data['env_scenarios']
            if (raw.shape != (report['steps'], len(scenarios), 160) or raw.dtype != np.float32
                    or not np.isfinite(raw).all() or max(train + validation) >= len(scenarios)
                    or any(str(scenarios[i]) not in source['training_cases'] for i in train + validation)
                    or int(data['seed']) != source['seed'] or float(data['grade_deg']) != 20.
                    or str(data['real2sim_model_sha256']) != physics
                    or Path(str(data['teacher_checkpoint'])).resolve() != checkpoint.resolve()):
                raise ValueError('Recorded pose sensors no longer match their collection')
        recall = coverage.get('candidate_recall')
        if (coverage['path'] != source['path'] or coverage['held_states'] != source['held_states']
                or coverage['eligible_default_states'] <= 0
                or not isinstance(recall, (float, int)) or not math.isfinite(recall) or not .98 <= recall <= 1.):
            raise ValueError('Recorded pose router lacks .98 held default-branch coverage')
        checked.append(dict(path=source['path'], sha256=source['sha256'], seed=source['seed'],
                            training_cases=source['training_cases'], coverage=coverage))
    return dict(datasets=checked, physics_sha256=physics,
                scope='Own-seed supported sensor classification only; conversion remains action identity, not physical admission')


def validate_conversion_state(before, after):
    """Check the structural identity, not just a finite sample's parity."""
    if not before or any(key not in after or not torch.equal(value,after[key]) for key,value in before.items()):
        raise ValueError("Conversion changed an existing policy, critic, noise or routing tensor")
    default=[key for key in before if key.startswith('actor.experts.0.')]
    if not default or any(key.replace('actor.experts.0.','actor.experts.3.') not in after
                          or not torch.equal(before[key],after[key.replace('actor.experts.0.','actor.experts.3.')])
                          for key in default):
        raise ValueError("New specialist is not an exact copy of the existing default")
    allowed={key.replace('actor.experts.0.','actor.experts.3.') for key in default}
    added=set(after)-set(before)
    if any(key not in allowed and not key.startswith('actor.split_router.')
           and key != 'actor.split_routing_confidence' for key in added):
        raise ValueError("Unexpected tensor added by default conversion")
    return dict(existing_tensors_unchanged=len(before),default_tensors_cloned=len(default),
                identity_scope='all finite sensor inputs: original routes preserved, default subdivision selects identical tensors')


def build(checkpoint, evaluation, router_path, output, baseline):
    torch.set_num_threads(2)
    info=checkpoint_info(checkpoint)
    if info['policy_class'] != ROUTED_CLASS or len(info['routed_parameters']['expert_hidden_dims']) != 3:
        raise ValueError('Default subdivision requires a qualified original three-expert policy')
    admission=qualify(checkpoint,evaluation,baseline)
    if not admission['transformer_level_recovered']:
        raise ValueError('Default subdivision source has not passed full paired physics acceptance')
    classifier=torch.load(router_path,map_location='cpu',weights_only=False)
    routing=classifier['infos']
    if digest(router_path.parent/'source.py')!=routing['source_sha256']:
        raise ValueError('Classifier training producer bytes changed')
    augmentation=validate_synthetic_augmentation(routing,checkpoint)
    recorded_augmentation=validate_recorded_pose_augmentation(routing,checkpoint)
    if (routing['status'] != 'completed' or routing['input_size'] != 160
            or routing['activation'] != 'elu' or routing['seed'] in (1234,4321)
            or routing['evaluation_data_used'] or routing['held_out_wrong_overrides']
            or routing['model_selection'] != 'fixed 80 epochs; no held or canonical outcome selection'
            or {row['grade_deg'] for row in routing['held_out_results']} != {0.,5.,10.,17.,20.}
            or any(row['wrong_override_states'] for row in routing['held_out_results'])
            or next(row for row in routing['held_out_results'] if row['grade_deg']==20)['candidate_recall'] < .98):
        raise ValueError('Binary sensor router lacks recorded whole-vehicle holdout coverage')
    original_base=Path(routing['base_checkpoint'])
    if digest(original_base) != routing['base_checkpoint_sha256']:
        raise ValueError('Classifier original routing checkpoint bytes changed')
    before=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
    router_base=torch.load(original_base,map_location='cpu',weights_only=False)['model_state_dict']
    if any(not torch.equal(value,router_base.get(key)) for key,value in before.items()
           if key.startswith('actor.router.') or key in ('actor.routing_confidence','actor.routing_load_threshold')):
        raise ValueError('Classifier must preserve the exact original three-way routing contract')
    state={key:value.clone() for key,value in before.items()}
    state.update({key.replace('actor.experts.0.','actor.experts.3.'):value.clone()
                  for key,value in before.items() if key.startswith('actor.experts.0.')})
    state.update({'actor.split_router.'+key:value.clone() for key,value in classifier['state_dict'].items()})
    state['actor.split_routing_confidence']=torch.tensor(routing['confidence'])
    identity=validate_conversion_state(before,state)
    config=yaml.safe_load((checkpoint.parent/'params/agent.yaml').read_text())
    policy=config['policy']
    policy.update(class_name=SPLIT_CLASS,expert_hidden_dims=[*policy['expert_hidden_dims'],policy['expert_hidden_dims'][0]],
                  split_router_hidden_dims=routing['hidden_dims'],split_routing_confidence=routing['confidence'],
                  trainable_expert_indices=[3])
    config.update(experiment_name=SPLIT_EXPERIMENT,run_name=output.name,seed=routing['seed'])
    spec=importlib.util.spec_from_file_location('split_identity_exporter',ROOT/'scripts/rsl_rl/export_onnx.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    original=exporter._build_actor(before)[0]
    actor=exporter._build_actor(state)[0]
    observations=[]
    for source in routing['datasets']:
        path=Path(source['path'])
        if digest(path) != source['sha256']:
            raise ValueError('Classifier holdout source bytes changed')
        with np.load(path,allow_pickle=False) as data:
            vehicles=(np.arange(data['policy_obs'].shape[1])%4==3) & ~np.isin(
                np.arange(data['policy_obs'].shape[1]),source['excluded_failed_vehicles'])
            observations.append(data['policy_obs'][:,vehicles].reshape(-1,160)[::16].copy())
    values=torch.from_numpy(np.concatenate(observations))
    max_delta=0.;counts=torch.zeros(4,dtype=torch.long)
    with torch.inference_mode():
        for part in values.split(512):
            max_delta=max(max_delta,float((actor(part)-original(part)).abs().max()))
            counts+=torch.bincount(actor.routing_indices(part),minlength=4)
    if max_delta != 0. or not counts.gt(0).all():
        raise ValueError('Conversion must give bit-identical actions and probe all four branches')
    output.mkdir(parents=True,exist_ok=False)
    params=output/'params';params.mkdir()
    for name in ('env.yaml','real2sim_model.json'):
        shutil.copyfile(checkpoint.parent/'params'/name,params/name)
    (params/'agent.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    result=dict(status='completed',kind='explicit sensor-only default specialist conversion',
        source_checkpoint=str(checkpoint.resolve()),source_sha256=digest(checkpoint),source_admission=admission,
        split_router=str(router_path.resolve()),split_router_sha256=digest(router_path),classifier_info=routing,
        synthetic_augmentation_verification=augmentation,
        recorded_pose_augmentation_verification=recorded_augmentation,
        structural_identity=identity,parity_samples=len(values),cpu_action_max_delta=max_delta,
        route_counts=counts.tolist(),trainable_experts=[3],
        closed_loop_acceptance='Conversion identity only; new trained checkpoint requires actual independent Isaac Sim evaluation')
    candidate=output/'model_0.pt'
    torch.save(dict(model_state_dict=state,optimizer_state_dict={},iter=0,infos=result),candidate)
    result.update(checkpoint_info(candidate))
    snapshots=[Path(__file__).relative_to(ROOT),'scripts/tools/deformable_mlp_guarded_train.py',
        'scripts/utils/deformable_checkpoint.py','scripts/rsl_rl/export_onnx.py',
        'source/agent_rl/agent_rl/rsl_rl/modules/sensor_routed_mlp.py',
        'source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_suspension_default_split_mlp.py',
        'source/agent_rl/agent_rl/rsl_rl/modules/__init__.py',
        'source/agent_tasks/agent_tasks/direct/deformable_suspension/agents/rsl_rl_ppo_cfg.py']
    result['source_sha256s']={}
    for relative in snapshots:
        destination=output/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/relative).read_bytes());result['source_sha256s'][str(relative)]=digest(destination)
    (output/'assembly.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--qualified-evaluation-dir',type=Path,required=True)
    parser.add_argument('--split-router',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,default=ROOT/'outputs/deformable_long2499_joint_reward_review_20261008/evaluation')
    args=parser.parse_args()
    result=build(args.checkpoint.resolve(),args.qualified_evaluation_dir.resolve(),args.split_router.resolve(),
                 args.output_dir.resolve(),args.baseline_dir.resolve())
    print(json.dumps({key:result[key] for key in ['status','sha256','policy_class','parity_samples','cpu_action_max_delta','route_counts']}),flush=True)


if __name__=='__main__':
    main()
