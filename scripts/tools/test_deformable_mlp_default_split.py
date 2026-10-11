"""Default specialization must preserve every existing nondefault branch."""
import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("default_split_test_module",
    ROOT/"source/agent_rl/agent_rl/rsl_rl/modules/sensor_routed_mlp.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.insert(0,str(ROOT/'scripts/tools'))
import deformable_mlp_default_split as assembly
import deformable_mlp_guarded_train as pipeline


def policy():
    torch.manual_seed(2423)
    actor=module.SensorDefaultSplitMLP([[12,8],[16,8],[20,8],[12,8]],[8],
        trainable_expert_indices=[3],split_router_hidden_dims=[8])
    with torch.no_grad():
        for parameter in (*actor.router.parameters(),*actor.split_router.parameters()):
            parameter.zero_()
        actor.router[-1].bias.copy_(torch.tensor([20.,-20.,-20.]))
        actor.split_router[-1].bias.copy_(torch.tensor([-20.,20.]))
    actor.experts[3].load_state_dict(actor.experts[0].state_dict())
    return actor


def test_default_clone_has_identical_actions_before_specialization():
    actor=policy();x=torch.randn(12,160);x[:,-14:-10]=.1
    assert torch.equal(actor(x),actor.experts[0](x))
    assert actor.routing_indices(x).eq(3).all()


def test_nondefault_routes_take_precedence_over_the_split_classifier():
    actor=policy();x=torch.randn(8,160);x[:,-14:-10]=.1
    for index in (1,2):
        with torch.no_grad():
            actor.router[-1].bias.fill_(-20.);actor.router[-1].bias[index]=20.
        assert actor.routing_indices(x).eq(index).all()
        assert torch.equal(actor(x),actor.experts[index](x))


def test_uncertain_split_and_unloaded_startup_retain_the_original_default():
    actor=policy();x=torch.randn(8,160);x[:,-14:-10]=0.
    assert torch.equal(actor(x),actor.experts[0](x))
    x[:,-14:-10]=.1
    with torch.no_grad(): actor.split_router[-1].bias.zero_()
    assert torch.equal(actor(x),actor.experts[0](x))


def test_actual_gradient_step_updates_only_the_new_default_specialist():
    actor=policy();x=torch.randn(8,160);x[:,-14:-10]=.1
    before={key:value.clone() for key,value in actor.state_dict().items()}
    optimizer=torch.optim.SGD([p for p in actor.parameters() if p.requires_grad],lr=.01)
    actor(x).square().mean().backward();optimizer.step()
    changed=[key for key,value in actor.state_dict().items() if not torch.equal(value,before[key])]
    assert changed and all(key.startswith('experts.3.') for key in changed)
    assert all(p.grad is None for p in (*actor.router.parameters(),*actor.split_router.parameters()))


def test_scripted_routing_preserves_selected_unclipped_outputs():
    actor=policy();x=torch.randn(8,160);x[:4,-14:-10]=.1;x[4:,-14:-10]=0.
    scripted=torch.jit.script(actor)
    assert torch.equal(scripted(x),actor(x))


def test_native_exporter_rebuilds_both_routers_with_no_action_change():
    actor=policy();x=torch.randn(8,160);x[:4,-14:-10]=.1;x[4:,-14:-10]=0.
    spec=importlib.util.spec_from_file_location('default_split_exporter',ROOT/'scripts/rsl_rl/export_onnx.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    rebuilt,inputs,outputs=exporter._build_actor({'actor.'+k:v for k,v in actor.state_dict().items()})
    assert (inputs,outputs)==(160,4)
    assert torch.equal(rebuilt(x),actor(x))
    assert torch.equal(torch.jit.script(rebuilt)(x),actor(x))


def test_invalid_expert_count_or_confidence_contract_is_rejected():
    with pytest.raises(ValueError,match='Default split'):
        module.SensorDefaultSplitMLP([[12],[16],[20]],[8])
    with pytest.raises(ValueError,match='Default split'):
        module.SensorDefaultSplitMLP([[12],[16],[20],[12]],[8],split_routing_confidence=float('nan'))


def test_saved_folded_specialist_width_survives_runtime_export_and_script():
    actor=module.SensorDefaultSplitMLP([[12,8],[16,8],[20,8],[18,12]],[8],
        trainable_expert_indices=[3],split_router_hidden_dims=[8])
    with torch.no_grad():
        for parameter in (*actor.router.parameters(),*actor.split_router.parameters()):parameter.zero_()
        actor.router[-1].bias.copy_(torch.tensor([20.,-20.,-20.]))
        actor.split_router[-1].bias.copy_(torch.tensor([-20.,20.]))
    x=torch.randn(8,160);x[:,-14:-10]=.1
    spec=importlib.util.spec_from_file_location('folded_specialist_export',ROOT/'scripts/rsl_rl/export_onnx.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    rebuilt,inputs,outputs=exporter._build_actor({'actor.'+k:v for k,v in actor.state_dict().items()})
    assert (inputs,outputs)==(160,4)
    assert rebuilt.experts[3][0].out_features==18
    assert torch.equal(rebuilt(x),actor(x))
    assert torch.equal(torch.jit.script(rebuilt)(x),actor(x))


@pytest.mark.parametrize('fault', [None, 'loader', 'canonical_seed', 'overlap',
                                  'unsupported', 'nan_recall', 'privileged', 'history_width'])
def test_recorded_pose_router_rejects_leakage_failures_and_stale_sources(tmp_path, fault):
    import numpy as np
    checkpoint, _, _ = checkpoint_fixture(tmp_path)
    physics = assembly.digest(checkpoint.parent / 'params/real2sim_model.json')
    teacher = tmp_path / 'teacher_source.py'; teacher.write_text('own training guide')
    evaluator = tmp_path / 'evaluator_source.py'; evaluator.write_text('own training evaluator')
    loader = tmp_path / 'data_loader_source.py'; loader.write_text('H5 sensors only')
    report_path, status_path, data_path = (tmp_path / name for name in ('report.json', 'status.json', 'data.npz'))
    seed = 1234 if fault == 'canonical_seed' else 2441
    report = dict(seed=seed, history=5, steps=600, command_frame='body', checkpoint=str(checkpoint),
                  terrain=dict(grade_deg=20.), real2sim=dict(model_sha256=physics),
                  training_data_controller='privileged_wheel_load_training_geometry_guide',
                  results=dict(POLICY=dict(static=dict(failure_adjusted_all_contact_rate=.97 if fault == 'unsupported' else 1.,
                                                       physical_terminated_resets=0, terrain_boundary_violations=0))))
    status = dict(status='completed', seed=seed, purpose='training collection',
                  experiment_teacher_sha256=assembly.digest(teacher), experiment_evaluator_sha256=assembly.digest(evaluator))
    report_path.write_text(json.dumps(report)); status_path.write_text(json.dumps(status))
    np.savez(data_path, policy_obs=np.zeros((600,4,128 if fault == 'history_width' else 160),np.float32),
             env_scenarios=np.asarray(['static']*4),seed=np.asarray(seed),grade_deg=np.asarray(20.),
             real2sim_model_sha256=np.asarray(physics),teacher_checkpoint=np.asarray(str(checkpoint)))
    source = dict(path=str(data_path),sha256=assembly.digest(data_path),report=str(report_path),
                  report_sha256=assembly.digest(report_path),collection_status=str(status_path),
                  status_sha256=assembly.digest(status_path),teacher_source=str(teacher),
                  teacher_source_sha256=assembly.digest(teacher),evaluator_source=str(evaluator),
                  evaluator_source_sha256=assembly.digest(evaluator),seed=seed,grade_deg=20.,physics_sha256=physics,
                  checkpoint_sha256=assembly.digest(checkpoint),action_targets_used=False,
                  privileged_critic_inputs_used=False,train_vehicles=[0,1,3] if fault == 'overlap' else [0,1,2],
                  held_vehicles=[3],excluded_failed_vehicles=[],training_cases=['static'],held_states=2400)
    coverage = dict(path=str(data_path),held_states=2400,eligible_default_states=2000,
                    candidate_recall=float('nan') if fault == 'nan_recall' else .999)
    augmentation = dict(evaluation_data_used=False,action_targets_used=False,
                        privileged_actor_inputs_used=fault == 'privileged',data_loader_source=str(loader),
                        data_loader_sha256=assembly.digest(loader),datasets=[source],held_out_results=[coverage])
    if fault == 'loader': loader.write_text('modified loader')
    routing = dict(recorded_pose_training_augmentation=augmentation)
    if fault is None:
        checked = assembly.validate_recorded_pose_augmentation(routing,checkpoint)
        assert checked['datasets'][0]['seed']==2441
        assert checked['physics_sha256']==physics
    else:
        with pytest.raises(ValueError,match='Recorded pose'):
            assembly.validate_recorded_pose_augmentation(routing,checkpoint)


def test_conversion_requires_unchanged_existing_tensors_and_an_exact_default_clone():
    before={'actor.experts.0.0.weight':torch.ones(4,160),'actor.router.0.weight':torch.ones(3,160),
            'critic.0.weight':torch.ones(1,40),'log_std':torch.ones(4)}
    after={key:value.clone() for key,value in before.items()}
    after['actor.experts.3.0.weight']=before['actor.experts.0.0.weight'].clone()
    after['actor.split_router.0.weight']=torch.ones(2,160)
    assert assembly.validate_conversion_state(before,after)['existing_tensors_unchanged']==4
    for key in ('actor.router.0.weight','critic.0.weight','log_std'):
        changed=copy.deepcopy(after);changed[key].add_(.01)
        with pytest.raises(ValueError,match='existing'):
            assembly.validate_conversion_state(before,changed)
    changed=copy.deepcopy(after);changed['actor.experts.3.0.weight'].add_(.01)
    with pytest.raises(ValueError,match='exact copy'):
        assembly.validate_conversion_state(before,changed)
    del changed['actor.experts.3.0.weight']
    with pytest.raises(ValueError,match='exact copy'):
        assembly.validate_conversion_state(before,changed)


def checkpoint_fixture(tmp_path):
    actor=policy()
    config=dict(class_name=pipeline.SPLIT_CLASS,history_length=5,actor_hidden_dims=[12,8],critic_hidden_dims=[16],
        expert_hidden_dims=[[12,8],[16,8],[20,8],[12,8]],router_hidden_dims=[8],routing_confidence=.995,
        routing_load_threshold=.02,split_router_hidden_dims=[8],split_routing_confidence=.995,
        trainable_expert_indices=[3],activation='elu',actor_obs_normalization=False,critic_obs_normalization=False)
    params=tmp_path/'params';params.mkdir()
    (params/'agent.yaml').write_text(yaml.safe_dump(dict(policy=config,algorithm=dict(class_name='DiagnosticPPO'))))
    (params/'real2sim_model.json').write_bytes(b'checked synthetic physics snapshot')
    env=dict(policy_history_length=5,observation_space=160,state_space=40,action_contract_version='minangle_physical_v3',
        real2sim_enabled=True,real2sim_observation_version='current_fraction_v2',
        real2sim_model_sha256=hashlib.sha256((params/'real2sim_model.json').read_bytes()).hexdigest())
    (params/'env.yaml').write_text(yaml.safe_dump(env))
    state={'actor.'+k:v for k,v in actor.state_dict().items()}
    state.update({'critic.'+k:v for k,v in torch.nn.Sequential(torch.nn.Linear(40,16),torch.nn.ELU(),torch.nn.Linear(16,1)).state_dict().items()})
    state['log_std']=torch.zeros(4)
    path=tmp_path/'model_0.pt';torch.save(dict(model_state_dict=state,iter=0),path)
    return path,config,state


def test_checkpoint_guard_checks_both_router_dimensions_and_saved_split_gates(tmp_path):
    path,config,state=checkpoint_fixture(tmp_path)
    info=pipeline.checkpoint_info(path)
    assert info['policy_class']==pipeline.SPLIT_CLASS
    assert info['routed_parameters']['split_router_hidden_dims']==[8]
    spec=importlib.util.spec_from_file_location('default_split_checkpoint',ROOT/'scripts/utils/deformable_checkpoint.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    checker.validate_deformable_policy(path,config)
    changed=copy.deepcopy(config);changed['split_routing_confidence']=.98
    with pytest.raises(ValueError,match='split_routing_confidence'):
        checker.validate_deformable_policy(path,changed)
    state['actor.split_router.2.weight']=torch.ones(3,8)
    torch.save(dict(model_state_dict=state,iter=0),path)
    with pytest.raises(ValueError,match='actor.split_router'):
        pipeline.checkpoint_info(path)


def test_checkpoint_guard_accepts_folded_width_only_when_saved_tensors_match(tmp_path):
    path, config, state = checkpoint_fixture(tmp_path)
    config['expert_hidden_dims'][3] = [18, 12]
    wider = module.elu_mlp(160, [18, 12], 4)
    for key, value in wider.state_dict().items():
        state['actor.experts.3.' + key] = value
    torch.save(dict(model_state_dict=state, iter=0), path)
    (path.parent / 'params/agent.yaml').write_text(yaml.safe_dump(dict(policy=config)))
    assert pipeline.checkpoint_info(path)['routed_parameters']['expert_hidden_dims'][3] == [18, 12]
    state['actor.experts.3.2.weight'] = torch.zeros(12, 17)
    torch.save(dict(model_state_dict=state, iter=0), path)
    with pytest.raises(ValueError, match='actor.experts.3'):
        pipeline.checkpoint_info(path)


def test_guarded_updates_reject_a_modified_frozen_split_router_or_other_expert(tmp_path):
    before,config,state=checkpoint_fixture(tmp_path)
    after=tmp_path/'model_1.pt'
    state={k:v.clone() for k,v in state.items()};state['actor.experts.3.4.bias'].add_(1.e-5)
    torch.save(dict(model_state_dict=state,iter=1),after)
    assert pipeline.actor_update_info(before,after)['policy_updated']
    state['actor.split_router.2.bias'].add_(1.e-5)
    torch.save(dict(model_state_dict=state,iter=1),after)
    with pytest.raises(ValueError,match='router changed'):
        pipeline.actor_update_info(before,after)
    state['actor.split_router.2.bias'].copy_(torch.zeros_like(state['actor.split_router.2.bias']))
    # Restore the actual frozen router, rather than treating zero as its baseline.
    original=torch.load(before,weights_only=False)['model_state_dict']
    state['actor.split_router.2.bias'].copy_(original['actor.split_router.2.bias'])
    state['actor.experts.0.4.bias'].add_(1.e-5)
    torch.save(dict(model_state_dict=state,iter=1),after)
    with pytest.raises(ValueError,match='Frozen expert'):
        pipeline.actor_update_info(before,after)


@pytest.mark.parametrize('fault',[None,'producer','physics','motor','wheel','wrong_override','nan_recall'])
def test_synthetic_classifier_cannot_bypass_provenance_limits_or_coverage(tmp_path,fault):
    params=tmp_path/'params';params.mkdir()
    model=dict(fitted_mechanism=dict(urdf_sha256='nominal_urdf'),
               simulation_effort_mapping=dict(motor_command_limit_current_raw=2048.))
    (params/'real2sim_model.json').write_text(json.dumps(model))
    producer=tmp_path/'poses.py';producer.write_text('nominal static witness producer')
    path=producer.with_suffix('.json')
    holding=dict(holding_feasible=True,nominal_motor_command_within_limit=True,wheel_torque_within_limit=True,
                 required_nominal_static_motor_current_raw=[500.]*4,required_wheel_joint_torque_nm=[3.]*4)
    document=dict(status='completed',source_sha256=assembly.digest(producer),
        real2sim_model_sha256=assembly.digest(params/'real2sim_model.json'),urdf_sha256='nominal_urdf',
        absolute_shaft_torque_calibrated=False,geometry_parity_max_m=1.e-6,cases=[dict(holding=holding)])
    if fault=='producer': document['source_sha256']='stale'
    if fault=='physics': document['real2sim_model_sha256']='different'
    if fault=='motor': holding['required_nominal_static_motor_current_raw'][0]=2049.
    if fault=='wheel': holding['required_wheel_joint_torque_nm'][0]=5.1
    path.write_text(json.dumps(document))
    held=[dict(grade_deg=17,wrong_override_states=0,candidate_recall=None),
          dict(grade_deg=20,wrong_override_states=0,candidate_recall=.999)]
    if fault=='wrong_override': held[0]['wrong_override_states']=1
    if fault=='nan_recall': held[1]['candidate_recall']=float('nan')
    info=dict(synthetic_training_augmentation=dict(path=str(path),sha256=assembly.digest(path),
              held_noise_seed=2428,held_noise_results=held))
    if fault:
        with pytest.raises(ValueError,match='Synthetic router'):
            assembly.validate_synthetic_augmentation(info,tmp_path/'model.pt')
    else:
        assert assembly.validate_synthetic_augmentation(info,tmp_path/'model.pt')['source_physics_sha256']==document['real2sim_model_sha256']


def command_repair_policy():
    original=policy()
    actor=module.SensorDefaultSplitMLP([[12,8],[16,8],[20,8],[12,8]],[8],
        trainable_expert_indices=[3],split_router_hidden_dims=[8],moving_expert_hidden_dims=[20,8])
    state={k:v.clone() for k,v in original.state_dict().items()}
    state.update({'moving_expert.'+k:v.clone() for k,v in original.experts[2].state_dict().items()})
    state['standing_command_tolerance']=torch.tensor(1.e-6)
    actor.load_state_dict(state,strict=True)
    with torch.no_grad():
        actor.experts[2][-1].bias.add_(2.)
        actor.experts[3][-1].bias.add_(3.)
    return original,actor


@pytest.mark.parametrize('primary_route',[0,1,2])
def test_command_repair_preserves_original_moving_means_and_static_priority(primary_route):
    original,actor=command_repair_policy()
    with torch.no_grad():
        for router in (original.router,actor.router):
            router[-1].bias.fill_(-20.);router[-1].bias[primary_route]=20.
    x=torch.randn(12,160)
    frames=x.reshape(-1,5,32);frames[:,:,1:4]=0.;frames[6:,-1,1]=.4
    x[:,-14:-10]=.1
    expected_static=actor.experts[3 if primary_route==0 else primary_route](x)
    result=actor(x)
    assert torch.equal(result[:6],expected_static[:6])
    assert torch.equal(result[6:],original(x)[6:])
    # A requested stop must retain original inference until old command
    # history disappears; the static fitter only saw five zero command rows.
    frames[6:,-1,1]=0.;frames[6:,0,3]=.1
    assert torch.equal(actor(x)[6:],original(x)[6:])
    assert torch.equal(torch.jit.script(actor)(x),actor(x))


def test_command_repair_native_export_and_saved_gate_are_exact():
    original,actor=command_repair_policy()
    x=torch.randn(8,160);x.reshape(-1,5,32)[:,:,1:4]=0.;x[4:,-31]=.4;x[:,-14:-10]=.1
    spec=importlib.util.spec_from_file_location('command_repair_export',ROOT/'scripts/rsl_rl/export_onnx.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    rebuilt,inputs,outputs=exporter._build_actor({'actor.'+k:v for k,v in actor.state_dict().items()})
    assert (inputs,outputs)==(160,4)
    assert torch.equal(rebuilt(x),actor(x))
    assert torch.equal(torch.jit.script(rebuilt)(x),actor(x))
    assert rebuilt.moving_expert[0].out_features==20
    assert 'standing_command_tolerance' not in original.state_dict()


@pytest.mark.parametrize('tolerance',[-1.e-6,1.e-5,float('nan')])
def test_command_repair_rejects_nonzero_or_nonfinite_standing_tolerance(tolerance):
    with pytest.raises(ValueError,match='numerical zero'):
        module.SensorDefaultSplitMLP([[12],[16],[20],[12]],[8],moving_expert_hidden_dims=[20],
                                    standing_command_tolerance=tolerance)


def test_command_repair_checkpoint_checks_saved_moving_architecture_and_zero_gate(tmp_path):
    path,config,state=checkpoint_fixture(tmp_path)
    config.update(moving_expert_hidden_dims=[20,8],standing_command_tolerance=1.e-6)
    original,actor=command_repair_policy()
    state={k:v for k,v in state.items() if not k.startswith('actor.')}
    state.update({'actor.'+k:v for k,v in actor.state_dict().items()})
    torch.save(dict(model_state_dict=state,iter=0),path)
    (path.parent/'params/agent.yaml').write_text(yaml.safe_dump(dict(
        policy=config, algorithm={'class_name': 'DiagnosticPPO'})))
    info=pipeline.checkpoint_info(path)
    assert info['routed_parameters']['moving_expert_hidden_dims']==[20,8]
    spec=importlib.util.spec_from_file_location('command_repair_metadata',ROOT/'scripts/utils/deformable_checkpoint.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    checker.validate_deformable_policy(path,config)
    changed=copy.deepcopy(config);changed['standing_command_tolerance']=.9e-6
    with pytest.raises(ValueError,match='standing_command_tolerance'):
        checker.validate_deformable_policy(path,changed)
    state['actor.standing_command_tolerance']=torch.tensor(.9e-6)
    torch.save(dict(model_state_dict=state,iter=0),path)
    with pytest.raises(ValueError,match='metadata'):
        pipeline.checkpoint_info(path)


def test_command_repair_guards_the_frozen_original_moving_expert(tmp_path):
    path,config,state=checkpoint_fixture(tmp_path)
    config.update(moving_expert_hidden_dims=[20,8],standing_command_tolerance=1.e-6)
    original,actor=command_repair_policy()
    state={k:v for k,v in state.items() if not k.startswith('actor.')}
    state.update({'actor.'+k:v for k,v in actor.state_dict().items()})
    torch.save(dict(model_state_dict=state,iter=0),path)
    (path.parent/'params/agent.yaml').write_text(yaml.safe_dump(dict(
        policy=config, algorithm={'class_name': 'DiagnosticPPO'})))
    candidate=tmp_path/'model_1.pt';state=copy.deepcopy(state)
    state['actor.experts.3.4.bias'].add_(1.e-5)
    torch.save(dict(model_state_dict=state,iter=1),candidate)
    assert pipeline.actor_update_info(path,candidate)['policy_updated']
    state['actor.moving_expert.4.bias'].add_(1.e-5)
    torch.save(dict(model_state_dict=state,iter=1),candidate)
    with pytest.raises(ValueError,match='Frozen moving expert'):
        pipeline.actor_update_info(path,candidate)
