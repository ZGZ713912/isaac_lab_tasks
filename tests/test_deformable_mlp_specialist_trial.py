"""A converted policy is never promoted using its source's physical reports."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import deformable_mlp_specialist_trial as trial
import deformable_mlp_guarded_train as pipeline


def setup_trial(monkeypatch,tmp_path):
    args=SimpleNamespace(checkpoint=tmp_path/'source.pt',qualified_evaluation_dir=tmp_path/'source_evaluation',
        split_router=tmp_path/'router.pt',reference_replay=tmp_path/'replay.npz',
        output_dir=tmp_path/'trial',baseline_dir=tmp_path/'baseline',iterations=100,num_envs=128,
        seed=53,device='cpu',timeout_hours=1.,learning_rate=5.e-6,preservation_weight=1.,
        reference_action_scale=.03,reference_replay_weight=.1,reference_replay_action_scale=.03,
        motion_relaxed_fraction=0.)
    calls=[]
    info=dict(policy_class=trial.SPLIT_CLASS,actor_hidden_dims=[12],critic_hidden_dims=[16],
              routed_parameters={},iteration=99,sha256='candidate')
    monkeypatch.setattr(trial,'build',lambda *unused:dict(source_sha256='source',structural_identity=True))
    monkeypatch.setattr(trial,'checkpoint_info',lambda *unused:info)
    monkeypatch.setattr(pipeline,'checkpoint_info',lambda *unused:info)
    monkeypatch.setattr(trial,'latest_checkpoint',lambda *unused:tmp_path/'trained/model_99.pt')
    monkeypatch.setattr(trial,'actor_update_info',lambda *unused:dict(policy_updated=True,changed_tensors=8))
    monkeypatch.setattr(trial,'run_child',lambda command,name,*unused:calls.append((name,command)))
    monkeypatch.setattr(trial,'qualify',lambda *unused:dict(transformer_level_recovered=True,strict_horizontal_goal_achieved=False))
    monkeypatch.setattr(trial,'audit_motion',lambda *unused:dict(motion_ready=False))
    monkeypatch.setattr(trial,'audit_tires',lambda *unused:dict(status='completed',cases=[]))
    return args,calls


@pytest.mark.parametrize('baseline_pass',[False,True])
def test_only_new_reports_promote_trained_bytes_and_failed_motion_cannot_enable_deployment(monkeypatch,tmp_path,baseline_pass):
    args,calls=setup_trial(monkeypatch,tmp_path)
    monkeypatch.setattr(trial,'qualify',lambda *unused:dict(transformer_level_recovered=baseline_pass,strict_horizontal_goal_achieved=False))
    assert trial.run_trial(args)==(0 if baseline_pass else 2)
    state=json.loads((args.output_dir/'status.json').read_text())
    assert [name for name,_ in calls]==['train_short','benchmark_short']
    assert state['initial_model_independently_evaluated'] is False
    assert state['last_accepted_checkpoint']==str(tmp_path/'trained/model_99.pt' if baseline_pass else args.checkpoint)
    assert not state['motion_ready'] and not state['deployment_ready']
    command=calls[0][1]
    assert 'agent.policy.trainable_expert_indices=[3]' in command
    assert 'agent.algorithm.reference_motion_relaxed_fraction=0.0' in command


def test_unchanged_actor_is_not_promoted_or_physically_benchmarked(monkeypatch,tmp_path):
    args,calls=setup_trial(monkeypatch,tmp_path)
    monkeypatch.setattr(trial,'actor_update_info',lambda *unused:dict(policy_updated=False))
    with pytest.raises(ValueError,match='actor did not update'):
        trial.run_trial(args)
    state=json.loads((args.output_dir/'status.json').read_text())
    assert [name for name,_ in calls]==['train_short']
    assert state['last_accepted_checkpoint']==str(args.checkpoint) and state['status']=='failed'


def test_missing_actual_tire_geometry_retains_source_and_completed_assessments(monkeypatch,tmp_path):
    args,calls=setup_trial(monkeypatch,tmp_path)
    def reject(*unused): raise ValueError('Actual tire-actuation geometry is missing')
    monkeypatch.setattr(trial,'audit_tires',reject)
    with pytest.raises(ValueError,match='geometry is missing'):
        trial.run_trial(args)
    state=json.loads((args.output_dir/'status.json').read_text())
    assert len(calls)==2 and state['last_accepted_checkpoint']==str(args.checkpoint)
    assert (args.output_dir/'assessment.json').is_file() and (args.output_dir/'motion_assessment.json').is_file()
    assert state['status']=='failed' and not state['deployment_ready']


def test_unqualified_conversion_source_cannot_start_training(monkeypatch,tmp_path):
    args,calls=setup_trial(monkeypatch,tmp_path)
    def reject(*unused): raise ValueError('source has not passed full paired physics acceptance')
    monkeypatch.setattr(trial,'build',reject)
    with pytest.raises(ValueError,match='full paired physics acceptance'):
        trial.run_trial(args)
    assert not calls
    assert json.loads((args.output_dir/'status.json').read_text())['last_accepted_checkpoint']==str(args.checkpoint)


def test_dry_run_does_not_convert_or_claim_acceptance(monkeypatch,tmp_path,capsys):
    calls=[];monkeypatch.setattr(trial,'run_trial',lambda *unused:calls.append(True))
    assert trial.main(['--checkpoint','source.pt','--qualified-evaluation-dir','evaluation',
                       '--split-router','router.pt','--reference-replay','replay.npz',
                       '--output-dir',str(tmp_path/'unused')])==0
    assert not calls and not (tmp_path/'unused').exists()
    output=capsys.readouterr().out
    assert 'no source acceptance' in output and 'run_gui.sh' in output and '--run' in output


def test_specialist_training_exposure_does_not_change_five_grade_benchmark(monkeypatch,tmp_path):
    args,calls=setup_trial(monkeypatch,tmp_path)
    args.training_grade=20.
    assert trial.run_trial(args)==0
    train,benchmark=(command for _,command in calls)
    assert 'env.terrain.terrain_generator.sub_terrains.periodic_slope.angle_choices=[20.0]' in train
    assert 'env.terrain.terrain_generator.sub_terrains.periodic_slope.angle_range=[20.0,20.0]' in train
    assert benchmark[benchmark.index('--grades')+1:benchmark.index('--grades')+6]==['0','5','10','17','20']
    assert not any(part.startswith('env.') for part in benchmark)
    state=json.loads((args.output_dir/'status.json').read_text())
    assert state['training_exposure']['grade_deg']==20.


def test_wrong_training_grade_cannot_train_expert_three(monkeypatch,tmp_path):
    args,calls=setup_trial(monkeypatch,tmp_path)
    args.training_grade=17.
    with pytest.raises(ValueError,match='20-degree training grade'):
        trial.run_trial(args)
    assert not calls
    assert json.loads((args.output_dir/'status.json').read_text())['last_accepted_checkpoint']==str(args.checkpoint)
