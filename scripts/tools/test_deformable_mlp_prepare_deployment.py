"""Staging must preserve unrelated work and reject incomplete long training."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parent))
import deformable_mlp_prepare_deployment as module
from deformable_mlp_prepare_deployment import replace_node_fields, prepare, run_logged, validate_joint_bindings, audit_training


def test_targeted_config_patch_preserves_other_nodes_and_comments():
    original = "# operator notes\nrl_bridge:\n  ros__parameters:\n    rl_obs_size: 256 # H8\n    history_length: 8\n\n# concurrent gimbal tuning\ngimbal_controller:\n  ros__parameters:\n    gain: 12.3\n"
    changed = replace_node_fields(original,"rl_bridge",dict(rl_obs_size=160,history_length=5))
    parsed = yaml.safe_load(changed)
    assert parsed["rl_bridge"]["ros__parameters"] == dict(rl_obs_size=160,history_length=5)
    assert changed.split("# concurrent gimbal tuning")[1] == original.split("# concurrent gimbal tuning")[1]
    assert changed.startswith("# operator notes\n")
    with pytest.raises(ValueError,match="found 0"):
        replace_node_fields(original,"rl_bridge",dict(nonexistent=5))


@pytest.mark.parametrize("status,updates,qualified",[("running",10100,True),("completed",100,True),("completed",10100,False)])
def test_release_cannot_bypass_training_and_all_stage_admission(tmp_path,status,updates,qualified):
    training = tmp_path/"training"
    training.mkdir()
    (training/"status.json").write_text(json.dumps(dict(status=status,requested_long_updates=10000,
        total_training_updates=updates,stages=[dict(qualified=qualified)])))
    output = tmp_path/"release"
    with pytest.raises(ValueError,match="fully qualified"):
        prepare(training,output,tmp_path/"rmcs")
    assert not output.exists()


def test_negative_contract_check_cannot_count_cli_errors_as_success(tmp_path):
    path = tmp_path/"negative.log"
    with pytest.raises(RuntimeError,match="outside contract"):
        run_logged([sys.executable,"-c","raise SystemExit(2)"],path,expected_success=False)
    run_logged([sys.executable,"-c","print('== SUMMARY: FAIL =='); raise SystemExit(1)"],
               path,expected_success=False)


def test_semantic_names_cannot_hide_wrong_physical_bindings():
    import copy
    joints=["right_front_joint","left_front_joint","left_back_joint","right_back_joint"]
    terms=[f"index={index} type={kind} joints={','.join(joints)} name=unchanged_semantic_name"
           for index,kind in ((10,"joint_pos"),(14,"joint_vel"))]
    terms += [f"index={18+i} path=/chassis/{joint}/feedback_current_raw name=current_{i}"
              for i,joint in enumerate(joints)]
    terms += [f"index={26+i} path=/chassis/{joint.replace('_joint','_wheel')}/rl_velocity name=wheel_{i}"
              for i,joint in enumerate(joints)]
    config=dict(rl_bridge=dict(ros__parameters=dict(joint_names=joints,joint_base_path="/chassis",
        joint_angle_suffix="/urdf_angle",joint_velocity_suffix="/urdf_velocity",observation_terms=terms)))
    validate_joint_bindings(yaml.safe_dump(config))
    for index in (0,1):
        bad=copy.deepcopy(config);bad["rl_bridge"]["ros__parameters"]["observation_terms"][index]=terms[index].replace(
            "joints=right_front_joint,left_front_joint","joints=left_front_joint,right_front_joint")
        with pytest.raises(ValueError,match="absolute order"):
            validate_joint_bindings(yaml.safe_dump(bad))
    bad=copy.deepcopy(config);bad["rl_bridge"]["ros__parameters"]["observation_terms"][2]=terms[2].replace(
        "/right_front_joint/","/left_front_joint/")
    with pytest.raises(ValueError,match="Current bindings"):
        validate_joint_bindings(yaml.safe_dump(bad))


def completed_state():
    return dict(status="completed",requested_long_updates=10000,total_training_updates=10100,
                last_accepted_checkpoint="long.pt",stages=[
                    dict(name="preflight",updates=100,qualified=True,checkpoint="short.pt"),
                    dict(name="long_01",updates=10000,qualified=True,checkpoint="long.pt")],jobs=[
                    dict(name="train_preflight",status="completed",command=["--checkpoint","source.pt","--max_iterations","100"]),
                    dict(name="train_long_01",status="completed",command=["--checkpoint","short.pt","--max_iterations","10000"])])


def test_release_rechecks_every_segment_and_its_actual_iteration():
    with patch.object(module,"qualify",return_value=dict(transformer_level_recovered=True)) as qualify, \
         patch.object(module,"actor_update_info",return_value=dict(policy_updated=True)), \
         patch.object(module,"checkpoint_info",side_effect=[dict(iteration=99),dict(iteration=9999)]):
        checkpoint,assessment,_=audit_training(completed_state(),Path("training"),Path("baseline"))
    assert checkpoint == Path("long.pt") and assessment["transformer_level_recovered"]
    assert [call.args[0] for call in qualify.call_args_list] == [Path("short.pt"),Path("long.pt")]
    with patch.object(module,"qualify",return_value=dict(transformer_level_recovered=False)), \
         patch.object(module,"actor_update_info",return_value=dict(policy_updated=True)), \
         patch.object(module,"checkpoint_info") as info:
        with pytest.raises(ValueError,match="no longer"):
            audit_training(completed_state(),Path("training"),Path("baseline"))
        info.assert_not_called()
    with patch.object(module,"qualify",return_value=dict(transformer_level_recovered=True)), \
         patch.object(module,"actor_update_info",return_value=dict(policy_updated=True)), \
         patch.object(module,"checkpoint_info",return_value=dict(iteration=49)):
        with pytest.raises(ValueError,match="claimed PPO updates"):
            audit_training(completed_state(),Path("training"),Path("baseline"))

    with patch.object(module,"actor_update_info",return_value=dict(policy_updated=False)):
        with pytest.raises(ValueError,match="unchanged actors"):
            audit_training(completed_state(),Path("training"),Path("baseline"))


def test_release_rejects_counter_only_claims_before_any_contract_work():
    state=completed_state();state["stages"][1]["updates"]=1000
    with patch.object(module,"qualify") as qualify:
        with pytest.raises(ValueError,match="consistent sequence"):
            audit_training(state,Path("training"),Path("baseline"))
        qualify.assert_not_called()
