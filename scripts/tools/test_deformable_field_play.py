"""Input persistence, camera geometry and real collision-surface queries."""

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controls = load("third_person_state_test", "scripts/rsl_rl/deformable_third_person_state.py")
surface = load("field_surface_test", "source/agent_tasks/agent_tasks/direct/deformable_suspension/field_surface.py")


def test_mouse_turns_physical_heading_only_with_right_button_and_release_keeps_target():
    state = controls.ThirdPersonState()
    state.mouse_move(100., 10.)
    assert state.target_yaw == 0. and state.pitch_deg == 20.
    state.right_button(True)
    state.mouse_move(100., 10.)
    assert state.target_yaw == pytest.approx(math.radians(-15.))
    assert state.command(0.)[2] < 0.
    state.right_button(False)
    state.mouse_move(-500., 10.)
    assert state.target_yaw == pytest.approx(math.radians(-15.))
    assert state.command(0.)[2] < 0.
    assert state.command(state.target_yaw)[2] == 0.


def test_wrap_crossing_does_not_command_a_full_rotation():
    state = controls.ThirdPersonState(target_yaw=math.radians(-179.))
    assert 0. < state.command(math.radians(179.))[2] < .2


def test_keys_cancel_and_release_preserves_other_held_keys():
    state = controls.ThirdPersonState()
    state.key("W", True, 0.)
    state.key("S", True, 0.)
    assert state.command(0.) == (0., 0., 0.)
    state.key("S", False, 0.)
    assert state.command(0.)[0] == .8
    state.key("A", True, 0.)
    vx, vy, _ = state.command(0.)
    assert vx > 0 and vy > 0 and math.hypot(vx/.8, vy/.5) == pytest.approx(1.)
    state.key("X", True, 0.)
    assert state.command(0.)[2] == 0.


@pytest.mark.parametrize("key", ["L", "ESCAPE", "R"])
def test_stop_and_reset_clear_pending_turn_and_keys(key):
    state = controls.ThirdPersonState(target_yaw=1.)
    state.pressed.add("W")
    state.right_button(True)
    state.key(key, True, .2)
    assert state.command(.2) == (0., 0., 0.)
    assert not state.right_held
    assert state.reset_requested == (key == "R")


def test_pitch_zoom_limits_and_camera_follows_actual_heading():
    state = controls.ThirdPersonState(target_yaw=0.)
    state.right_button(True)
    state.mouse_move(0., -10000.)
    assert state.pitch_deg == 75.
    state.mouse_move(0., 10000.)
    assert state.pitch_deg == 5.
    state.scroll(100.)
    assert state.distance == 1.
    state.scroll(-100.)
    assert state.distance == 8.
    eye, pivot = state.camera_pose((0., 0., 0.), math.pi/2)
    assert abs(eye[0]-pivot[0]) < 1.e-6 and eye[1] < pivot[1] and eye[2] > pivot[2]
    eye, pivot = state.camera_pose((0., 0., 0.), 0., lambda *args: .6)
    assert np.linalg.norm(eye-pivot) == pytest.approx(.45)


def plane(slope=.2):
    vertices = np.array([[-2.,-2.,-.4], [2.,-2.,.4], [2.,2.,.4], [-2.,2.,-.4]])
    return surface.FieldSurface(vertices, [[0,1,2],[0,2,3]])


def test_static_surface_height_normal_no_hit_and_tick_cache():
    ground = plane()
    points = torch.tensor([[[.5,.3,1.], [4.,4.,1.]]])
    height, normals, valid = ground.support(points, tick=1)
    assert height[0,0] == pytest.approx(.1,abs=1.e-6)
    assert normals[0,0].numpy() == pytest.approx(np.array([-.2,0.,1.])/math.sqrt(1.04),abs=1.e-6)
    assert valid.tolist() == [[True,False]]
    assert torch.isfinite(height).all() and torch.isfinite(normals).all()
    ground.support(points.clone(), tick=1)
    assert ground.query_count == 1 and ground.cache_hits == 1
    ground.support(points, tick=2)
    assert ground.query_count == 2
    assert ground.segment((0.,0.,1.), (0.,0.,-1.), 2.) == pytest.approx(1.,abs=1.e-6)


def test_asset_contract_and_missing_external_entry(tmp_path):
    package = dict(units="metres",usd_up_axis="Z",entries={"isaac_static_collision_usd":"scene.usda"})
    (tmp_path/"asset_package.json").write_text(json.dumps(package))
    with pytest.raises(FileNotFoundError):
        surface.resolve_field_asset(tmp_path)
    (tmp_path/"scene.usda").write_text("#usda 1.0\n")
    _, entry, digest = surface.resolve_field_asset(tmp_path)
    assert entry == tmp_path/"scene.usda" and len(digest) == 64
    package["entries"]["isaac_static_collision_usd"] = "../outside.usda"
    (tmp_path/"asset_package.json").write_text(json.dumps(package))
    with pytest.raises(ValueError,match="inside"):
        surface.resolve_field_asset(tmp_path)
    package["units"] = "millimetres"
    (tmp_path/"asset_package.json").write_text(json.dumps(package))
    with pytest.raises(ValueError,match="metres"):
        surface.resolve_field_asset(tmp_path)
