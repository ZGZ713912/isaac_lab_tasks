"""Exercise production termination decisions without launching Isaac Sim."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "src/agent_tasks/direct/deformable_suspension"


def load_function(path, name, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body if class_name is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, "math": math, "du": SimpleNamespace(BODY_TOP_OFFSET=.1)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


get_dones = load_function(ENV_DIR / "env.py", "_get_dones", "DeformableSuspensionEnv")
configure_play = load_function(ROOT / "scripts/rsl_rl/play_deformable.py", "_configure_play_termination")


def make_env(n=5):
    # Read the real training default: a play override must never change it.
    config_tree = ast.parse((ENV_DIR / "env_cfg.py").read_text())
    enabled = next(node.value for node in ast.walk(config_tree) if isinstance(node, ast.AnnAssign)
                   and isinstance(node.target, ast.Name) and node.target.id == "chassis_ground_termination_enabled")
    cfg = SimpleNamespace(chassis_ground_termination_enabled=ast.literal_eval(enabled),
        termination_roll_deg=30., termination_pitch_deg=30., terminate_base_height_low=.004,
        terminate_chassis_clearance=0., terminate_body_top=None,
        training_progress_steps_per_iteration=48, base_contact_death_after_iterations=2000,
        boundary_reset_enabled=True, boundary_reset_margin=.5, scene=SimpleNamespace(env_spacing=4.))
    data = SimpleNamespace(projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
        joint_pos=torch.zeros(n, 4), root_lin_vel_b=torch.zeros(n, 3), root_ang_vel_b=torch.zeros(n, 3),
        root_pos_w=torch.zeros(n, 3))
    return SimpleNamespace(cfg=cfg, robot=SimpleNamespace(data=data), num_envs=n, device="cpu",
        contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(n, 1, 3))),
        _base_contact_idx=[0], common_step_counter=2000*48, base_height=torch.full((n,), .15),
        chassis_clearance=torch.full((n,), .01), episode_length_buf=torch.zeros(n, dtype=torch.long),
        max_episode_length=4000, _periodic=False, scene=SimpleNamespace(env_origins=torch.zeros(n, 3)))


@pytest.mark.parametrize("play", [False, True])
def test_contact_clearance_and_low_height_do_not_reset_play_but_still_terminate_training(play):
    env = make_env()
    env.contact_sensor.data.net_forces_w[[0, 3], 0, 2] = 100.
    env.chassis_clearance[[1, 3]] = -.01
    env.base_height[[2, 3]] = -.01
    if play:
        configure_play(env.cfg)
    terminated, timeout = get_dones(env)
    assert terminated.tolist() == ([False]*5 if play else [True, True, True, True, False])
    assert not timeout.any()
    # Contact telemetry remains available even when play permits grounding.
    assert env._base_contact.tolist() == [True, False, False, True, False]


@pytest.mark.parametrize("iteration,dead", [(1999, False), (2000, True)])
def test_training_contact_curriculum_is_preserved(iteration, dead):
    env = make_env(1)
    env.common_step_counter = iteration*48
    env.contact_sensor.data.net_forces_w[:, 0, 2] = 100.
    terminated, _ = get_dones(env)
    assert terminated.item() is dead


def test_other_termination_and_timeout_decisions_remain_effective_in_play():
    env = make_env(8)
    configure_play(env.cfg)
    env.contact_sensor.data.net_forces_w[:, 0, 2] = 100.
    env.chassis_clearance.fill_(-.01)
    env.robot.data.projected_gravity_b[1, 1] = .8
    env.robot.data.projected_gravity_b[2, 0] = .8
    env.robot.data.joint_pos[3, 0] = float("nan")
    env.robot.data.root_lin_vel_b[4, 0] = float("inf")
    env.robot.data.root_ang_vel_b[5, 0] = float("nan")
    env.cfg.terminate_body_top = .3
    env.base_height[6] = .25
    env.episode_length_buf[7] = env.max_episode_length
    terminated, timeout = get_dones(env)
    assert terminated.tolist() == [False, True, True, True, True, True, True, False]
    assert timeout.tolist() == [False]*7 + [True]


def test_play_can_drive_outside_training_cells_without_reset():
    env = make_env(1)
    env._periodic = True
    env.robot.data.root_pos_w[:, 0] = 20.
    assert get_dones(env)[1].item()
    configure_play(env.cfg)
    assert not get_dones(env)[1].item()
