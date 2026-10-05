"""CPU mock tests of the production reset method, without Isaac Sim imports."""

import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension"


def quat_apply(quat, vector):
    twice = 2 * torch.cross(quat[..., 1:], vector, dim=-1)
    return vector + quat[..., :1] * twice + torch.cross(quat[..., 1:], twice, dim=-1)


def load_reset():
    terrains = ModuleType("agent_world.terrains")
    terrains.periodic_slope_angle = lambda *args: 0.0
    spec = importlib.util.spec_from_file_location("reset_test_utils", ENV_DIR / "cfg_utils.py")
    du = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"agent_world.terrains": terrains}):
        spec.loader.exec_module(du)
    tree = ast.parse((ENV_DIR / "dynamic_env.py").read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    reset = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "_reset_idx")

    class Base:
        def _reset_idx(self, ids):
            self.extras = {"log": {}}
            self.robot.data.joint_pos[ids, :4] = self.q_cmd[ids, None]
            self.leg_target[ids] = self.q_cmd[ids, None]
            self.actions[ids] = 0
            self.last_actions[ids] = 0
            self._prev2_actions[ids] = 0

    namespace = {"torch": torch, "du": du, "quat_apply": quat_apply, "Base": Base}
    node = ast.ClassDef(name="ResetEnv", bases=[ast.Name(id="Base", ctx=ast.Load())],
                        keywords=[], body=[reset], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(ENV_DIR / "dynamic_env.py"), "exec"), namespace)
    return namespace["ResetEnv"], du


def mock_env(x, yaw, threshold=0.005, periodic=True, buffer=0.0):
    cls, du = load_reset()
    env = cls()
    n = len(x)
    env.device = "cpu"
    env.cfg = SimpleNamespace(leg_target_upper_limit=1.0563, chassis_ground_threshold=threshold,
                              reset_height_buffer=buffer, leg_action_scale=0.35,
                              tire_friction_range=(0.5, 0.8), encoder_bias_std=0., gyro_bias_std=0.,
                              max_sensor_delay_steps=0)
    env._periodic = periodic
    env._period_seg = 1.0
    env._profile_x_offset = 4.0
    env._slope_angle_table = torch.tensor([5., -5., 3., -2., 5.])
    env._bottom_samples = torch.cartesian_prod(torch.linspace(-.12, .12, 7), torch.linspace(-.12, .12, 7))
    env._bottom_samples = torch.cat((env._bottom_samples, torch.full((49, 1), -.01)), -1)
    env.q_cmd = torch.full((n,), 1.01229)
    env._legs_idx = list(range(4))
    names = list(du.ORDERED_LEG_JOINT_NAMES + du.ORDERED_WS_JOINT_NAMES + du.ORDERED_UPPER_LEG_JOINT_NAMES)
    data = SimpleNamespace(joint_pos=torch.zeros(n, 12), root_link_pos_w=torch.zeros(n, 3),
                           root_link_quat_w=torch.zeros(n, 4))
    data.root_link_pos_w[:, 0] = torch.tensor(x)
    data.root_link_quat_w[:, 0] = torch.cos(torch.tensor(yaw) / 2)
    data.root_link_quat_w[:, 3] = torch.sin(torch.tensor(yaw) / 2)

    def write_joints(joints, velocity, env_ids):
        assert velocity.count_nonzero() == 0
        data.joint_pos[env_ids] = joints

    def write_pose(pose, env_ids):
        data.root_link_pos_w[env_ids] = pose[:, :3]
        data.root_link_quat_w[env_ids] = pose[:, 3:]

    env.robot = SimpleNamespace(data=data, joint_names=names, write_joint_state_to_sim=write_joints,
                                write_root_pose_to_sim=write_pose,
                                set_external_force_and_torque=lambda *args, **kwargs: None)
    env.leg_target = torch.zeros(n, 4)
    env.actions = torch.zeros(n, 4)
    env.last_actions = torch.zeros(n, 4)
    env._prev2_actions = torch.zeros(n, 4)
    env._metric_names = ("mock",)
    env._metrics = torch.zeros(n, 1)
    env._metric_steps = torch.zeros(n)
    env._wheel_body_ids = [0, 1, 2, 3]
    env._wheel_drive = SimpleNamespace(reset=lambda ids: None)
    env._leg_adrc = SimpleNamespace(reset=lambda ids, q, target: torch.testing.assert_close(q, target))
    env._leg_actuator = None
    for name, width in (("_drive_command", 3), ("_last_wheel_slip", 4), ("_tire_deflection", 4),
                        ("_friction", 1), ("_encoder_bias", 4), ("_gyro_bias", 3)):
        setattr(env, name, torch.zeros(n, width))
    env._delay = torch.zeros(n, dtype=torch.long)
    env._history_valid = torch.zeros(n, dtype=torch.bool)
    env._history = torch.zeros(n, 1, 32)
    env._sensor_fifo = torch.zeros(n, 1, 32)
    # Subset resets must never call all-environment broadcasting helpers.
    env._ground_height = lambda *args: pytest.fail("all-env ground helper used")
    return env, du


def test_best_effort_infeasible_reset_starts_tangent_to_slope():
    env, du = mock_env([0.5, 4.5], [0.0, 1.0])
    env.cfg.best_effort_leveling = True
    env._slope_angle_table.fill_(15.0)
    env._reset_idx(torch.tensor([0, 1]))
    assert env.extras["log"]["dynamic/reset_infeasible_fraction"] == 1.0
    quat = env.robot.data.root_link_quat_w
    up = quat_apply(quat, torch.tensor([[0., 0., 1.]]).expand(2, -1))
    expected = torch.tensor([-math.sin(math.radians(15)), 0., math.cos(math.radians(15))])
    torch.testing.assert_close(up, expected.expand(2, -1), atol=1e-5, rtol=1e-5)
    assert env.extras["log"]["dynamic/reset_penetration_max"] < 1e-5
    assert env.actions.abs().max() <= 1


def geometry(env, du, ids):
    q = env.robot.data.joint_pos[ids, :4]
    centers, _ = du.wheel_geometry(q)
    world = env.robot.data.root_link_pos_w[ids, None] + quat_apply(
        env.robot.data.root_link_quat_w[ids, None].expand(-1, 4, -1), centers)
    x = world[..., 0] + env._profile_x_offset
    ground = du.periodic_slope_height_torch(x, env._period_seg, env._slope_angle_table)
    # Independent one-sided derivative, not the production normal computation.
    eps = 1.e-4
    gradient = (du.periodic_slope_height_torch(x + eps, env._period_seg, env._slope_angle_table) - ground) / eps
    gap = world[..., 2] - ground - du.WHEEL_RADIUS * torch.sqrt(1 + gradient.square())
    bottom = env.robot.data.root_link_pos_w[ids, None] + quat_apply(
        env.robot.data.root_link_quat_w[ids, None].expand(-1, 49, -1),
        env._bottom_samples[None].expand(len(ids), -1, -1))
    clearance = bottom[..., 2] - du.periodic_slope_height_torch(
        bottom[..., 0] + env._profile_x_offset, env._period_seg, env._slope_angle_table)
    return q, gap, clearance


def test_subset_slopes_yaw_and_sphere_contacts():
    env, du = mock_env([.5, -1.5, 4.5, 6.5, 9.5], [0., .4, 1.2, 2.3, math.pi])
    ids = torch.tensor([0, 2, 3])
    untouched = env.robot.data.root_link_pos_w[[1, 4]].clone()
    env._reset_idx(ids)
    q, gap, clearance = geometry(env, du, ids)
    assert (q >= 0).all() and (q <= env.cfg.leg_target_upper_limit).all()
    assert gap.abs().max() < 3.e-6
    assert clearance.min() >= env.cfg.chassis_ground_threshold
    assert env.extras["log"]["dynamic/reset_infeasible_fraction"] == 0
    torch.testing.assert_close(env.robot.data.root_link_pos_w[[1, 4]], untouched)
    torch.testing.assert_close(env.robot.data.joint_pos[ids, 4:8], q)
    torch.testing.assert_close(env.robot.data.joint_pos[ids, 8:12], -q)
    torch.testing.assert_close(du.suspension_target(env.actions[ids], env.q_cmd[ids, None],
                                                 env.cfg.leg_target_upper_limit), q)
    assert env.actions[ids].abs().max() <= 1.0
    torch.testing.assert_close(env.last_actions[ids], env.actions[ids])
    torch.testing.assert_close(env._prev2_actions[ids], env.actions[ids])


@pytest.mark.parametrize("threshold", [.04, .3])
def test_clearance_constraint_and_infeasible_fallback(threshold):
    env, du = mock_env([.5, 2.5], [.2, 1.3], threshold=threshold, buffer=.01)
    ids = torch.arange(2)
    env._reset_idx(ids)
    _, gap, clearance = geometry(env, du, ids)
    assert gap.min() >= .01 - 3.e-6
    assert clearance.min() >= threshold
    assert env.extras["log"]["dynamic/reset_penetration_max"] == 0
    if threshold == .3:
        assert env.extras["log"]["dynamic/reset_infeasible_fraction"] == 1


def test_breakpoints_and_random_phases():
    torch.manual_seed(17)
    x = torch.cat((torch.arange(0., 12., .125), torch.rand(256) * 12))
    yaw = torch.rand(len(x)) * 2 * math.pi
    env, du = mock_env(x.tolist(), yaw.tolist())
    ids = torch.arange(len(x))
    env._reset_idx(ids)
    q, gap, clearance = geometry(env, du, ids)
    assert torch.isfinite(q).all()
    assert gap.min() > -4.e-6
    assert clearance.min() >= env.cfg.chassis_ground_threshold
    assert env.extras["log"]["dynamic/reset_penetration_max"] == 0


def test_flat_subset_and_empty_reset():
    env, _ = mock_env([0., 1., 2.], [0., .1, .2], periodic=False)
    env._reset_idx(torch.tensor([1]))
    torch.testing.assert_close(env.leg_target[1], env.q_cmd[1].expand(4))
    before = env.robot.data.joint_pos.clone()
    env._reset_idx(torch.empty(0, dtype=torch.long))
    torch.testing.assert_close(env.robot.data.joint_pos, before)
