"""Bounded Isaac integration probe; results are not a training acceptance test."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for package in ('agent_world', 'agent_tasks', 'agent_rl'):
    sys.path.insert(0, str(ROOT / 'source' / package))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Robotics-Deformable-Suspension-BestEffort-Precision-Real2Sim-v2')
parser.add_argument('--grade-deg', type=float, default=20.)
parser.add_argument('--steps', type=int, default=200)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.steps <= 1000 or not 1 <= args.num_envs <= 16:
    parser.error('probe requires 1..1000 steps and 1..16 environments')
app = AppLauncher(args).app
try:
    import torch
    import gymnasium as gym
    import agent_tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from scripts.tools.deformable_suspension_eval import configure_grade

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 42
    cfg.external_cmd_override = True
    cfg.spawn_phase_stratify = False
    configure_grade(cfg, args.grade_deg)
    env = gym.make(args.task, cfg=cfg)
    try:
        u = env.unwrapped
        u.scene.env_origins[:, 0] = 90. - u._profile_x_offset
        u.scene.env_origins[:, 1] = 3. * torch.arange(u.num_envs, device=u.device)
        u.scene.env_origins[:, 2] = 0.
        obs, _ = env.reset(seed=42)
        initial_q = u.robot.data.joint_pos[:, u._legs_idx].clone()
        count = 0
        samples = []
        for step in range(args.steps):
            obs, reward, terminated, timeout, _ = env.step(torch.zeros(u.num_envs, 4, device=u.device))
            assert all(torch.isfinite(value).all() for value in obs.values()), 'non-finite observations'
            assert torch.isfinite(reward).all(), 'non-finite rewards'
            count += int((terminated | timeout).sum())
            q = u.robot.data.joint_pos[:, u._legs_idx]
            tilt = torch.rad2deg(torch.atan2(u.robot.data.projected_gravity_b[:, :2].norm(dim=-1),
                                          -u.robot.data.projected_gravity_b[:, 2]))
            samples.append([float(tilt.mean()), float((q-u.leg_target).abs().mean()),
                            float((u.wheel_normal_forces > cfg.wheel_contact_force_threshold).all(-1).float().mean()),
                            float(u.robot.data.applied_torque[:, u._legs_idx].abs().max())])
        result = {'purpose': 'integration_probe_zero_action_not_policy_acceptance',
                  'task': args.task, 'grade_deg': args.grade_deg, 'steps': args.steps,
                  'num_envs': u.num_envs, 'resets': count, 'initial_q': initial_q.tolist(),
                  'q_cmd': u.q_cmd.tolist(),
                  'mean_after_settling': torch.tensor(samples[50:]).mean(0).tolist() if len(samples) > 50 else None,
                  'sample_columns': ['tilt_deg','target_error_rad','all_contact_fraction','max_leg_effort_nm'],
                  'final_sample': samples[-1], 'finite_observations': True,
                  'real2sim_observation_version': getattr(cfg, 'real2sim_observation_version', None)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print('SMOKE_RESULT '+json.dumps(result), flush=True)
    finally:
        env.close()
finally:
    app.close()
