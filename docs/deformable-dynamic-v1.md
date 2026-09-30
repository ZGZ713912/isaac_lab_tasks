# Deformable Dynamic Suspension V1

## Contract

- Policy: 8 frames, oldest first, 32 features per frame (256 total).
- Frame: baseline q(1), body command vx/vy/wz(3, yaw scaled by 0.25), gyro(3, x0.5), gravity(3), leg q(4), leg velocity(4, x0.1), commanded/applied leg effort(4, x0.05), previous action(4), wheel encoder speed(4, x0.05), encoder-derived vx/vy(2).
- Critic: current frame(32), true body velocity(3), highest-wheel-ground-relative body top(1), wheel normal forces(4).
- Transformer: 5 spatial tokens per frame plus learned time embedding; current-frame tokens output four actions. History is updated once per policy step and refilled on reset.
- Action: q_target = clamp(q_baseline - Q_LOW * max(-action, 0), 0, q_baseline). Zero action holds the low reference; negative action extends a corner. This direction is the simulator convention, not an assertion about hardware encoder minangle.
- Physics: 200 Hz, policy: 100 Hz. Cascade PI integrals use physics dt. V1 adds nominal gravity-load feedforward; deployment must reproduce the controller or re-identify gains before transfer.
- Height: transformed body mesh maximum world z minus highest terrain height under the four wheel contact locations. Airborne wheels do not raise the reference. The training penalty starts at 255 mm; termination is at 280 mm after 0.5 s settling. **255 mm is not a hard safety guarantee.**
- Clearance: conservative 7x7 samples over the transformed bottom bounding rectangle. This is not an exact mesh-to-terrain distance.

## Dynamics

The sphere handles normal collision only. Wheel sphere material friction is zeroed at startup. Tire longitudinal force comes from encoder rim speed minus wheel-set contact-point velocity, bounded with transverse roller drag by a per-wheel friction circle. Actual wheel joints receive bounded motor effort. External tire wrenches are expressed at each wheel COM in its current local frame, including contact-point moment. No force or torque is applied to the base.

The COM conversion avoids stale global link-pose caching in the installed Isaac Lab 2.3.2 permanent wrench composer. Ground height/normals currently support planes and the existing periodic x-profile terrain only. Do not use arbitrary rough meshes without replacing this ground-query implementation.

Tire friction (0.5-0.9), slip stiffness, transverse drag, encoder/IMU noise and 0-2 step delays are provisional identification parameters. Encoder odometry is deliberately not ground truth; sensor noise alone does not replace real slip validation. IMU bias is sampled per episode; long-term drift and actuator transport delay remain future identification work.

## Tasks

- `Robotics-Deformable-Suspension-Flat-History-Transformer-v1`
- `Robotics-Deformable-Suspension-Rough-History-Transformer-v1` (2-8 degree periodic slopes)
- `Robotics-Deformable-Suspension-Rough-Steep-History-Transformer-v1` (8-17 degree slopes)
- `Robotics-Deformable-Suspension-Rough-Keyboard-Play-History-Transformer-v1`

Training commands progress from standing through translation and spin to combined motion over 4000 iterations. Both spin signs and exact full-spin samples are included. Translation is world-frame in training, body-frame for keyboard play. Terrain stages are separate tasks, not an automatic slope curriculum.

## Run

Activate the `isaaclab` environment first. Start fresh: V0 checkpoints cannot load into V1.

```bash
python scripts/rsl_rl/train.py --task=Robotics-Deformable-Suspension-Flat-History-Transformer-v1 --num_envs=128 --max_iterations=10000 --headless --device=cuda:0
python scripts/rsl_rl/train.py --task=Robotics-Deformable-Suspension-Rough-History-Transformer-v1 --num_envs=128 --max_iterations=10000 --headless --device=cuda:0
./run_gui.sh python scripts/rsl_rl/play_deformable.py --task=Robotics-Deformable-Suspension-Rough-Keyboard-Play-History-Transformer-v1 --checkpoint=<v1-checkpoint> --device=cuda:0 --vx_max=1 --vy_max=1
python scripts/tools/deformable_dynamic_probe.py --device=cuda:0 --steps=600 --strict
python -m pytest scripts/tools/test_deformable_dynamic.py -q
```

For 16-frame ablations set all three together: `env.policy_history_length=16 env.observation_space=512 agent.policy.history_length=16`. For single-frame ablations use `1`, `32`, `1` respectively. Changed time embeddings require a fresh model.

Episode metrics are logged as `dynamic/*`: simultaneous contact, tilt squared, 255 mm violation fraction, mean per-step minimum clearance, velocity/yaw error, slip and leg torque saturation. Their first 0.5 s is excluded. These metrics are episode means, not P95 or worst-case safety bounds. The existing `deformable_eval_tilt.py` also now measures simultaneous normal-load contact instead of average per-wheel force magnitude.

## Verification (2026-09-30)

Three CPU unit tests passed (geometry/odometry, airborne friction limit, temporal gradients and TorchScript tracing). GPU flat and rough PPO smoke runs completed. Fixed-low-pose plane tests with four robots, 600 steps per command, passed history/reset and airborne-zero-traction assertions:

| Command | Measured late-window mean | Four-wheel contact | Body top | Minimum sampled clearance |
| --- | --- | --- | --- | --- |
| Standing | near zero | 100% | 242.7 mm | 9.7 mm |
| vx=1 m/s | vx=0.895 m/s | 100% | 242.7 mm | 9.7 mm |
| wz=+2pi rad/s | +6.28276 rad/s | 100% | 242.8 mm | 9.8 mm |
| wz=-2pi rad/s | -6.28283 rad/s | 100% | 242.8 mm | 9.8 mm |
| vx=0.5, wz=+2pi | vx=0.444, vy=0.044, wz=6.28276 | 100% | 242.8 mm | 9.8 mm |

These are physical motion sanity checks, **not trained slope suspension results**. No long training, real-robot validation, arbitrary terrain validation or 2 m/s benchmark has been completed. Existing V0 dynamics remain legacy; only the shared PID timestep bug was corrected there.
