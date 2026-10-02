# Deformable Dynamic Suspension V1

## Contract

- Policy: 8 frames, oldest first, 32 features per frame (256 total).
- Frame: baseline q(1), body command vx/vy/wz(3, yaw scaled by 0.25), gyro(3, x0.5), gravity(3), leg q(4), leg velocity(4, x0.1), commanded/applied leg effort(4, x0.05), previous action(4), wheel encoder speed(4, x0.05), encoder-derived vx/vy(2).
- Critic: current frame(32), true body velocity(3), highest-wheel-ground-relative body top(1), wheel normal forces(4).
- Transformer: 5 spatial tokens per frame plus learned time embedding; current-frame tokens output four actions. History is updated once per policy step and refilled on reset.
- Action: q_target = clamp(q_baseline - Q_LOW * max(-action, 0), 0, q_baseline). Zero action holds the low reference; negative action extends a corner. This direction is the simulator convention, not an assertion about hardware encoder minangle.
- Physics and leg ADRC: 1000 Hz, policy: 100 Hz (decimation 10). Legs use the RMCS TD + ESO + NLESF implementation in `adrc.py`; the previous cascade PI and nominal-load feedforward are no longer executed by V1. The current wheel drive uses bounded PI, rotor-inertia feedforward and command acceleration limits; see [low-slip V2](deformable-low-slip-v2.md).
- Physical angle: `alpha = radians(maxangle) - q_urdf`, velocity `alpha_dot = -q_dot`. No q_max/span normalization is used. ADRC runs entirely in physical-angle radians; positive motor effort increases URDF q, consistent with physical-angle b0=-1. The configurable maxangle defaults to 75 degrees; Q_LOW remains the CAD-derived safe reference, not a claim that the hardware minangle calibration is final.
- The current minangle-residual experiment uses the hardware physical minimum angle of 17 degrees as `Q_MINANGLE = 75 degrees - 17 degrees = 1.01229 rad` in the direct URDF-q coordinate. `Q_LOW=1.0563` remains the measured chassis-safe reference; it must not be treated as the hardware normalized minangle.
- ADRC uses RMCS dt=td_h=0.001, td_r=50, eso_w0=250, auto-beta=(3*w0,3*w0^2,w0^3), k1=30, k2=17, alpha1=0.75, alpha2=0.7, delta=0.02, kt=1. Stability experiments set the simulation default b0=-10 and `adrc_feedback_applied_torque=True`. These are **simulation tuning, not identified hardware parameters**. Raw RMCS b0=-1/pre-saturation feedback remains reproducible with Hydra overrides `env.adrc_b0=-1 env.adrc_feedback_applied_torque=False`. Internal output is +/-200, actual motor effort +/-25 Nm; calibrated ESO feeds back the bounded commanded effort (not a measured torque sensor). TD/ESO states reset independently per environment.
- Height: transformed body mesh maximum world z minus highest terrain height under the four wheel contact locations. Airborne wheels do not raise the reference. The training penalty starts at 255 mm; termination is at 280 mm after 0.5 s settling. **255 mm is not a hard safety guarantee.**
- Clearance: conservative 7x7 samples over the transformed bottom bounding rectangle. This is not an exact mesh-to-terrain distance.

## Dynamics

The sphere handles normal collision only. Wheel sphere material friction is zeroed at startup. Tire longitudinal force combines a bounded elastic tread deflection with damping of encoder rim speed minus wheel-set contact-point velocity. This supports static slope loads without persistent sliding. Longitudinal force and transverse roller drag share a per-wheel friction circle. Actual wheel joints receive bounded motor effort. External tire wrenches are expressed at each wheel COM in its current local frame, including contact-point moment. No force or torque is applied to the base.

The COM conversion avoids stale global link-pose caching in the installed Isaac Lab 2.3.2 permanent wrench composer. Ground height/normals currently support planes and the existing periodic x-profile terrain only. Do not use arbitrary rough meshes without replacing this ground-query implementation.

Tire friction (currently 0.8-1.0), contact elasticity, slip damping, transverse drag, encoder/IMU noise and 0-2 step delays are simulator identification parameters. Encoder odometry is deliberately not ground truth; sensor noise alone does not replace real slip validation. IMU bias is sampled per episode; long-term drift and actuator transport delay remain future identification work.

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

## Export (2026-10-01)

`scripts/rsl_rl/export_onnx.py` rebuilds the actor from a checkpoint without loading Isaac. History transformers export rank-3 `[1, 8, 32]` per the RMCS convention; single-frame MLP/transformer checkpoints stay rank-2 `[1, obs]`, and `--flat` forces rank-2 as a fallback (deployment `auto` then infers `mlp`, not `transformer`).

```bash
python scripts/rsl_rl/export_onnx.py --checkpoint=logs/rsl_rl/deformable_dynamic_adrc_history_v1/<ts>/model_3999.pt
cp <ckpt-dir>/exported/policy.onnx <rmcs-ws>/rmcs_ws/src/rmcs_rl/models/deformable_v2_transformer.onnx
python <rmcs-ws>/rmcs_ws/src/rmcs_rl/tool/check_policy_contract.py <rmcs-model> --model-type transformer
```

Contract: single input `obs` float32 `(1, 8, 32)`, single output `actions` float32 `(1, 4)`, opset 18, weights inlined into one file, and **no `rmcs_*` metadata** — layout, `rmcs_model_type`, `rmcs_history_length` and `rmcs_obs_frame_size` are written by the RMCS stamp tool, which also gates the contract check for rank-3 inputs. The 8×32 tensor is a reshape of the flat 256 observation: rank-3 and rank-2 exports of the same checkpoint agree to 3.6e-7, and torch/ONNX/TorchScript agree to 2.4e-7.

`model_3999.pt` of run `2026-09-30_21-36-47` (Rough-Steep, 4000 iterations) exports to `model_id 0xe3314b6d0e387404`. Deployment-side work still outstanding: `deformable-infantry-omni-rl.yaml` keeps `rl_obs_size: 26` (needs 256) and no `history_length: 8`, `expected_model_id` still holds the previous model, and the file must be stamped before `policy_server` will load it.

## ADRC Verification (2026-09-30)

After stability tuning, the unchanged strict 600-step-per-command plane probe passes: no resets, 100% simultaneous contact, height 242.9 mm, clearance 9.9 mm, no late-window motor saturation. Standing q converges to approximately 1.056304 rad (target 1.0563). Positive/negative spin are +6.282873/-6.282949 rad/s; vx=1 achieves 0.89357 m/s; vx=0.5 plus spin achieves vx=0.44268, vy=0.04383, wz=6.282879. This validates fixed-reference plane dynamics, not slope policy acceptance.

A 64-env, 100-iteration, seed-42 static-slope run completed at `logs/rsl_rl/deformable_dynamic_adrc_history_v1/2026-09-30_17-27-17/model_99.pt`. Training reward improved from -223.96 to 937.23 and mean episode length from 18.67 to 257.28 policy steps. **The policy is not converged/accepted:** episodes remain short relative to 20 seconds, contacts vary, and final terrain occupancy is 76.6% flat. A separate 32-env/20-iteration accelerated-motion run also completed without sustained actuator saturation. Do not equate reward improvement with slope capability. Future comparisons must stratify slope/flat terrain and use deterministic evaluation.

Metric logging now excludes episodes with no samples after the 0.5-second settling interval, and reports `dynamic/valid_episode_fraction`; previously zero-filled short episodes diluted all dynamic averages. Runs above predate this logging correction.

### Historical Raw-Parameter Failure

Four CPU tests pass, including 300-step float64 parity against scalar formulas matching RMCS, output/motor saturation, and selective environment reset. GPU plane probes (180 steps per command) and a rough PPO smoke run (4 environments, 2 iterations) complete with finite observations. **Performance acceptance fails:** plane probe four-wheel contact rate was 0 at policy sampling instants, combined motion caused resets, and rough smoke episodes lasted about 9 steps with high effort saturation. These indicate model/controller/contact-timing mismatch requiring diagnosis, not a trained suspension result. No gains were silently retuned to recover the old PI scores.

Current low-slip runs use `deformable_dynamic_low_slip_history_v1`; the ADRC-only historical runs below used `deformable_dynamic_adrc_history_v1`. The initial ADRC smoke checkpoint was generated before that directory rename under `logs/rsl_rl/deformable_dynamic_history_v1/2026-09-30_17-03-33/`; it is debug-only. Use fresh training for the changed drive dynamics. RMCS itself has not been modified in these simulation changes; its observation/action contract still requires separate alignment before deployment.

## Historical PI Verification (2026-09-30)

Three CPU unit tests passed (geometry/odometry, airborne friction limit, temporal gradients and TorchScript tracing). GPU flat and rough PPO smoke runs completed. Fixed-low-pose plane tests with four robots, 600 steps per command, passed history/reset and airborne-zero-traction assertions:

| Command | Measured late-window mean | Four-wheel contact | Body top | Minimum sampled clearance |
| --- | --- | --- | --- | --- |
| Standing | near zero | 100% | 242.7 mm | 9.7 mm |
| vx=1 m/s | vx=0.895 m/s | 100% | 242.7 mm | 9.7 mm |
| wz=+2pi rad/s | +6.28276 rad/s | 100% | 242.8 mm | 9.8 mm |
| wz=-2pi rad/s | -6.28283 rad/s | 100% | 242.8 mm | 9.8 mm |
| vx=0.5, wz=+2pi | vx=0.444, vy=0.044, wz=6.28276 | 100% | 242.8 mm | 9.8 mm |

These are historical PI physical motion sanity checks, **not current ADRC results or trained slope suspension results**. No long training, real-robot validation, arbitrary terrain validation or 2 m/s benchmark has been completed. Existing V0 dynamics remain legacy; only the shared PID timestep bug was corrected there.
