# Deformable low-slip drive V2 (2026-10-01)

The weak wheel P loop and velocity-only tire law allowed wheel spin with almost
no uphill motion. The new drive matches the requested low-slip behavior while
keeping actual motor effort, normal contact, wheel reactions and friction limits.

## Model

- All V1 training and keyboard-play tasks inherit the same drive configuration.
- At 1 kHz, chassis translation ramps at 2 m/s² and yaw at 4 rad/s², in the command
  frame. Filtering the command before inverse kinematics preserves simultaneous
  world translation and full yaw; a low wheel slew limit alone cannot do this.
- Wheel speed PI: kp=0.8, ki=2.0 (near-critical damping under chassis load),
  conditional anti-windup, ±5 Nm motor effort,
  ±60 rad/s wheel speed. A secondary 200 rad/s² wheel slew bound and URDF axial
  inertia 0.002092387 kg m² supply bounded acceleration feedforward.
- Tire longitudinal force is `5000 * tread_deflection + 500 * slip`, in newtons.
  Deflection integrates contact slip at 1 ms and is bounded by `mu * N / 5000`.
  This elastic force can hold a stationary chassis on a grade; a viscous-only law
  necessarily requires continuous sliding to support a load.
- `mu` is sampled in [0.8, 1.0]. Longitudinal and passive roller forces share the
  original friction circle. Transverse roller drag remains 0.2 N/(m/s).
- Separated wheels have zero force and zero tread deflection. Unloaded wheels
  clear the PI integral. Episode reset clears the command ramp, PI state,
  deflection and external wheel wrenches independently for each environment.
- Policy/critic/action dimensions remain 256/40/4. Wheel dynamics do not add true
  chassis velocity to policy observations or apply a base servo.

These are simulator settings consistent with the reported low-slip hardware,
not parameters identified from hardware measurements. Large demands, low contact
loads or airborne wheels can still exceed the available physical grip.

## Runs

Use the `isaaclab` environment. New training logs default to
`logs/rsl_rl/deformable_dynamic_low_slip_history_v1/`.

```bash
python scripts/rsl_rl/train.py --task=Robotics-Deformable-Suspension-Rough-History-Transformer-v1 --num_envs=128 --max_iterations=10000 --headless --device=cuda:0
./run_gui.sh python scripts/rsl_rl/play_deformable.py --task=Robotics-Deformable-Suspension-Rough-Keyboard-Play-History-Transformer-v1 --checkpoint=<low-slip-checkpoint> --device=cuda:0 --vx_max=1 --vy_max=1 --fixed_camera --debug_motion
python -m pytest scripts/tools/test_deformable_dynamic.py -q
python scripts/tools/deformable_dynamic_probe.py --device=cuda:0 --steps=600 --strict
python scripts/tools/deformable_grade_probe.py --device=cuda:0 --grade=0 --steps=600 --full-spin --strict
python scripts/tools/deformable_grade_probe.py --device=cuda:0 --grade=8 --steps=900 --transitions --strict
```

The old 3999 checkpoint can still load for comparison, but learned the old tire
and controller response. Start fresh training for acceptance on the new dynamics.
The fixed-reference probes isolate drive physics; they do not establish learned
suspension performance over every slope transition.

## Verification with the final kp=0.8, ki=2.0 configuration

The 8-degree ramp probe uses five robots, seed 42, 900 policy steps: start,
reverse, then stop (300 steps per phase). Suspension stays at Q_LOW; means are
computed over the last 150 steps of each phase. Positive x is uphill. This run
measured root COM velocity; full-spin plane checks measure base_link origin
velocity to exclude the normal COM orbit caused by its offset.

| Initial command | Start phase | Reverse phase | Stop phase |
| --- | --- | --- | --- |
| vx=+1 m/s | +0.981580 m/s | -0.993639 m/s | +0.001103 m/s |
| vx=-1 m/s | -0.981950 m/s | +0.993686 m/s | -0.001114 m/s |
| vy=+1 m/s | +1.001194 m/s | -1.013337 m/s | +0.001168 m/s |
| wz=+0.6 rad/s | +0.600155 rad/s | -0.600539 rad/s | +0.000168 rad/s |

No resets; all sampled wheel loads exceed 3 N. Mean absolute slip is below
0.0009 m/s for every robot across settled phase windows. Maximum per-phase
velocity standard deviation is 0.01996. Command increments remain bounded by
0.02 m/s and 0.04 rad/s per policy step (ten physics updates). The uphill/downhill
horizontal speed includes the existing command projection onto the pitched body
frame; low tire slip and a small horizontal velocity error can coexist.

Seven CPU tests pass, including loaded wheel tracking, stalled/airborne integral
behavior, friction limits, selective reset, ADRC and policy history. An 8-env,
3-iteration rough training smoke run with accelerated motion curriculum completed
and saved `model_2.pt` at
`logs/rsl_rl/deformable_low_slip_smoke/2026-10-01_20-14-24_critical_elastic_drive_v2/`.
The smoke run checks training integration, not policy convergence or terrain
acceptance. Its initially random suspension can still unload wheels and produce
slip. Learned suspension and changing slope contacts require fresh training and
separate evaluation.

### Plane, including simultaneous translation and full spin

Eight robots, seed 42, 600 policy steps; means over the last 300 steps.
Linear velocity is measured at the base_link origin.

| Command | Measured velocity |
| --- | --- |
| vx=1, vy=0 m/s; wz=0.000000 rad/s | vx=0.999881, vy=-0.000025 m/s; wz=-0.000010 rad/s |
| vx=-1, vy=0 m/s; wz=0.000000 rad/s | vx=-0.999922, vy=-0.000068 m/s; wz=-0.000002 rad/s |
| vx=0, vy=1 m/s; wz=0.000000 rad/s | vx=-0.000053, vy=0.999909 m/s; wz=0.000012 rad/s |
| vx=0, vy=0 m/s; wz=6.283185 rad/s | vx=0.000026, vy=-0.000033 m/s; wz=6.283671 rad/s |
| vx=0, vy=0 m/s; wz=0.000000 rad/s | vx=-0.000087, vy=-0.000069 m/s; wz=0.000000 rad/s |
| vx=0, vy=0 m/s; wz=-6.283185 rad/s | vx=-0.000021, vy=0.000042 m/s; wz=-6.283688 rad/s |
| vx=0.5, vy=0 m/s; wz=6.283185 rad/s | vx=0.499390, vy=0.000273 m/s; wz=6.283674 rad/s |
| vx=-0.5, vy=0 m/s; wz=-6.283185 rad/s | vx=-0.499389, vy=0.000343 m/s; wz=-6.283667 rad/s |

All strict speed, slip, contact, height, clearance and smoothness checks pass.
Maximum settled mean absolute slip is 0.00007931 m/s;
maximum velocity standard deviation is 0.000748.
No resets; body top is approximately 242.9 mm and clearance approximately 9.9 mm.

Raw final-configuration measurements:
`logs/debug/deformable_low_slip_v2_validation.json`.
