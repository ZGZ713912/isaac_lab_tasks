# Minangle V2 Suspension Evaluation

## Implementation And Current Evidence

Physical angles and deployment-normalized q are separate coordinates. With
17-degree minangle and 75-degree maxangle, the direct CAD coordinate baseline is
`q_urdf = radians(75 - 17) = 1.01229`. The deployment-normalized coordinate at
17 degrees is 1.36, which must never be sent directly to the CAD articulation.
This angle offset remains a hardware/CAD calibration assumption, not identified
sim-to-real evidence. Q_LOW=1.0563 is the measured simulator clearance reference.

V2 signed actions cover the full available stroke on each side of the baseline:
negative actions interpolate toward q=0, positive actions toward Q_LOW. Zero
holds minangle. Targets slew at 2 rad/s. Policy frames remain 32 features;
privileged critic height/forces use fixed physical scaling. Tunnel height is
disabled by default; contact, horizontal attitude and chassis safety take priority.
PPO uses log-std with a 0.03 floor, fixed 1e-4 LR, value coefficient 1, and
separate actor/critic gradient clipping. DiagnosticPPO logs pre-clip gradients,
effective std, KL and explained variance. V1 preserves old action and baseline
semantics, but reset improvements mean it is not a bitwise historical replay.

CPU regression: 43 tests pass. GPU flat and rough 8-env/3-update smoke runs pass.
A 32-env, 50-update static rough foundation completed at
`logs/rsl_rl/deformable_minangle_residual_v2/2026-10-02_21-35-57_static_foundation_50/model_49.pt`.
Paired held-out static evaluation (seed 1234, four environments, 100 steps,
last 50 settled steps per environment) produced:

| Metric | Policy | Zero baseline |
| --- | --- | --- |
| Four-wheel contact | 1.00 | 1.00 |
| Contact and tilt <3 degrees | 0.75 | 0.75 |
| Tilt RMS, degrees | 1.8813 | 1.9722 |
| Minimum clearance, m | 0.01093 | 0.00340 |
| Minimum normal load, N | 52.67 | 18.84 |
| Terminations | 0 | 0 |

**Acceptance failed.** This short run does not establish convergence, dynamic
performance or architecture superiority. The complete multi-seed ablation and
long training have not been executed. Geometry-only 24-yaw-bin scans at 6 mm
normal chassis clearance give 24/24 feasible directions at 0, 2 and 5 degrees,
16/24 at 8 degrees, 3/24 at 10 degrees and 0/24 at 17 degrees. These are not
dynamic performance guarantees; V2 starts at 0-5 degrees for this reason.

`scripts/tools/deformable_suspension_eval.py` evaluates the
`minangle_residual_v2` action contract, without training, exporting or writing
reports. The default registered task is
`Robotics-Deformable-Suspension-Rough-History-Transformer-v2`.
The evaluator does not silently fall back to V1. An explicitly selected task must expose the same action contract,
32-feature policy frames, 40-feature critic and physical signal interface.

## Commands

Run in the repaired `isaaclab` conda environment, from the repository root.
Do not launch alongside another GPU simulation.

```bash
conda run --no-capture-output -n isaaclab python -B scripts/tools/deformable_suspension_eval.py --checkpoint logs/rsl_rl/deformable_minangle_residual_v2/RUN/model_9999.pt --num_envs 32 --steps 600 --seed 1234 --device cuda:0 --strict
conda run --no-capture-output -n isaaclab python -B scripts/tools/deformable_suspension_eval.py --num_envs 32 --steps 600 --seed 1234 --history 8 --device cuda:0
```

Use `--task TASK` to select an available compatible task. `--scenarios static`
restricts the budget to standing; the default evaluates six scenarios sequentially.
`--agent-yaml PATH` selects the actual saved YAML if it is not next to the checkpoint
under `params/agent.yaml`. Only use trusted checkpoints. YAML uses safe loading.

The YAML supplies the entire runner/policy/algorithm configuration, including
normalization and action clipping. No current registry agent defaults are merged.
The evaluator constructs `OnPolicyRunner` with that saved mapping, imports both
`agent_rl.rsl_rl.modules` and `agent_rl.rsl_rl.algorithms` to register custom
classes including `DiagnosticPPO`, and loads the checkpoint without its optimizer.
Incompatible state dictionaries fail rather than being partially loaded.

Saved `policy.history_length` must be 1, 4 or 8. Before creating the environment,
the evaluator sets `policy_history_length=H` and `observation_space=32*H`.
`--history H` asserts a checkpoint's history rather than overriding it; without a
checkpoint it selects baseline history. History ablations require separately
trained matching checkpoints, not reshaping an H=8 checkpoint into H=1 or H=4.

## Protocol

The bound is `steps * num_envs * scenarios * modes` policy transitions. Settling
is included in `--steps`; no additional warmup is run. Modes are `POLICY` and
`ZERO` with a checkpoint, otherwise `ZERO` only. Zero actions retain the task's
normal suspension reference/controller and wheel drive, not zero actuator torque.

All commands are world-frame `(vx, vy, wz)` in m/s and rad/s:

| Scenario | Command |
| --- | --- |
| static | (0, 0, 0) |
| forward | (1, 0, 0) |
| lateral | (0, 1, 0) |
| spin_positive | (0, 0, +2*pi) |
| spin_negative | (0, 0, -2*pi) |
| dynamic | Four equal-budget phases: stop, (0.5, 0, +2*pi), (-0.5, 0, -2*pi), stop |

Commands are applied before inference, including updating the newest observation
frame's command fields without advancing history twice. The task's acceleration
filters remain active. The dynamic schedule follows scenario time, not episode
time, so failures cannot restart or skip a difficult command phase. No settling
exclusion is added at command transitions.

Terrain/configuration and initial reset seeds match between modes. Each environment's
nth reset gets a seed determined by scenario, environment ID and episode ordinal;
reset timing cannot consume the other environments' random streams. Initial robot
state hashes must match. Observation noise, bias and delay are disabled and tire
friction is fixed at the configured range midpoint. Safety and boundary/episode
timeouts are retained. No play-mode safety relaxation is enabled.

PyTorch deterministic algorithms are required and TF32 is disabled. This defines
a repeatable protocol on the same software/hardware, not a claim of bitwise PhysX
GPU reproducibility across drivers, devices or Isaac Sim releases.

## JSON And Acceptance

Stdout contains one final JSON document; Python simulator/runner diagnostics are
redirected to stderr. Native simulator logging can still be platform dependent.
No evaluation artifacts are written. Isaac Sim itself may maintain its normal
external runtime caches/logs. Errors abort with a nonzero exit rather than emit a
partial success report.

Metrics are sampled inside `_get_dones`, before auto-reset. Samples at episode age
at most 0.5 seconds are excluded, independently after every reset. Terminal settled
states remain included. Reports include all completed episodes and nonempty final
budget-censored episodes, including failures with zero settled samples. Such short
failures add a failed observation to the failure-adjusted contact denominators and
remain failed in the episode-success denominator. Raw settled rates are also
reported; they must not be interpreted alone as survival or acceptance.

- `all_contact_rate`: all four normal loads strictly exceed the task's threshold.
- `contact_and_horizontal_rate`: all contact and true gravity tilt strictly below 3 degrees.
- Roll/pitch RMS and absolute p50/p95/p99/max use individual settled observations,
  not angles reconstructed from batch-averaged gravity. Tilt uses acos of normalized
  negative gravity z, so upside-down poses cannot appear horizontal.
- `min_load_n` and `clearance_min_m` are minima over settled samples, not batch means.
- Leg/wheel torque saturation is the fraction of joint-time samples whose applied
  torque reaches 99% of the configured effort limit, sampled at policy-step end.
  This is not a physics-substep peak saturation measure.
- Terminated resets and timeout resets are separate; a simultaneous event appears
  in both counters but only once in `reset_events`. Final budget truncation is not
  a simulator timeout. Each episode record labels these outcomes explicitly.

`--strict` checks every selected scenario for the POLICY mode, or ZERO when no
checkpoint is supplied. The comparison baseline is reported but need not pass.
Acceptance requires failure-adjusted all-contact >= 0.98, joint contact/horizontal
success >= 0.95, episode success >= 0.95, no terminated resets, nonempty settled
samples, and both tilt RMS and absolute p95 < 3 degrees. Episode success uses the
same contact/joint/tilt thresholds and rejects every terminated or no-sample episode.
Timeouts are recorded rather than automatically called physical failures. Clearance,
load and saturation are diagnostics, not additional hidden acceptance gates.

Exit status is 1 for strict acceptance failure and 0 for a completed non-strict run,
even when `passed` is false. No GPU evaluation results are claimed by this document.

## Minimal Ablation

`scripts/tools/deformable_ablation.py` defaults to rendering commands only:
MLP and Transformer, H1/H4/H8, with shared seeds `42 123 2026` (18 jobs).
No simulator is imported by this entrypoint. Explicit `--run` launches the
jobs sequentially with the current `sys.executable`, from the repository root.
Use the Isaac Lab Python environment when actually launching training.

```bash
python scripts/tools/deformable_ablation.py --dry-run --evaluate
python scripts/tools/deformable_ablation.py --run --iterations 100 --num-envs 128 --models mlp transformer --seeds 42 123 2026 --timeout 3600 --evaluate
python -m pytest scripts/tools/test_deformable_ablation.py -q
```

Default training budget is 100 iterations and 128 environments; `--num-envs`
must be in 1..128. `--iterations`, `--seeds`, and `--models` change the matrix
budget. All histories use the registered Rough-History-Transformer-v2 task,
with matching `agent.policy.history_length`, `env.policy_history_length`, and
`env.observation_space=32*H` overrides. All other task/algorithm settings remain
shared. Both policies use log noise, initial standard deviation 0.3 and floor
0.03; both receive hidden dimensions `[256,128,64]` (ignored by Transformer).
The MLP flattens the entire actor history, keeps the current 40-feature critic,
inherits the installed RSL-RL 3 ActorCritic interface and normalization, and
intentionally discards the shared Transformer-only configuration fields.
This is not a parameter-count-matched comparison.

Every job gets a unique `--run_name` in `deformable_minangle_v2_ablation`.
After successful training the runner resolves that suffix under
`logs/rsl_rl/deformable_minangle_v2_ablation` and selects the numerically latest
`model_*.pt`, requiring its saved `params/agent.yaml`. `--evaluate` invokes the
existing evaluator on that new checkpoint, asserting its history and using the
same seed/environment count. Dry-run evaluation paths are placeholders until
training creates a run. Evaluations use the existing 600-step, six-scenario,
POLICY/ZERO protocol; non-strict acceptance results are printed, not converted
to subprocess failures.

`--timeout` defaults to 3600 seconds per subprocess, separately for training
and evaluation, not for the whole matrix. Child stdout/stderr are streamed;
the final JSON job summary includes checkpoint paths, elapsed time and failures.
Failed/timed-out jobs do not stop remaining jobs, but any failure returns exit 1.
No GPU training or simulation validation was performed for this entrypoint.
