"""Recover a standalone five-frame MLP from a verified suspension teacher.

Collection runs actual Isaac Sim rollouts using a training seed. The student sees
only the most recent five 32D sensor frames. Normalization is folded into its
first linear layer, so inference and the existing exporter need no teacher or
extra state. Held-out entire vehicles, rather than adjacent frames, select the
supervised checkpoint. Independent closed-loop evaluation remains mandatory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/tools"))
from deformable_precision_pipeline import extract_report, run_child, timestamp, write_json


def collect(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", started_at=timestamp(), jobs=[], teacher=str(args.teacher),
                 teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(), seed=args.seed,
                 purpose="training data, excluded from independent acceptance", grades=args.grades,
                 geometry_guide=args.geometry_guide)
    state["source_sha256"] = {}
    for relative in ("scripts/tools/deformable_mlp_distill.py", "scripts/tools/deformable_geometry_teacher.py",
                     "scripts/tools/deformable_suspension_eval.py",
                     "source/agent_tasks/agent_tasks/direct/deformable_suspension/cfg_utils.py",
                     "source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_env.py"):
        data = (ROOT / relative).read_bytes()
        target = args.output_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        state["source_sha256"][relative] = hashlib.sha256(data).hexdigest()
    write_json(args.output_dir / "status.json", state)
    try:
        for grade in args.grades:
            name = f"grade_{grade:g}"
            command = [str(ROOT / "run_gui.sh"), sys.executable, "-B", "-u",
                       "scripts/tools/deformable_suspension_eval.py", "--task", args.task,
                       "--checkpoint", str(args.teacher), "--policy-only", "--num_envs", "16",
                       "--batch-scenarios", "--steps", str(args.steps), "--seed", str(args.seed),
                       "--device", args.device, "--grade-deg", str(grade), "--command-profile", "play",
                       "--command-frame", "body", "--headless", "--policy-data-dir",
                       str(args.output_dir / name)]
            if args.student:
                command += ["--student-checkpoint", str(args.student.resolve())]
            if args.geometry_guide:
                command += ["--geometry-guide", "--geometry-correction-limit", str(args.geometry_correction_limit),
                            "--geometry-correction-smoothing", str(args.geometry_correction_smoothing)]
                if args.geometry_yaw_support_gate:
                    command.append("--geometry-yaw-support-gate")
            log = run_child(command, name, args, state)
            report = extract_report(log)
            write_json(args.output_dir / f"{name}.json", report)
        state.update(status="completed", ended_at=timestamp(), current_job=None)
    except BaseException as error:
        state.update(status="failed", ended_at=timestamp(), error=str(error))
        raise
    finally:
        write_json(args.output_dir / "status.json", state)


def fold_input_normalization(layer, mean, scale):
    """Preserve f((x-mean)/scale) exactly as an ordinary f_raw(x)."""
    import torch
    with torch.no_grad():
        weight = layer.weight.double() / scale.double()[None]
        bias = layer.bias.double() - weight @ mean.double()
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)


def join_mlp_actors(base, residual):
    """Fold a sum of equal-depth ELU MLPs into one ordinary sequential MLP."""
    import torch
    from torch import nn
    if (len(base) < 3 or len(base) != len(residual) or len(base) % 2 != 1
            or any(not isinstance(base[i], nn.Linear) or not isinstance(residual[i], nn.Linear)
                   for i in range(0, len(base), 2))
            or any(not isinstance(base[i], nn.ELU) or not isinstance(residual[i], nn.ELU)
                   or base[i].alpha != residual[i].alpha for i in range(1, len(base), 2))
            or base[0].in_features != residual[0].in_features
            or base[-1].out_features != residual[-1].out_features):
        raise ValueError("Residual folding requires matching input/output and equal-depth ELU MLPs")
    layers = []
    for i in range(0, len(base), 2):
        a, b = base[i], residual[i]
        first, last = i == 0, i == len(base)-1
        inputs = a.in_features if first else a.in_features + b.in_features
        outputs = a.out_features if last else a.out_features + b.out_features
        layer = nn.Linear(inputs, outputs).to(device=a.weight.device, dtype=a.weight.dtype)
        with torch.no_grad():
            if first:
                layer.weight.copy_(torch.cat((a.weight, b.weight), 0))
                layer.bias.copy_(torch.cat((a.bias, b.bias), 0))
            elif last:
                layer.weight.copy_(torch.cat((a.weight, b.weight), 1))
                layer.bias.copy_(a.bias + b.bias)
            else:
                layer.weight.zero_()
                layer.weight[:a.out_features, :a.in_features].copy_(a.weight)
                layer.weight[a.out_features:, a.in_features:].copy_(b.weight)
                layer.bias.copy_(torch.cat((a.bias, b.bias), 0))
        layers.append(layer)
        if not last:
            layers.append(copy.deepcopy(base[i+1]))
    return nn.Sequential(*layers)


def validate_training_rollout_safety(report):
    rows = report["results"]["POLICY"]
    from deformable_mlp_repair_assess import SCENARIOS
    if set(rows) != SCENARIOS or any(row["physical_terminated_resets"] or row["terrain_boundary_violations"]
                                   or row["failure_adjusted_all_contact_rate"] < .98 for row in rows.values()):
        raise ValueError("Unsafe teacher rollouts must not become imitation targets")


def validate_geometry_training_report(report):
    if report.get("training_data_controller") != "sensor_only_geometry_guide":
        raise ValueError("Geometry targets require a recorded geometry teacher rollout")
    validate_training_rollout_safety(report)


def startup_sample_weights(steps, envs, step_dt, duration, weight):
    import numpy as np
    if (min(steps, envs) <= 0 or not all(math.isfinite(v) for v in (step_dt, duration, weight))
            or step_dt <= 0 or duration <= 0 or weight < 1):
        raise ValueError("Startup weighting requires positive dimensions/time and finite weight >= 1")
    by_step = np.where(np.arange(steps)*step_dt < duration, weight, 1.).astype(np.float32)
    return np.broadcast_to(by_step[:, None], (steps, envs)).reshape(-1).copy()


def weighted_mse(predicted, target, weights):
    return ((predicted-target).square().mean(-1)*weights).sum()/weights.sum()


def selected_feedback_targets(teacher, frozen, selected):
    import torch
    return torch.where(selected[:, None], teacher, frozen)


def reference_residual_targets(reference, frozen, selected):
    """Training-only regional mentor; retain the exact frozen raw mean elsewhere."""
    import torch
    return torch.where(selected[:, None], reference-frozen, torch.zeros_like(frozen)).detach()


def validate_anchor_reference(checkpoint, evaluation_dir, grades):
    from deformable_mlp_guarded_train import checkpoint_info, qualify
    from deformable_mlp_repair_assess import GRADES
    if not grades or any(g not in GRADES for g in grades):
        raise ValueError("Anchor mentor grades must belong to the paired five-grade protocol")
    info = checkpoint_info(checkpoint)
    baseline = ROOT / "outputs/deformable_long2499_joint_reward_review_20261008/evaluation"
    result = qualify(checkpoint, evaluation_dir, baseline)
    if not grades or any(str(int(g)) not in result["grades"]
                         or not result["grades"][str(int(g))]["recovered"] for g in grades):
        raise ValueError("Anchor mentor requires completed paired recovery on every selected grade")
    return dict(checkpoint=str(checkpoint.resolve()), sha256=info["sha256"],
                evaluation_dir=str(evaluation_dir.resolve()), grades=grades,
                purpose="raw feedback mentor only on independently recovered grades")


def fit(args):
    import numpy as np
    import torch
    import yaml
    sys.path.insert(0, str(ROOT / "source/agent_rl"))
    from agent_rl.rsl_rl.modules.actor_critic_suspension_mlp import ActorCriticSuspensionMLP

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_sha256 = {}
    for relative in ("scripts/tools/deformable_mlp_distill.py", "scripts/tools/deformable_geometry_teacher.py",
                     "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_suspension_mlp.py"):
        data = (ROOT / relative).read_bytes()
        destination = args.output_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        source_sha256[relative] = hashlib.sha256(data).hexdigest()
    paths = sorted({path for directory in [args.data_dir, *args.extra_data_dirs]
                    for path in directory.glob("grade_*/*.npz")
                    if (directory / f"{path.parent.name}.json").is_file()})
    if not paths:
        raise ValueError("No completed teacher rollout datasets")
    rows, provenance, guide_configs = [], [], []
    for path in paths:
        report = json.loads((path.parent.parent / f"{path.parent.name}.json").read_text())
        geometric = report.get("training_data_controller") == "sensor_only_geometry_guide"
        guide_source_sha256 = None
        if getattr(args, "raw_feedback_teacher", False):
            validate_training_rollout_safety(report)
            collected = json.loads((path.parent.parent / "status.json").read_text())
            if (collected["status"] != "completed"
                    or collected["teacher_sha256"] != hashlib.sha256(args.teacher.read_bytes()).hexdigest()):
                raise ValueError("Raw feedback requires completed collection with the matching teacher bytes")
        if args.geometry_guide or geometric:
            validate_geometry_training_report(report)
            if not args.geometry_guide:
                raise ValueError("Use --geometry-guide to preserve the actual teacher target provenance")
            collected = json.loads((path.parent.parent / "status.json").read_text())
            relative = "scripts/tools/deformable_geometry_teacher.py"
            guide_source_sha256 = collected["source_sha256"][relative]
            if collected["teacher_sha256"] != hashlib.sha256(args.teacher.read_bytes()).hexdigest():
                raise ValueError("Geometry feedback requires the actual collected fallback checkpoint")
            if (args.teacher_jitter > 0
                    and guide_source_sha256 != hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()):
                raise ValueError("Geometry feedback labeling requires the same teacher source as actual collection")
        with np.load(path) as data:
            obs = data["policy_obs"]
            if obs.shape[-1] < 160 or obs.shape[-1] % 32:
                raise ValueError("Teacher history must contain at least five 32D frames")
            steps, envs, _ = obs.shape
            # The same environment is entirely in one partition at every time.
            holdout = np.broadcast_to(np.arange(envs)[None] % 4 == 3, (steps, envs)).reshape(-1)
            rows.append((obs[..., -160:].reshape(-1, 160), data["teacher_actions"].reshape(-1, 4),
                         data["critic_obs"].reshape(-1, 40), data["teacher_values"].reshape(-1, 1), holdout,
                         obs.reshape(-1, obs.shape[-1]),
                         startup_sample_weights(steps, envs, float(data["step_dt_s"]),
                                                getattr(args, "startup_duration_s", 2.),
                                                getattr(args, "startup_loss_weight", 1.)),
                         np.broadcast_to(np.isin(data["env_scenarios"], args.teacher_scenarios)
                                         if getattr(args, "teacher_scenarios", None) else np.ones(envs, dtype=bool),
                                         (steps, envs)).reshape(-1).copy()))
            provenance.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   teacher=str(data["teacher_checkpoint"]), seed=int(data["seed"]),
                                   controller=str(data["controller_checkpoint"]) if "controller_checkpoint" in data else None,
                                   geometry_teacher_source_sha256=guide_source_sha256,
                                   grade_deg=float(data["grade_deg"]), transitions=steps * envs))
            guide_configs.append(str(data["geometry_guide_config"]) if "geometry_guide_config" in data else "null")
    if args.geometry_guide and (len(set(guide_configs)) != 1 or guide_configs[0] == "null"):
        raise ValueError("Geometric feedback targets require one matching, recorded teacher configuration")
    device = torch.device(args.device)
    x, target, critic_x, value, held = (
        torch.from_numpy(np.concatenate([row[i] for row in rows])).to(device) for i in range(5))
    sample_weights = torch.from_numpy(np.concatenate([row[6] for row in rows])).to(device)
    teacher_selected = torch.from_numpy(np.concatenate([row[7] for row in rows])).to(device)
    if not all(torch.isfinite(t).all() for t in (x, target, critic_x, value)):
        raise ValueError("Nonfinite teacher data")
    train_ids, val_ids = (~held).nonzero().flatten(), held.nonzero().flatten()
    mean, scale = x[train_ids].mean(0), x[train_ids].std(0).clamp_min(.05)
    cmean, cscale = critic_x[train_ids].mean(0), critic_x[train_ids].std(0).clamp_min(.05)
    vmean, vscale = value[train_ids].mean(), value[train_ids].std().clamp_min(1.)
    xn, cn, vn = (x - mean) / scale, (critic_x - cmean) / cscale, (value - vmean) / vscale
    teacher, teacher_x, jitter_scale = None, None, None
    if args.teacher_jitter > 0:
        import importlib.util
        spec = importlib.util.spec_from_file_location("distill_teacher", ROOT / "scripts/rsl_rl/export_onnx.py")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        teacher_state = torch.load(args.teacher, map_location="cpu", weights_only=False)["model_state_dict"]
        teacher = (exporter._build_transformer_actor(teacher_state, str(args.teacher))[0]
                   if exporter._is_transformer(teacher_state) else exporter._build_actor(teacher_state)[0])
        if getattr(args, "raw_feedback_teacher", False):
            config = yaml.safe_load((args.teacher.parent / "params/agent.yaml").read_text())["policy"]
            if (config["class_name"] != "ActorCriticSuspensionMLP" or config["history_length"] != 5
                    or config.get("actor_obs_normalization", False)):
                raise ValueError("Raw feedback requires the matching unnormalized five-frame MLP")
        elif args.geometry_guide:
            from deformable_geometry_teacher import GeometryTeacher
            from deformable_feasibility import utilities
            teacher = GeometryTeacher(teacher, utilities(), **json.loads(guide_configs[0]))
        teacher = teacher.to(device)
        teacher.eval().requires_grad_(False)
        teacher_x = torch.from_numpy(np.concatenate([row[5] for row in rows])).to(device)
        jitter_scale = x.new_tensor([0.] * 4 + [.02] * 3 + [.01, .01, 0.] + [.01] * 4
                                    + [.01] * 4 + [.01] * 4 + [.005] * 4 + [.02] * 4 + [.02] * 2)
    critic_dims = args.hidden_dims
    if args.resume_student:
        resumed_config = yaml.safe_load((args.resume_student.parent / "params/agent.yaml").read_text())
        critic_dims = resumed_config["policy"]["critic_hidden_dims"]
    pcfg = dict(class_name="ActorCriticSuspensionMLP", history_length=5,
                actor_hidden_dims=args.hidden_dims, critic_hidden_dims=critic_dims,
                activation="elu", actor_obs_normalization=False, critic_obs_normalization=False,
                init_noise_std=.015, min_noise_std=.005, noise_std_type="log")
    groups = dict(policy=["policy"], critic=["critic"])
    policy = ActorCriticSuspensionMLP({"policy": xn[:2], "critic": cn[:2]}, groups, 4,
                                    **{k: v for k, v in pcfg.items() if k != "class_name"}).to(device)
    if args.resume_student:
        state = torch.load(args.resume_student, map_location=device, weights_only=False)
        policy.load_state_dict(state["model_state_dict"], strict=True)
        with torch.no_grad():
            for layer, offset, divisor in ((policy.actor[0], mean, scale),
                                           (policy.critic[0], cmean, cscale)):
                raw_weight = layer.weight.double().clone()
                layer.bias.add_((raw_weight @ offset.double()).to(layer.bias.dtype))
                layer.weight.copy_(raw_weight * divisor.double()[None])
            policy.critic[-1].weight.div_(vscale)
            policy.critic[-1].bias.sub_(vmean).div_(vscale)
    residual_dims = getattr(args, "residual_hidden_dims", None)
    if residual_dims:
        if not args.resume_student or len(residual_dims) != len(args.hidden_dims) or min(residual_dims) <= 0:
            raise ValueError("Residual fitting requires a resumed MLP and matching hidden-layer count")
        from torch import nn
        base = policy.actor.eval().requires_grad_(False)
        widths = [160, *residual_dims, 4]
        layers = []
        for i, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(inputs, outputs))
            if i < len(widths)-2:
                layers.append(nn.ELU())
        residual = nn.Sequential(*layers).to(device)
        nn.init.zeros_(residual[-1].weight)
        nn.init.zeros_(residual[-1].bias)

        class ResidualActor(nn.Module):
            def __init__(self, base, residual):
                super().__init__()
                self.base, self.residual = base, residual

            def forward(self, observations):
                return self.base(observations) + self.residual(observations)

        policy.actor = ResidualActor(base, residual)
        pcfg["actor_hidden_dims"] = [a+b for a, b in zip(args.hidden_dims, residual_dims)]
    if getattr(args, "teacher_scenarios", None):
        if not residual_dims:
            raise ValueError("Selected feedback correction requires a frozen resumed MLP")
        with torch.no_grad():
            inactive = (~teacher_selected).nonzero().flatten()
            for ids in inactive.split(args.batch_size):
                # Preserve raw means, including saturated ones: clipping the
                # frozen mean would alter its feedback despite unchanged actions.
                target[ids] = policy.actor.base(xn[ids])
            if getattr(args, "raw_feedback_teacher", False):
                for ids in teacher_selected.nonzero().flatten().split(args.batch_size):
                    target[ids] = teacher(teacher_x[ids])
    anchor_x, anchor_train_ids, anchor_val_ids, anchor_probe, anchor_provenance = None, None, None, None, []
    anchor_target, anchor_probe_target, anchor_reference, anchor_reference_info = None, None, None, None
    if getattr(args, "anchor_data_dir", None):
        if not residual_dims:
            raise ValueError("Frozen-feedback anchors require residual fitting")
        observations, partitions, reference_masks = [], [], []
        if getattr(args, "anchor_reference_checkpoint", None):
            anchor_reference_info = validate_anchor_reference(
                args.anchor_reference_checkpoint, args.anchor_reference_evaluation_dir, args.anchor_reference_grades)
            reference_state = torch.load(args.anchor_reference_checkpoint, map_location="cpu",
                                         weights_only=False)["model_state_dict"]
            anchor_reference = exporter._build_actor(reference_state)[0].to(device).eval().requires_grad_(False)
        reference_grades = getattr(args, "anchor_reference_grades", None) or []
        for grade in args.anchor_grades:
            directory = args.anchor_data_dir / f"grade_{grade:g}"
            for path in sorted(directory.glob("*.npz")):
                with np.load(path) as data:
                    raw = data["policy_obs"]
                    steps, envs, _ = raw.shape
                    observations.append(raw[..., -160:].reshape(-1, 160))
                    partitions.append(np.broadcast_to(np.arange(envs)[None] % 4 == 3,
                                                       (steps, envs)).reshape(-1))
                    reference_masks.append(np.full(steps*envs, grade in reference_grades, dtype=bool))
                anchor_provenance.append(dict(path=str(path.resolve()),
                                              sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                              grade_deg=grade,
                                              purpose=("validated raw feedback mentor" if grade in reference_grades
                                                       else "retain frozen base feedback, zero residual target")))
        if not observations:
            raise ValueError("No frozen-feedback anchor observations")
        anchor_raw = torch.from_numpy(np.concatenate(observations)).to(device)
        anchor_held = torch.from_numpy(np.concatenate(partitions)).to(device)
        if not torch.isfinite(anchor_raw).all():
            raise ValueError("Nonfinite frozen-feedback anchor observations")
        anchor_x = (anchor_raw-mean)/scale
        anchor_reference_mask = torch.from_numpy(np.concatenate(reference_masks)).to(device)
        anchor_target = torch.zeros(len(anchor_x), 4, device=device)
        if anchor_reference is not None:
            with torch.no_grad():
                for ids in torch.arange(len(anchor_x), device=device).split(args.batch_size):
                    anchor_target[ids] = reference_residual_targets(
                        anchor_reference(anchor_raw[ids]), policy.actor.base(anchor_x[ids]),
                        anchor_reference_mask[ids])
        anchor_train_ids, anchor_val_ids = (~anchor_held).nonzero().flatten(), anchor_held.nonzero().flatten()
        probe_ids = anchor_val_ids[torch.linspace(0, len(anchor_val_ids)-1, 512, device=device).long()]
        anchor_probe = anchor_x[probe_ids]
        anchor_probe_target = anchor_target[probe_ids]
        if teacher is not None:
            frames = anchor_raw[probe_ids].reshape(len(probe_ids), 5, 32)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                torch.manual_seed(args.seed+1203)
                probe = frames + args.teacher_jitter*jitter_scale*(
                    torch.randn(len(probe_ids), 1, 32, device=device)+.25*torch.randn_like(frames))
            probe[..., 9] = -(1-probe[..., 7:9].square().sum(-1)).clamp_min(.01).sqrt()
            anchor_probe = (probe.flatten(1)-mean)/scale
            if anchor_reference is not None:
                with torch.no_grad():
                    anchor_probe_target = reference_residual_targets(
                        anchor_reference(probe.flatten(1)), policy.actor.base(anchor_probe),
                        anchor_reference_mask[probe_ids])
    local_probe, local_probe_target, local_probe_weights = None, None, None
    if teacher is not None:
        probe_ids = val_ids[torch.linspace(0, len(val_ids) - 1, 512, device=device).long()]
        frames = teacher_x[probe_ids].reshape(len(probe_ids), -1, 32)
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed + 919)
            perturbation = torch.randn(len(probe_ids), 1, 32, device=device) + .25 * torch.randn_like(frames)
        probe = frames + args.teacher_jitter * jitter_scale * perturbation
        probe[..., 9] = -(1. - probe[..., 7:9].square().sum(-1)).clamp_min(.01).sqrt()
        with torch.no_grad():
            local_probe_target = teacher(probe.flatten(1))
            if not getattr(args, "raw_feedback_teacher", False):
                local_probe_target = local_probe_target.clamp(-1., 1.)
        local_probe = (probe[:, -5:].flatten(1) - mean) / scale
        if getattr(args, "teacher_scenarios", None):
            with torch.no_grad():
                local_probe_target = selected_feedback_targets(
                    local_probe_target, policy.actor.base(local_probe), teacher_selected[probe_ids])
        local_probe_weights = sample_weights[probe_ids].clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    best, best_state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        permutation = train_ids[torch.randperm(len(train_ids), device=device)]
        policy.train()
        total, count = 0., 0
        for ids in permutation.split(args.batch_size):
            pred = policy.actor(xn[ids])
            actor_loss = weighted_mse(pred, target[ids], sample_weights[ids])
            critic_loss = (policy.critic(cn[ids]) - vn[ids]).square().mean()
            loss = actor_loss + .01 * critic_loss
            if teacher is not None:
                # Matching only a trajectory's values leaves feedback gains
                # underdetermined. Teacher-label small neighborhoods too, so
                # sensor/action perturbations do not acquire arbitrary gains.
                selected = ids[:min(512, len(ids))]
                frames = teacher_x[selected].reshape(len(selected), -1, 32)
                coherent = torch.randn(len(selected), 1, 32, device=device)
                temporal = .25 * torch.randn_like(frames)
                perturbed = frames + args.teacher_jitter * jitter_scale * (coherent + temporal)
                perturbed[..., 9] = -(1. - perturbed[..., 7:9].square().sum(-1)).clamp_min(.01).sqrt()
                with torch.no_grad():
                    local_target = teacher(perturbed.flatten(1))
                    if not getattr(args, "raw_feedback_teacher", False):
                        local_target = local_target.clamp(-1., 1.)
                    local_input = (perturbed[:, -5:].flatten(1) - mean) / scale
                    if getattr(args, "teacher_scenarios", None):
                        local_target = selected_feedback_targets(
                            local_target, policy.actor.base(local_input), teacher_selected[selected])
                local_pred = policy.actor(local_input)
                loss = loss + args.feedback_weight * weighted_mse(local_pred, local_target,
                                                                 sample_weights[selected])
            if anchor_x is not None:
                selected = anchor_train_ids[torch.randint(len(anchor_train_ids), (512,), device=device)]
                anchor = anchor_x[selected]
                loss = loss + args.anchor_weight * (policy.actor.residual(anchor)-anchor_target[selected]).square().mean()
                if teacher is not None:
                    frames = anchor_raw[selected].reshape(len(selected), 5, 32)
                    probe = frames + args.teacher_jitter*jitter_scale*(
                        torch.randn(len(selected), 1, 32, device=device)+.25*torch.randn_like(frames))
                    probe[..., 9] = -(1-probe[..., 7:9].square().sum(-1)).clamp_min(.01).sqrt()
                    anchor = (probe.flatten(1)-mean)/scale
                    local_anchor_target = anchor_target[selected]
                    if anchor_reference is not None:
                        with torch.no_grad():
                            local_anchor_target = reference_residual_targets(
                                anchor_reference(probe.flatten(1)), policy.actor.base(anchor),
                                anchor_reference_mask[selected])
                    loss = loss + args.anchor_weight*(policy.actor.residual(anchor)-local_anchor_target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step()
            total += actor_loss.item() * len(ids)
            count += len(ids)
        policy.eval()
        with torch.no_grad():
            errors = torch.cat([policy.actor(xn[ids]) - target[ids] for ids in val_ids.split(args.batch_size)])
            validation = (errors.square().mean(-1)*sample_weights[val_ids]).sum().div(sample_weights[val_ids].sum()).item()
            feedback_validation = (weighted_mse(policy.actor(local_probe), local_probe_target,
                                               local_probe_weights).item()
                                   if local_probe is not None else 0.)
            anchor_validation = 0.
            if anchor_x is not None:
                probe_ids = anchor_val_ids[torch.linspace(0, len(anchor_val_ids)-1, 512, device=device).long()]
                anchor_validation = ((policy.actor.residual(anchor_x[probe_ids])-anchor_target[probe_ids]).square().mean()
                                     + (policy.actor.residual(anchor_probe)-anchor_probe_target).square().mean()).item()
            selection_score = validation + args.feedback_weight * feedback_validation + args.anchor_weight*anchor_validation
            entry = dict(epoch=epoch + 1, train_mse=total / count, validation_mse=validation,
                         validation_unweighted_mse=errors.square().mean().item(),
                         validation_abs_p95=torch.quantile(errors.abs().flatten(), .95).item(),
                         feedback_validation_mse=feedback_validation, selection_score=selection_score)
            entry["anchor_validation_mse"] = anchor_validation
        history.append(entry)
        if selection_score < best:
            best = selection_score
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(json.dumps(entry), flush=True)
            write_json(args.output_dir / "fit_progress.json", dict(status="training", history=history))
    policy.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        probe_ids = val_ids[torch.linspace(0, len(val_ids)-1, 512, device=device).long()]
        probes = x[probe_ids]
        reference = policy.actor((probes - mean) / scale).clone()
    if residual_dims:
        policy.actor = join_mlp_actors(policy.actor.base, policy.actor.residual)
    fold_input_normalization(policy.actor[0], mean, scale)
    fold_input_normalization(policy.critic[0], cmean, cscale)
    with torch.no_grad():
        policy.critic[-1].weight.mul_(vscale)
        policy.critic[-1].bias.mul_(vscale).add_(vmean)
        error = (policy.actor(probes) - reference).abs().max().item()
    if error > 2.e-5:
        raise RuntimeError(f"Normalization folding changed inference: {error}")
    params = args.output_dir / "params"
    params.mkdir()
    teacher_params = args.teacher.parent / "params"
    env_text = (teacher_params / "env.yaml").read_text()
    import re
    env_text = re.sub(r"(?m)^policy_history_length: \d+$", "policy_history_length: 5", env_text)
    env_text = re.sub(r"(?m)^observation_space: \d+$", "observation_space: 160", env_text)
    (params / "env.yaml").write_text(env_text)
    shutil.copyfile(teacher_params / "real2sim_model.json", params / "real2sim_model.json")
    acfg = yaml.safe_load((teacher_params / "agent.yaml").read_text())
    acfg.update(policy=pcfg, obs_groups=groups, experiment_name="deformable_mlp5_recovered_v3",
                run_name=args.output_dir.name, seed=args.seed, save_interval=50)
    acfg.pop("launch_checkpoint", None)
    acfg["algorithm"].update(learning_rate=3.e-5, entropy_coef=.0001,
                             steep_preservation_weight=0., flat_posture_weight=0.)
    (params / "agent.yaml").write_text(yaml.safe_dump(acfg, sort_keys=False))
    info = dict(kind="five_frame_sensor_only_teacher_distillation", datasets=provenance,
                source_sha256=source_sha256,
                training_transitions=len(train_ids), held_out_vehicle_transitions=len(val_ids),
                validation_selection_score=best, input_normalization_fold_max_error=error,
                teacher_neighborhood_jitter=args.teacher_jitter,
                feedback_weight=args.feedback_weight,
                startup_loss_weight=getattr(args, "startup_loss_weight", 1.),
                startup_duration_s=getattr(args, "startup_duration_s", 2.),
                teacher_scenarios=getattr(args, "teacher_scenarios", None),
                feedback_label_source=("raw collected fallback MLP" if getattr(args, "raw_feedback_teacher", False)
                                       else "recorded controller actions and matching teacher neighborhoods"),
                inactive_scenario_targets=("raw frozen actor, zero correction"
                                           if getattr(args, "teacher_scenarios", None) else None),
                residual_hidden_dims=residual_dims,
                frozen_actor_checkpoint=str(args.resume_student) if residual_dims else None,
                frozen_feedback_anchor_datasets=anchor_provenance,
                frozen_feedback_anchor_weight=getattr(args, "anchor_weight", 0.),
                anchor_reference=anchor_reference_info,
                geometry_guide_config=json.loads(guide_configs[0]) if args.geometry_guide else None,
                closed_loop_acceptance="not yet evaluated", history=history)
    state = policy.state_dict()
    if not all(torch.isfinite(t).all() for t in state.values()):
        raise RuntimeError("Nonfinite student checkpoint")
    checkpoint = args.output_dir / "model_0.pt"
    torch.save(dict(model_state_dict=state, optimizer_state_dict={}, iter=0, infos=info), checkpoint)
    write_json(args.output_dir / "distillation.json", info)
    write_json(args.output_dir / "fit_progress.json", dict(
        status="completed", history=history, checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        validation_selection_score=best, closed_loop_acceptance="not yet evaluated"))
    print(f"CHECKPOINT {checkpoint.resolve()}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("collect", "fit"))
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--extra-data-dirs", type=Path, nargs="*", default=[])
    p.add_argument("--student", type=Path, help="Optional controller for teacher-labeled data aggregation")
    p.add_argument("--geometry-guide", action="store_true",
                   help="Training-only sensor geometry targets, including teacher-labeled feedback neighborhoods")
    p.add_argument("--geometry-correction-limit", type=float, default=.04)
    p.add_argument("--geometry-correction-smoothing", type=float, default=1.)
    p.add_argument("--geometry-yaw-support-gate", action="store_true")
    p.add_argument("--resume-student", type=Path, help="Initialize from a previously distilled raw-input student")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task", default="Robotics-Deformable-Suspension-Support-Leveling-Mixed-Real2Sim-v3")
    p.add_argument("--grades", type=float, nargs="+", default=[0, 5, 10, 17, 20])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=2401)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timeout-hours", type=float, default=2.)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=.0005)
    p.add_argument("--teacher-jitter", type=float, default=0.,
                   help="Teacher-label local sensor/state neighborhoods to preserve feedback sensitivity")
    p.add_argument("--feedback-weight", type=float, default=.5,
                   help="Weight local teacher responses during training and held-out selection")
    p.add_argument("--startup-loss-weight", type=float, default=1.,
                   help="Extra actor/feedback weight for early trajectory samples, without changing validation physics")
    p.add_argument("--startup-duration-s", type=float, default=2.)
    p.add_argument("--teacher-scenarios", nargs="+", choices=("static", "forward", "lateral",
                                                            "spin_positive", "spin_negative", "dynamic"),
                   help="Learn teacher feedback only on these training scenarios; retain the frozen actor elsewhere")
    p.add_argument("--raw-feedback-teacher", action="store_true",
                   help="Re-label selected feedback with the raw fallback MLP mean, preserving saturation gains")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--residual-hidden-dims", type=int, nargs="+",
                   help="Freeze the resumed feedback and fit an additive MLP, folded into a plain MLP at save")
    p.add_argument("--anchor-data-dir", type=Path,
                   help="Sensor histories on which the frozen base actor should retain zero correction")
    p.add_argument("--anchor-grades", type=float, nargs="+", default=[0, 17, 20])
    p.add_argument("--anchor-weight", type=float, default=1.)
    p.add_argument("--anchor-reference-checkpoint", type=Path,
                   help="Optional raw MLP mentor on selected anchor grades with completed paired recovery")
    p.add_argument("--anchor-reference-evaluation-dir", type=Path)
    p.add_argument("--anchor-reference-grades", type=float, nargs="+")
    args = p.parse_args()
    args.teacher, args.output_dir = args.teacher.resolve(), args.output_dir.resolve()
    if not args.teacher.is_file():
        p.error("teacher checkpoint does not exist")
    if args.mode == "fit" and args.data_dir is None:
        p.error("fit requires --data-dir")
    if args.mode == "collect" and args.seed in (1234, 4321):
        p.error("Reserve acceptance seeds 1234 and 4321 for independent evaluation")
    if (not math.isfinite(args.startup_loss_weight) or args.startup_loss_weight < 1
            or not math.isfinite(args.startup_duration_s) or args.startup_duration_s <= 0):
        p.error("Startup loss weight must be finite >= 1, and its duration finite > 0")
    if args.teacher_scenarios and (args.mode != "fit" or not args.resume_student or not args.residual_hidden_dims):
        p.error("Selected teacher scenarios require residual fitting from a frozen resumed actor")
    if args.raw_feedback_teacher and (not args.teacher_scenarios or args.teacher_jitter <= 0):
        p.error("Raw feedback requires selected teacher scenarios and positive feedback jitter")
    if args.anchor_reference_checkpoint:
        if (args.mode != "fit" or not args.anchor_data_dir or not args.resume_student
                or not args.residual_hidden_dims or args.teacher_jitter <= 0
                or not args.anchor_reference_evaluation_dir or not args.anchor_reference_grades
                or not set(args.anchor_reference_grades) <= set(args.anchor_grades)):
            p.error("Anchor mentor requires residual fitting, positive jitter, paired reports and selected anchor grades")
    elif args.anchor_reference_evaluation_dir or args.anchor_reference_grades:
        p.error("Anchor mentor reports/grades require its checkpoint")
    if args.mode == "collect":
        collect(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
