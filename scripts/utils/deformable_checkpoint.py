"""Reject silent changes to deformable action/current observation semantics."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import yaml


def configure_recovery_precision(policy_cfg, algorithm_cfg):
    """Use deployment FP32 math for contact-sensitive policy preservation."""
    if (policy_cfg.get("class_name") not in ("ActorCriticSuspensionMLP", "ActorCriticSuspensionRoutedMLP", "ActorCriticSuspensionDefaultSplitMLP")
            or policy_cfg.get("history_length") != 5 or not algorithm_cfg.get("steep_preservation_weight",0.)):
        return None
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return dict(actor_math="float32",allow_tf32=False,
                purpose="Match independent Isaac Sim evaluation and native ONNX inference")


def load_deformable_agent_config(checkpoint, device, *, params_dir=None):
    """Use the saved policy architecture rather than a play task's defaults."""
    params = Path(params_dir) if params_dir is not None else Path(checkpoint).resolve().parent / "params"
    path = params / "agent.yaml"
    if not path.is_file():
        raise ValueError(f"Keep params/agent.yaml with the deformable checkpoint: {path}")
    saved = yaml.safe_load(path.read_text())
    if not isinstance(saved, dict):
        raise ValueError("Saved deformable agent configuration must be a mapping")
    for field in ("policy", "algorithm"):
        if not isinstance(saved.get(field), dict) or not saved[field].get("class_name"):
            raise ValueError(f"Saved {field}.class_name is required")
    history = saved["policy"].get("history_length", 1)
    if type(history) is not int or history not in (1, 4, 5, 8):
        raise ValueError("Saved deformable history must be 1, 4, 5 or 8")
    if saved.get("class_name", "OnPolicyRunner") != "OnPolicyRunner":
        raise ValueError("Deformable play supports saved OnPolicyRunner configs only")
    saved["device"] = str(device)
    return saved


def snapshot_real2sim_model(cfg, log_dir):
    if not getattr(cfg, "real2sim_enabled", False):
        return
    source = Path(cfg.real2sim_model_path)
    target = Path(log_dir) / "params" / "real2sim_model.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    cfg.real2sim_model_path = str(target.resolve())
    cfg.real2sim_model_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()


def validate_deformable_policy(checkpoint, policy_cfg, *, params_dir=None):
    """MLP fine-tunes require the same architecture, including history width."""
    saved = load_deformable_agent_config(checkpoint, "cpu", params_dir=params_dir)["policy"]
    mlp_classes = ("ActorCriticSuspensionMLP", "ActorCriticSuspensionRoutedMLP", "ActorCriticSuspensionDefaultSplitMLP")
    if not any(name in mlp_classes for name in (saved["class_name"], policy_cfg["class_name"])):
        return  # Existing Transformer geometry-input expansion remains supported.
    keys = ["class_name", "history_length", "actor_hidden_dims", "critic_hidden_dims",
            "activation", "actor_obs_normalization", "critic_obs_normalization"]
    routed_classes = ("ActorCriticSuspensionRoutedMLP", "ActorCriticSuspensionDefaultSplitMLP")
    if any(name in routed_classes for name in (saved["class_name"], policy_cfg["class_name"])):
        keys.extend(("expert_hidden_dims", "router_hidden_dims", "routing_confidence", "routing_load_threshold"))
    if "ActorCriticSuspensionDefaultSplitMLP" in (saved["class_name"], policy_cfg["class_name"]):
        keys.extend(("split_router_hidden_dims", "split_routing_confidence"))
        if any(config.get("moving_expert_hidden_dims") is not None for config in (saved,policy_cfg)):
            keys.extend(("moving_expert_hidden_dims", "standing_command_tolerance"))
    for key in keys:
        if saved.get(key) != policy_cfg.get(key):
            raise ValueError(
                f"Deformable MLP checkpoint {key} differs from the requested policy. "
                "Start the new MLP without --checkpoint; weights cannot be silently converted."
            )


def validate_deformable_checkpoint(cfg, checkpoint, policy_cfg=None, *, allow_runtime_guard_transition=False,
                                   params_dir=None):
    if not hasattr(cfg, "action_contract_version"):
        return  # Other robot families have their own contracts.
    if policy_cfg is not None:
        validate_deformable_policy(checkpoint, policy_cfg, params_dir=params_dir)
    params = Path(params_dir) if params_dir is not None else Path(checkpoint).resolve().parent / "params"
    env_path = params / "env.yaml"
    if not env_path.is_file():
        raise ValueError(f"Keep params/env.yaml with the deformable checkpoint: {env_path}")
    # BaseLoader reads config tags as data, without constructing Python objects.
    saved = yaml.load(env_path.read_text(), Loader=yaml.BaseLoader)
    if not isinstance(saved, dict) or saved.get("action_contract_version") != cfg.action_contract_version:
        raise ValueError("Deformable checkpoint action contract differs from the selected task")
    expected = getattr(cfg, "real2sim_observation_version", None) if getattr(cfg, "real2sim_enabled", False) else None
    actual = saved.get("real2sim_observation_version") if saved.get("real2sim_enabled") == "true" else None
    if expected != actual:
        raise ValueError(
            f"Deformable observation contract mismatch: checkpoint={actual!r}, task={expected!r}. "
            "Legacy torque and Real2Sim current observations cannot share a checkpoint silently."
        )
    saved_guard = saved.get("runtime_guard_enabled", "false") == "true"
    requested_guard = getattr(cfg, "runtime_guard_enabled", False)
    if saved_guard != requested_guard:
        if not (allow_runtime_guard_transition and requested_guard and not saved_guard):
            raise ValueError("Runtime guard contract differs; transfer requires an explicit fine-tune/evaluation flag")
        saved_period = float(saved["sim"]["dt"]) * int(saved["decimation"])
        if (saved.get("policy_history_length") != "5"
                or abs(saved_period - cfg.policy_history_period_s) > 1.e-9):
            raise ValueError("Guard transfer must preserve the checkpoint's five-frame history timing")
    if saved_guard and not allow_runtime_guard_transition:
        for field in ("runtime_guard_version", "policy_history_period_s", "guard_enter_deg", "guard_exit_deg",
                      "guard_plane_tolerance_m", "guard_confirmation_s", "guard_attitude_tau_s",
                      "guard_acceleration_tau_s", "guard_acceleration_fusion_scale", "guard_flat_return_rate_limit",
                      "guard_flat_return_acceleration_limit", "guard_flat_velocity_reference", "guard_flat_terminal_braking",
                      "guard_flat_terminal_acceleration_limit", "guard_startup_trajectory",
                      "guard_coordinated_return", "guard_flat_return_minimum_duration_s"):
            requested = getattr(cfg, field)
            serialized = str(requested).lower() if isinstance(requested, bool) else str(requested)
            if serialized != str(saved.get(field)):
                raise ValueError(f"Saved runtime guard contract differs: {field}")
        if (int(saved["decimation"]) != cfg.decimation
                or abs(float(saved["sim"]["dt"]) - cfg.sim.dt) > 1.e-9):
            raise ValueError("Saved guarded policy control timing differs from the task")
    if expected:
        model_path = params / "real2sim_model.json"
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.is_file() else None
        if digest is None or digest != saved.get("real2sim_model_sha256"):
            raise ValueError("Real2Sim checkpoint requires its matching params/real2sim_model.json snapshot")
        cfg.real2sim_model_path = str(model_path)
        cfg.real2sim_model_sha256 = digest
