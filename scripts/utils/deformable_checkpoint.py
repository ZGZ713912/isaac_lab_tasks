"""Reject silent changes to deformable action/current observation semantics."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import yaml


def configure_recovery_precision(policy_cfg, algorithm_cfg):
    """Use deployment FP32 math for contact-sensitive policy preservation."""
    if (policy_cfg.get("class_name") not in ("ActorCriticSuspensionMLP", "ActorCriticSuspensionRoutedMLP")
            or policy_cfg.get("history_length") != 5 or not algorithm_cfg.get("steep_preservation_weight",0.)):
        return None
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return dict(actor_math="float32",allow_tf32=False,
                purpose="Match independent Isaac Sim evaluation and native ONNX inference")


def load_deformable_agent_config(checkpoint, device):
    """Use the saved policy architecture rather than a play task's defaults."""
    path = Path(checkpoint).resolve().parent / "params" / "agent.yaml"
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


def validate_deformable_policy(checkpoint, policy_cfg):
    """MLP fine-tunes require the same architecture, including history width."""
    saved = load_deformable_agent_config(checkpoint, "cpu")["policy"]
    mlp_classes = ("ActorCriticSuspensionMLP", "ActorCriticSuspensionRoutedMLP")
    if not any(name in mlp_classes for name in (saved["class_name"], policy_cfg["class_name"])):
        return  # Existing Transformer geometry-input expansion remains supported.
    keys = ["class_name", "history_length", "actor_hidden_dims", "critic_hidden_dims",
            "activation", "actor_obs_normalization", "critic_obs_normalization"]
    if "ActorCriticSuspensionRoutedMLP" in (saved["class_name"], policy_cfg["class_name"]):
        keys.extend(("expert_hidden_dims", "router_hidden_dims", "routing_confidence", "routing_load_threshold"))
    for key in keys:
        if saved.get(key) != policy_cfg.get(key):
            raise ValueError(
                f"Deformable MLP checkpoint {key} differs from the requested policy. "
                "Start the new MLP without --checkpoint; weights cannot be silently converted."
            )


def validate_deformable_checkpoint(cfg, checkpoint, policy_cfg=None):
    if not hasattr(cfg, "action_contract_version"):
        return  # Other robot families have their own contracts.
    if policy_cfg is not None:
        validate_deformable_policy(checkpoint, policy_cfg)
    params = Path(checkpoint).resolve().parent / "params"
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
    if expected:
        model_path = params / "real2sim_model.json"
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.is_file() else None
        if digest is None or digest != saved.get("real2sim_model_sha256"):
            raise ValueError("Real2Sim checkpoint requires its matching params/real2sim_model.json snapshot")
        cfg.real2sim_model_path = str(model_path)
        cfg.real2sim_model_sha256 = digest
