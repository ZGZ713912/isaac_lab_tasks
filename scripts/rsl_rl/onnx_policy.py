"""Stateless ONNX inference over the environment's existing, oldest-first history."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


def resolve_policy_config(model, directory=None):
    """Find params beside a model or in its run; never fall back to another run."""
    model = Path(model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"ONNX model does not exist: {model}")
    if directory is not None:
        root = Path(directory).expanduser().resolve()
        candidates = (root, root / "params")
    else:
        candidates = (model.parent / "params", model.parent.parent / "params")
    for params in candidates:
        if all((params / name).is_file() for name in ("env.yaml", "agent.yaml")):
            return params
    raise ValueError("ONNX needs its matching params/env.yaml and params/agent.yaml; "
                     "keep <run>/exported/model.onnx with <run>/params, or use --policy-config <run-or-params>")


def load_policy_config(params):
    # Isaac YAML includes object tags; BaseLoader treats those as ordinary data.
    saved = yaml.load((params / "env.yaml").read_text(), Loader=yaml.BaseLoader)
    agent = yaml.safe_load((params / "agent.yaml").read_text())
    if not isinstance(saved, dict) or not isinstance(agent, dict) or not isinstance(agent.get("policy"), dict):
        raise ValueError("ONNX policy config must contain environment and policy mappings")
    if agent.get("empirical_normalization") or agent["policy"].get("actor_obs_normalization"):
        raise ValueError("Normalized policies need an explicit embedded-normalizer contract; "
                         "this field adapter accepts unnormalized actor observations only")
    if agent.get("obs_groups", {}).get("policy", ["policy"]) != ["policy"]:
        raise ValueError("ONNX field play expects the actor's policy observation group only")
    return saved, agent


def validate_saved_fields(cfg, saved, fields):
    """Reject same-sized tensors whose scales, controls or history clocks differ."""
    for name in fields:
        requested = getattr(cfg, name)
        serialized = str(requested).lower() if isinstance(requested, bool) else str(requested)
        if name not in saved or str(saved[name]) != serialized:
            raise ValueError(f"ONNX/task contract mismatch: {name} (saved={saved.get(name)!r}, task={serialized!r})")
    if not math.isclose(float(saved["sim"]["dt"]), cfg.sim.dt, rel_tol=0., abs_tol=1.e-12):
        raise ValueError("ONNX/task physics timestep mismatch")


class OnnxPolicy:
    def __init__(self, model, *, observation_dim, action_dim, history_length, frame_dim, saved_env=None):
        import onnxruntime as ort

        self.model = Path(model).expanduser().resolve()
        self.runtime_version = ort.__version__
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.history_length, self.frame_dim = history_length, frame_dim
        if history_length * frame_dim != observation_dim:
            raise ValueError("ONNX history/frame size does not match the environment observation width")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("ONNX field play requires one observation input and one action output; "
                             "extra inputs/recurrent state need a separate explicit adapter")
        source, target = inputs[0], outputs[0]
        if source.type != "tensor(float)" or target.type != "tensor(float)":
            raise ValueError("ONNX observations and actions must be float32")
        if len(source.shape) == 2:
            self.input_shape = (1, observation_dim)
        elif len(source.shape) == 3:
            self.input_shape = (1, history_length, frame_dim)
        else:
            raise ValueError(f"ONNX observation rank must be 2 or 3, got {source.shape}")
        for actual, expected in zip(source.shape, self.input_shape):
            if isinstance(actual, int) and actual != expected:
                raise ValueError(f"ONNX input shape {source.shape} differs from task {self.input_shape}")
        if len(target.shape) != 2 or any(isinstance(a, int) and a != e
                for a, e in zip(target.shape, (1, action_dim))):
            raise ValueError(f"ONNX action shape {target.shape} differs from task (1, {action_dim})")
        self.input_name, self.output_name = source.name, target.name
        self.metadata = self.session.get_modelmeta().custom_metadata_map
        for key, expected in (("rmcs_history_length", history_length), ("rmcs_obs_frame_size", frame_dim)):
            if key in self.metadata and int(self.metadata[key]) != expected:
                raise ValueError(f"ONNX metadata mismatch: {key}")
        layout = self.metadata.get("rmcs_obs_layout", "")
        if layout.startswith("v3-history="):
            if int(layout.split("|", 1)[0].split("=", 1)[1]) != history_length:
                raise ValueError("ONNX layout history differs from task")
            if sum(int(field.rsplit("@", 1)[1]) for field in layout.split("|")[1:]) != frame_dim:
                raise ValueError("ONNX layout frame size differs from task")
        if self.metadata.get("training_obs_history_order", "oldest_to_newest") != "oldest_to_newest":
            raise ValueError("ONNX requires a different history order")
        if saved_env is not None:
            for key, field in (("training_observation_version", "real2sim_observation_version"),
                               ("training_real2sim_model_sha256", "real2sim_model_sha256")):
                if key in self.metadata and self.metadata[key] != saved_env.get(field):
                    raise ValueError(f"ONNX metadata mismatch: {key}")
            if "training_action_contract" in self.metadata:
                contract = json.loads(self.metadata["training_action_contract"])
                period = float(saved_env["sim"]["dt"]) * int(saved_env["decimation"])
                if (contract.get("version") != saved_env.get("action_contract_version") or
                        not math.isclose(float(contract["policy_period_s"]), period, abs_tol=1.e-9)):
                    raise ValueError("ONNX metadata action contract or policy period differs")
        self.calls = 0
        self._run(np.zeros((1, observation_dim), dtype=np.float32))

    def _run(self, observations):
        if observations.shape != (1, self.observation_dim) or observations.dtype != np.float32:
            raise ValueError(f"Expected float32 environment observations (1, {self.observation_dim})")
        if not np.isfinite(observations).all():
            raise ValueError("Nonfinite ONNX observations")
        # The environment owns sampling/reset; reshaping never adds history frames.
        values = np.ascontiguousarray(observations.reshape(self.input_shape))
        actions = self.session.run([self.output_name], {self.input_name: values})[0]
        if actions.shape != (1, self.action_dim) or actions.dtype != np.float32 or not np.isfinite(actions).all():
            raise ValueError("ONNX produced invalid or nonfinite actions")
        return actions

    def __call__(self, observations):
        import torch

        if not isinstance(observations, torch.Tensor):
            observations = observations["policy"]
        actions = self._run(observations.detach().cpu().numpy())
        self.calls += 1
        return torch.from_numpy(actions).to(device=observations.device)

    def audit(self):
        return dict(backend="onnxruntime", model=str(self.model),
                    model_sha256=hashlib.sha256(self.model.read_bytes()).hexdigest(),
                    runtime_version=self.runtime_version,
                    providers=self.session.get_providers(), input_name=self.input_name,
                    input_shape=list(self.input_shape), output_name=self.output_name,
                    output_shape=[1, self.action_dim], policy_calls=self.calls,
                    history_order="oldest_to_newest", history_owner="environment")
