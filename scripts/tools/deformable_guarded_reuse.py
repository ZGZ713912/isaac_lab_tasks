"""Reuse a completed short PPO run when only its acceptance procedure changes."""

import ast
import hashlib
import json
from pathlib import Path
import re

from deformable_mlp_guarded_train import actor_update_info, checkpoint_info
from deformable_precision_pipeline import ROOT


# These files do not define the policy, observations, simulator, reward or PPO.
ACCEPTANCE_ONLY_SOURCES = frozenset((
    "scripts/tools/deformable_guarded_assess.py",
    "scripts/tools/deformable_guarded_train.py",
    "scripts/tools/deformable_guarded_reuse.py",
))

TRAINING_CONFIG = "source/agent_tasks/agent_tasks/direct/deformable_suspension/agents/rsl_rl_ppo_cfg.py"


def evaluation_config_equivalent(old, new):
    """Only two numeric optimizer/loss settings may differ in cached physics."""
    def signature(source):
        tree = ast.parse(source)
        found = set()
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "DeformableGuardedRoutedMLPPPORunnerCfg":
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "__post_init__":
                        kept = []
                        for statement in method.body:
                            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                                target = ast.unparse(statement.targets[0])
                                if target in ("self.algorithm.learning_rate", "self.algorithm.flat_hold_startup_weight"):
                                    value = statement.value
                                    if (not isinstance(value, ast.Constant) or type(value.value) not in (int, float)
                                            or not 0 < value.value < float("inf")):
                                        raise ValueError("Cached evaluation needs literal positive training settings")
                                    found.add(target)
                                    continue
                            kept.append(statement)
                        method.body = kept
        if len(found) != 2:
            raise ValueError("Guarded training configuration was not identified")
        return ast.dump(tree, include_attributes=False)
    return signature(old) == signature(new)


def evaluation_sources_match(directory, previous, current, root=ROOT):
    """Validate saved bytes and current physics; training-only changes are explicit."""
    directory, root = Path(directory), Path(root)
    ignored = ACCEPTANCE_ONLY_SOURCES | {"scripts/tools/run_deformable_guarded_experiment.py"}
    checked = 0
    for name, expected in current.items():
        if name in ignored:
            continue
        snapshot, actual = directory / "source" / name, root / name
        if (not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != previous.get(name)
                or not actual.is_file() or hashlib.sha256(actual.read_bytes()).hexdigest() != expected):
            return False
        if previous.get(name) != expected:
            if name != TRAINING_CONFIG:
                return False
            try:
                equivalent = evaluation_config_equivalent(snapshot.read_text(), actual.read_text())
            except (ValueError, SyntaxError):
                return False
            if not equivalent:
                return False
        checked += 1
    return checked > 0 and any(name.endswith("/dynamic_env.py") for name in current)


def training_checkpoint(log, experiment, name, root=ROOT):
    """Select the exact run printed by this subprocess, including its timestamp."""
    text = Path(log).read_text(errors="replace")
    stamps = re.findall(r"^Exact experiment name requested from command line: (\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", text, re.M)
    directories = re.findall(r"^\[INFO\] Logging experiment in directory: (.+)$", text, re.M)
    directory = Path(root).resolve() / "logs/rsl_rl" / experiment
    if len(stamps) != 1 or len(directories) != 1 or Path(directories[0]).resolve() != directory:
        raise ValueError("Training log does not identify exactly one expected experiment")
    run = directory / f"{stamps[0]}_{name}"
    candidates = [path for path in run.glob("model_*.pt") if path.stem[6:].isdigit()]
    if not candidates or not (run / "params/agent.yaml").is_file():
        raise ValueError(f"Incomplete exact training run: {run}")
    return max(candidates, key=lambda path: int(path.stem[6:]))


def reuse_short_run(reference, directory, sources, updates, seed):
    directory = Path(directory).resolve()
    reference = Path(reference).resolve()
    state = json.loads((directory / "status.json").read_text())
    if state["status"] not in ("needs_iteration", "completed", "failed"):
        raise ValueError("Short run must have stopped before checkpoint reuse")
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    if Path(state["checkpoint"]).resolve() != reference or state["checkpoint_sha256"] != digest:
        raise ValueError("Reused short run has a different reference checkpoint")
    checked = {}
    for name, expected in sources.items():
        if name in ACCEPTANCE_ONLY_SOURCES:
            continue
        if state["sources"].get(name) != expected:
            raise ValueError(f"Short-run training/runtime source differs: {name}")
        snapshot = directory / "source" / name
        if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Short-run source snapshot differs: {name}")
        checked[name] = expected
    if not checked or not any(name.endswith("/dynamic_env.py") for name in checked):
        raise ValueError("Short-run reuse requires the complete runtime source manifest")
    jobs = [job for job in state["jobs"] if job["name"].endswith("_short")]
    if len(jobs) != 1 or jobs[0]["status"] != "completed" or jobs[0].get("exit_code") != 0:
        raise ValueError("Short training subprocess did not complete successfully")
    command = jobs[0]["command"]
    for flag, value in (("--max_iterations", updates), ("--seed", seed), ("--checkpoint", reference)):
        if flag not in command or command[command.index(flag) + 1] != str(value):
            raise ValueError(f"Reused short training command differs: {flag}")
    records = state["training"]
    recovered = False
    if not records and state["status"] == "failed":
        if "--run_name" not in command or command[command.index("--run_name") + 1] != jobs[0]["name"]:
            raise ValueError("Recovery requires the exact subprocess run name")
        candidate = training_checkpoint(jobs[0]["log"], "deformable_guarded_mlp5_v4", jobs[0]["name"])
        info = checkpoint_info(candidate)
        record = dict(checkpoint=str(candidate), checkpoint_sha256=info["sha256"], updates=updates, seed=seed)
        recovered = True
    elif len(records) == 1 and records[0]["updates"] == updates and records[0]["seed"] == seed:
        record = records[0].copy()
    else:
        raise ValueError("Reuse requires the exact completed short training budget and seed")
    candidate = Path(record["checkpoint"]).resolve()
    info = checkpoint_info(candidate)
    if info["sha256"] != record["checkpoint_sha256"] or info["iteration"] != updates - 1:
        raise ValueError("Reused short checkpoint bytes/budget differ from the training record")
    update = actor_update_info(reference, candidate)
    if not update["policy_updated"]:
        raise ValueError("Reused short checkpoint did not update the actor")
    record.update(actor_update=update, provenance=dict(
        reused_from=str(directory), original_status=state["status"],
        training_sources=checked, additional_short_updates=0,
        recovered_record_from_exact_log=recovered))
    return candidate, record
