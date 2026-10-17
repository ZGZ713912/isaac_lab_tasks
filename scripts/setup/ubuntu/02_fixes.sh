#!/usr/bin/env bash
# Compatibility entry: validate the installed project without rewriting source.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
python tools/check_repository.py
python - <<'PYTHON'
import importlib.util
from pathlib import Path
import agent_world
from agent_rl.rsl_rl.env import VecEnv, RslRlVecEnvWrapper
from scripts.utils.velocity_trace_html import build_velocity_trace_html

assert importlib.util.find_spec("agent_tasks") is not None
assert Path(agent_world.RobotAssetPath).is_dir()
print("Project packages, vector environment interface and robot assets are available.")
PYTHON
