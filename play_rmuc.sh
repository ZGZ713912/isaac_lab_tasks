#!/usr/bin/env bash
# Play the installed RMUC map with a selected robot, from any working directory.
set -euo pipefail
rmuc_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -B "$rmuc_repo_dir/scripts/rsl_rl/play_rmuc.py" "$@"
