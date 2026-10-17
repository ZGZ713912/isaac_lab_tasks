"""Check the Git file boundary and source syntax without simulator dependencies."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "CONTRIBUTING.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
    "pyproject.toml", "Makefile", ".github/workflows/tests.yml",
    "assets/README.md", "docs/README.md", "models/README.md", "references/README.md",
    "src/agent_world/__init__.py", "src/agent_tasks/__init__.py", "src/agent_rl/__init__.py",
    "src/agent_rl/rsl_rl/env/__init__.py", "src/agent_rl/rsl_rl/env/vec_env.py",
    "src/agent_rl/rsl_rl/env/rsl_rl_vec_env.py",
)


def artifact_reason(name: str) -> str | None:
    """Describe disallowed tracked output, allowing actual source env packages."""
    path = Path(name)
    if path.parts[0] in {"参考例程", "pretrained", "runs", "outputs", "logs", "reports", "debug", ".local"}:
        return "local reference or runtime output"
    if path.parts[0].startswith("runs_v"):
        return "legacy runtime output"
    if path.parts[0] in {"models", "references"} and name != f"{path.parts[0]}/README.md":
        return "local model or reference checkout"
    if "__pycache__" in path.parts or any(part.endswith(".egg-info") for part in path.parts):
        return "generated Python metadata"
    if path.suffix in {".pt", ".pth", ".onnx", ".mp4", ".zip", ".log", ".ses", ".pyc"}:
        return "checkpoint, archive or generated file"
    if path.name.startswith("events.out.tfevents.") or ".usdshim" in path.parts:
        return "workstation runtime state"
    if path.parts[0] == "source":
        return "obsolete source layout"
    return None


def main() -> int:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
    names = [name for name in result.stdout.decode().split("\0") if name]
    errors = []
    for required in REQUIRED:
        if required not in names or not (ROOT / required).is_file():
            errors.append(f"Required file is missing from Git: {required}")
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"], cwd=ROOT,
        input=result.stdout, capture_output=True, check=False,
    )
    if ignored.returncode not in (0, 1):
        errors.append(ignored.stderr.decode().strip())
    for name in ignored.stdout.decode().split("\0"):
        if name:
            errors.append(f"Ignored file is still tracked: {name}")
    python_count = shell_count = 0
    for name in names:
        reason = artifact_reason(name)
        if reason:
            errors.append(f"{name}: {reason}")
            continue
        path = ROOT / name
        if not path.is_file():
            errors.append(f"Tracked file is missing: {name}")
            continue
        if path.suffix == ".py":
            python_count += 1
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
                for node in ast.walk(tree):
                    imports = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
                        [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                    )
                    if any(item == "scripts.tools" or item.startswith("scripts.tools.") for item in imports):
                        errors.append(f"{name}:{node.lineno}: obsolete scripts.tools import")
            except (SyntaxError, UnicodeError) as error:
                errors.append(f"{name}: {error}")
        elif path.suffix == ".sh":
            shell_count += 1
            shell = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            if shell.returncode:
                errors.append(f"{name}: {shell.stderr.strip()}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Repository OK: {len(names)} tracked files; {python_count} Python and {shell_count} shell files checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
