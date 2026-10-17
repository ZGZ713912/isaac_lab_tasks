"""Choose a trained robot and play it on the installed RMUC field."""

import argparse
import json
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "deformable": (
        "play_deformable.py", "Robotics-Deformable-Suspension-Guarded-History-MLP-Real2Sim-v4",
        "logs/rsl_rl/deformable_guarded_mlp5_v4/2026-10-14_12-40-18_training_long_05/model_999.pt", "third-person"),
    "wheel-leg": (
        "play_wheel_leg_field.py", "Robotics-Wheel-Leg-V4-Flat-Play-v0",
        "logs/rsl_rl/wheel_leg_v4_v5_230/2026-09-22_23-32-52/model_19999.pt", "follow"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Other arguments are passed to the robot's play script.")
    parser.add_argument("robot", choices=DEFAULTS, nargs="?", default="deformable")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--checkpoint", default=None, help="Policy path: .pt checkpoint or .onnx model.")
    policy.add_argument("--onnx", "--model", dest="onnx", default=None, help="Run this ONNX directly, without loading a .pt checkpoint.")
    parser.add_argument("--policy-config", default=None, help="Matching training run or params directory for ONNX.")
    parser.add_argument("--task", default=None)
    parser.add_argument("--field-root", default=str(ROOT / "assets/terrains/RMUC2026_V2_gray_fence_revision_03"))
    parser.add_argument("--camera-mode", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python", default=os.environ.get("ISAACLAB_PYTHON", "/home/noir/miniconda3/envs/isaaclab/bin/python"))
    parser.add_argument("--dry-run", action="store_true", help="Print the launch command without starting Isaac Sim.")
    args, extra = parser.parse_known_args()
    # Keep the familiar play interface: --checkpoint accepts .pt or .onnx.
    if args.checkpoint and Path(args.checkpoint).suffix.lower() == ".onnx":
        args.onnx, args.checkpoint = args.checkpoint, None
    script, task, model, camera = DEFAULTS[args.robot]
    selected = Path(args.onnx or args.checkpoint).expanduser().resolve() if args.onnx or args.checkpoint else ROOT / model
    field = Path(args.field_root).expanduser().resolve()
    if not Path(args.python).is_file():
        parser.error("Isaac Lab Python is missing; set --python or ISAACLAB_PYTHON")
    if not selected.is_file():
        parser.error(f"Policy missing: {selected}; use --onnx PATH or --checkpoint PATH")
    if args.policy_config and not args.onnx:
        parser.error("--policy-config is for --onnx; checkpoints use their own params directory")
    package = field / "asset_package.json"
    if not package.is_file():
        parser.error(f"Map package missing: {package}; see docs/rmuc-field-play.md")
    manifest = json.loads(package.read_text())
    entry = (field / manifest["entries"]["isaac_static_collision_usd"]).resolve()
    if not entry.is_relative_to(field) or not entry.is_file():
        parser.error("The field collision USD must exist inside the package")
    command = [str(ROOT / "run_gui.sh"), args.python, "-u", "-B", str(ROOT / "scripts/rsl_rl" / script),
               "--task", args.task or task, "--onnx" if args.onnx else "--checkpoint", str(selected), "--field-root", str(field),
               "--camera-mode", args.camera_mode or camera, "--device", args.device, *extra]
    if args.policy_config:
        command += ["--policy-config", str(Path(args.policy_config).expanduser().resolve())]
    print("[RMUC] " + shlex.join(command), flush=True)
    if not args.dry_run:
        os.chdir(ROOT)
        os.execv(command[0], command)


if __name__ == "__main__":
    main()
