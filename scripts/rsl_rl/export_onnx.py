# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
# =============================================================================

"""独立、任务无关的 ONNX 导出（部署用）。

直接从 rsl_rl 的 ``model_XXXX.pt`` 里重建 actor MLP 并导出，**不加载 Isaac 环境**，
因此不依赖任何任务代码/键盘分支，也不会触发仿真。

产物（默认写到 checkpoint 同目录的 ``exported/``）：
    policy.onnx   : 输入 ``obs`` (1, obs_dim)，输出 ``actions`` (1, act_dim)，opset 18
    policy.pt     : 对应的 TorchScript

用法：
    python scripts/rsl_rl/export_onnx.py \
        --checkpoint logs/rsl_rl/deformable_suspension_direct/<ts>/model_999.pt

    # 自定义输出目录/文件名/opset：
    python scripts/rsl_rl/export_onnx.py --checkpoint=<...>.pt \
        --output_dir=/tmp/exp --filename=policy.onnx --opset=18
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn


def _load_state_dict(checkpoint_path: str) -> dict:
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")


def _build_actor(state_dict: dict) -> tuple[nn.Sequential, int, int]:
    """从 ``actor.*`` 权重重建 rsl_rl ActorCritic 的 actor MLP（Linear+ELU）。"""
    lin_keys = sorted(
        (k for k in state_dict if k.startswith("actor.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    if not lin_keys:
        raise ValueError("checkpoint 中没有 actor.* 权重，无法导出。")
    idxs = [int(k.split(".")[1]) for k in lin_keys]
    in_dims = [state_dict[f"actor.{i}.weight"].shape[1] for i in idxs]
    out_dims = [state_dict[f"actor.{i}.weight"].shape[0] for i in idxs]
    dims = in_dims + [out_dims[-1]]

    layers: list[nn.Module] = []
    for n in range(len(dims) - 1):
        layers.append(nn.Linear(dims[n], dims[n + 1]))
        if n < len(dims) - 2:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)
    actor.load_state_dict({k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")})
    actor.eval()
    return actor, dims[0], dims[-1]


def _verify_onnx(onnx_path: str, actor: nn.Module, obs_dim: int, device: torch.device) -> None:
    """打印 ONNX 输入/输出并做校验；若装了 onnxruntime 再做数值一致性对比。"""
    import onnx

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    def _shape(t):
        return [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]

    inputs = [(i.name, _shape(i)) for i in model.graph.input]
    outputs = [(o.name, _shape(o)) for o in model.graph.output]
    print(f"[VERIFY] onnx inputs : {inputs}")
    print(f"[VERIFY] onnx outputs: {outputs}")
    assert inputs and tuple(inputs[0][1]) == (1, obs_dim), f"input shape mismatch: {inputs}"
    print("[VERIFY] onnx.checker: OK")

    try:
        import onnxruntime as ort
    except ImportError:
        print("[VERIFY] onnxruntime 未安装，跳过数值一致性对比（可 pip install onnxruntime 后再跑）。")
        return

    x = torch.randn(1, obs_dim, device=device)
    with torch.no_grad():
        ref = actor(x).cpu().numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {inputs[0][0]: x.cpu().numpy()})[0]
    max_err = float(abs(ref - got).max())
    print(f"[VERIFY] max|torch-onnx| = {max_err:.3e}")
    assert max_err < 1e-4, f"ONNX 与 torch 输出不一致: {max_err}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rsl_rl checkpoint actor to ONNX (task-agnostic).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_XXXX.pt")
    parser.add_argument("--output_dir", type=str, default=None, help="Default: <checkpoint dir>/exported")
    parser.add_argument("--filename", type=str, default="policy.onnx", help="ONNX file name.")
    parser.add_argument("--jit_filename", type=str, default="policy.pt", help="TorchScript file name.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument(
        "--external_data",
        action="store_true",
        default=False,
        help="Keep weights as a separate policy.onnx.data sidecar (default: inline into a single self-contained .onnx).",
    )
    parser.add_argument("--no_jit", action="store_true", default=False, help="Skip TorchScript export.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose torch.onnx.export.")
    args = parser.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")
    out_dir = args.output_dir or os.path.join(os.path.dirname(ckpt), "exported")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, args.filename)
    jit_path = os.path.join(out_dir, args.jit_filename)

    state_dict = _load_state_dict(ckpt)
    actor, obs_dim, act_dim = _build_actor(state_dict)
    print(f"[INFO] checkpoint : {ckpt}")
    print(f"[INFO] actor      : obs_dim={obs_dim}  act_dim={act_dim}  layers={[m for m in actor]}")
    print(f"[INFO] output_dir : {out_dir}")

    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(
        actor,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes=None,
        verbose=args.verbose,
    )
    print(f"[OK] ONNX  -> {onnx_path}")

    # torch.onnx.export 可能把权重外置成 policy.onnx.data；默认内联成自包含单文件，
    # 避免部署只拷贝 .onnx 导致加载失败（常见“导出问题”根因）。
    if not args.external_data:
        import onnx

        model = onnx.load(onnx_path)  # 连带读取同目录 .data
        onnx.save_model(model, onnx_path, save_as_external_data=False)
        data_sidecar = onnx_path + ".data"
        if os.path.exists(data_sidecar):
            os.remove(data_sidecar)
        print(f"[OK] 已内联为自包含单文件（{os.path.getsize(onnx_path)} bytes）")

    if not args.no_jit:
        scripted = torch.jit.script(actor)
        scripted.save(jit_path)
        print(f"[OK] TorchScript -> {jit_path}")

    _verify_onnx(onnx_path, actor, obs_dim, torch.device("cpu"))


if __name__ == "__main__":
    main()
