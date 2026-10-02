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
    policy.onnx   : 输入 ``obs`` float32，输出 ``actions`` float32，opset 18，单文件内联
                    - 单帧 MLP / 单帧 transformer: obs (1, obs_dim)
                    - 历史 transformer: obs (1, history_length, frame_dim)，即 RMCS 约定的 rank-3
    policy.pt     : 对应的 TorchScript

不写任何 ``rmcs_*`` metadata（布局/模型类型由 RMCS 的 stamp 工具盖章）。
历史 transformer 如需 rank-2 兜底，加 ``--flat``（部署端 auto 会按 mlp 推断）。

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


def _is_transformer(state_dict: dict) -> bool:
    return "actor.global_embed.weight" in state_dict


def _build_transformer_actor(state_dict: dict, checkpoint_path: str) -> tuple[nn.Module, int, int, int]:
    """重建 ActorCriticTransformer 的 actor；超参从同 run 的 params/agent.yaml 读取。"""
    for pkg in ("agent_rl",):
        pkg_root = os.path.join(_REPO_ROOT, "source", pkg)
        if pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)
    import yaml

    from agent_rl.rsl_rl.modules.actor_critic_transformer import (
        DEFORMABLE_ACTOR_LAYOUT,
        LegTokenTransformer,
    )

    agent_yaml = os.path.join(os.path.dirname(checkpoint_path), "params", "agent.yaml")
    if not os.path.isfile(agent_yaml):
        raise SystemExit(f"transformer checkpoint 需要 {agent_yaml} 以读取网络超参")
    with open(agent_yaml) as f:
        pcfg = yaml.unsafe_load(f)["policy"]

    layout = pcfg.get("actor_layout") or DEFORMABLE_ACTOR_LAYOUT
    # 历史策略的观测是 history_length 个 32 维帧拼成的扁平向量（V1 合同 = 8 * 32 = 256）。
    # 单帧旧 checkpoint 没有 history_length 键，退回 1，行为与之前一致。
    history_length = int(pcfg.get("history_length", 1))
    frame_dim = max(list(layout["global"]) + [i for l in layout["legs"] for i in l]) + 1
    obs_dim = history_length * frame_dim
    act_dim = int(state_dict["std"].shape[0]) if "std" in state_dict else int(state_dict["log_std"].shape[0])
    actor = LegTokenTransformer(
        obs_dim,
        act_dim,
        layout,
        d_model=int(pcfg.get("d_model", 64)),
        nhead=int(pcfg.get("nhead", 4)),
        num_layers=int(pcfg.get("num_layers", 2)),
        dim_ff=int(pcfg.get("dim_ff", 128)),
        head_hidden=int(pcfg.get("head_hidden", 64)),
        head=str(pcfg.get("actor_head", "per_leg")),
        history_length=history_length,
    )
    actor.load_state_dict({k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")})
    actor.eval()
    return actor, obs_dim, act_dim, history_length


def _verify_onnx(onnx_path: str, actor: nn.Module, input_shape: tuple, act_dim: int, device: torch.device) -> None:
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
    assert inputs and inputs[0][0] == "obs", f"primary input must be 'obs': {inputs}"
    assert tuple(inputs[0][1]) == tuple(input_shape), f"input shape mismatch: {inputs} != {input_shape}"
    assert len(outputs) == 1 and outputs[0][0] == "actions", f"single output must be 'actions': {outputs}"
    assert tuple(outputs[0][1])[-1] == act_dim, f"output feature mismatch: {outputs}"
    element_types = {i.type.tensor_type.elem_type for i in model.graph.input} | {
        o.type.tensor_type.elem_type for o in model.graph.output
    }
    assert element_types == {onnx.TensorProto.FLOAT}, f"obs/actions must be float32: {element_types}"
    rmcs_keys = [p.key for p in model.metadata_props if p.key.startswith("rmcs_")]
    assert not rmcs_keys, f"导出文件不得携带 rmcs_* metadata（盖章工具负责写）: {rmcs_keys}"
    print("[VERIFY] onnx.checker: OK  (obs/actions float32, 无 rmcs_* metadata)")

    try:
        import onnxruntime as ort
    except ImportError:
        print("[VERIFY] onnxruntime 未安装，跳过数值一致性对比（可 pip install onnxruntime 后再跑）。")
        return

    x = torch.randn(*input_shape, device=device)
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
    parser.add_argument(
        "--flat",
        action="store_true",
        default=False,
        help="历史 transformer 也按 rank-2 [1, obs_dim] 导出（兜底；RMCS 端 auto 只能推断成 mlp）。",
    )
    args = parser.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")
    out_dir = args.output_dir or os.path.join(os.path.dirname(ckpt), "exported")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, args.filename)
    jit_path = os.path.join(out_dir, args.jit_filename)

    state_dict = _load_state_dict(ckpt)
    transformer = _is_transformer(state_dict)
    history_length = 1
    if transformer:
        actor, obs_dim, act_dim, history_length = _build_transformer_actor(state_dict, ckpt)
        desc = f"LegTokenTransformer(history_length={history_length})"
    else:
        actor, obs_dim, act_dim = _build_actor(state_dict)
        desc = str([m for m in actor])
    # RMCS 部署约定：历史 transformer 导出 rank-3 [1, history, frame]（auto 推断成 transformer）；
    # 单帧模型仍为 rank-2 [1, obs]。--flat 可强制历史模型也走 rank-2 作为兜底。
    if transformer and history_length > 1 and not args.flat:
        input_shape = (1, history_length, obs_dim // history_length)
    else:
        input_shape = (1, obs_dim)
    print(f"[INFO] checkpoint : {ckpt}")
    print(f"[INFO] actor      : obs_dim={obs_dim}  act_dim={act_dim}  arch={desc}")
    print(f"[INFO] obs_shape  : {list(input_shape)}  (rank-{len(input_shape)})")
    print(f"[INFO] output_dir : {out_dir}")

    dummy = torch.zeros(*input_shape)
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
        scripted = torch.jit.trace(actor, dummy) if transformer else torch.jit.script(actor)
        scripted.save(jit_path)
        print(f"[OK] TorchScript -> {jit_path}")

    _verify_onnx(onnx_path, actor, input_shape, act_dim, torch.device("cpu"))


if __name__ == "__main__":
    main()
