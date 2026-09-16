#!/usr/bin/env python3
# =============================================================================
# URDF → USD 转换（保留 <mimic> 闭链），deformable_V2 专用。
#
# 为什么不用 IsaacLab 自带的 scripts/tools/convert_urdf.py：
#   IsaacLab 的 UrdfConverter 把配置字段
#       cfg.convert_mimic_joints_to_normal_joints
#   直接喂给 importer 的
#       import_config.set_parse_mimic(...)
#   （见 isaaclab/sim/converters/urdf_converter.py）。
#   而 importer 侧 `parse_mimic=True` 的真实语义是「**创建** PhysxMimicJointAPI」，
#   不是「把 mimic 变成普通关节」——字段名与语义相反。
#   convert_urdf.py 没有暴露该字段，默认 False → **mimic 会被静默忽略**，
#   平四闭链丢失且不报错。本脚本显式置 True，并在转换后断言 mimic 真的写进去了。
#
# 用法（仓库根目录，或由 convert_deformable_v2_urdf.sh 调用）：
#   ./isaaclab.sh -p scripts/tools/convert_urdf_mimic.py <in.urdf> <out.usd> \
#       --joint-target-type none [--headless]
# =============================================================================
"""Convert a URDF to USD while preserving <mimic> joints as PhysxMimicJointAPI."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=str, help="输入 URDF 路径")
parser.add_argument("output", type=str, help="输出 USD 路径")
parser.add_argument("--fix-base", action="store_true", default=False, help="把 base 固定到导入位姿")
parser.add_argument("--merge-joints", action="store_true", default=False, help="合并 fixed 关节")
parser.add_argument("--joint-stiffness", type=float, default=100.0, help="关节驱动 stiffness")
parser.add_argument("--joint-damping", type=float, default=1.0, help="关节驱动 damping")
parser.add_argument(
    "--joint-target-type",
    type=str,
    default="none",
    choices=["position", "velocity", "none"],
    help="关节驱动目标类型（deformable_V2 用 none = effort 模式）",
)
parser.add_argument(
    "--mimic-nf",
    type=float,
    default=1000.0,
    help="mimic 耦合自然频率 naturalFrequency（默认 1000；importer 默认仅 25，过软）",
)
parser.add_argument(
    "--mimic-damping",
    type=float,
    default=1.0,
    help="mimic 耦合阻尼比 dampingRatio（默认 1.0；importer 默认 0.005，几乎无阻尼会振荡）",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path


def _configure_mimic(usd_path: str, natural_frequency: float, damping_ratio: float) -> int:
    """调 mimic 耦合参数并补 referenceJointAxis。

    PhysX 的 mimic 关节是**弹性耦合**（naturalFrequency + dampingRatio），没有真正的刚性
    模式：nf 越大、dr 越接近 1，越接近 θ_mimic = gearing·θ_ref + offset。URDF importer 的
    默认是 nf=25 / dr=0.005（很软、几乎无阻尼 → 从动边漂移并振荡），且不写 referenceJointAxis。
    """
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    axis_tokens = {
        "X": UsdPhysics.Tokens.rotX,
        "Y": UsdPhysics.Tokens.rotY,
        "Z": UsdPhysics.Tokens.rotZ,
    }
    stage = Usd.Stage.Open(usd_path)
    count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        for token in (UsdPhysics.Tokens.rotX, UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ):
            if not prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, token):
                continue
            api = PhysxSchema.PhysxMimicJointAPI(prim, token)
            prim.CreateAttribute(
                f"physxMimicJoint:{token}:naturalFrequency", Sdf.ValueTypeNames.Float
            ).Set(natural_frequency)
            prim.CreateAttribute(
                f"physxMimicJoint:{token}:dampingRatio", Sdf.ValueTypeNames.Float
            ).Set(damping_ratio)
            targets = api.GetReferenceJointRel().GetTargets()
            if targets:
                ref_axis = stage.GetPrimAtPath(targets[0]).GetAttribute("physics:axis").Get()
                if ref_axis in axis_tokens:
                    api.GetReferenceJointAxisAttr().Set(axis_tokens[str(ref_axis)])
            count += 1
    stage.GetRootLayer().Save()
    return count


def main() -> None:
    urdf_path = args_cli.input if os.path.isabs(args_cli.input) else os.path.abspath(args_cli.input)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid URDF path: {urdf_path}")
    dest_path = args_cli.output if os.path.isabs(args_cli.output) else os.path.abspath(args_cli.output)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        # ↓ 名字是反的：True 才会让 importer 保留 <mimic> 并生成 PhysxMimicJointAPI
        convert_mimic_joints_to_normal_joints=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.joint_stiffness,
                damping=args_cli.joint_damping,
            ),
            target_type=args_cli.joint_target_type,
        ),
    )

    converter = UrdfConverter(cfg)
    print(f">>> Generated USD file: {converter.usd_path}")

    n = _configure_mimic(converter.usd_path, args_cli.mimic_nf, args_cli.mimic_damping)
    print(f">>> mimic 调参: {n} 处  naturalFrequency={args_cli.mimic_nf}  "
          f"dampingRatio={args_cli.mimic_damping}")

    # ---- 断言：mimic 真的落地成 PhysxMimicJointAPI ----
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = Usd.Stage.Open(converter.usd_path)
    found: dict[str, tuple[float, float, str, float, float]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        for token in (UsdPhysics.Tokens.rotX, UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ):
            if not prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, token):
                continue
            api = PhysxSchema.PhysxMimicJointAPI(prim, token)
            name = prim.GetName()
            ref_name = "?"
            targets = api.GetReferenceJointRel().GetTargets()
            if targets:
                ref_name = targets[0].name
            gearing = api.GetGearingAttr().Get()
            offset = api.GetOffsetAttr().Get()
            nf_attr = prim.GetAttribute(f"physxMimicJoint:{token}:naturalFrequency")
            dr_attr = prim.GetAttribute(f"physxMimicJoint:{token}:dampingRatio")
            nf = nf_attr.Get() if nf_attr and nf_attr.HasAuthoredValue() else None
            dr = dr_attr.Get() if dr_attr and dr_attr.HasAuthoredValue() else None
            found[name] = (gearing, offset, ref_name, nf, dr)

    print(f">>> PhysxMimicJointAPI: {len(found)} 个")
    for name, (g, o, ref, nf, dr) in sorted(found.items()):
        print(f"    {name:22s} gearing={g:+.3f} offset={o:+.3f} ref={ref:12s}"
              f" nf={nf} dr={dr}")

    # 校验：8 处、|gearing|=1、offset=0、referenceJoint 指向同角 joint_leg_N。
    # 注意 gearing 的符号由 importer 按物理轴方向自动修正（ws=−1 / upper=+1），
    # 不要求等于 URDF 的 multiplier 符号；物理正确性由 validate_mimic 仿真验证。
    n = None
    expected_ref = {
        **{f"joint_wheel_set_{i}": f"joint_leg_{i}" for i in range(1, 5)},
        **{f"joint_upper_leg_{i}": f"joint_leg_{i}" for i in range(1, 5)},
    }
    missing = [k for k in expected_ref if k not in found]
    bad = [
        k for k, ref in expected_ref.items()
        if k in found and (abs(abs(found[k][0]) - 1.0) > 1e-6 or abs(found[k][1]) > 1e-6
                           or found[k][2] != ref)
    ]
    if missing or bad:
        raise SystemExit(f"[FAIL] mimic 校验不通过: missing={missing} bad={bad}")
    print(">>> [OK] 平四 mimic 闭链 8 处结构正确（|gearing|=1, offset=0, ref 对齐）")


if __name__ == "__main__":
    main()
    simulation_app.close()
