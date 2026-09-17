# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# deformable 键盘 play 的可视化：
#   A. 接触标记：接触中的 body 显示红球（用 ContactSensor 自带 debug 可视化）
#   B. 车身水平：base 处两根细杆 —— 绿色=世界竖直参考，红色=车身 z 轴（随姿态倾斜）
#   C. HUD 文字：roll/pitch、离地高度、四轮接触力、q_cmd、命令
# =============================================================================
"""Play-time visualization helpers for the deformable active-suspension task."""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class DeformablePlayVis:
    """A（接触红球）+ B（水平两杆）+ C（HUD）可视化。"""

    def __init__(
        self,
        env,
        enable_contact: bool = True,
        enable_level: bool = True,
        enable_hud: bool = True,
        hud_interval: int = 5,
    ):
        self._env = env
        self._unwrapped = env.unwrapped
        self._robot = self._unwrapped.robot
        self._counter = 0
        self._hud_interval = max(1, int(hud_interval))
        self._labels: dict = {}
        self._window = None
        self._ui = None
        self._level_markers: VisualizationMarkers | None = None

        # ---- A. 接触标记 ----
        if enable_contact:
            try:
                ok = self._unwrapped.contact_sensor.set_debug_vis(True)
                print(f"[VIS] 接触可视化(红球=接触): {'已开启' if ok else '传感器不支持'}")
            except Exception as exc:  # noqa: BLE001
                print(f"[VIS] 接触可视化开启失败: {exc}")

        # ---- B. 车身水平两杆 ----
        if enable_level:
            cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/DeformableBodyLevel",
                markers={
                    "world_up": sim_utils.CylinderCfg(
                        radius=0.006,
                        height=0.5,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    "body_up": sim_utils.CylinderCfg(
                        radius=0.006,
                        height=0.5,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._level_markers = VisualizationMarkers(cfg)
            print("[VIS] 车身水平标记: 绿杆=世界竖直参考, 红杆=车身 z 轴（倾斜即夹角）")

        # ---- C. HUD ----
        if enable_hud:
            try:
                import omni.ui as ui

                self._ui = ui
                self._window = ui.Window("Deformable Play HUD", width=380, height=250)
                with self._window.frame:
                    with ui.VStack(spacing=4):
                        for key in ("mode", "q_cmd", "cmd", "roll", "pitch", "height", "force", "contact"):
                            self._labels[key] = ui.Label("", height=18)
                print("[VIS] HUD 已创建")
            except Exception as exc:  # noqa: BLE001
                print(f"[VIS] HUD 创建失败: {exc}")

    # ------------------------------------------------------------------
    def update(self) -> None:
        self._counter += 1
        if self._level_markers is not None:
            self._update_level_markers()
        if self._window is not None and self._counter % self._hud_interval == 0:
            try:
                self._update_hud()
            except Exception:  # noqa: BLE001
                pass

    def _update_level_markers(self) -> None:
        data = self._robot.data
        base_pos = data.root_pos_w  # (N,3)
        quat = data.root_quat_w  # (N,4) wxyz
        device = base_pos.device
        if base_pos.shape[0] != 1:
            # 只可视化第一个 env，避免实例数不停变化
            base_pos = base_pos[:1]
            quat = quat[:1]
        up = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        pos = base_pos + torch.tensor([[0.0, 0.0, 0.25]], device=device)
        translations = torch.cat([pos, pos], dim=0)
        orientations = torch.cat([up, quat], dim=0)
        self._level_markers.visualize(
            translations=translations, orientations=orientations, marker_indices=[0, 1]
        )

    def _update_hud(self) -> None:
        unwrapped = self._unwrapped
        pgb = self._robot.data.projected_gravity_b[0]
        roll = math.degrees(math.atan2(-pgb[1].item(), -pgb[2].item()))
        pitch = math.degrees(math.atan2(pgb[0].item(), -pgb[2].item()))
        height = float(unwrapped.base_height[0].item())
        forces = unwrapped.wheel_contact_forces[0]
        q_cmd = float(unwrapped.q_cmd[0].item())
        low = q_cmd > float(unwrapped.cfg.low_mode_q_threshold)
        cmd = unwrapped.cmd_buf[0]

        self._labels["mode"].text = f"模式: {'低车身' if low else '高车身'}"
        self._labels["q_cmd"].text = f"q_cmd: {q_cmd:.4f} rad"
        self._labels["cmd"].text = (
            f"cmd: vx={cmd[0].item():+.2f} vy={cmd[1].item():+.2f} wz={cmd[2].item():+.2f}"
        )
        self._labels["roll"].text = f"roll (车身横滚): {roll:+.1f}°"
        self._labels["pitch"].text = f"pitch (车身俯仰): {pitch:+.1f}°"
        self._labels["height"].text = f"base 离地高度: {height:.3f} m"
        self._labels["force"].text = (
            "四轮接触力: " + "  ".join(f"{f.item():5.1f}" for f in forces) + " N"
        )
        self._labels["contact"].text = (
            "四轮触地: " + "  ".join("●" if f.item() > 1.0 else "○" for f in forces)
        )
