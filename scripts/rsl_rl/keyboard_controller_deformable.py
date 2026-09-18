# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Deformable 主动悬挂键盘遥控（play 用）。
#
# 与 wheelbipe 的区别：deformable 的“运动”由底盘速度伺服实现（对 base_link 施力/力矩），
# 命令写在 env.cmd_buf=[vx,vy,ωz]，高度基准写在 env.q_cmd（两档：高/低车身）。
#
# 键位：
#   W / S       前进 / 后退（vx，按住给值、松开归零）
#   A / D       左移 / 右移（vy 横向平移，按住给值、松开归零）
#   X / Z       自旋角速度 ωz 加 / 减（增量式，松开保持）
#   Q           低车身训练阶段无效（保留按键，避免旧脚本误操作）
#   L           所有命令归零
# =============================================================================
"""Keyboard teleop for the deformable active-suspension task (chassis servo + q_cmd)."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field

import carb
import omni


class DeformableKeyboard:
    """键盘遥控：写入 env.cmd_buf / env.q_cmd。"""

    def __init__(self, cfg: "DeformableKeyboardCfg"):
        self.cfg = cfg
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        self._env = None
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._q_idx = 0  # 0=高车身(Q_HIGH)，1=低车身(Q_LOW)
        self._pressed: set[str] = set()

    def __del__(self):
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_sub"):
            if self._sub is not None:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
                self._sub = None

    def __str__(self) -> str:
        return (
            "Deformable Keyboard Teleop\n"
            "\tW/S: 前进/后退 (vx)   A/D: 左移/右移 (vy)\n"
            "\tX/Z: 自旋 +/-(ωz, 增量)   Q: 切换高/低车身基准角   L: 全部归零\n"
        )

    # ------------------------------------------------------------------
    def set_env(self, env) -> None:
        self._env = env

    @property
    def q_cmd(self) -> float:
        return float(self.cfg.q_choices[self._q_idx])

    def reset(self) -> None:
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._pressed.clear()
        self.apply()

    def apply(self) -> None:
        """把当前键盘状态写入环境。"""
        if self._env is None:
            return
        unwrapped = self._env.unwrapped
        if hasattr(unwrapped, "cmd_buf"):
            unwrapped.cmd_buf[:, 0] = self._vx
            unwrapped.cmd_buf[:, 1] = self._vy
            unwrapped.cmd_buf[:, 2] = self._wz
        if hasattr(unwrapped, "q_cmd"):
            unwrapped.q_cmd[:] = self.q_cmd

    def status(self) -> str:
        return (
            f"vx={self._vx:+.2f} vy={self._vy:+.2f} wz={self._wz:+.2f} "
            f"q_cmd={self.q_cmd:.4f}({'高车身' if self._q_idx == 0 else '低车身'})"
        )

    # ------------------------------------------------------------------
    def _on_keyboard_event(self, event, *args, **kwargs):
        name = event.input if isinstance(event.input, str) else event.input.name
        et = event.type

        if et == carb.input.KeyboardEventType.KEY_PRESS:
            cfg = self.cfg
            if name == "W":
                self._vx = +cfg.vx_max
                self._pressed.add("W")
            elif name == "S":
                self._vx = -cfg.vx_max
                self._pressed.add("S")
            elif name == "A":
                self._vy = +cfg.vy_max
                self._pressed.add("A")
            elif name == "D":
                self._vy = -cfg.vy_max
                self._pressed.add("D")
            elif name == "X":
                self._wz = min(cfg.wz_max, self._wz + cfg.wz_step)
                self._pressed.add("X")
            elif name == "Z":
                self._wz = max(-cfg.wz_max, self._wz - cfg.wz_step)
                self._pressed.add("Z")
            elif name == "Q":
                if len(cfg.q_choices) > 1:
                    self._q_idx = 1 - self._q_idx
                    print(f"[键盘] 基准角切换 → q_cmd={self.q_cmd:.4f} "
                          f"({'高车身' if self._q_idx == 0 else '低车身'})")
                else:
                    print(f"[键盘] 当前固定低车身 q_cmd={self.q_cmd:.4f}")
            elif name == "L":
                self.reset()
                print("[键盘] 命令全部归零")
            self.apply()

        elif et == carb.input.KeyboardEventType.KEY_RELEASE:
            if name in ("W", "S") and name in self._pressed:
                self._vx = 0.0
                self._pressed.discard(name)
            elif name in ("A", "D") and name in self._pressed:
                self._vy = 0.0
                self._pressed.discard(name)
            elif name in ("Z", "X"):
                self._pressed.discard(name)  # ωz 保持
            self.apply()

        return True


@dataclass
class DeformableKeyboardCfg:
    """键盘遥控配置。"""

    vx_max: float = 3.0  # 前进/后退最大速度（m/s）
    vy_max: float = 1.6  # 横移最大速度（m/s）
    wz_max: float = 4.5  # 自旋最大角速度（rad/s）
    wz_step: float = 0.6  # 每次 Z/X 的角速度增量（rad/s）
    q_choices: tuple[float, float] = field(default_factory=lambda: (0.0, 1.0563))
