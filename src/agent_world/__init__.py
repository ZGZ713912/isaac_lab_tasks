# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

"""Package containing robot assets, actuators and terrain generators for the wheelbipe platform."""

import os
from pathlib import Path

# Parent Path
RootPath = os.path.dirname(os.path.abspath(__file__))
# Robot descriptions and field data are repository assets, separate from code.
# An editable install uses the checkout. A wheel install can point at an asset
# checkout with WHEELED_LEGGED_ASSETS_DIR.
AssetPath = str(Path(os.environ.get(
    "WHEELED_LEGGED_ASSETS_DIR", Path(__file__).resolve().parents[2] / "assets"
)).expanduser().resolve())
RobotAssetPath = os.path.join(AssetPath, "robots")
TerrainAssetPath = os.path.join(AssetPath, "terrains")

ROBOT_UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
