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

from .height_field import (
    FourQuadrantTerrainCfg,
    HfCliffInvertedPyramidStairsTerrainCfg,
    HfCustomDirectionalWaveTerrainCfg,
    HfCustomGridBarsTerrainCfg,
    HfCustomNpyTerrainCfg,
    HfCustomPeriodicSlopeTerrainCfg,
    HfCustomRaisedInvertedPyramidSlopedTerrainCfg,
    HfCustomTruncatedSlopedTerrainCfg,
    MeshCustomGridBarsTerrainCfg,
    MeshCustomSplitGridBarsTerrainCfg,
    periodic_slope_angle,
    periodic_slope_height,
)

__all__ = [
    "FourQuadrantTerrainCfg",
    "HfCliffInvertedPyramidStairsTerrainCfg",
    "HfCustomDirectionalWaveTerrainCfg",
    "HfCustomGridBarsTerrainCfg",
    "HfCustomNpyTerrainCfg",
    "HfCustomPeriodicSlopeTerrainCfg",
    "HfCustomRaisedInvertedPyramidSlopedTerrainCfg",
    "HfCustomTruncatedSlopedTerrainCfg",
    "MeshCustomGridBarsTerrainCfg",
    "MeshCustomSplitGridBarsTerrainCfg",
    "periodic_slope_angle",
    "periodic_slope_height",
]
