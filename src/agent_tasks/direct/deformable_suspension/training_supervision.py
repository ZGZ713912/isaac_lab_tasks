"""Simulation labels for PPO only; never inputs to the deployed actor."""

import math

import torch


def flat_ground_label(points_w, normals_w, ground_height, grade_deg=3., tolerance_m=.006):
    """Do not preserve the old raised pose during IMU startup/free fall.

    All four local surfaces and their fitted support plane must be near
    horizontal. A centered fit also rejects large nonplanar height changes
    even when the separate surface normals all point upwards.
    """
    valid = (points_w.isfinite().all(dim=(1, 2)) & normals_w.isfinite().all(dim=(1, 2))
             & ground_height.isfinite().all(-1))
    grade = torch.atan2(normals_w[..., :2].norm(dim=-1), normals_w[..., 2]) * (180. / math.pi)
    xy = torch.nan_to_num(points_w[..., :2])
    xy = xy - xy.mean(1, keepdim=True)
    height = torch.nan_to_num(ground_height)
    height = height - height.mean(1, keepdim=True)
    matrix = torch.cat((xy, torch.ones_like(height[..., None])), -1)
    transpose = matrix.transpose(-1, -2)
    gram = transpose @ matrix + torch.eye(3, device=xy.device, dtype=xy.dtype) * 1.e-8
    coefficients = torch.linalg.solve(gram, transpose @ height[..., None])
    residual = (matrix @ coefficients).squeeze(-1).sub(height).abs().amax(-1)
    fitted_grade = torch.atan(coefficients[:, :2, 0].norm(dim=-1)) * (180. / math.pi)
    return (valid & (normals_w[..., 2] > 0.).all(-1) & (grade <= grade_deg + 1.e-5).all(-1)
            & (fitted_grade <= grade_deg + 1.e-5) & (residual <= tolerance_m))
