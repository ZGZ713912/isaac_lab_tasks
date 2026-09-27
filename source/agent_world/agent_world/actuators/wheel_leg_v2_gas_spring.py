"""Curve-based gas-spring force model for Wheel_leg_V2.

The supplied BKB catalogue gives a force-vs-stroke curve.  For the current
10 MPa estimate we use a linear interpolation between the user-selected
endpoints and keep damping optional until measured damping is available.

A gas spring pushes harder the more it is compressed, so the force must
decrease as ``length_m`` grows: ``force_at_min_n`` (shortest, most compressed)
must be LARGER than ``force_at_max_n`` (longest, most extended), e.g. the
BKB0.45-063-172 10 MPa endpoints are 347 N @ 109 mm and 279 N @ 172 mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WheelLegV2GasSpringModel:
    """Axial gas-spring model with a one-dimensional force curve."""

    pressure_mpa: float = 10.0
    min_length_m: float = 0.109
    max_length_m: float = 0.172
    force_at_min_n: float = 347.0
    force_at_max_n: float = 279.0
    damping_n_s_per_m: float = 0.0

    @property
    def stroke_m(self) -> float:
        return self.max_length_m - self.min_length_m

    def force_magnitude(self, length_m: torch.Tensor, length_rate_mps: torch.Tensor | None = None) -> torch.Tensor:
        """Return outward axial force magnitude in newtons.

        ``length_rate_mps`` is positive while extending.  Damping opposes the
        extension rate and is clipped so the gas spring never pulls in this
        unilateral model.
        """

        stroke_ratio = ((length_m - self.min_length_m) / self.stroke_m).clamp(0.0, 1.0)
        force = self.force_at_min_n + (self.force_at_max_n - self.force_at_min_n) * stroke_ratio
        if length_rate_mps is not None and self.damping_n_s_per_m > 0.0:
            force = force - self.damping_n_s_per_m * length_rate_mps
        return force.clamp_min(0.0)

    def force_vector(
        self,
        length_m: torch.Tensor,
        axis_01_w: torch.Tensor,
        length_rate_mps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return force on body 1, with ``axis_01_w`` pointing body0 -> body1."""

        return self.force_magnitude(length_m, length_rate_mps).unsqueeze(-1) * axis_01_w
