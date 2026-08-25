# -*- coding: utf-8 -*-
"""Convert multi-offset affinities into a watershed boundary probability."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _soft_dilated_support(
    boundary: torch.Tensor,
    radius: int,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    radius = int(radius)
    if radius < 0:
        raise ValueError("support radius must be non-negative")
    if float(temperature) <= 0:
        raise ValueError("support temperature must be positive")
    if radius:
        boundary = F.max_pool2d(
            boundary, kernel_size=2 * radius + 1, stride=1, padding=radius
        )
    return torch.sigmoid(
        (boundary - float(threshold)) / float(temperature)
    )


def affinity_boundary_probability(
    affinity_logits: torch.Tensor,
    *,
    mode: str = "mean",
    distance2_weight: float = 0.50,
    distance4_weight: float = 0.25,
    support_threshold: float = 0.20,
    support_temperature: float = 0.05,
) -> torch.Tensor:
    """Return one boundary channel from the eight standard affinity channels.

    ``mean`` preserves the original ``1 - mean(affinity)`` behavior.
    ``short`` uses only the four unit-offset channels.
    ``gated`` lets distance-2/4 channels reinforce or bridge only the dilated
    neighborhood of a short-range boundary. Long-range texture responses
    therefore cannot create standalone fog far from a localized interface.
    """
    if affinity_logits.ndim != 4 or affinity_logits.shape[1] != 8:
        raise ValueError(
            f"expected standard affinity logits [B,8,H,W], got {affinity_logits.shape}"
        )
    probability = torch.sigmoid(affinity_logits)
    if mode == "mean":
        return 1.0 - probability.mean(dim=1, keepdim=True)

    short = 1.0 - probability[:, :4].mean(dim=1, keepdim=True)
    if mode == "short":
        return short
    if mode != "gated":
        raise ValueError(f"unknown affinity fusion mode: {mode}")
    if float(distance2_weight) < 0 or float(distance4_weight) < 0:
        raise ValueError("distance weights must be non-negative")

    distance2 = 1.0 - probability[:, 4:6].mean(dim=1, keepdim=True)
    distance4 = 1.0 - probability[:, 6:8].mean(dim=1, keepdim=True)
    support2 = _soft_dilated_support(
        short, radius=2, threshold=support_threshold,
        temperature=support_temperature,
    )
    support4 = _soft_dilated_support(
        short, radius=4, threshold=support_threshold,
        temperature=support_temperature,
    )
    weighted2 = float(distance2_weight) * support2
    weighted4 = float(distance4_weight) * support4
    fused = (
        short + weighted2 * distance2 + weighted4 * distance4
    ) / (1.0 + weighted2 + weighted4)
    return fused.clamp(0.0, 1.0)
