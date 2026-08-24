# -*- coding: utf-8 -*-
"""Coupled center-heatmap and dense-offset loss."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def center_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    probability = torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)
    valid = valid_mask.bool()
    positive = (target >= 1.0 - 1e-6) & valid
    negative = (~positive) & valid
    positive_loss = -torch.log(probability) * (1.0 - probability).pow(alpha)
    negative_weight = (1.0 - target).pow(beta)
    negative_loss = (
        -torch.log(1.0 - probability) * probability.pow(alpha) * negative_weight
    )
    positive_count = positive.sum().clamp_min(1)
    return (
        positive_loss[positive].sum() + negative_loss[negative].sum()
    ) / positive_count


def center_offset_loss(
    prediction: Dict[str, torch.Tensor],
    center_target: torch.Tensor,
    offset_target: torch.Tensor,
    foreground: torch.Tensor,
    valid_content: torch.Tensor,
    instance_map: torch.Tensor | None = None,
    *,
    center_weight: float = 1.0,
    offset_weight: float = 5.0,
    smooth_l1_beta: float = 0.02,
    offset_reduction: str = "pixel_mean",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    center_logits = prediction["center_logits"]
    offsets = prediction["offsets"]
    center_loss = center_focal_loss(center_logits, center_target, valid_content)
    offset_mask = foreground.bool() & valid_content.bool()
    component_mask = offset_mask.expand_as(offsets)
    raw_offset = F.smooth_l1_loss(
        offsets, offset_target, reduction="none", beta=float(smooth_l1_beta)
    )
    if offset_reduction == "instance_balanced":
        if instance_map is None:
            raise ValueError("instance_map is required for instance-balanced offset loss")
        per_pixel_loss = raw_offset.mean(dim=1)
        instance_losses = []
        for batch_index in range(offsets.shape[0]):
            mask = offset_mask[batch_index, 0]
            labels = instance_map[batch_index][mask].long()
            values = per_pixel_loss[batch_index][mask]
            if labels.numel() == 0:
                continue
            unique_labels, inverse = torch.unique(
                labels, sorted=False, return_inverse=True
            )
            sums = torch.zeros(
                unique_labels.shape[0], device=values.device, dtype=values.dtype
            ).scatter_add_(0, inverse, values)
            counts = torch.zeros_like(sums).scatter_add_(
                0, inverse, torch.ones_like(values)
            )
            instance_losses.append(sums / counts.clamp_min(1.0))
        offset_loss = (
            torch.cat(instance_losses).mean()
            if instance_losses else raw_offset.sum() * 0
        )
    elif offset_reduction == "pixel_mean":
        offset_loss = (
            raw_offset[component_mask].mean()
            if component_mask.any() else raw_offset.sum() * 0
        )
    else:
        raise ValueError(f"unknown offset_reduction: {offset_reduction}")
    total = float(center_weight) * center_loss + float(offset_weight) * offset_loss
    with torch.no_grad():
        probability = torch.sigmoid(center_logits)
        peak_mask = center_target >= 1.0 - 1e-6
        background = (center_target < 0.01) & valid_content.bool()
        offset_mae = (
            (offsets - offset_target).abs()[component_mask].mean()
            if component_mask.any() else offsets.new_tensor(0.0)
        )
        metrics = {
            "loss": total.detach(),
            "center_loss": center_loss.detach(),
            "offset_loss": offset_loss.detach(),
            "center_peak_probability": (
                probability[peak_mask].mean() if peak_mask.any()
                else probability.new_tensor(0.0)
            ),
            "center_background_probability": (
                probability[background].mean() if background.any()
                else probability.new_tensor(0.0)
            ),
            "offset_mae_normalized": offset_mae,
        }
    return total, metrics
