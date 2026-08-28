# -*- coding: utf-8 -*-
"""Semantic-only training utilities for instance-aware supervision.

The competition metric treats grains as instances, while the historical
semantic objective averages every pixel.  Large grains therefore dominate
small grains.  This module adds a GPU-friendly instance-balanced loss and a
target-aware dark-rim augmentation without changing instance geometry.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _resize_label_map(label_map: torch.Tensor, size) -> torch.Tensor:
    if tuple(label_map.shape[-2:]) == tuple(size):
        return label_map
    return F.interpolate(
        label_map.unsqueeze(1).float(), size=size, mode="nearest"
    ).squeeze(1).long()


def instance_balanced_core_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    instance_map: torch.Tensor,
    boundary_target: torch.Tensor | None = None,
    *,
    core_radius: int = 3,
    boundary_threshold: float = 0.20,
    min_core_pixels: int = 12,
    class_balance: bool = True,
    ferrite_class_weight: float = 1.0,
    hard_instance_gamma: float = 0.0,
    hard_instance_floor: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Average semantic BCE per annotated instance instead of per pixel.

    Pixels close to a polygon/boundary target are removed from sufficiently
    large instance cores.  If erosion would erase a thin or tiny instance, the
    whole instance is retained as a fallback, so small grains still contribute.
    The final optional class balancing gives ferrite and pearlite equal weight
    even when their annotated instance counts differ.
    """

    if logits.ndim != 3 or target.ndim != 3 or instance_map.ndim != 3:
        raise ValueError("logits, target and instance_map must be [B,H,W]")
    if logits.shape != target.shape:
        raise ValueError(f"logits {logits.shape} != target {target.shape}")

    labels = _resize_label_map(instance_map.long(), logits.shape[-2:])
    if labels.shape != logits.shape:
        raise ValueError(f"instance_map {labels.shape} != logits {logits.shape}")

    foreground = labels > 0
    if not bool(foreground.any()):
        zero = logits.sum() * 0.0
        return zero, {
            "instances": 0.0,
            "core_instances": 0.0,
            "fallback_instances": 0.0,
            "selected_pixels": 0.0,
        }

    radius = max(0, int(core_radius))
    if boundary_target is None:
        core_pixels = foreground
    else:
        boundary = boundary_target
        if tuple(boundary.shape[-2:]) != tuple(logits.shape[-2:]):
            boundary = F.interpolate(
                boundary.unsqueeze(1).float(),
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        boundary = (boundary >= float(boundary_threshold)).to(logits.dtype)
        if radius > 0:
            boundary = F.max_pool2d(
                boundary.unsqueeze(1),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            ).squeeze(1)
        core_pixels = foreground & (boundary < 0.5)

    # Offset IDs per image so equal polygon IDs from different batch items do
    # not collide during scatter aggregation.
    stride = int(labels.detach().max().item()) + 1
    offsets = (
        torch.arange(labels.shape[0], device=labels.device, dtype=labels.dtype)
        .view(-1, 1, 1)
        * stride
    )
    global_ids = labels + offsets
    slots = labels.shape[0] * stride
    flat_ids = global_ids.reshape(-1)

    ones = torch.ones_like(target, dtype=logits.dtype)
    total_counts = torch.zeros(slots, device=logits.device, dtype=logits.dtype)
    total_counts = total_counts.scatter_add(
        0, flat_ids[foreground.reshape(-1)], ones.reshape(-1)[foreground.reshape(-1)]
    )
    core_counts = torch.zeros_like(total_counts)
    core_counts = core_counts.scatter_add(
        0,
        flat_ids[core_pixels.reshape(-1)],
        ones.reshape(-1)[core_pixels.reshape(-1)],
    )

    use_core_by_id = core_counts >= float(max(1, min_core_pixels))
    use_core_pixel = use_core_by_id[global_ids]
    selected = foreground & ((core_pixels & use_core_pixel) | (~use_core_pixel))

    per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    selected_flat = selected.reshape(-1)
    selected_ids = flat_ids[selected_flat]
    selected_loss = per_pixel.reshape(-1)[selected_flat]
    selected_target = target.reshape(-1)[selected_flat]

    loss_sums = torch.zeros(slots, device=logits.device, dtype=logits.dtype)
    loss_sums = loss_sums.scatter_add(0, selected_ids, selected_loss)
    selected_counts = torch.zeros_like(loss_sums)
    selected_counts = selected_counts.scatter_add(
        0, selected_ids, torch.ones_like(selected_loss)
    )
    target_sums = torch.zeros_like(loss_sums)
    target_sums = target_sums.scatter_add(0, selected_ids, selected_target)
    probabilities = torch.sigmoid(logits).reshape(-1)[selected_flat]
    true_confidence = torch.where(
        selected_target >= 0.5, probabilities, 1.0 - probabilities
    )
    confidence_sums = torch.zeros_like(loss_sums)
    confidence_sums = confidence_sums.scatter_add(
        0, selected_ids, true_confidence
    )

    valid = selected_counts > 0
    per_instance_loss = loss_sums[valid] / selected_counts[valid].clamp_min(1.0)
    per_instance_confidence = (
        confidence_sums[valid] / selected_counts[valid].clamp_min(1.0)
    )
    per_instance_class = (
        target_sums[valid] / selected_counts[valid].clamp_min(1.0) >= 0.5
    )

    gamma = max(0.0, float(hard_instance_gamma))
    if gamma > 0:
        focus = float(max(0.0, hard_instance_floor)) + (
            1.0 - per_instance_confidence.detach()
        ).clamp(0.0, 1.0).pow(gamma)
    else:
        focus = torch.ones_like(per_instance_loss)

    def focused_mean(values, weights):
        return (values * weights).sum() / weights.sum().clamp_min(1.0e-6)

    if class_balance and bool(per_instance_class.any()) and bool((~per_instance_class).any()):
        ferrite_weight = max(0.0, float(ferrite_class_weight))
        ferrite_loss = focused_mean(
            per_instance_loss[per_instance_class], focus[per_instance_class]
        )
        pearlite_loss = focused_mean(
            per_instance_loss[~per_instance_class], focus[~per_instance_class]
        )
        loss = (
            pearlite_loss + ferrite_weight * ferrite_loss
        ) / max(1.0e-6, 1.0 + ferrite_weight)
    else:
        class_weight = torch.where(
            per_instance_class,
            torch.full_like(per_instance_loss, max(0.0, float(ferrite_class_weight))),
            torch.ones_like(per_instance_loss),
        )
        loss = focused_mean(per_instance_loss, focus * class_weight)

    valid_ids = torch.nonzero(total_counts > 0, as_tuple=False).flatten()
    core_instance_count = int(use_core_by_id[valid_ids].sum().item())
    instance_count = int(valid_ids.numel())
    stats = {
        "instances": float(instance_count),
        "core_instances": float(core_instance_count),
        "fallback_instances": float(instance_count - core_instance_count),
        "selected_pixels": float(selected.sum().item()),
        "ferrite_instances": float(per_instance_class.sum().item()),
        "mean_true_confidence": float(per_instance_confidence.detach().mean()),
        "mean_hard_focus": float(focus.detach().mean()),
    }
    return loss, stats


def dark_boundary_contamination(
    image: torch.Tensor,
    boundary_target: torch.Tensor,
    *,
    width: int,
    opacity: float,
    threshold: float = 0.20,
) -> torch.Tensor:
    """Darken a soft band around annotated boundaries without moving labels."""

    if image.ndim != 3 or boundary_target.ndim != 2:
        raise ValueError("image must be [3,H,W] and boundary_target must be [H,W]")
    if tuple(image.shape[-2:]) != tuple(boundary_target.shape[-2:]):
        raise ValueError("image and boundary_target spatial shapes must match")

    band = (boundary_target >= float(threshold)).to(image.dtype)
    radius = max(0, int(width))
    if radius > 0:
        band = F.max_pool2d(
            band.unsqueeze(0).unsqueeze(0),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        ).squeeze(0).squeeze(0)
    # A small average-pool feather avoids teaching a perfectly binary rim.
    band = F.avg_pool2d(
        band.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze(0).squeeze(0)
    factor = 1.0 - float(opacity) * band.clamp(0.0, 1.0)
    return (image * factor.unsqueeze(0)).clamp(0.0, 1.0)
