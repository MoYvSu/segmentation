# -*- coding: utf-8 -*-
"""Dense local-affinity target construction and class-balanced loss."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS


def _edge_slices(height: int, width: int, dy: int, dx: int):
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    return (
        (slice(source_y0, source_y1), slice(source_x0, source_x1)),
        (
            slice(source_y0 + dy, source_y1 + dy),
            slice(source_x0 + dx, source_x1 + dx),
        ),
    )


def build_affinity_targets_torch(
    instance_map: torch.Tensor,
    valid_content: torch.Tensor,
    offsets: Sequence[Tuple[int, int]] = DEFAULT_AFFINITY_OFFSETS,
):
    if instance_map.ndim != 3:
        raise ValueError(f"instance_map must be [B,H,W], got {instance_map.shape}")
    if valid_content.ndim == 4 and valid_content.shape[1] == 1:
        valid_pixels = valid_content[:, 0].bool()
    elif valid_content.ndim == 3:
        valid_pixels = valid_content.bool()
    else:
        raise ValueError(f"valid_content must be [B,1,H,W], got {valid_content.shape}")
    if tuple(valid_pixels.shape) != tuple(instance_map.shape):
        raise ValueError(
            f"valid shape {valid_pixels.shape} != labels {instance_map.shape}"
        )
    batch, height, width = instance_map.shape
    target = torch.zeros(
        (batch, len(offsets), height, width),
        dtype=torch.float32,
        device=instance_map.device,
    )
    edge_valid = torch.zeros_like(target, dtype=torch.bool)
    for channel, (dy, dx) in enumerate(offsets):
        source, destination = _edge_slices(height, width, int(dy), int(dx))
        source_index = (slice(None), *source)
        destination_index = (slice(None), *destination)
        source_label = instance_map[source_index]
        destination_label = instance_map[destination_index]
        pair_valid = (
            valid_pixels[source_index]
            & valid_pixels[destination_index]
            & (source_label > 0)
            & (destination_label > 0)
        )
        edge_valid[(slice(None), channel, *source)] = pair_valid
        target[(slice(None), channel, *source)] = (
            pair_valid & (source_label == destination_label)
        ).float()
    return target, edge_valid


def balanced_affinity_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    edge_valid: torch.Tensor,
):
    if logits.shape != target.shape or logits.shape != edge_valid.shape:
        raise ValueError(
            f"shape mismatch logits={logits.shape} target={target.shape} "
            f"valid={edge_valid.shape}"
        )
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    channel_losses = []
    for channel in range(logits.shape[1]):
        valid = edge_valid[:, channel]
        positive = valid & (target[:, channel] > 0.5)
        negative = valid & ~positive
        terms = []
        if positive.any():
            terms.append(raw[:, channel][positive].mean())
        if negative.any():
            terms.append(raw[:, channel][negative].mean())
        if terms:
            channel_losses.append(torch.stack(terms).mean())
    loss = (
        torch.stack(channel_losses).mean()
        if channel_losses else logits.sum() * 0.0
    )
    with torch.no_grad():
        probability = torch.sigmoid(logits)
        prediction = probability >= 0.5
        positive = edge_valid & (target > 0.5)
        negative = edge_valid & ~positive
        true_positive = int((prediction & positive).sum())
        false_positive = int((prediction & negative).sum())
        true_negative = int((~prediction & negative).sum())
        false_negative = int((~prediction & positive).sum())
        metrics: Dict[str, torch.Tensor | float] = {
            "loss": loss.detach(),
            "precision": true_positive / max(1, true_positive + false_positive),
            "recall": true_positive / max(1, true_positive + false_negative),
            "specificity": true_negative / max(1, true_negative + false_positive),
            "positive_probability": (
                probability[positive].mean() if positive.any() else probability.new_tensor(0.0)
            ),
            "negative_probability": (
                probability[negative].mean() if negative.any() else probability.new_tensor(0.0)
            ),
            "positive_edges": int(positive.sum()),
            "negative_edges": int(negative.sum()),
        }
    return loss, metrics
