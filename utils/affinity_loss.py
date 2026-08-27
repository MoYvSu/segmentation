# -*- coding: utf-8 -*-
"""Dense local-affinity target construction and class-balanced loss."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple
import math

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
    uncovered_as_boundary: torch.Tensor | None = None,
    return_uncovered_mask: bool = False,
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
    if uncovered_as_boundary is None:
        boundary_samples = torch.zeros(
            (instance_map.shape[0],), dtype=torch.bool, device=instance_map.device
        )
    else:
        boundary_samples = uncovered_as_boundary.to(
            device=instance_map.device, dtype=torch.bool
        ).reshape(-1)
        if boundary_samples.numel() != instance_map.shape[0]:
            raise ValueError(
                "uncovered_as_boundary must contain one flag per batch sample"
            )
    batch, height, width = instance_map.shape
    target = torch.zeros(
        (batch, len(offsets), height, width),
        dtype=torch.float32,
        device=instance_map.device,
    )
    edge_valid = torch.zeros_like(target, dtype=torch.bool)
    uncovered_mask = torch.zeros_like(target, dtype=torch.bool)
    for channel, (dy, dx) in enumerate(offsets):
        source, destination = _edge_slices(height, width, int(dy), int(dx))
        source_index = (slice(None), *source)
        destination_index = (slice(None), *destination)
        source_label = instance_map[source_index]
        destination_label = instance_map[destination_index]
        labeled_pair = (
            valid_pixels[source_index]
            & valid_pixels[destination_index]
            & (source_label > 0)
            & (destination_label > 0)
        )
        uncovered_pair = (
            boundary_samples[:, None, None]
            & valid_pixels[source_index]
            & valid_pixels[destination_index]
            & ((source_label > 0) ^ (destination_label > 0))
        )
        pair_valid = labeled_pair | uncovered_pair
        edge_valid[(slice(None), channel, *source)] = pair_valid
        uncovered_mask[(slice(None), channel, *source)] = uncovered_pair
        target[(slice(None), channel, *source)] = (
            pair_valid & (source_label == destination_label)
        ).float()
    if return_uncovered_mask:
        return target, edge_valid, uncovered_mask
    return target, edge_valid


def balanced_affinity_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    edge_valid: torch.Tensor,
    negative_weight: float = 1.0,
    hard_negative_weight: float = 0.0,
    hard_negative_gamma: float = 2.0,
    edge_weight: torch.Tensor | None = None,
):
    if logits.shape != target.shape or logits.shape != edge_valid.shape:
        raise ValueError(
            f"shape mismatch logits={logits.shape} target={target.shape} "
            f"valid={edge_valid.shape}"
        )
    if edge_weight is None:
        supervision_weight = torch.ones_like(logits)
    else:
        if edge_weight.shape != logits.shape:
            raise ValueError(
                f"edge_weight shape {edge_weight.shape} != logits {logits.shape}"
            )
        supervision_weight = edge_weight.to(device=logits.device, dtype=logits.dtype)
        if torch.any(supervision_weight < 0):
            raise ValueError("edge_weight must be non-negative")
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    channel_losses = []
    for channel in range(logits.shape[1]):
        valid = edge_valid[:, channel]
        positive = valid & (target[:, channel] > 0.5)
        negative = valid & ~positive
        terms = []
        term_weights = []
        if positive.any():
            positive_raw = raw[:, channel][positive]
            positive_weights = supervision_weight[:, channel][positive]
            terms.append(
                (positive_raw * positive_weights).sum()
                / positive_weights.sum().clamp_min(1.0e-12)
            )
            term_weights.append(1.0)
        if negative.any():
            negative_raw = raw[:, channel][negative]
            weights = supervision_weight[:, channel][negative]
            if hard_negative_weight > 0:
                hardness = torch.sigmoid(logits[:, channel][negative]).detach()
                weights = weights * (
                    1.0 + float(hard_negative_weight) * hardness.pow(
                        float(hard_negative_gamma)
                    )
                )
            negative_term = (
                (negative_raw * weights).sum()
                / weights.sum().clamp_min(1.0e-12)
            )
            terms.append(negative_term)
            term_weights.append(float(negative_weight))
        if terms:
            weights = logits.new_tensor(term_weights)
            channel_losses.append(
                (torch.stack(terms) * weights).sum() / weights.sum()
            )
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


def negative_affinity_tail_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    edge_valid: torch.Tensor,
    *,
    margin: float = 0.45,
    top_fraction: float = 0.05,
    sample_mask: torch.Tensor | None = None,
    edge_weight: torch.Tensor | None = None,
):
    """Penalize only the worst high-affinity real-boundary leakage edges.

    A single false connection can merge two complete instances. Mean BCE can
    hide that sparse tail, so this CVaR-like term selects only the largest
    margin violations among supervised negative edges.
    """
    if logits.shape != target.shape or logits.shape != edge_valid.shape:
        raise ValueError("logits, target, and edge_valid must share shape")
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction must be within (0, 1]")
    if not 0.0 <= float(margin) <= 1.0:
        raise ValueError("margin must be within [0, 1]")
    negative = edge_valid.bool() & (target <= 0.5)
    if sample_mask is not None:
        mask = sample_mask.to(device=logits.device, dtype=torch.bool).reshape(-1)
        if mask.numel() != logits.shape[0]:
            raise ValueError("sample_mask must contain one value per batch sample")
        negative = negative & mask[:, None, None, None]
    probability = torch.sigmoid(logits)
    violation = torch.relu(probability - float(margin)).square()
    if edge_weight is not None:
        if edge_weight.shape != logits.shape:
            raise ValueError("edge_weight must share logits shape")
        violation = violation * edge_weight.to(
            device=logits.device, dtype=logits.dtype
        )
    selected = violation[negative]
    if selected.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, {
            "tail_loss": zero.detach(),
            "tail_edges": 0,
            "tail_selected": 0,
            "tail_max_affinity": 0.0,
        }
    count = max(1, int(math.ceil(selected.numel() * float(top_fraction))))
    tail = torch.topk(selected, count, sorted=False).values
    loss = tail.mean()
    with torch.no_grad():
        affinity_values = probability[negative]
        metrics = {
            "tail_loss": loss.detach(),
            "tail_edges": int(selected.numel()),
            "tail_selected": int(count),
            "tail_max_affinity": float(affinity_values.max()),
        }
    return loss, metrics
