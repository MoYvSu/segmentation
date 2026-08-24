# -*- coding: utf-8 -*-
"""Center-seeded reconstruction for direct center-offset geometry."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def extract_center_peaks(
    center_probability: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.25,
    nms_radius: int = 3,
    max_centers: int = 255,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Extract deterministic center seeds from a probability heatmap."""
    probability = np.asarray(center_probability, dtype=np.float32)
    if probability.ndim != 2:
        raise ValueError("center_probability must be 2-D")
    if valid_mask is None:
        valid = np.ones(probability.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != probability.shape:
            raise ValueError("center_probability/valid_mask shape mismatch")
    radius = max(0, int(nms_radius))
    diameter = 2 * radius + 1
    pooled = cv2.dilate(probability, np.ones((diameter, diameter), np.uint8))
    candidates = valid & (probability >= float(threshold)) & (
        probability >= pooled - 1e-7
    )
    component_count, components = cv2.connectedComponents(
        candidates.astype(np.uint8), connectivity=8
    )
    peaks = []
    for component_id in range(1, component_count):
        ys, xs = np.nonzero(components == component_id)
        if ys.size == 0:
            continue
        scores = probability[ys, xs]
        best = int(np.argmax(scores))
        peaks.append((float(scores[best]), int(ys[best]), int(xs[best])))
    peaks.sort(key=lambda value: (-value[0], value[1], value[2]))
    raw_count = len(peaks)
    peaks = peaks[: max(0, int(max_centers))]
    centers = np.asarray([(y, x) for _, y, x in peaks], dtype=np.int32)
    if centers.size == 0:
        centers = np.empty((0, 2), dtype=np.int32)
    scores = np.asarray([score for score, _, _ in peaks], dtype=np.float32)
    audit = {
        "raw_center_count": raw_count,
        "kept_center_count": len(peaks),
        "dropped_for_cap": max(0, raw_count - len(peaks)),
    }
    return centers, scores, audit


def assign_endpoints_to_centers(
    endpoint_y: np.ndarray,
    endpoint_x: np.ndarray,
    foreground: np.ndarray,
    centers_yx: np.ndarray,
    max_assignment_distance: float | None = 8.0,
    min_instance_area: int = 1,
    chunk_size: int = 16384,
) -> Tuple[np.ndarray, Dict[str, float | int]]:
    """Assign every accepted foreground endpoint to its nearest center seed."""
    foreground = np.asarray(foreground, dtype=bool)
    if endpoint_y.shape != foreground.shape or endpoint_x.shape != foreground.shape:
        raise ValueError("endpoint/foreground shape mismatch")
    centers = np.asarray(centers_yx, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1:] != (2,):
        raise ValueError("centers_yx must have shape [N, 2]")
    result = np.zeros(foreground.shape, dtype=np.int32)
    ys, xs = np.nonzero(foreground)
    if ys.size == 0 or centers.shape[0] == 0:
        return result, {
            "center_count": int(centers.shape[0]),
            "kept_instances": 0,
            "dropped_small_instances": 0,
            "assigned_foreground_pixels": 0,
            "unassigned_foreground_pixels": int(ys.size),
            "mean_assignment_distance": 0.0,
            "max_assignment_distance": 0.0,
        }
    points = np.stack(
        [endpoint_y[ys, xs], endpoint_x[ys, xs]], axis=1
    ).astype(np.float32, copy=False)
    nearest = np.empty(points.shape[0], dtype=np.int32)
    nearest_distance = np.empty(points.shape[0], dtype=np.float32)
    chunk = max(1, int(chunk_size))
    for start in range(0, points.shape[0], chunk):
        stop = min(points.shape[0], start + chunk)
        delta = points[start:stop, None, :] - centers[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        indices = np.argmin(squared, axis=1)
        nearest[start:stop] = indices
        nearest_distance[start:stop] = np.sqrt(
            squared[np.arange(stop - start), indices]
        )
    accepted = np.ones(points.shape[0], dtype=bool)
    if max_assignment_distance is not None:
        accepted &= nearest_distance <= float(max_assignment_distance)
    labels = nearest + 1
    support = np.bincount(labels[accepted], minlength=centers.shape[0] + 1)
    keep = [
        label for label in range(1, centers.shape[0] + 1)
        if int(support[label]) >= int(min_instance_area)
    ]
    remap = np.zeros(centers.shape[0] + 1, dtype=np.int32)
    for new_id, old_id in enumerate(keep, start=1):
        remap[old_id] = new_id
    assigned_labels = np.zeros(points.shape[0], dtype=np.int32)
    assigned_labels[accepted] = remap[labels[accepted]]
    result[ys, xs] = assigned_labels
    assigned_distances = nearest_distance[assigned_labels > 0]
    audit = {
        "center_count": int(centers.shape[0]),
        "kept_instances": len(keep),
        "dropped_small_instances": int(centers.shape[0] - len(keep)),
        "assigned_foreground_pixels": int(np.sum(assigned_labels > 0)),
        "unassigned_foreground_pixels": int(np.sum(assigned_labels == 0)),
        "mean_assignment_distance": (
            float(assigned_distances.mean()) if assigned_distances.size else 0.0
        ),
        "max_assignment_distance": (
            float(assigned_distances.max()) if assigned_distances.size else 0.0
        ),
    }
    return result, audit
