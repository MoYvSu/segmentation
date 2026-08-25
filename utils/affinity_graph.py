# -*- coding: utf-8 -*-
"""Local instance-affinity targets and deterministic graph reconstruction."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# Six axial links at radii 1/2/4 plus the two unit diagonals.  Only one
# direction of each undirected edge is stored.
DEFAULT_AFFINITY_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
    (0, 2),
    (2, 0),
    (0, 4),
    (4, 0),
)


def _edge_slices(height: int, width: int, dy: int, dx: int):
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    source = (slice(source_y0, source_y1), slice(source_x0, source_x1))
    target = (
        slice(source_y0 + dy, source_y1 + dy),
        slice(source_x0 + dx, source_x1 + dx),
    )
    return source, target


def build_affinity_targets(
    instance_map: np.ndarray,
    valid_content: np.ndarray | None = None,
    offsets: Sequence[Tuple[int, int]] = DEFAULT_AFFINITY_OFFSETS,
):
    """Return same-instance targets and valid-edge masks for local pairs.

    Only pairs whose two endpoints have positive instance IDs are supervised.
    Pixels left uncovered by LabelMe polygons therefore do not become invented
    negative boundaries.
    """
    labels = np.asarray(instance_map)
    if labels.ndim != 2:
        raise ValueError(f"instance_map must be 2-D, got {labels.shape}")
    height, width = labels.shape
    valid_pixels = (
        np.ones((height, width), dtype=bool)
        if valid_content is None else np.asarray(valid_content, dtype=bool)
    )
    if valid_pixels.shape != labels.shape:
        raise ValueError(
            f"valid_content shape {valid_pixels.shape} != labels {labels.shape}"
        )
    affinity = np.zeros((len(offsets), height, width), dtype=np.float32)
    edge_valid = np.zeros_like(affinity, dtype=bool)
    for channel, (dy, dx) in enumerate(offsets):
        source, target = _edge_slices(height, width, int(dy), int(dx))
        source_label, target_label = labels[source], labels[target]
        pair_valid = (
            valid_pixels[source]
            & valid_pixels[target]
            & (source_label > 0)
            & (target_label > 0)
        )
        edge_valid[channel][source] = pair_valid
        affinity[channel][source] = (
            pair_valid & (source_label == target_label)
        ).astype(np.float32)
    return affinity, edge_valid


def reconstruct_affinity_components(
    foreground: np.ndarray,
    affinity: np.ndarray,
    offsets: Sequence[Tuple[int, int]] = DEFAULT_AFFINITY_OFFSETS,
    threshold: float | Sequence[float] = 0.5,
    max_instances: int | None = 255,
):
    """Partition foreground pixels by thresholded undirected affinity edges.

    The largest ``max_instances - 1`` components are retained when the raw
    graph exceeds the competition cap; all remaining fragments are assigned
    to one overflow instance.  This deliberately simple cap policy is audited
    and is not expected to activate for a perfect labeled affinity graph.
    """
    mask = np.asarray(foreground, dtype=bool)
    scores = np.asarray(affinity, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"foreground must be 2-D, got {mask.shape}")
    if scores.shape != (len(offsets), *mask.shape):
        raise ValueError(
            f"affinity shape {scores.shape} != {(len(offsets), *mask.shape)}"
        )
    threshold_array = np.asarray(threshold, dtype=np.float32)
    if threshold_array.ndim == 0:
        threshold_array = np.full(
            len(offsets), float(threshold_array), dtype=np.float32
        )
    if threshold_array.shape != (len(offsets),):
        raise ValueError(
            f"threshold must be scalar or {len(offsets)} values, "
            f"got {threshold_array.shape}"
        )
    if max_instances is not None and (
        int(max_instances) < 1 or int(max_instances) > 255
    ):
        raise ValueError("max_instances must be within [1, 255]")

    height, width = mask.shape
    foreground_flat = np.flatnonzero(mask.ravel())
    output_dtype = np.int32 if max_instances is None else np.uint8
    output = np.zeros((height, width), dtype=output_dtype)
    if foreground_flat.size == 0:
        return output, {
            "raw_component_count": 0,
            "kept_instance_count": 0,
            "merged_components_for_cap": 0,
            "foreground_pixels": 0,
        }

    node_lut = np.full(height * width, -1, dtype=np.int32)
    node_lut[foreground_flat] = np.arange(foreground_flat.size, dtype=np.int32)
    rows, columns = [], []
    flat_index = np.arange(height * width, dtype=np.int64).reshape(height, width)
    for channel, (dy, dx) in enumerate(offsets):
        source, target = _edge_slices(height, width, int(dy), int(dx))
        selected = (
            (scores[channel][source] >= float(threshold_array[channel]))
            & mask[source]
            & mask[target]
        )
        if not np.any(selected):
            continue
        source_flat = flat_index[source][selected]
        target_flat = flat_index[target][selected]
        rows.append(node_lut[source_flat])
        columns.append(node_lut[target_flat])
    if rows:
        row = np.concatenate(rows)
        column = np.concatenate(columns)
        graph = coo_matrix(
            (np.ones(row.size, dtype=np.uint8), (row, column)),
            shape=(foreground_flat.size, foreground_flat.size),
        )
        raw_count, component = connected_components(
            graph, directed=False, return_labels=True
        )
    else:
        raw_count = int(foreground_flat.size)
        component = np.arange(raw_count, dtype=np.int32)

    areas = np.bincount(component, minlength=raw_count)
    order = np.argsort(-areas, kind="stable")
    component_to_instance = np.zeros(raw_count, dtype=np.uint16)
    if max_instances is None:
        component_to_instance[order] = np.arange(1, raw_count + 1)
        kept_count = raw_count
        merged_for_cap = 0
    elif raw_count <= int(max_instances):
        component_to_instance[order] = np.arange(1, raw_count + 1)
        kept_count = raw_count
        merged_for_cap = 0
    elif int(max_instances) == 1:
        component_to_instance[:] = 1
        kept_count = 1
        merged_for_cap = raw_count - 1
    else:
        retained = order[: int(max_instances) - 1]
        overflow = order[int(max_instances) - 1 :]
        component_to_instance[retained] = np.arange(1, int(max_instances))
        component_to_instance[overflow] = int(max_instances)
        kept_count = int(max_instances)
        merged_for_cap = int(raw_count - kept_count)
    output.ravel()[foreground_flat] = component_to_instance[component].astype(
        output_dtype
    )
    return output, {
        "raw_component_count": int(raw_count),
        "kept_instance_count": int(kept_count),
        "merged_components_for_cap": int(merged_for_cap),
        "foreground_pixels": int(foreground_flat.size),
    }


def audit_instance_recovery(gt_map: np.ndarray, prediction: np.ndarray):
    """Count graph-induced GT splits and prediction merges."""
    gt = np.asarray(gt_map)
    pred = np.asarray(prediction)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt={gt.shape} pred={pred.shape}")
    gt_ids = [int(value) for value in np.unique(gt) if int(value) != 0]
    pred_ids = [int(value) for value in np.unique(pred) if int(value) != 0]
    split_gt = {
        value: [int(item) for item in np.unique(pred[gt == value]) if int(item) != 0]
        for value in gt_ids
    }
    merged_pred = {
        value: [int(item) for item in np.unique(gt[pred == value]) if int(item) != 0]
        for value in pred_ids
    }
    split_count = sum(len(values) != 1 for values in split_gt.values())
    merge_count = sum(len(values) != 1 for values in merged_pred.values())
    exact_gt = sum(
        len(values) == 1 and len(merged_pred.get(values[0], [])) == 1
        for values in split_gt.values()
    )
    foreground = gt > 0
    return {
        "gt_instance_count": len(gt_ids),
        "pred_instance_count": len(pred_ids),
        "split_gt_instance_count": int(split_count),
        "merged_pred_instance_count": int(merge_count),
        "exact_gt_instance_count": int(exact_gt),
        "exact_gt_instance_fraction": float(exact_gt / max(1, len(gt_ids))),
        "unassigned_gt_pixels": int(np.sum(foreground & (pred == 0))),
        "exact_partition": bool(
            split_count == 0
            and merge_count == 0
            and len(gt_ids) == len(pred_ids)
            and np.all(pred[foreground] > 0)
        ),
    }
