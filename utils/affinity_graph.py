# -*- coding: utf-8 -*-
"""Local instance-affinity targets and deterministic graph reconstruction."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
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


def regularize_affinity_components(
    component_map: np.ndarray,
    min_component_area: int,
):
    """Keep affinity cores and uniquely assign cut-band pixels to a core.

    Thresholded affinity graphs often turn the low-affinity boundary belt into
    thousands of one-pixel components.  Those are not plausible instances.
    Components below ``min_component_area`` are therefore treated as an
    unassigned cut band and completed by nearest retained component (a
    deterministic Voronoi assignment).  This preserves graph-derived topology
    while guaranteeing one instance ID for every input pixel.
    """
    source = np.asarray(component_map)
    if source.ndim != 2:
        raise ValueError(f"component_map must be 2-D, got {source.shape}")
    source = source.astype(np.int32, copy=False)
    labels, counts = np.unique(source[source > 0], return_counts=True)
    if labels.size == 0:
        return np.zeros(source.shape, dtype=np.int32), {
            "raw_component_count": 0,
            "retained_core_count": 0,
            "removed_fragment_count": 0,
            "reassigned_pixels": 0,
        }

    minimum = max(1, int(min_component_area))
    retained = labels[counts >= minimum]
    if retained.size == 0:
        retained = labels[np.argmax(counts)].reshape(1)
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lookup[retained] = np.arange(1, retained.size + 1, dtype=np.int32)
    partition = lookup[source]
    cut_band = partition == 0
    reassigned = int(np.count_nonzero(cut_band))
    if reassigned:
        nearest = distance_transform_edt(
            cut_band,
            return_distances=False,
            return_indices=True,
        )
        partition[cut_band] = partition[
            nearest[0][cut_band], nearest[1][cut_band]
        ]
    return partition, {
        "raw_component_count": int(labels.size),
        "retained_core_count": int(retained.size),
        "removed_fragment_count": int(labels.size - retained.size),
        "reassigned_pixels": reassigned,
    }


def regularize_affinity_components_by_affinity(
    component_map: np.ndarray,
    affinity: np.ndarray,
    min_component_area: int,
    offsets: Sequence[Tuple[int, int]] = DEFAULT_AFFINITY_OFFSETS,
    max_components: int | None = 255,
    assignment_channels: int = 4,
):
    """Assign small graph components with a seeded maximum spanning forest.

    Large components are immutable seeds.  Small components are agglomerated
    through the strongest *mean* affinity between adjacent raw components.
    Edges joining two different seeded cores are rejected, so the operation
    cannot merge retained instances.  Only local radius-1 channels participate
    by default; longer links remain useful for the initial graph reconstruction
    but cannot jump a cut band during cleanup.
    """
    source = np.asarray(component_map)
    scores = np.asarray(affinity, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"component_map must be 2-D, got {source.shape}")
    if scores.shape != (len(offsets), *source.shape):
        raise ValueError(
            f"affinity shape {scores.shape} != {(len(offsets), *source.shape)}"
        )
    channel_count = int(assignment_channels)
    if channel_count < 1 or channel_count > len(offsets):
        raise ValueError(
            f"assignment_channels must be within [1, {len(offsets)}]"
        )
    if max_components is not None and int(max_components) < 1:
        raise ValueError("max_components must be positive or None")

    source = source.astype(np.int32, copy=False)
    labels, counts = np.unique(source[source > 0], return_counts=True)
    if labels.size == 0:
        return np.zeros(source.shape, dtype=np.int32), {
            "raw_component_count": 0,
            "retained_core_count": 0,
            "removed_fragment_count": 0,
            "removed_core_count_for_cap": 0,
            "reassigned_pixels": 0,
            "adjacency_pair_count": 0,
            "blocked_seed_edge_count": 0,
            "promoted_disconnected_group_count": 0,
            "assignment_method": "affinity_seeded_maximum_spanning_forest",
        }

    minimum = max(1, int(min_component_area))
    eligible = np.flatnonzero(counts >= minimum)
    if eligible.size == 0:
        eligible = np.array([int(np.argmax(counts))], dtype=np.int64)
    cap_removed = 0
    if max_components is not None and eligible.size > int(max_components):
        order = np.lexsort((labels[eligible], -counts[eligible]))
        cap_removed = int(eligible.size - int(max_components))
        eligible = eligible[order[: int(max_components)]]
    eligible = eligible[np.argsort(labels[eligible], kind="stable")]

    label_to_node = np.full(int(labels.max()) + 1, -1, dtype=np.int32)
    label_to_node[labels] = np.arange(labels.size, dtype=np.int32)
    node_map = np.full(source.shape, -1, dtype=np.int32)
    positive = source > 0
    node_map[positive] = label_to_node[source[positive]]

    edge_keys = []
    edge_scores = []
    node_base = int(labels.size)
    for channel in range(channel_count):
        dy, dx = offsets[channel]
        source_slice, target_slice = _edge_slices(
            source.shape[0], source.shape[1], int(dy), int(dx)
        )
        left = node_map[source_slice]
        right = node_map[target_slice]
        valid = (left >= 0) & (right >= 0) & (left != right)
        if not np.any(valid):
            continue
        low = np.minimum(left[valid], right[valid]).astype(np.int64)
        high = np.maximum(left[valid], right[valid]).astype(np.int64)
        edge_keys.append(low * node_base + high)
        edge_scores.append(scores[channel][source_slice][valid].astype(np.float64))

    if edge_keys:
        keys = np.concatenate(edge_keys)
        values = np.concatenate(edge_scores)
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        sums = np.bincount(inverse, weights=values)
        contacts = np.bincount(inverse)
        means = sums / np.maximum(contacts, 1)
        edge_order = np.lexsort((unique_keys, -means))
        edge_left = (unique_keys // node_base).astype(np.int32)
        edge_right = (unique_keys % node_base).astype(np.int32)
    else:
        unique_keys = np.empty(0, dtype=np.int64)
        edge_order = np.empty(0, dtype=np.int64)
        edge_left = np.empty(0, dtype=np.int32)
        edge_right = np.empty(0, dtype=np.int32)

    parent = np.arange(labels.size, dtype=np.int32)
    tree_size = counts.astype(np.int64, copy=True)
    seed = np.zeros(labels.size, dtype=np.int32)
    seed[eligible] = np.arange(1, eligible.size + 1, dtype=np.int32)

    def find(node: int) -> int:
        root = int(node)
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[node]) != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    blocked = 0
    for edge_index in edge_order:
        left_root = find(int(edge_left[edge_index]))
        right_root = find(int(edge_right[edge_index]))
        if left_root == right_root:
            continue
        left_seed = int(seed[left_root])
        right_seed = int(seed[right_root])
        if left_seed and right_seed and left_seed != right_seed:
            blocked += 1
            continue
        if tree_size[left_root] < tree_size[right_root] or (
            tree_size[left_root] == tree_size[right_root]
            and left_root > right_root
        ):
            left_root, right_root = right_root, left_root
            left_seed, right_seed = right_seed, left_seed
        parent[right_root] = left_root
        tree_size[left_root] += tree_size[right_root]
        seed[left_root] = left_seed or right_seed

    roots = np.array([find(index) for index in range(labels.size)], dtype=np.int32)
    node_seed = seed[roots]
    promoted = 0
    for root in np.unique(roots[node_seed == 0]):
        promoted += 1
        seed[int(root)] = int(eligible.size + promoted)
    if promoted:
        node_seed = seed[roots]

    unique_seeds = np.unique(node_seed[node_seed > 0])
    seed_lookup = np.zeros(int(unique_seeds.max()) + 1, dtype=np.int32)
    seed_lookup[unique_seeds] = np.arange(1, unique_seeds.size + 1, dtype=np.int32)
    node_partition = seed_lookup[node_seed]
    label_partition = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    label_partition[labels] = node_partition
    partition = label_partition[source]
    retained_labels = labels[eligible]
    retained_lookup = np.zeros(int(labels.max()) + 1, dtype=bool)
    retained_lookup[retained_labels] = True
    reassigned = int(np.count_nonzero(positive & ~retained_lookup[source]))
    return partition, {
        "raw_component_count": int(labels.size),
        "retained_core_count": int(eligible.size),
        "removed_fragment_count": int(labels.size - eligible.size),
        "removed_core_count_for_cap": cap_removed,
        "reassigned_pixels": reassigned,
        "adjacency_pair_count": int(unique_keys.size),
        "blocked_seed_edge_count": int(blocked),
        "promoted_disconnected_group_count": int(promoted),
        "assignment_method": "affinity_seeded_maximum_spanning_forest",
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
