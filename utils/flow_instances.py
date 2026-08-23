# -*- coding: utf-8 -*-
"""Dense instance-flow targets and model-independent mask reconstruction.

The functions in this module deliberately do not depend on a neural network.
They are shared by the O0 oracle experiment and the future flow-head inference
path, so the oracle cannot use GT instance ids during reconstruction.

Flow channel order is ``(dy, dx)`` throughout.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import cv2
import numpy as np


def resize_label_map(label_map: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize for integer instance labels."""
    height, width = map(int, shape)
    resized = cv2.resize(
        label_map.astype(np.int32, copy=False),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.int32, copy=False)


def stride_shape(shape: Tuple[int, int], stride: int) -> Tuple[int, int]:
    """Return the decoder-grid shape for a full-resolution image."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    height, width = map(int, shape)
    return max(1, math.ceil(height / stride)), max(1, math.ceil(width / stride))


def _instance_bbox(mask: np.ndarray, padding: int = 2) -> Tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return slice(0, 0), slice(0, 0)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(mask.shape[0], int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(mask.shape[1], int(xs.max()) + padding + 1)
    return slice(y0, y1), slice(x0, x1)


def build_center_offset_target(instance_map: np.ndarray) -> np.ndarray:
    """Build direct per-pixel offsets to an interior EDT-maximum center."""
    flow = np.zeros((2, *instance_map.shape), dtype=np.float32)
    for instance_id in np.unique(instance_map):
        if int(instance_id) == 0:
            continue
        mask = instance_map == int(instance_id)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        y_slice, x_slice = _instance_bbox(mask)
        local = mask[y_slice, x_slice].astype(np.uint8)
        distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
        center_local_y, center_local_x = np.unravel_index(
            int(np.argmax(distance)), distance.shape
        )
        center_y = float(y_slice.start + center_local_y)
        center_x = float(x_slice.start + center_local_x)
        flow[0, ys, xs] = center_y - ys
        flow[1, ys, xs] = center_x - xs
    return flow


def _tile_starts(length: int, tile_size: int, overlap: int):
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError(
            f"invalid tile geometry: size={tile_size}, overlap={overlap}"
        )
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size - overlap))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def build_tile_local_center_offset_target(
    instance_map: np.ndarray,
    tile_size: int,
    overlap: int,
    blend_floor: float = 0.05,
) -> np.ndarray:
    """Blend offsets whose centers are recomputed from each visible tile crop.

    This is a deliberately pessimistic oracle for tiled inference. It models a
    head that cannot infer a center outside its current tile and therefore aims
    each clipped fragment at a separate local center.
    """
    height, width = instance_map.shape
    tile_height = min(int(tile_size), height)
    tile_width = min(int(tile_size), width)
    y_starts = _tile_starts(
        height, tile_height, min(int(overlap), tile_height - 1)
    )
    x_starts = _tile_starts(
        width, tile_width, min(int(overlap), tile_width - 1)
    )
    wy = (
        np.hanning(tile_height).astype(np.float32)
        if tile_height > 1 else np.ones(1, np.float32)
    )
    wx = (
        np.hanning(tile_width).astype(np.float32)
        if tile_width > 1 else np.ones(1, np.float32)
    )
    weight = np.maximum(np.outer(wy, wx), float(blend_floor)).astype(np.float32)
    accumulated = np.zeros((2, height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    for top in y_starts:
        for left in x_starts:
            bottom, right = top + tile_height, left + tile_width
            local_map = instance_map[top:bottom, left:right]
            local_offset = build_center_offset_target(local_map)
            accumulated[:, top:bottom, left:right] += local_offset * weight
            weight_sum[top:bottom, left:right] += weight
    valid = weight_sum > 0
    accumulated[:, valid] /= weight_sum[valid]
    return accumulated


def normalize_flow(flow: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Normalize non-zero vectors to unit length inside ``foreground``."""
    result = np.asarray(flow, dtype=np.float32).copy()
    norm = np.sqrt(result[0] ** 2 + result[1] ** 2)
    valid = foreground & (norm > 1e-6)
    result[0, valid] /= norm[valid]
    result[1, valid] /= norm[valid]
    result[:, ~foreground] = 0.0
    return result


def build_edt_flow_target(
    instance_map: np.ndarray,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Build an inward unit flow from the gradient of each instance EDT.

    The field is local and converges to a medial-axis attractor rather than
    requiring every pixel to regress one global centroid.
    """
    flow = np.zeros((2, *instance_map.shape), dtype=np.float32)
    for instance_id in np.unique(instance_map):
        if int(instance_id) == 0:
            continue
        mask = instance_map == int(instance_id)
        y_slice, x_slice = _instance_bbox(mask)
        local_mask = mask[y_slice, x_slice]
        if not np.any(local_mask):
            continue
        distance = cv2.distanceTransform(
            local_mask.astype(np.uint8), cv2.DIST_L2, 5
        ).astype(np.float32)
        if smooth_sigma > 0:
            distance = cv2.GaussianBlur(
                distance, (0, 0), sigmaX=float(smooth_sigma),
                sigmaY=float(smooth_sigma), borderType=cv2.BORDER_REPLICATE,
            )
        grad_y, grad_x = np.gradient(distance)
        local_flow = np.stack([grad_y, grad_x]).astype(np.float32)
        local_flow = normalize_flow(local_flow, local_mask)
        target = flow[:, y_slice, x_slice]
        target[:, local_mask] = local_flow[:, local_mask]
    return flow


def perturb_unit_flow(
    flow: np.ndarray,
    foreground: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add component noise and renormalize a local unit-vector field."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if noise_std == 0:
        return np.asarray(flow, dtype=np.float32).copy()
    noisy = np.asarray(flow, dtype=np.float32).copy()
    noise = rng.normal(0.0, float(noise_std), size=noisy.shape).astype(np.float32)
    noisy[:, foreground] += noise[:, foreground]
    return normalize_flow(noisy, foreground)


def integrate_flow(
    flow: np.ndarray,
    foreground: np.ndarray,
    steps: int = 192,
    step_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Euler-integrate every foreground grid point through a predicted flow."""
    if flow.shape != (2, *foreground.shape):
        raise ValueError(f"flow/foreground shape mismatch: {flow.shape}, {foreground.shape}")
    height, width = foreground.shape
    pos_y, pos_x = np.indices((height, width), dtype=np.float32)
    active = foreground.astype(bool, copy=False)
    for _ in range(int(steps)):
        sample_y = cv2.remap(
            flow[0], pos_x, pos_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        sample_x = cv2.remap(
            flow[1], pos_x, pos_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        pos_y[active] += float(step_size) * sample_y[active]
        pos_x[active] += float(step_size) * sample_x[active]
        np.clip(pos_y, 0.0, float(height - 1), out=pos_y)
        np.clip(pos_x, 0.0, float(width - 1), out=pos_x)
    return pos_y, pos_x


def endpoints_from_center_offsets(
    offsets: np.ndarray,
    foreground: np.ndarray,
    endpoint_noise_px: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply direct center offsets, optionally with endpoint regression noise."""
    if offsets.shape != (2, *foreground.shape):
        raise ValueError("offset/foreground shape mismatch")
    if endpoint_noise_px < 0:
        raise ValueError("endpoint_noise_px must be non-negative")
    pos_y, pos_x = np.indices(foreground.shape, dtype=np.float32)
    pos_y += offsets[0]
    pos_x += offsets[1]
    if endpoint_noise_px > 0:
        pos_y[foreground] += rng.normal(
            0.0, float(endpoint_noise_px), size=int(foreground.sum())
        ).astype(np.float32)
        pos_x[foreground] += rng.normal(
            0.0, float(endpoint_noise_px), size=int(foreground.sum())
        ).astype(np.float32)
    np.clip(pos_y, 0.0, float(foreground.shape[0] - 1), out=pos_y)
    np.clip(pos_x, 0.0, float(foreground.shape[1] - 1), out=pos_x)
    return pos_y, pos_x


def cluster_endpoints(
    endpoint_y: np.ndarray,
    endpoint_x: np.ndarray,
    foreground: np.ndarray,
    close_radius: int = 1,
    min_instance_area: int = 1,
    max_instances: int = 255,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Cluster endpoint occupancy without using GT ids or a GT instance count."""
    height, width = foreground.shape
    end_y = np.rint(endpoint_y).astype(np.int32)
    end_x = np.rint(endpoint_x).astype(np.int32)
    np.clip(end_y, 0, height - 1, out=end_y)
    np.clip(end_x, 0, width - 1, out=end_x)

    occupancy = np.zeros((height, width), dtype=np.uint8)
    occupancy[end_y[foreground], end_x[foreground]] = 1
    if close_radius > 0:
        diameter = 2 * int(close_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)

    component_count, endpoint_components = cv2.connectedComponents(
        occupancy, connectivity=8
    )
    assignments = np.zeros((height, width), dtype=np.int32)
    assignments[foreground] = endpoint_components[end_y[foreground], end_x[foreground]]
    support = np.bincount(assignments[foreground], minlength=component_count)
    candidate_ids = [
        component_id for component_id in range(1, component_count)
        if int(support[component_id]) >= int(min_instance_area)
    ]
    candidate_ids.sort(key=lambda value: (-int(support[value]), value))
    dropped_for_cap = max(0, len(candidate_ids) - int(max_instances))
    candidate_ids = candidate_ids[: int(max_instances)]

    result = np.zeros_like(assignments, dtype=np.int32)
    for new_id, component_id in enumerate(candidate_ids, start=1):
        result[assignments == component_id] = new_id
    audit = {
        "raw_endpoint_components": int(component_count - 1),
        "kept_instances": len(candidate_ids),
        "dropped_small_components": int(
            component_count - 1 - len(candidate_ids) - dropped_for_cap
        ),
        "dropped_for_cap": int(dropped_for_cap),
        "unassigned_foreground_pixels": int(np.sum(foreground & (result == 0))),
    }
    return result, audit


def majority_class_map(
    predicted_instances: np.ndarray,
    gt_instances: np.ndarray,
    gt_class_map: Mapping[int, int],
) -> Dict[int, int]:
    """Assign each oracle basin its majority GT semantic class."""
    gt_class_image = np.full(gt_instances.shape, -1, dtype=np.int8)
    for instance_id, class_id in gt_class_map.items():
        gt_class_image[gt_instances == int(instance_id)] = int(class_id)
    result: Dict[int, int] = {}
    for predicted_id in np.unique(predicted_instances):
        if int(predicted_id) == 0:
            continue
        values = gt_class_image[predicted_instances == int(predicted_id)]
        values = values[values >= 0]
        if values.size == 0:
            continue
        counts = np.bincount(values, minlength=2)
        result[int(predicted_id)] = int(np.argmax(counts))
    return result
