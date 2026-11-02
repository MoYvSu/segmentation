# -*- coding: utf-8 -*-
"""固定C算法的纯CPU读出：种子、分水岭、实例投票和最终小区合并。"""
from __future__ import annotations
from typing import Tuple
from collections.abc import Mapping
import cv2
import numpy as np
import torch
from torch.nn import functional as F


def _boundary_skeleton_belt(
    boundary_mask: np.ndarray,
    bridge_width: int,
    dilate_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a binary boundary map into its skeleton and barrier belt."""
    boundary_binary = (boundary_mask > 0).astype(np.uint8) * 255
    if bridge_width > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * bridge_width + 1, 2 * bridge_width + 1)
        )
        boundary_binary = cv2.dilate(boundary_binary, k)
    if not np.any(boundary_binary):
        skeleton = boundary_binary
    else:
        try:
            skeleton = cv2.ximgproc.thinning(
                boundary_binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
        except (AttributeError, cv2.error):
            from skimage.morphology import skeletonize as sk_skeletonize

            skeleton = (sk_skeletonize(boundary_binary > 0) * 255).astype(np.uint8)

    if dilate_width > 0:
        kernel_size = 2 * dilate_width + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        skeleton_belt = cv2.dilate(skeleton, kernel)
    else:
        skeleton_belt = skeleton
    return skeleton, skeleton_belt

def reconstruct_marker_boundary(
    boundary_probability: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    max_steps: int,
) -> np.ndarray:
    """Locally grow strong marker edges through nearby weak responses.

    Unlike unbounded hysteresis, the finite growth radius cannot absorb an
    entire foggy weak-edge component merely because it touches one strong edge.
    The reconstructed mask is used for marker topology only; the watershed
    elevation/barrier continues to use the high-threshold boundary mask.
    """
    probability = np.asarray(boundary_probability, dtype=np.float32)
    strong = probability > float(high_threshold)
    steps = max(0, int(max_steps))
    if steps == 0 or float(low_threshold) >= float(high_threshold):
        return strong.astype(np.uint8)

    weak = probability > float(low_threshold)
    reconstructed = strong.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(steps):
        grown = cv2.dilate(reconstructed, kernel)
        grown = ((grown > 0) & weak).astype(np.uint8)
        if np.array_equal(grown, reconstructed):
            break
        reconstructed = grown
    return reconstructed

def _region_adjacency(region_map: np.ndarray, radius: int = 2):
    """Build a contact-weighted graph, bridging one-pixel watershed seams."""
    labels = [int(v) for v in np.unique(region_map) if int(v) > 0]
    graph = {label: {} for label in labels}
    h, w = region_map.shape
    radius = max(1, int(radius))
    for dy in range(0, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx <= 0:
                continue
            y0a, y1a = 0, h - dy
            y0b, y1b = dy, h
            if dx >= 0:
                x0a, x1a = 0, w - dx
                x0b, x1b = dx, w
            else:
                x0a, x1a = -dx, w
                x0b, x1b = 0, w + dx
            a = region_map[y0a:y1a, x0a:x1a]
            b = region_map[y0b:y1b, x0b:x1b]
            valid = (a > 0) & (b > 0) & (a != b)
            if not np.any(valid):
                continue
            pairs = np.stack((a[valid], b[valid]), axis=1).astype(np.int64)
            pairs.sort(axis=1)
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            for (left, right), count in zip(unique_pairs, counts):
                left, right, count = int(left), int(right), int(count)
                graph[left][right] = graph[left].get(right, 0) + count
                graph[right][left] = graph[right].get(left, 0) + count
    return graph

def _merge_region_map_to_cap(
    region_map: np.ndarray,
    max_regions: int,
) -> Tuple[np.ndarray, list]:
    """Agglomerate smallest adjacent regions without creating a global ID."""
    source_map = np.asarray(region_map, dtype=np.int32)
    labels, counts = np.unique(source_map[source_map > 0], return_counts=True)
    active = {int(label): int(count) for label, count in zip(labels, counts)}
    max_regions = max(1, int(max_regions))
    if len(active) <= max_regions:
        return source_map.copy(), []

    graph = _region_adjacency(source_map, radius=2)
    parent = {label: label for label in active}
    merge_edges = []
    while len(active) > max_regions:
        source = min(active, key=lambda label: (active[label], label))
        neighbours = {
            label: contact for label, contact in graph.get(source, {}).items()
            if label in active and label != source
        }
        if neighbours:
            target = max(
                neighbours,
                key=lambda label: (neighbours[label], active[label], -label),
            )
            parent[source] = target
            merge_edges.append((source, target))
            active[target] += active[source]
            for neighbour, contact in list(graph.get(source, {}).items()):
                if neighbour not in active or neighbour == target:
                    continue
                graph[target][neighbour] = graph[target].get(neighbour, 0) + contact
                graph[neighbour][target] = graph[neighbour].get(target, 0) + contact
                graph[neighbour].pop(source, None)
            graph[target].pop(source, None)
        else:
            parent[source] = 0
        graph.pop(source, None)
        active.pop(source)

    def root(label):
        path = []
        while parent[label] not in (0, label):
            path.append(label)
            label = parent[label]
        resolved = parent[label] if parent[label] == 0 else label
        for item in path:
            parent[item] = resolved
        return resolved

    label_lookup = np.zeros(int(source_map.max()) + 1, dtype=np.int32)
    for label in parent:
        resolved = root(label)
        if resolved > 0:
            label_lookup[label] = resolved
    merged = label_lookup[source_map]

    # Vectorized one-pixel seam fill. A zero pixel is filled only when at least
    # two distinct original regions touch it and all touching regions now share
    # the same agglomerated root. Boundaries between retained roots stay zero.
    zero = source_map == 0
    root_min = np.full(source_map.shape, np.iinfo(np.int32).max, dtype=np.int32)
    root_max = np.zeros(source_map.shape, dtype=np.int32)
    source_min = root_min.copy()
    source_max = np.zeros(source_map.shape, dtype=np.int32)
    merged_pad = np.pad(merged, 1, mode="constant")
    source_pad = np.pad(source_map, 1, mode="constant")
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbour_root = merged_pad[
                1 + dy : 1 + dy + merged.shape[0],
                1 + dx : 1 + dx + merged.shape[1],
            ]
            neighbour_source = source_pad[
                1 + dy : 1 + dy + merged.shape[0],
                1 + dx : 1 + dx + merged.shape[1],
            ]
            valid = zero & (neighbour_root > 0)
            root_min[valid] = np.minimum(root_min[valid], neighbour_root[valid])
            root_max[valid] = np.maximum(root_max[valid], neighbour_root[valid])
            source_min[valid] = np.minimum(
                source_min[valid], neighbour_source[valid]
            )
            source_max[valid] = np.maximum(
                source_max[valid], neighbour_source[valid]
            )
    fill = (
        zero
        & (root_max > 0)
        & (root_min == root_max)
        & (source_min < source_max)
    )
    merged[fill] = root_max[fill]
    return merged, merge_edges

def _classes(classes, labels):
    if not isinstance(classes, Mapping):
        raise ValueError('Classes must be a mapping')
    normalized = {}
    for key, value in classes.items():
        if type(key) is int:
            label = key
        elif isinstance(key, str) and key.isascii() and key.isdecimal() and str(int(key)) == key:
            label = int(key)
        else:
            raise ValueError('Class IDs must be canonical positive integers')
        if not 1 <= label <= 65535 or str(label) in normalized:
            raise ValueError('Invalid or duplicate class ID')
        if type(value) is not int or value not in (0, 1):
            raise ValueError('Classes must be integer 0/1')
        normalized[str(label)] = value
    if set(normalized) != {str(int(label)) for label in labels}:
        raise ValueError('Classes mismatch positive instance IDs')
    return normalized

def _tiny_coordinates(source, tiny):
    """一次扫描收集小区像素，避免每个 ID 各扫描一次整图。"""
    choose = np.zeros(65536, bool)
    choose[tiny] = True
    y, x = np.nonzero(choose[source])
    if not len(y):
        return {}
    labels = source[y, x]
    order = np.argsort(labels, kind='stable')
    labels, y, x = labels[order], y[order], x[order]
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return {int(labels[a]): (y[a:b], x[a:b]) for a, b in zip(starts[:-1], starts[1:])}

def _contacts(source, y, x, retained):
    """等价于 radius=2 无向接触图，仅统计所需的小区到原大区的边。"""
    h, w = source.shape
    result = {}
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == dx == 0:
                continue
            yy, xx = y + dy, x + dx
            good = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            values = source[yy[good], xx[good]]
            values = values[retained[values]]
            keys, counts = np.unique(values, return_counts=True)
            for key, count in zip(keys, counts):
                key = int(key)
                result[key] = result.get(key, 0) + int(count)
    return result

def _seam_candidates(source, coordinates):
    """可回填缝必在某个被并小区的一像素邻域内，只缓存这些零像素。"""
    h, w = source.shape
    if not coordinates:
        return np.empty(0, np.int64), np.empty((0, 8), np.uint16)
    pieces = []
    for y, x in coordinates:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                valid = (yy > 0) & (yy < h - 1) & (xx > 0) & (xx < w - 1)
                yy, xx = yy[valid], xx[valid]
                zero = source[yy, xx] == 0
                pieces.append(yy[zero].astype(np.int64) * w + xx[zero])
    flat = np.unique(np.concatenate(pieces)) if pieces else np.empty(0, np.int64)
    y, x = np.divmod(flat, w)
    neighbors = np.stack([source[y + dy, x + dx] for dy in (-1, 0, 1)
                          for dx in (-1, 0, 1) if dy or dx], axis=1)
    return flat, neighbors

def _apply_mapping(source, mapping, seam_flat, neighbors):
    lookup = np.arange(65536, dtype=np.uint16)
    for label, target in mapping.items():
        lookup[label] = target
    merged = lookup[source]
    fill = np.zeros(source.shape, bool)
    if mapping and len(seam_flat):
        roots = lookup[neighbors].astype(np.int32)
        original = neighbors.astype(np.int32)
        sentinel = 65536
        root_min = np.min(np.where(roots > 0, roots, sentinel), axis=1)
        root_max = roots.max(axis=1)
        source_min = np.min(np.where(original > 0, original, sentinel), axis=1)
        source_max = original.max(axis=1)
        selected = (root_max > 0) & (root_min == root_max) & (source_min < source_max)
        flat = seam_flat[selected]
        merged.flat[flat] = root_max[selected].astype(np.uint16)
        fill.flat[flat] = True
    return merged, fill

def _attached(source, merged, label, target, coordinates):
    y, x = coordinates[label]
    y0, y1 = max(0, int(y.min()) - 3), min(source.shape[0], int(y.max()) + 4)
    x0, x1 = max(0, int(x.min()) - 3), min(source.shape[1], int(x.max()) + 4)
    old, new = source[y0:y1, x0:x1], merged[y0:y1, x0:x1]
    _, components = cv2.connectedComponents((new == target).astype(np.uint8), connectivity=8)
    from_small = set(int(v) for v in np.unique(components[old == label]))
    from_target = set(int(v) for v in np.unique(components[old == target])) - {0}
    return bool(from_small) and 0 not in from_small and from_small <= from_target

def _label_counts(source):
    """返回升序ID、整数面积和逐像素索引；稀疏ID不申请无界计数表。"""
    maximum = int(source.max())
    if maximum <= (1 << 20):
        counts = np.bincount(source.ravel())
        return np.arange(len(counts), dtype=np.int32), counts, source
    labels, inverse, counts = np.unique(source, return_inverse=True, return_counts=True)
    return labels, counts, inverse.reshape(source.shape)


def probability_to_logit(probability):
    value = probability.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(value) - torch.log1p(-value)


def _markers(belt):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(belt), connectivity=8)
    markers = np.zeros(belt.shape, np.int32)
    valid = 0
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < 50:
            continue
        valid += 1
        markers[labels == label] = valid
    return markers, valid


def _seal(belt):
    width = min(2, max(1, min(belt.shape) // 2))
    belt[:width, :] = 255; belt[-width:, :] = 255
    belt[:, :width] = 255; belt[:, -width:] = 255
    return belt


def markers_and_barrier(after, reference):
    probability = torch.sigmoid(probability_to_logit(torch.from_numpy(after))).numpy()
    high = (probability > .65).astype(np.uint8)
    _, barrier = _boundary_skeleton_belt(high, 1, 1)
    reconstructed = reconstruct_marker_boundary(probability, .45, .65, 8)
    _, original_belt = _boundary_skeleton_belt(reconstructed, 1, 1)
    _, original_count = _markers(_seal(original_belt))

    bridged = (reconstructed > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bridged = cv2.dilate(bridged, kernel)
    padded = cv2.copyMakeBorder(bridged, 32, 32, 32, 32, cv2.BORDER_REFLECT)
    expanded, _ = _boundary_skeleton_belt(padded, 0, 0)
    height, width = after.shape
    skeleton = expanded[32:32 + height, 32:32 + width].copy()
    belt = _seal(cv2.dilate(skeleton, kernel))
    markers, _ = _markers(belt)

    ids, inverse = np.unique(markers.ravel(), return_inverse=True)
    low_counts = np.bincount(inverse[reference.ravel() <= .45], minlength=len(ids))
    keep = (ids == 0) | (low_counts > 0)
    if np.any(ids > 0) and not np.any((ids > 0) & keep):
        raise ValueError('No seed has support in the original boundary map')
    selected = np.where(keep[inverse], markers.ravel(), np.int32(0)).reshape(markers.shape)
    selected = np.array(selected, dtype=np.int32, copy=True, order='C')
    if not original_count:
        raise ValueError('No original watershed seeds remain after the area threshold')
    return selected, barrier


def finish_watershed(watershed):
    source = np.maximum(watershed, 0).copy()
    candidate, _ = _merge_region_map_to_cap(source, 65535)
    labels, counts, index = _label_counts(candidate)
    positive = np.flatnonzero((labels > 0) & (counts > 0))
    ordered = positive[np.argsort(-counts[positive], kind='stable')]
    if len(ordered) > 65535:
        raise RuntimeError('Instance count exceeds uint16 range')
    lookup = np.zeros(len(counts), dtype=np.uint16)
    lookup[ordered] = np.arange(1, len(ordered) + 1, dtype=np.uint16)
    return lookup[index]


def watershed_instances(semantic, after, reference):
    if (not torch.is_tensor(semantic) or semantic.dtype != torch.float32
            or tuple(semantic.shape) != (1, 1, *after.shape) or not bool(torch.isfinite(semantic).all())):
        raise ValueError('Semantic logits must be finite native FP32 B1HW')
    for value in (after, reference):
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != after.shape
                or value.ndim != 2 or not value.size or not np.isfinite(value).all()
                or np.any((value < 0) | (value > 1))):
            raise ValueError('Boundary must be native finite FP32 HW probability')
    selected, barrier = markers_and_barrier(after, reference)
    semantic = semantic.detach().cpu().contiguous()
    boundary = probability_to_logit(torch.from_numpy(after)[None, None])
    packed = F.interpolate(torch.cat([semantic, boundary], dim=1), size=after.shape,
                           mode='bilinear', align_corners=True)
    probability = torch.sigmoid(packed[0, 0]).numpy()
    image = np.full((*after.shape, 3), 128, np.uint8)
    image[probability > .5] = 200
    overlay = image.copy()
    overlay[barrier > 0] = 255
    image = cv2.addWeighted(image, .7, overlay, .3, 0)
    return finish_watershed(cv2.watershed(image, selected.copy()))


def vote_instances(instances, probability):
    if (instances.dtype != np.uint16 or instances.ndim != 2 or probability.shape != instances.shape
            or probability.dtype != np.float32 or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError('Instance vote requires native uint16 IDs and finite FP32 probabilities')
    classes = {}
    for value in np.unique(instances):
        if value == 0:
            continue
        mask = instances == value
        ys, xs = np.nonzero(mask)
        region = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        # 保留原先bbox内、逐行选取后的float32 np.mean，不能改float64分组累加。
        score = float(np.mean(probability[region][mask[region]]))
        classes[str(int(value))] = 1 if score > .5 else 0
    return classes


def merge_final_instances(ids, classes, min_area=200):
    source = np.asarray(ids)
    labels, counts = np.unique(source[source > 0], return_counts=True)
    original_classes = _classes(classes, labels)
    sizes = {int(key): int(value) for key, value in zip(labels, counts)}
    tiny = sorted((key for key, area in sizes.items() if area < min_area), key=lambda key: (sizes[key], key))
    coords = _tiny_coordinates(source, tiny) if tiny else {}
    retained = np.zeros(65536, bool)
    retained[[key for key, area in sizes.items() if area >= min_area]] = True
    mapping = {}
    for label in tiny:
        y, x = coords[label]
        if np.any((y == 0) | (y == source.shape[0] - 1) | (x == 0) | (x == source.shape[1] - 1)):
            continue
        neighbors = _contacts(source, y, x, retained)
        if neighbors:
            mapping[label] = max(neighbors, key=lambda key: (neighbors[key], sizes[key], -key))
    seam_flat, neighbors = _seam_candidates(source, [coords[label] for label in mapping])
    while True:
        merged, _ = _apply_mapping(source, mapping, seam_flat, neighbors)
        unsafe = [label for label, target in mapping.items()
                  if not _attached(source, merged, label, target, coords)]
        if not unsafe:
            break
        for label in unsafe:
            del mapping[label]
    final_classes = {key: value for key, value in original_classes.items() if int(key) not in mapping}
    return merged, final_classes


def verify_output(ids, classes, shape):
    if (not isinstance(ids, np.ndarray) or ids.dtype != np.uint16 or ids.ndim != 2
            or ids.shape != tuple(shape) or int(ids.max()) > 65535):
        raise ValueError('Final instance map must be native single-channel uint16')
    _classes(classes, np.unique(ids[ids > 0]))
