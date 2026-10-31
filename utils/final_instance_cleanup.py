# -*- coding: utf-8 -*-
"""最终原生实例小区合并：仅用几何邻接，不读模型、GT或平均面积目标。

该步骤位于最终语义投票之后。小区继承被并入的大区类别，不能重新投票。
未找到可靠连接时保留小区，避免删除像素或留下同 ID 的分离小岛。
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import cv2
import numpy as np


ACTION = 'neighbor_merge_v1'


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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


def merge_final_instances(ids, classes, min_area=0, *, action=ACTION):
    """返回独立 uint16 实例图、字符串键类别表和可序列化审计记录。

    面积使用最终原图像素。候选面积严格小于 ``min_area``，只能并入
    原本面积大于等于该值的直接邻区；按接触长度、原面积、较小 ID 决胜。
    ``min_area=0`` 为恒等。该函数不改变种子、不重新执行分水岭或语义票。
    """
    if type(min_area) is not int or min_area < 0:
        raise ValueError('min_area must be a nonnegative integer')
    if action != ACTION:
        raise ValueError('Unknown final small-instance action: ' + str(action))
    source = np.asarray(ids)
    if source.ndim != 2 or source.dtype != np.uint16 or not all(source.shape):
        raise ValueError('Expected a nonempty original uint16 instance map')
    labels, counts = np.unique(source[source > 0], return_counts=True)
    original_classes = _classes(classes, labels)
    original_hash = _hash(source)
    sizes = {int(k): int(v) for k, v in zip(labels, counts)}
    tiny = sorted((k for k, area in sizes.items() if area < min_area), key=lambda k: (sizes[k], k))
    coords = _tiny_coordinates(source, tiny) if tiny else {}
    retained = np.zeros(65536, bool)
    retained[[k for k, area in sizes.items() if area >= min_area]] = True
    mapping, decisions = {}, []
    for label in tiny:
        record = dict(source_id=label, original_area=sizes[label],
                      original_class=original_classes[str(label)], merged=False)
        y, x = coords[label]
        if np.any((y == 0) | (y == source.shape[0] - 1) | (x == 0) | (x == source.shape[1] - 1)):
            record['reason'] = 'touches_original_image_frame'
        else:
            neighbors = _contacts(source, y, x, retained)
            if neighbors:
                target = max(neighbors, key=lambda k: (neighbors[k], sizes[k], -k))
                mapping[label] = target
                record.update(target_id=target, target_class=original_classes[str(target)],
                              contact_score=neighbors[target], neighbours=neighbors, merged=True,
                              reason='longest_local_contact_to_original_ge_threshold_neighbour')
            else:
                record['reason'] = 'no_direct_retained_large_neighbour'
        decisions.append(record)
    seam_flat, neighbors = _seam_candidates(source, [coords[label] for label in mapping])
    rejected = set()
    while True:
        merged, fill = _apply_mapping(source, mapping, seam_flat, neighbors)
        unsafe = [label for label, target in mapping.items()
                  if not _attached(source, merged, label, target, coords)]
        if not unsafe:
            break
        for label in unsafe:
            rejected.add(label)
            del mapping[label]
    for record in decisions:
        if record['source_id'] in rejected:
            record.update(merged=False, reason='would_leave_disconnected_ID_or_unfilled_seam')
    final_classes = {key: value for key, value in original_classes.items() if int(key) not in mapping}
    changed_labels = np.zeros(65536, bool)
    changed_labels[list(mapping)] = True
    allowed = changed_labels[source] | fill
    contract = (np.array_equal(merged[~allowed], source[~allowed])
                and np.array_equal(merged[[0, -1], :], source[[0, -1], :])
                and np.array_equal(merged[:, [0, -1]], source[:, [0, -1]])
                and set(final_classes) == {str(int(i)) for i in np.unique(merged) if i}
                and original_hash == _hash(source))
    if not contract:
        raise RuntimeError('Final merge violated immutable-pixel or ID/class contract')
    audit = dict(action=ACTION, threshold=min_area, final_min_instance_area=min_area,
        threshold_unit='native pixels of original final instance', class_agnostic=True,
        raw_probabilities_available=False, semantic_policy='target class inherited; no revote',
        no_target_count_or_baseline_area_fitting=True, input_ID_sha256=original_hash,
        output_ID_sha256=_hash(merged), candidate_small_count=len(tiny), merged_count=len(mapping),
        retained_ambiguous_count=len(tiny) - len(mapping), merged_ids=sorted(mapping), decisions=decisions,
        filled_seam_pixels=int(fill.sum()), changed_pixels=int(np.count_nonzero(merged != source)),
        retained_original_pixels_exact=True, retained_classes_exact=True, image_frame_exact=True,
        no_new_disconnected_islands=True, source_input_unchanged=True, final_contract_valid=True)
    return merged, final_classes, audit
