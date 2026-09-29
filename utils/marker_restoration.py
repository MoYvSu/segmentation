# -*- coding: utf-8 -*-
"""恢复骨架化前已存在的种子分隔，不添加边框或连接新边界。"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def restore_marker_partitions(markers: np.ndarray, marker_boundary_mask: np.ndarray,
                              bridge_width: int, min_core_area: int = 50) -> tuple[np.ndarray, dict]:
    """在每个旧正种子内部，恢复粗边界背景已经分开的多个可靠核心。

    参考区域来自整图 marker_boundary_mask 经既有椭圆桥接膨胀后的背景8连通域。
    每个参考区域与旧种子的交集只取最大连通块；至少两个不同参考区域分别达到
    min_core_area 才拆分。最大核心保留旧ID，其余新增ID，旧种子的剩余像素置0。
    未触发拆分的旧种子逐像素保留。本恢复步骤不改输入、不跨旧种子合并、不删除旧ID。
    后续统一面积过滤独立执行，仍可删除任意小区域的种子，不受这里的ID保留约束。
    输出为 watershed 所需的 int32 种子；audit 的 splits 记录恢复阶段的来源关系。
    """
    markers = np.asarray(markers)
    source = np.asarray(marker_boundary_mask)
    if markers.ndim != 2 or source.shape != markers.shape or not markers.size:
        raise ValueError('Markers and boundary mask must be nonempty two-dimensional arrays of the same shape')
    if not np.issubdtype(markers.dtype, np.integer) or np.any(markers < 0):
        raise ValueError('Marker IDs must be nonnegative integers')
    maximum = int(markers.max())
    if maximum > np.iinfo(np.int32).max:
        raise ValueError('Marker IDs must fit int32')
    if not np.all(np.isfinite(source)):
        raise ValueError('Boundary mask must be finite')
    bridge_width, min_core_area = int(bridge_width), int(min_core_area)
    if bridge_width < 0 or min_core_area < 1:
        raise ValueError('bridge_width must be nonnegative and min_core_area positive')
    boundary = (source > 0).astype(np.uint8)
    if bridge_width > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * bridge_width + 1, 2 * bridge_width + 1))
        boundary = cv2.dilate(boundary, kernel)
    reference_count, reference = cv2.connectedComponents((boundary == 0).astype(np.uint8), connectivity=8)
    restored = markers.astype(np.int32, copy=True)
    boxes = ndimage.find_objects(markers)
    next_id = maximum + 1
    splits = []
    old_count = 0
    for parent_id, box in enumerate(boxes, 1):
        if box is None:
            continue
        old_count += 1
        parent = markers[box] == parent_id
        reference_crop = reference[box]
        covered = parent & (reference_crop > 0)
        if not covered.any():
            continue
        # 不同全图参考区域不可能在这里相接，故一次连通域计算便可取得全部交集分块。
        count, parts, stats, _ = cv2.connectedComponentsWithStats(covered.astype(np.uint8), connectivity=8)
        reference_ids = np.zeros(count, dtype=np.int32)
        reference_ids[parts[covered]] = reference_crop[covered]
        components = [part_id for part_id in range(1, count)
                      if int(stats[part_id, cv2.CC_STAT_AREA]) >= min_core_area]
        # 面积相同时以位置决定顺序；每个参考区域只保留第一项，即最大连通交集。
        components.sort(key=lambda part_id: (
            -int(stats[part_id, cv2.CC_STAT_AREA]),
            int(stats[part_id, cv2.CC_STAT_TOP]),
            int(stats[part_id, cv2.CC_STAT_LEFT]),
            part_id,
        ))
        selected = []
        seen = set()
        for part_id in components:
            reference_id = int(reference_ids[part_id])
            if reference_id not in seen:
                selected.append(part_id)
                seen.add(reference_id)
        if len(selected) < 2:
            continue
        if next_id + len(selected) - 2 > np.iinfo(np.int32).max:
            raise ValueError('Restored marker IDs exceed int32')
        children_ids = [parent_id] + list(range(next_id, next_id + len(selected) - 1))
        next_id += len(selected) - 1
        destination = restored[box]
        destination[parent] = 0
        for part_id, child_id in zip(selected, children_ids):
            destination[parts == part_id] = child_id
        parent_area = int(parent.sum())
        core_areas = [int(stats[part_id, cv2.CC_STAT_AREA]) for part_id in selected]
        core_total = sum(core_areas)
        splits.append({
            'parent_id': parent_id, 'parent_area': parent_area,
            'children_ids': children_ids,
            'reference_ids': [int(reference_ids[part_id]) for part_id in selected],
            'core_areas': core_areas, 'core_total_area': core_total,
            'released_pixels': parent_area - core_total,
        })
    added = next_id - maximum - 1
    audit = {
        'version': 'marker_partition_restore_v1',
        'bridge_width': bridge_width, 'min_core_area': min_core_area,
        'reference_components': int(reference_count - 1),
        'old_marker_count': old_count, 'new_marker_count': old_count + added,
        'split_parent_count': len(splits), 'preserved_parent_count': old_count - len(splits),
        'added_marker_count': added,
        'released_marker_pixels': sum(item['released_pixels'] for item in splits),
        'splits': splits,
    }
    return restored, audit
