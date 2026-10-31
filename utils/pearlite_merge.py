# -*- coding: utf-8 -*-
"""把原预测中相邻的珠光体区域连接起来，不加载模型、不改变铁素体。

P代表珠光体，类别0；F代表铁素体，类别1。这里的连通关系只来自原实例图，
不依据RGB猜测材料，不放大邻接范围，也不试图纠正语义误判。
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import numpy as np


FORMAT = 'adjacent_pearlite_merge_v1'


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _class_sha(classes):
    return hashlib.sha256(json.dumps(classes, sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()


def _validate(instances, classes):
    if (not isinstance(instances, np.ndarray) or np.ma.isMaskedArray(instances)
            or instances.ndim != 2 or instances.dtype != np.uint16 or not all(instances.shape)):
        raise ValueError('Expected a nonempty two-dimensional uint16 instance map')
    if not isinstance(classes, Mapping):
        raise ValueError('Classes must be a canonical str-ID mapping')
    result = dict(classes)
    labels = {str(int(v)) for v in np.unique(instances) if v}
    if (set(result) != labels
            or any(not isinstance(k, str) or not k.isascii() or not k.isdecimal()
                   or str(int(k)) != k or not 1 <= int(k) <= 65535 for k in result)
            or any(type(value) is not int or value not in (0, 1) for value in result.values())):
        raise ValueError('Classes must exactly match positive IDs using canonical str keys and integer 0/1')
    return result


def _edge_keys(a, b):
    """使用uint64编码原始ID对，避免uint16乘法溢出。"""
    a, b = a.astype(np.uint64), b.astype(np.uint64)
    return np.minimum(a, b)*65536+np.maximum(a, b)


def _summary(instances, classes):
    areas = np.bincount(instances.ravel(), minlength=65536)
    result = {}
    for phase in (0, 1):
        labels = [int(k) for k, value in classes.items() if value == phase]
        pixels = sum(int(areas[label]) for label in labels)
        result[str(phase)] = dict(instances=len(labels), pixels=pixels,
                                  mean_area=pixels/len(labels) if labels else None)
    return result, areas


def merge_adjacent_pearlite(instances, classes):
    """返回 ``(new_instances, new_classes, audit)``。

    只采用原图四邻接P/P接触，以及恰经一个内部零像素的P/P接触；零像素的
    所有正四邻居必须全是P，至少来自两个不同原ID。三P交汇允许连接，
    但P/P/F交点、宽零缝及外框零均不连接或填充。

    与单次配对不同，本规则允许A-B-C依次接触的整个P组归为一个ID。
    每组保留最小原ID，F像素/ID/类别逐值不变；不重新投票、不运行分水岭、
    不按面积筛选、不重编号。原来同ID已有分离小岛时保持原义，不强修拓扑。
    """
    original_classes = _validate(instances, classes)
    source = instances
    before_hash = _sha(source)
    before_class_hash = _class_sha(original_classes)
    before_summary, areas = _summary(source, original_classes)
    p_ids = sorted(int(k) for k, value in original_classes.items() if value == 0)
    p_lookup = np.zeros(65536, bool)
    p_lookup[p_ids] = True
    f_lookup = np.zeros(65536, bool)
    f_lookup[[int(k) for k, value in original_classes.items() if value == 1]] = True

    direct_keys = np.empty(0, np.uint64)
    bridge_keys = np.empty(0, np.uint64)
    safe_flat = np.empty(0, np.int64)
    safe_low = np.empty(0, np.uint16)
    if len(p_ids) >= 2:
        direct_parts = []
        for a, b in ((source[:, :-1], source[:, 1:]), (source[:-1], source[1:])):
            choose = p_lookup[a] & p_lookup[b] & (a != b)
            direct_parts.append(_edge_keys(a[choose], b[choose]))
        direct_keys = np.unique(np.concatenate(direct_parts))

        # 一次只看原始零像素，禁止填缝后继续扩散到更宽背景。
        y, x = np.nonzero(source == 0)
        h, w = source.shape
        interior = (y > 0) & (y < h-1) & (x > 0) & (x < w-1)
        y, x = y[interior], x[interior]
        neighbors = np.stack((source[y-1, x], source[y+1, x],
                              source[y, x-1], source[y, x+1]), axis=1)
        positive = neighbors > 0
        low = np.min(np.where(positive, neighbors.astype(np.int32), 65536), axis=1)
        high = neighbors.max(axis=1)
        safe = ((low < high) & np.all(~positive | p_lookup[neighbors], axis=1))
        neighbors = neighbors[safe]
        safe_flat = (y[safe].astype(np.int64)*w+x[safe]).astype(np.int64)
        safe_low = low[safe].astype(np.uint16)
        bridge_parts = []
        for i in range(4):
            for j in range(i+1, 4):
                a, b = neighbors[:, i], neighbors[:, j]
                choose = (a > 0) & (b > 0) & (a != b)
                bridge_parts.append(_edge_keys(a[choose], b[choose]))
        bridge_keys = np.unique(np.concatenate(bridge_parts))

    parent = np.arange(65536, dtype=np.int32)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    all_keys = np.union1d(direct_keys, bridge_keys)
    for key in all_keys:
        a, b = int(key)//65536, int(key)%65536
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    lookup = np.arange(65536, dtype=np.uint16)
    groups = {}
    for label in p_ids:
        root = find(label)
        lookup[label] = root
        groups.setdefault(root, []).append(label)
    output = lookup[source]
    output.flat[safe_flat] = lookup[safe_low]
    output_classes = {k: value for k, value in original_classes.items()
                      if value == 1 or lookup[int(k)] == int(k)}
    after_summary, final_areas = _summary(output, output_classes)

    fill_counts = np.bincount(lookup[safe_low], minlength=65536)
    group_records = [dict(retained_id=root, original_ids=members,
                          original_areas=[int(areas[label]) for label in members],
                          filled_seam_pixels=int(fill_counts[root]),
                          final_area=int(final_areas[root]))
                     for root, members in sorted(groups.items())]
    frame_zero_preserved = all(np.all(new[old == 0] == 0) for old, new in
        ((source[0], output[0]), (source[-1], output[-1]),
         (source[:, 0], output[:, 0]), (source[:, -1], output[:, -1])))
    original_f = f_lookup[source]
    output_f_lookup = np.zeros(65536, bool)
    output_f_lookup[[int(k) for k, value in output_classes.items() if value == 1]] = True
    f_exact = bool(np.array_equal(original_f, output_f_lookup[output])
                   and np.array_equal(output[original_f], source[original_f])
                   and {k: v for k, v in original_classes.items() if v == 1}
                       == {k: v for k, v in output_classes.items() if v == 1})
    fill_mask = np.zeros(source.shape, bool)
    fill_mask.flat[safe_flat] = True
    final_valid = (output.dtype == np.uint16
                   and set(output_classes) == {str(int(v)) for v in np.unique(output) if v}
                   and np.array_equal(output[source > 0], lookup[source[source > 0]])
                   and np.all(output[(source == 0) & ~fill_mask] == 0)
                   and np.all(source.flat[safe_flat] == 0)
                   and frame_zero_preserved and f_exact
                   and _sha(source) == before_hash and dict(classes) == original_classes)
    if not final_valid:
        raise RuntimeError('Pearlite merge violated final ID/class/F/immutable-input contract')

    def pairs(keys):
        return [[int(key)//65536, int(key)%65536] for key in keys]

    audit = dict(format=FORMAT,
                 definition='original direct4 P/P or one interior zero with only P positive four-neighbors; full connected P groups',
                 before_class_summary=before_summary, after_class_summary=after_summary,
                 direct4_pair_count=len(direct_keys), direct4_pairs=pairs(direct_keys),
                 zero_bridge_pair_count=len(bridge_keys), zero_bridge_pairs=pairs(bridge_keys),
                 union_pair_count=len(all_keys), P_groups=group_records,
                 merged_group_count=sum(len(g['original_ids']) > 1 for g in group_records),
                 removed_P_instance_count=len(p_ids)-len(groups),
                 filled_seam_pixels=len(safe_flat), changed_pixels=int(np.count_nonzero(output != source)),
                 input_ID_sha256=before_hash, output_ID_sha256=_sha(output),
                 input_classes_sha256=before_class_hash, output_classes_sha256=_class_sha(output_classes),
                 F_final_ID_map_and_classes_exact=f_exact,
                 original_outer_frame_zero_pixels_unchanged=frame_zero_preserved,
                 existing_multicomponent_IDs_not_repaired=True,
                 chain_merging=True, no_compaction=True, no_model_forward=True,
                 no_scores=True, no_GT=True, no_WS=True, no_semantic_revote=True,
                 no_area_filter=True, no_growth_after_original_one_zero_fill=True,
                 source_input_unchanged=True, final_contract_valid=True,
                 score_gain_or_accuracy_claimed=False)
    return output, output_classes, audit
