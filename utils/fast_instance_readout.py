# -*- coding: utf-8 -*-
"""独立整数读出候选：计数与查表代替逐实例整图扫描，旧入口默认不变。

仅替换 compact_instance_readout.finish_watershed。分水岭、浮点类别投票和
最终面积清理不在本模块执行；返回的全0占位类别仍须由原raw投票覆盖。
"""
from __future__ import annotations

import numpy as np

from utils.instance_area_policy import resolve_instance_area_policy
from utils.post_process import _merge_region_map_to_cap


FORMAT = 'integer_lookup_instance_readout_v1'
# 有界计数表约8 MiB；超大稀疏int32 ID改用unique，不按max ID申请内存。
_DENSE_LABEL_LIMIT = 1 << 20


def _label_counts(source):
    """返回升序ID、整数面积和逐像素索引；稀疏ID不申请无界计数表。"""
    maximum = int(source.max())
    if maximum <= _DENSE_LABEL_LIMIT:
        counts = np.bincount(source.ravel())
        return np.arange(len(counts), dtype=np.int32), counts, source
    labels, inverse, counts = np.unique(source, return_inverse=True, return_counts=True)
    return labels, counts, inverse.reshape(source.shape)


def finish_watershed(watershed, inference):
    """与旧函数同签名、同ID、同占位类别、同审计字段的整数快路径。

    cap仍接收原ID的过滤结果并调用旧算法；既不改变cap分支的数据域，
    也不改变其失败方式。只有cap完成后才生成最终连续uint16 ID。
    """
    if (not isinstance(watershed, np.ndarray) or watershed.dtype != np.int32
            or watershed.ndim != 2 or not watershed.size):
        raise ValueError('Expected nonempty HW watershed labels')
    source = np.maximum(watershed, 0).copy()
    area = resolve_instance_area_policy(inference)
    maximum = int(inference.get('max_instance_id', 65535))
    if not 1 <= maximum <= 65535:
        raise ValueError('max_instance_id must be within [1, 65535]')

    minimum = area['pre_vote_final_min_area']
    if minimum == 0:
        candidate = source
    else:
        labels, counts, index = _label_counts(source)
        keep = (labels > 0) & (counts >= minimum)
        candidate = np.where(keep, labels, 0).astype(np.int32, copy=False)[index]
    candidate, cap_merges = _merge_region_map_to_cap(candidate, maximum)
    labels, counts, index = _label_counts(candidate)
    positive = np.flatnonzero((labels > 0) & (counts > 0))
    # labels已升序；稳定降序使同面积区域仍按原ID升序编号。
    ordered = positive[np.argsort(-counts[positive], kind='stable')]
    if len(ordered) > maximum:
        raise RuntimeError('Compact watershed output exceeded the configured instance cap')
    lookup = np.zeros(len(counts), dtype=np.uint16)
    lookup[ordered] = np.arange(1, len(ordered) + 1, dtype=np.uint16)
    instances = lookup[index]
    placeholders = {current: 0 for current in range(1, len(ordered) + 1)}
    if int(instances.max()) > maximum:
        raise RuntimeError('Compact watershed output exceeded the configured instance cap')
    return instances, placeholders, dict(
        pre_vote_final_min_area=area['pre_vote_final_min_area'],
        maximum_id=maximum, cap_merge_edges=[list(pair) for pair in cap_merges],
        sorting='original stable descending area; equal areas retain ascending original label',
        temporary_semantic_vote_calls=0,
        temporary_classes='P0 placeholders; mandatory raw revote before final cleanup')
