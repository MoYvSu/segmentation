# -*- coding: utf-8 -*-
"""固定关系评分的一次纯几何合并；不读模型/GT，不重跑WS或语义投票。"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import math
from numbers import Integral, Real

import cv2
import numpy as np

FORMAT = 'boundary_relation_merge_v1'


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _key(a, b):
    return min(a, b)*65536+max(a, b)


def _geometry(source):
    """原图direct4/one-zero4邻接，以及严格只有两种原正ID的可填零像素。"""
    direct = set()
    for left, right in ((source[:, :-1], source[:, 1:]), (source[:-1], source[1:])):
        mask = (left > 0) & (right > 0) & (left != right)
        a, b = left[mask].astype(np.uint64), right[mask].astype(np.uint64)
        direct.update(int(v) for v in np.unique(np.minimum(a, b)*65536+np.maximum(a, b)))
    y, x = np.nonzero(source == 0)
    neighbors = np.zeros((len(y), 4), np.uint16)
    h, w = source.shape
    for i, (dy, dx) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        yy, xx = y+dy, x+dx
        good = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        neighbors[good, i] = source[yy[good], xx[good]]
    adjacent = direct.copy()
    for i in range(4):
        for j in range(i+1, 4):
            a, b = neighbors[:, i], neighbors[:, j]
            good = (a > 0) & (b > 0) & (a != b)
            aa, bb = a[good].astype(np.uint64), b[good].astype(np.uint64)
            adjacent.update(int(v) for v in np.unique(np.minimum(aa, bb)*65536+np.maximum(aa, bb)))
    positive = neighbors.astype(np.int32)
    low = np.min(np.where(positive > 0, positive, 65536), axis=1)
    high = positive.max(axis=1)
    # 原图外框背景不填；邻接定义仍保留这些接触，只限制可实际填入的零缝。
    interior = (y > 0) & (y < h-1) & (x > 0) & (x < w-1)
    safe = (interior & (low < high) & np.all((positive == 0) | (positive == low[:, None]) | (positive == high[:, None]), axis=1))
    keys = low[safe].astype(np.uint64)*65536+high[safe].astype(np.uint64)
    flat = (y[safe].astype(np.int64)*w+x[safe]).astype(np.int64)
    order = np.argsort(keys, kind='stable');keys, flat = keys[order], flat[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1, len(keys)]
    seams = {int(keys[a]): flat[a:b] for a, b in zip(starts[:-1], starts[1:]) if a < b}
    return adjacent, direct, seams


def _component_counts(mask):
    return {str(connectivity): int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)[0]-1)
            for connectivity in (4, 8)}


def _summary(ids, classes):
    areas = np.bincount(ids.ravel(), minlength=65536)
    result = {}
    for cls in (0, 1):
        labels = [int(key) for key, value in classes.items() if value == cls]
        pixels = sum(int(areas[label]) for label in labels)
        result[str(cls)] = dict(instances=len(labels), pixels=pixels,
                                mean_area=pixels/len(labels) if labels else None)
    return result, areas


def _candidate(value, index):
    record = dict(candidate_index=index, pair_ids=None, keep_score=None)
    if not isinstance(value, Mapping):
        return dict(record, reason='invalid_candidate_mapping')
    pair = value.get('pair_ids')
    if (not isinstance(pair, (list, tuple)) or len(pair) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or not 1 <= int(v) <= 65535 for v in pair)
            or int(pair[0]) == int(pair[1])):
        return dict(record, reason='invalid_pair_ids')
    record['pair_ids'] = sorted(int(v) for v in pair)
    score = value.get('keep_score')
    if (isinstance(score, (bool, np.bool_)) or not isinstance(score, Real)
            or not math.isfinite(float(score)) or not 0 <= float(score) <= 1):
        return dict(record, reason='invalid_keep_score', invalid_score_repr=repr(score)[:160])
    record['keep_score'] = float(score)
    return record


def merge_relation_pairs(instances, classes, candidates, *, threshold=.1):
    """返回 ``(new_ids, new_classes, audit)``，只执行一次非重叠同类配对。

    类别表须使用规范字符串正整数ID及Python整数0/1。坏候选分数、重复pair、
    跨类及非邻接逐条skip并记录；实例dtype/类别表错误直接拒绝。重复pair全部
    拒绝，不能通过输入顺序或挑最低分决定。原图外框零像素保持0；阈值0/空候选
    逐值恒等且不compact。
    """
    if (not isinstance(instances, np.ndarray) or np.ma.isMaskedArray(instances)
            or instances.ndim != 2 or instances.dtype != np.uint16 or not all(instances.shape)):
        raise ValueError('Expected a nonempty native uint16 instance map')
    if (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
            or not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1):
        raise ValueError('threshold must be finite in [0,1]')
    if not isinstance(classes, Mapping):
        raise ValueError('classes must be a complete canonical str-ID mapping')
    source = instances
    labels = {str(int(v)) for v in np.unique(source) if v}
    original_classes = dict(classes)
    if (set(original_classes) != labels or any(not isinstance(k, str) or not k.isascii() or not k.isdecimal()
            or str(int(k)) != k or not 1 <= int(k) <= 65535 for k in original_classes)
            or any(type(v) is not int or v not in (0, 1) for v in original_classes.values())):
        raise ValueError('classes must exactly match PNG positive IDs with canonical str keys and integer0/1')
    if not isinstance(candidates, (list, tuple)):
        raise ValueError('candidates must be a fixed list/tuple, not a changing stream')
    before_hash = _sha(source)
    before_summary, areas = _summary(source, original_classes)
    records = [_candidate(value, i) for i, value in enumerate(candidates)]
    occurrences = Counter(tuple(row['pair_ids']) for row in records if row['pair_ids'] is not None)
    rejected, eligible = [], []
    for row in records:
        pair = row['pair_ids']
        reason = row.get('reason')
        if pair is not None and occurrences[tuple(pair)] > 1:
            reason = 'duplicate_pair_all_occurrences_rejected'
        if reason is None:
            a, b = pair
            if not areas[a] or not areas[b]:reason = 'unknown_instance_id'
            elif float(threshold) == 0:reason = 'threshold_disabled'
            elif original_classes[str(a)] != original_classes[str(b)]:reason = 'different_predicted_class'
            elif row['keep_score'] >= threshold:reason = 'score_not_below_threshold'
        if reason is not None:rejected.append(dict(row, reason=reason))
        else:eligible.append(row)
    adjacent, direct, seams = _geometry(source) if eligible else (set(), set(), {})
    selected, used = [], set()
    for row in sorted(eligible, key=lambda r: (r['keep_score'], *r['pair_ids'])):
        a, b = row['pair_ids'];key = _key(a, b)
        if key not in adjacent:reason = 'not_original_adjacent'
        elif key not in direct and key not in seams:reason = 'no_safe_binary_contact'
        elif a in used or b in used:reason = 'instance_already_used_in_single_pass'
        else:reason = None
        if reason is not None:
            rejected.append(dict(row, reason=reason));continue
        used.update((a, b))
        selected.append(dict(row, retained_id=a, removed_id=b, predicted_class=original_classes[str(a)],
            original_areas=[int(areas[a]), int(areas[b])], direct4_contact=key in direct,
            filled_seam_pixels=int(len(seams.get(key, [])))))
    lookup = np.arange(65536, dtype=np.uint16)
    output_classes = original_classes.copy()
    for row in selected:
        lookup[row['removed_id']] = row['retained_id'];del output_classes[str(row['removed_id'])]
    output = lookup[source]
    filled = np.zeros(source.shape, bool)
    for row in selected:
        key = _key(*row['pair_ids']);flat = seams.get(key, np.empty(0, np.int64))
        output.flat[flat] = row['retained_id'];filled.flat[flat] = True
    for row in selected:
        a, b = row['pair_ids'];yy, xx = np.nonzero((source == a) | (source == b))
        box = np.s_[int(yy.min()):int(yy.max())+1, int(xx.min()):int(xx.max())+1]
        old, new = source[box], output[box]
        old_counts = {str(v): _component_counts(old == v) for v in (a, b)}
        new_counts = _component_counts(new == a)
        if new_counts['4'] >= old_counts[str(a)]['4']+old_counts[str(b)]['4']:
            raise RuntimeError('Selected merge did not create actual A/B four-connected contact')
        row.update(components_before=old_counts, components_after=new_counts,
                   merged_area=int(np.count_nonzero(new == a)), actual_A_B_connection_created=True)
    after_summary, _ = _summary(output, output_classes)
    frame_zero_preserved = all(not np.any(new[old == 0]) for old, new in
        ((source[0], output[0]), (source[-1], output[-1]), (source[:, 0], output[:, 0]), (source[:, -1], output[:, -1])))
    if (output.dtype != np.uint16 or {str(int(v)) for v in np.unique(output) if v} != set(output_classes)
            or np.any(filled & (source != 0)) or np.any(output[(source == 0) & ~filled] != 0)
            or not np.array_equal(output[source > 0], lookup[source[source > 0]]) or not frame_zero_preserved or _sha(source) != before_hash):
        raise RuntimeError('Relation merge violated immutable source/zero-fill/ID-class contract')
    audit = dict(format=FORMAT, threshold=float(threshold), selection='score ascending, then canonical pair IDs; each original ID at most once',
        candidate_count=len(records), selected_count=len(selected), rejected_count=len(rejected), selected=selected,
        rejected=sorted(rejected, key=lambda r:r['candidate_index']), rejection_reasons=dict(Counter(r['reason']for r in rejected)),
        before_class_summary=before_summary, after_class_summary=after_summary,
        filled_seam_pixels=int(filled.sum()), changed_pixels=int(np.count_nonzero(output != source)),
        original_instances=len(original_classes), final_instances=len(output_classes),
        input_ID_sha256=before_hash, output_ID_sha256=_sha(output),
        component_counts_scope='selected original IDs only; untouched IDs retain their exact pixels',
        original_positive_pixels_only_ID_remapped=True, zero_fill_uses_original_exact_two_positive_4neighbors=True,
        original_outer_frame_zero_pixels_unchanged=frame_zero_preserved,
        no_chain_merging=True, no_model_forward=True, no_rescoring=True, no_WS=True, no_semantic_revote=True,
        no_compaction=True, classes_inherited_exact=True, source_input_unchanged=True,
        score_gain_or_accuracy_claimed=False, final_contract_valid=True)
    return output, output_classes, audit
