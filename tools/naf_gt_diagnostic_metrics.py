# -*- coding: utf-8 -*-
"""NAF新增线与处理后GT的CPU机制诊断，不是竞赛代理评分器。

来源分组只用于统计，不分别切断original/filled区域来制造连通差异。
GT可能漏标；本模块的“内部”与“切分”均指记录GT及其已知安全域。
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def _inputs(instance_map, original_covered, filled, residual_unknown, safe_margin):
    ids = np.asarray(instance_map)
    if ids.ndim != 2 or ids.dtype.kind not in 'iu' or ids.size == 0 or np.any(ids < 0):
        raise ValueError('Nonnegative integer HW instance map required')
    if type(safe_margin) is not int or safe_margin < 0:
        raise ValueError('safe_margin must be a nonnegative native pixel integer')
    arrays = [np.asarray(v) for v in (original_covered, filled, residual_unknown)]
    if any(v.shape != ids.shape or not np.isin(v, [0, 1]).all() for v in arrays):
        raise ValueError('Binary provenance masks must match instance map')
    original, fill, unknown = [v.astype(bool) for v in arrays]
    known = ids > 0
    if np.any(original & ~known) or not np.array_equal(fill, known & ~original) or not np.array_equal(unknown, ~known):
        raise ValueError('Processed GT provenance is inconsistent')
    size = 2 * safe_margin + 1
    safe = cv2.erode(known.astype(np.uint8), np.ones((size, size), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return ids, original, fill, unknown, safe


def _probability(a, shape, name):
    a = np.asarray(a)
    if a.shape != shape or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError(name + ' must be a finite aligned HW probability or binary mask')
    return a


def _distance(mask):
    # EDT对没有任何零点的数组不会返回inf，故空边界必须显式处理。
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return ndi.distance_transform_edt(~mask).astype(np.float32)


def gt_boundary_provenance(instance_map, original_covered):
    """已知不同实例的4邻界面，两侧落点均标记；允许同落点有多种邻接来源。"""
    ids = np.asarray(instance_map)
    original = np.asarray(original_covered, bool)
    oo = np.zeros(ids.shape, bool)
    involving_fill = np.zeros_like(oo)
    for axis in [0, 1]:
        lower = [slice(None), slice(None)]; upper = lower.copy()
        lower[axis] = slice(1, None); upper[axis] = slice(None, -1)
        a, b = tuple(lower), tuple(upper)
        different = (ids[a] > 0) & (ids[b] > 0) & (ids[a] != ids[b])
        raw = different & original[a] & original[b]
        filled = different & ~raw
        oo[a] |= raw; oo[b] |= raw
        involving_fill[a] |= filled; involving_fill[b] |= filled
    return oo, involving_fill


def added_boundary_diagnostic(input_skeleton, output_probability, instance_map,
                              original_covered, filled, residual_unknown, *,
                              threshold=.65, new_distance=2., boundary_tolerance=4.,
                              deep_distance=8., safe_margin=8):
    """返回(summary,masks)，按骨架长度统计新增响应，避免把正常线宽增加算作新线。

支持来源按容差邻域内可支持的GT邻接分OO-only/含filled-only/both，
避免最近边界等距时任意选择来源。near_gt仅代表与记录GT一致，不是物理真值。
"""
    ids, original, fill, unknown, safe = _inputs(
        instance_map, original_covered, filled, residual_unknown, safe_margin)
    if not 0 < threshold < 1 or not 0 <= new_distance or not 0 <= boundary_tolerance < deep_distance:
        raise ValueError('Invalid thresholds or distance ordering')
    inp = _probability(input_skeleton, ids.shape, 'input_skeleton') > .5
    output = _probability(output_probability, ids.shape, 'output_probability') >= threshold
    input_line = skeletonize(inp)
    output_line = skeletonize(output)
    distance_input = _distance(input_line)
    raw_added = output_line & (distance_input > new_distance)
    added = raw_added & safe
    oo, has_fill = gt_boundary_provenance(ids, original)
    distance_gt = _distance(oo | has_fill)
    near = added & (distance_gt <= boundary_tolerance)
    deep = added & (distance_gt > deep_distance)
    buffer = added & ~near & ~deep
    near_oo = near & (_distance(oo) <= boundary_tolerance)
    near_fill = near & (_distance(has_fill) <= boundary_tolerance)
    groups = {}
    for name, domain in [('all_known_safe', safe), ('original_covered', safe & original), ('filled', safe & fill)]:
        groups[name] = dict(evaluation_pixels=int(domain.sum()),
            added_skeleton_pixels=int((added & domain).sum()),
            near_recorded_boundary_pixels=int((near & domain).sum()),
            intermediate_pixels=int((buffer & domain).sum()),
            deep_recorded_interior_pixels=int((deep & domain).sum()),
            supported_oo_only=int((near_oo & ~near_fill & domain).sum()),
            supported_with_fill_only=int((near_fill & ~near_oo & domain).sum()),
            supported_both_origins=int((near_fill & near_oo & domain).sum()))
        assert groups[name]['added_skeleton_pixels'] == sum(groups[name][k] for k in [
            'near_recorded_boundary_pixels', 'intermediate_pixels', 'deep_recorded_interior_pixels'])
    summary = dict(format='naf_added_boundary_diagnostic_v1', threshold=float(threshold),
        new_distance=float(new_distance), boundary_tolerance=float(boundary_tolerance),
        deep_distance=float(deep_distance), safe_margin=safe_margin,
        input_skeleton_pixels=int(input_line.sum()), output_skeleton_pixels=int(output_line.sum()),
        raw_added_pixels=int(raw_added.sum()), ignored_unknown_or_margin_pixels=int((raw_added & ~safe).sum()),
        groups=groups, support_origin_policy='within_tolerance_OO_only_with_fill_only_or_both',
        interpretation='Agreement with recorded processed GT, not independent truth or official score.')
    masks = dict(added=added, raw_added=raw_added, near_gt=near, deep_interior=deep,
                 buffer=buffer, valid=safe, input_skeleton=input_line, output_skeleton=output_line,
                 gt_boundary=oo | has_fill, gt_boundary_oo=oo, gt_boundary_with_fill=has_fill)
    return summary, masks


def known_domain_splits(before_barrier, after_barrier, instance_map, original_covered,
                        filled, residual_unknown, *, safe_margin=8,
                        min_piece_pixels=50, min_piece_fraction=.05, deep_distance=8.,
                        new_distance=2.):
    """比较每个GT内同一旧开放分量是否被新增屏障切开；全部子块与显著子块均保留。

actual使用真实after屏障，addition_only用before|after以隔离新增屏障的作用。
所有连通只在整个GT安全已知域计算；original/filled只统计子块来源，禁止先按来源切域。
unknown附近不作“全晶粒已切开”的推断；输出只称known-domain separation。
    原substantial_deep字段仅表共现；因果字段另通过移除新增深部屏障后是否重新连通核验。
"""
    ids, original, fill, unknown, safe = _inputs(
        instance_map, original_covered, filled, residual_unknown, safe_margin)
    if type(min_piece_pixels) is not int or min_piece_pixels < 1 or not 0 <= min_piece_fraction < .5:
        raise ValueError('Invalid diagnostic piece thresholds')
    before = _probability(before_barrier, ids.shape, 'before_barrier') > .5
    after = _probability(after_barrier, ids.shape, 'after_barrier') > .5
    oo, has_fill = gt_boundary_provenance(ids, original)
    far_gt = _distance(oo | has_fill) > deep_distance
    far_before = _distance(skeletonize(before)) > new_distance
    newly_deep = after & ~before & safe & far_gt & far_before
    boxes = ndi.find_objects(ids.astype(np.int32, copy=False))
    output = {}
    for mode, barrier in [('actual', after), ('addition_only', before | after)]:
        cases = []
        for gt, box in enumerate(boxes, 1):
            if box is None:
                continue
            domain = (ids[box] == gt) & safe[box]
            if not domain.any():
                continue
            old, old_count = ndi.label(domain & ~before[box], structure=np.ones((3, 3), bool))
            new, new_count = ndi.label(domain & ~barrier[box], structure=np.ones((3, 3), bool))
            if old_count == 0 or new_count < 2:
                continue
            counts = np.bincount(old.ravel(), minlength=old_count + 1)
            code = old.astype(np.int64) * (new_count + 1) + new
            table = np.bincount(code.ravel(), minlength=(old_count + 1) * (new_count + 1)).reshape(old_count + 1, new_count + 1)
            deep_counts = np.bincount(old[newly_deep[box]], minlength=old_count + 1)
            reverted = None
            for parent in range(1, old_count + 1):
                children = np.flatnonzero(table[parent, 1:] > 0) + 1
                if len(children) < 2:
                    continue
                pieces = []
                for child in children:
                    region = (old == parent) & (new == child)
                    pixels = int(table[parent, child])
                    substantial = pixels >= min_piece_pixels and pixels >= min_piece_fraction * counts[parent]
                    pieces.append(dict(component=int(child), pixels=pixels,
                        original_pixels=int((region & original[box]).sum()),
                        filled_pixels=int((region & fill[box]).sum()), substantial=bool(substantial)))
                pieces.sort(key=lambda r: (-r['pixels'], r['component']))
                substantial_pieces = [r for r in pieces if r['substantial']]
                strong = len(substantial_pieces) >= 2 and deep_counts[parent] > 0
                supported = [r for r in substantial_pieces if r['original_pixels'] >= min_piece_pixels]
                provenance = ('two_original_supported_children' if len(supported) >= 2 else
                              'one_original_supported_child' if len(supported) == 1 else
                              'filled_supported_children')
                # 仅删新增深部屏障，不删原界/近GT带，验证深部新增与切分不是偶然共现。
                # after自由域是reverted自由域的子集，每个after子块必须整体落在一个reverted分量。
                reverted_components = []
                if strong:
                    if reverted is None:
                        reverted_barrier = barrier[box] & ~newly_deep[box]
                        reverted = ndi.label(domain & ~reverted_barrier, structure=np.ones((3, 3), bool))[0]
                    for piece in substantial_pieces:
                        region = (old == parent) & (new == piece['component'])
                        labels = np.unique(reverted[region])
                        labels = labels[labels > 0]
                        if len(labels) != 1:
                            raise RuntimeError('Removing a barrier unexpectedly disconnected an existing child')
                        reverted_components.append(int(labels[0]))
                contributes = strong and len(set(reverted_components)) < len(substantial_pieces)
                required = strong and len(set(reverted_components)) == 1
                cases.append(dict(gt=int(gt), parent_component=int(parent), parent_pixels=int(counts[parent]),
                    bbox_xyxy=[int(box[1].start), int(box[0].start), int(box[1].stop), int(box[0].stop)],
                    after_children=len(pieces), substantial_children=len(substantial_pieces),
                    all_children=pieces, newly_deep_barrier_pixels=int(deep_counts[parent]),
                    substantial_deep_known_domain_split=bool(strong),
                    substantial_split_with_deep_additions=bool(strong),
                    deep_addition_contributes_to_split=bool(contributes),
                    deep_addition_causally_required_for_split=bool(required),
                    reverted_components_of_substantial_children=reverted_components,
                    provenance=provenance,
                    secondary_pixels=sum(r['pixels'] for r in pieces[1:])))
        strong = [r for r in cases if r['substantial_deep_known_domain_split']]
        causal = [r for r in cases if r['deep_addition_causally_required_for_split']]
        output[mode] = dict(parent_components_with_multiple_children=len(cases),
            substantial_deep_split_parents=len(strong),
            substantial_deep_split_gt_instances=len({r['gt'] for r in strong}),
            original_supported_split_parents=sum(r['provenance'] == 'two_original_supported_children' for r in strong),
            deep_addition_contributes_parents=sum(r['deep_addition_contributes_to_split'] for r in cases),
            deep_addition_causally_required_parents=len(causal),
            deep_addition_causally_required_gt_instances=len({r['gt'] for r in causal}),
            original_supported_deep_causal_split_parents=sum(r['provenance'] == 'two_original_supported_children' for r in causal),
            cases=cases)
    output.update(format='naf_known_domain_splits_v1', safe_margin=safe_margin,
        min_piece_pixels=min_piece_pixels, min_piece_fraction=min_piece_fraction,
        deep_distance=deep_distance, new_distance=new_distance,
        interpretation='GT-known-domain connectivity diagnostic; provenance holes do not define separate domains. Deep cooccurrence differs from causal barrier-removal checks. All pieces retained; no prediction filtering or deployment rollback; not whole-grain physical correctness.')
    return output
