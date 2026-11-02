# -*- coding: utf-8 -*-
"""原生全图的异实例内核分隔诊断；没有训练、模型或部署后处理。

原始覆盖只提供可信内核，processed GT 提供待审查的界面位置。filled 不
生成内核，未知不生成界面。输出是相对既有标注的关系诊断，不是准确率。
所有 radius/area 均用原图像素；每个实例的形态学操作只在其 bbox 内完成。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import combinations
from numbers import Integral

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _labels(value, name, shape=None):
    value = np.asarray(value)
    if (value.ndim != 2 or not value.size or value.dtype == np.bool_
            or value.dtype.kind not in 'iu' or np.any(value < 0)):
        raise ValueError(f'{name} must be a nonempty nonnegative integer HW map')
    if shape is not None and value.shape != tuple(shape):
        raise ValueError(f'{name} shape differs from the original native image')
    if int(value.max()) > np.iinfo(np.int32).max:
        raise ValueError(f'{name} IDs exceed int32')
    return value


def _gt_original(gt, original_covered):
    gt = _labels(gt, 'GT')
    if int(gt.max()) > 65535:
        raise ValueError('GT IDs must fit the project uint16 instance contract')
    original = np.asarray(original_covered)
    if original.shape != gt.shape or not np.isin(original, (0, 1)).all():
        raise ValueError('original_covered must be a matching binary mask')
    original = original.astype(bool)
    if np.any(original & (gt == 0)):
        raise ValueError('Original coverage cannot mark unknown GT=0 as known')
    return gt, original


def _disk(radius):
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    return (xx*xx + yy*yy <= radius*radius).astype(np.uint8)


def _sha(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode('ascii'))
    h.update(str(array.shape).encode('ascii'))
    h.update(array.tobytes())
    return h.hexdigest()


def _box_list(box):
    return [int(box[0].start), int(box[1].start), int(box[0].stop), int(box[1].stop)]


def build_trusted_cores(gt, original_covered, *, radius=4, min_component_pixels=64):
    """逐实例侵蚀 ``original_covered & (GT==id)``，保留全部合格8连通组件。

    返回 core_map（值仍是 GT ID）、独立 component_map、每组件 bbox、每GT
    统计及身份。原标孔洞、未知和填补区不会被联合域侵蚀偷偷桥接。
    """
    gt, original = _gt_original(gt, original_covered)
    radius = _integer(radius, 'radius')
    minimum = _integer(min_component_pixels, 'min_component_pixels', 1)
    kernel = _disk(radius)
    core = np.zeros(gt.shape, np.int32)
    component_map = np.zeros(gt.shape, np.int32)
    components, rows = [], []
    # 一次扫描建立所有实例 bbox，避免每个ID重复扫描完整原图。
    boxes = find_objects(gt)
    next_component = 0
    for gid, box in enumerate(boxes, 1):
        if box is None:
            continue
        local_gt = gt[box]
        source = original[box] & (local_gt == gid)
        eroded = cv2.erode(source.astype(np.uint8), kernel,
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        local_components, count = label(eroded, structure=np.ones((3, 3), np.uint8))
        sizes = np.bincount(local_components.ravel(), minlength=count+1)
        keep = [cid for cid in range(1, count+1) if sizes[cid] >= minimum]
        core_view, component_view = core[box], component_map[box]
        component_boxes = find_objects(local_components)
        ids = []
        for cid in keep:
            next_component += 1
            selected = local_components == cid
            core_view[selected] = gid
            component_view[selected] = next_component
            cb = component_boxes[cid-1]
            bbox = [box[0].start+cb[0].start, box[1].start+cb[1].start,
                    box[0].start+cb[0].stop, box[1].start+cb[1].stop]
            components.append(dict(gt_id=gid, component_id=next_component,
                                   pixels=int(sizes[cid]), bbox_yxyx=list(map(int, bbox))))
            ids.append(next_component)
        kept_pixels = int(sum(sizes[cid] for cid in keep))
        rows.append(dict(gt_id=gid, gt_pixels=int((local_gt == gid).sum()),
                         original_pixels=int(source.sum()), eroded_pixels=int(eroded.sum()),
                         core_pixels=kept_pixels, raw_component_count=int(count),
                         retained_component_count=len(keep), removed_component_count=int(count-len(keep)),
                         removed_small_pixels=int(eroded.sum())-kept_pixels,
                         component_ids=ids, bbox_yxyx=_box_list(box)))
    assert not np.any((core > 0) & ~original)
    assert np.array_equal(core[core > 0], gt[core > 0])
    summary = dict(format='boundary_trusted_cores_v1', native_shape=list(gt.shape),
                   radius=radius, erosion='Euclidean disk; constant-zero outside instance bbox',
                   min_component_pixels=minimum, connectivity=8, gt_count=len(rows),
                   gt_with_cores=sum(r['core_pixels'] > 0 for r in rows),
                   core_pixels=int((core > 0).sum()), retained_components=len(components),
                   filled_used_for_cores=False, unknown_used=False,
                   gt_sha256=_sha(gt), original_covered_sha256=_sha(original),
                   core_map_sha256=_sha(core), component_map_sha256=_sha(component_map))
    return dict(core_map=core, component_map=component_map, components=components,
                gt_stats=rows, summary=summary)


def adjacent_instance_pairs(gt):
    """全部4邻接、双方known且不同ID的无向邻接对；不跨未知或斜角补邻接。"""
    gt = _labels(gt, 'GT')
    if int(gt.max()) > 65535:
        raise ValueError('GT IDs must fit the project uint16 instance contract')
    pieces = []
    for a, b in ((gt[1:], gt[:-1]), (gt[:, 1:], gt[:, :-1])):
        valid = (a > 0) & (b > 0) & (a != b)
        if valid.any():
            pieces.append(np.sort(np.stack([a[valid], b[valid]], axis=1), axis=1))
    if not pieces:
        return []
    return [tuple(map(int, row)) for row in np.unique(np.concatenate(pieces), axis=0)]


def _distribution(counts, total):
    counts = {int(k):int(v) for k, v in counts.items() if v > 0}
    counts.setdefault(0, 0)
    positive = {k:v for k, v in counts.items() if k > 0}
    dominant = min(positive, key=lambda k:(-positive[k], k)) if positive else None
    covered = int(sum(positive.values()))
    return dict(core_pixels=int(total), marker_counts={str(k):counts[k] for k in sorted(counts)},
                zero_pixels=counts[0], positive_pixels=covered,
                coverage=covered/total if total else None,
                positive_marker_ids=sorted(positive), positive_marker_count=len(positive),
                dominant_marker=dominant, dominant_pixels=positive.get(dominant, 0),
                dominant_fraction=positive.get(dominant, 0)/total if total else None)


def core_marker_statistics(cores, marker_labels):
    """每个GT的所有保留内核对整图marker的贡献；0明确保留在分母中。

    不在原标或GT并域裁断marker图。仅对已生成整图marker取值，因而不把
    经过未知区的绕行误认成已分隔；绕行来源需由runner另行标记。
    """
    core = _labels(cores['core_map'], 'core_map')
    cm = _labels(cores['component_map'], 'component_map', core.shape)
    markers = _labels(marker_labels, 'marker_labels', core.shape)
    if not np.array_equal(core > 0, cm > 0):
        raise ValueError('core_map and component_map support differs')
    if (_sha(core) != cores['summary']['core_map_sha256']
            or _sha(cm) != cores['summary']['component_map_sha256']):
        raise ValueError('Trusted core maps changed after construction')
    by_gt = {int(r['gt_id']):Counter() for r in cores['gt_stats']}
    per_gt_components = defaultdict(list)
    for item in cores['components']:
        y0, x0, y1, x1 = item['bbox_yxyx']
        box = np.s_[y0:y1, x0:x1]
        selected = cm[box] == item['component_id']
        if int(selected.sum()) != item['pixels'] or np.any(core[box][selected] != item['gt_id']):
            raise ValueError('Component metadata and core identity disagree')
        ids, counts = np.unique(markers[box][selected], return_counts=True)
        count_map = dict(zip(map(int, ids), map(int, counts)))
        by_gt[item['gt_id']].update(count_map)
        per_gt_components[item['gt_id']].append({**item, **_distribution(count_map, item['pixels'])})
    rows = {}
    for row in cores['gt_stats']:
        gid, total = int(row['gt_id']), int(row['core_pixels'])
        if sum(by_gt[gid].values()) != total:
            raise ValueError('Component contributions do not cover all retained core pixels')
        rows[str(gid)] = dict(gt_id=gid, **_distribution(by_gt[gid], total),
                             retained_component_count=row['retained_component_count'],
                             components=per_gt_components[gid])
    total = int(sum(r['core_pixels'] for r in rows.values()))
    zero = int(sum(r['zero_pixels'] for r in rows.values()))
    return dict(format='boundary_core_marker_statistics_v1',
                core_identity={k:cores['summary'][k] for k in
                               ('native_shape', 'core_map_sha256', 'component_map_sha256')},
                marker_sha256=_sha(markers), gt=rows,
                summary=dict(gt_count=len(rows), gt_with_cores=sum(v['core_pixels'] > 0 for v in rows.values()),
                             core_pixels=total, zero_pixels=zero, positive_pixels=total-zero,
                             coverage=(total-zero)/total if total else None,
                             all_components_retained=True, zero_is_not_separation=True))


def _pair(pair):
    if not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2:
        raise ValueError('pair must contain two different positive GT IDs')
    a, b = (_integer(x, 'GT ID', 1) for x in pair)
    if a == b:
        raise ValueError('pair must contain two different GT IDs')
    return tuple(sorted((a, b)))


def _fraction(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be a finite fraction in [0,1]')
    return float(value)


def pair_core_relation(statistics, pair, *, min_shared_pixels=32, min_shared_fraction=.1, dominance=.9):
    """共享像素为同marker内双方core像素之和，不是两张core的空间交集。"""
    a, b = _pair(pair)
    minimum = _integer(min_shared_pixels, 'min_shared_pixels', 1)
    fraction = _fraction(min_shared_fraction, 'min_shared_fraction')
    dominance = _fraction(dominance, 'dominance')
    if str(a) not in statistics['gt'] or str(b) not in statistics['gt']:
        raise ValueError('Requested GT pair is not in the core cohort')
    left, right = statistics['gt'][str(a)], statistics['gt'][str(b)]
    common = sorted(set(left['positive_marker_ids']) & set(right['positive_marker_ids']))
    shared = []
    for marker in common:
        ac, bc = left['marker_counts'][str(marker)], right['marker_counts'][str(marker)]
        af, bf = ac/left['core_pixels'], bc/right['core_pixels']
        shared.append(dict(marker_id=marker, a_pixels=ac, b_pixels=bc, a_fraction=af, b_fraction=bf,
                           significant=ac >= minimum and bc >= minimum and af >= fraction and bf >= fraction))
    significant = [v for v in shared if v['significant']]
    def aggregate(values):
        return dict(marker_count=len(values), marker_ids=[v['marker_id'] for v in values],
                    a_pixels=sum(v['a_pixels'] for v in values), b_pixels=sum(v['b_pixels'] for v in values),
                    shared_core_pixels=sum(v['a_pixels']+v['b_pixels'] for v in values))
    same_dominant = bool(left['dominant_marker'] is not None
                         and left['dominant_marker'] == right['dominant_marker']
                         and left['dominant_fraction'] >= dominance and right['dominant_fraction'] >= dominance)
    return dict(pair=[a, b], a_core_pixels=left['core_pixels'], b_core_pixels=right['core_pixels'],
                a_zero_pixels=left['zero_pixels'], b_zero_pixels=right['zero_pixels'],
                a_coverage=left['coverage'], b_coverage=right['coverage'],
                eligible_both_have_cores=left['core_pixels'] > 0 and right['core_pixels'] > 0,
                both_have_positive_markers=bool(left['positive_marker_ids'] and right['positive_marker_ids']),
                all=aggregate(shared), significant=aggregate(significant), shared_markers=shared,
                all_positive_labels_disjoint=not common, dominant_same_at_least_threshold=same_dominant,
                thresholds=dict(min_shared_pixels=minimum, min_shared_fraction=fraction, dominance=dominance),
                shared_pixels_definition='sum of core pixels from both GTs assigned to shared positive markers; not spatial intersection')


def candidate_interface(gt, original_covered, pair, *, support_policy='original', radius=2, thin_radius=4):
    """按双端支持筛完整4邻接界面，膨胀后仍限制在目标两ID及支持域。

    thin4 对每个ID单独计算到其自身原标的距离，不向其它ID借用支持。
    processed_known_optimistic 是机制参照，明确不具备训练可靠性结论。
    """
    gt, original = _gt_original(gt, original_covered)
    pair = _pair(pair)
    radius = _integer(radius, 'radius')
    thin_radius = _integer(thin_radius, 'thin_radius', 1)
    if support_policy not in ('original', 'thin4', 'processed_known_optimistic'):
        raise ValueError('Unknown interface support policy')
    # 此策略名绑定4个原生像素；拒绝静默改变含义。
    if support_policy == 'thin4' and thin_radius != 4:
        raise ValueError('thin4 requires exactly four native pixels')
    locations = np.argwhere((gt == pair[0]) | (gt == pair[1]))
    if not len(locations) or any(not np.any(gt == gid) for gid in pair):
        raise ValueError('Both pair IDs must occur in GT')
    y0, x0 = locations.min(axis=0);y1, x1 = locations.max(axis=0)+1
    box = np.s_[y0:y1, x0:x1]
    local_gt, local_original = gt[box], original[box]
    union = (local_gt == pair[0]) | (local_gt == pair[1])
    supported = local_original & union
    if support_policy == 'processed_known_optimistic':
        supported = union.copy()
    elif support_policy == 'thin4':
        # pair bbox可能较大；每个ID的EDT仍进一步缩到该实例自身bbox。
        for gid in pair:
            positions = np.argwhere(local_gt == gid)
            ay, ax = positions.min(axis=0);by, bx = positions.max(axis=0)+1
            sub = np.s_[ay:by, ax:bx]
            own = (local_gt[sub] == gid) & local_original[sub]
            if own.any():
                distance = distance_transform_edt(~own)
                supported[sub] |= (local_gt[sub] == gid) & (distance <= thin_radius)
    full_contacts = np.zeros(local_gt.shape, bool)
    contacts = np.zeros(local_gt.shape, bool)
    full_count, supported_count = 0, 0
    for left, right in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        a, b = local_gt[left], local_gt[right]
        edge = ((a == pair[0]) & (b == pair[1])) | ((a == pair[1]) & (b == pair[0]))
        selected = edge & supported[left] & supported[right]
        full_contacts[left] |= edge;full_contacts[right] |= edge
        contacts[left] |= selected;contacts[right] |= selected
        full_count += int(edge.sum());supported_count += int(selected.sum())
    expanded = cv2.dilate(contacts.astype(np.uint8), _disk(radius),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    selected = expanded & union & supported
    arrays = {}
    for key, value in [('mask', selected), ('contact_mask', contacts),
                       ('full_contact_mask', full_contacts), ('support_mask', supported)]:
        arrays[key] = np.zeros(gt.shape, bool);arrays[key][box] = value
    assert not np.any(arrays['mask'] & (gt == 0))
    provenance = dict(mask_pixels=int(selected.sum()), original_pixels=int((selected & local_original).sum()),
                      filled_pixels=int((selected & ~local_original).sum()), unknown_pixels=0,
                      supported_original_pixels=int((supported & local_original).sum()),
                      supported_filled_pixels=int((supported & ~local_original).sum()),
                      contact_original_centers=int((contacts & local_original).sum()),
                      contact_filled_centers=int((contacts & ~local_original).sum()))
    summary = dict(format='boundary_core_interface_v1', pair=list(pair), support_policy=support_policy,
                   radius_native=radius, thin_radius_native=thin_radius,
                   bbox_yxyx=list(map(int, [y0, x0, y1, x1])), full_contact_edges=full_count,
                   supported_contact_edges=supported_count, excluded_contact_edges=full_count-supported_count,
                   full_interface_supported=bool(full_count and supported_count == full_count),
                   write_support='both endpoints supported before dilation; union and support after dilation',
                   filled_semantics='derived processed-GT support, not new human truth',
                   optimistic_only=support_policy == 'processed_known_optimistic',
                   reliable_for_training=False, no_training=True,
                   mask_sha256=_sha(arrays['mask']))
    return dict(**arrays, provenance=provenance, summary=summary)


def _shared_pair_set(statistics):
    memberships = defaultdict(list)
    for key, row in statistics['gt'].items():
        for marker in row['positive_marker_ids']:
            memberships[marker].append(int(key))
    return {pair for ids in memberships.values() for pair in combinations(sorted(ids), 2)}


def compare_core_separation(before_stats, after_stats, pair, *, min_coverage=.9):
    """比较固定全部core，不删失败分母；两个种子点分开不等于两组core分开。"""
    coverage = _fraction(min_coverage, 'min_coverage')
    if (before_stats['core_identity'] != after_stats['core_identity']
            or set(before_stats['gt']) != set(after_stats['gt'])):
        raise ValueError('Before/after must use exactly the same trusted core cohort')
    before_pair = pair_core_relation(before_stats, pair)
    after_pair = pair_core_relation(after_stats, pair)
    eligible = after_pair['eligible_both_have_cores']
    covered = bool(eligible and after_pair['a_coverage'] >= coverage and after_pair['b_coverage'] >= coverage)
    separated = bool(covered and after_pair['both_have_positive_markers'] and after_pair['all_positive_labels_disjoint'])
    gt_changes, component_changes = [], []
    for key in sorted(before_stats['gt'], key=int):
        old, new = before_stats['gt'][key], after_stats['gt'][key]
        if old['core_pixels'] != new['core_pixels']:
            raise ValueError('GT core denominator changed')
        gt_changes.append(dict(gt_id=int(key), core_pixels=old['core_pixels'],
                               before_coverage=old['coverage'], after_coverage=new['coverage'],
                               before_zero_pixels=old['zero_pixels'], after_zero_pixels=new['zero_pixels'],
                               zero_pixel_delta=new['zero_pixels']-old['zero_pixels'],
                               before_positive_markers=old['positive_marker_count'], after_positive_markers=new['positive_marker_count'],
                               before_dominant_fraction=old['dominant_fraction'], after_dominant_fraction=new['dominant_fraction']))
        old_components = {c['component_id']:c for c in old['components']}
        new_components = {c['component_id']:c for c in new['components']}
        if set(old_components) != set(new_components):
            raise ValueError('Retained component cohort changed')
        for cid, a in old_components.items():
            b = new_components[cid]
            if a['pixels'] != b['pixels']:
                raise ValueError('Component pixel denominator changed')
            component_changes.append(dict(gt_id=int(key), component_id=cid, pixels=a['pixels'],
                before_positive_markers=a['positive_marker_count'], after_positive_markers=b['positive_marker_count'],
                before_zero_pixels=a['zero_pixels'], after_zero_pixels=b['zero_pixels'],
                zero_pixel_delta=b['zero_pixels']-a['zero_pixels'],
                before_dominant_fraction=a['dominant_fraction'], after_dominant_fraction=b['dominant_fraction'],
                newly_split=a['positive_marker_count'] == 1 and b['positive_marker_count'] > 1,
                split_count_increased=b['positive_marker_count'] > a['positive_marker_count'],
                newly_unseeded=a['positive_pixels'] > 0 and b['positive_pixels'] == 0,
                before_unseeded=a['positive_pixels'] == 0, after_unseeded=b['positive_pixels'] == 0))
    old_shared, new_shared = _shared_pair_set(before_stats), _shared_pair_set(after_stats)
    return dict(format='boundary_core_separation_comparison_v1', pair=after_pair['pair'],
                target_relation=dict(before=before_pair, after=after_pair),
                min_coverage=coverage, target_coverage_pass=covered,
                target_separated_and_covered=separated,
                target_safety_is_separate_from_separation=True,
                all_gt_changes=gt_changes, component_changes=component_changes,
                new_cross_gt_shared_pairs=[list(p) for p in sorted(new_shared-old_shared)],
                removed_cross_gt_shared_pairs=[list(p) for p in sorted(old_shared-new_shared)],
                summary=dict(core_gt_count=len(gt_changes), retained_components=len(component_changes),
                             newly_split_components=sum(c['newly_split'] for c in component_changes),
                             split_count_increased_components=sum(c['split_count_increased'] for c in component_changes),
                             newly_unseeded_components=sum(c['newly_unseeded'] for c in component_changes),
                             core_components_with_more_zero=sum(c['zero_pixel_delta'] > 0 for c in component_changes),
                             zero_pixel_delta=sum(c['zero_pixel_delta'] for c in component_changes),
                             new_shared_gt_pairs=len(new_shared-old_shared)),
                limitations=['Core agreement is relative to processed GT identity, not material truth or accuracy.',
                             'A target separation pass does not imply absence of new fragments or neighbor harm.',
                             'Marker labels must come from the full native image; ROI connectivity is not substituted.'])


def compare_marker_atoms(cores, before_markers, after_markers):
    """逐像素追踪旧正marker原子，不用显著比例阈值吞掉微小副作用。

    atom=(GT ID, 保留core组件ID, 旧正marker ID)。零像素不生成旧atom，
    但每个旧atom被新0吞掉的像素必须报告。GT不同且旧marker不同的两个
    atom若共享新正marker，才记新增跨GT合并；旧共有marker不算新增。
    """
    core = _labels(cores['core_map'], 'core_map')
    cm = _labels(cores['component_map'], 'component_map', core.shape)
    before = _labels(before_markers, 'before_markers', core.shape)
    after = _labels(after_markers, 'after_markers', core.shape)
    if (_sha(core) != cores['summary']['core_map_sha256']
            or _sha(cm) != cores['summary']['component_map_sha256']):
        raise ValueError('Trusted core maps changed after construction')
    atoms, membership = [], defaultdict(list)
    old_zero_pixels, old_zero_now_positive = 0, 0
    for component in cores['components']:
        y0, x0, y1, x1 = component['bbox_yxyx']
        box = np.s_[y0:y1, x0:x1]
        selected = cm[box] == component['component_id']
        old, new = before[box][selected], after[box][selected]
        if len(old) != component['pixels']:
            raise ValueError('Atom component pixel denominator changed')
        old_zero_pixels += int((old == 0).sum())
        old_zero_now_positive += int(((old == 0) & (new > 0)).sum())
        # 每组件只排序一次二维关系，再逐旧marker归并；不逐atom扫描全图。
        relations, counts = np.unique(np.stack([old, new], axis=1), axis=0, return_counts=True)
        by_old = defaultdict(Counter)
        for (old_id, new_id), count in zip(relations, counts):
            if old_id > 0:
                by_old[int(old_id)][int(new_id)] += int(count)
        for old_id in sorted(by_old):
            count_map = by_old[old_id]
            total = int(sum(count_map.values()))
            distribution = _distribution(count_map, total)
            aid = len(atoms)+1
            atom = dict(atom_id=aid, gt_id=component['gt_id'], component_id=component['component_id'],
                        old_marker_id=old_id, pixels=total, **distribution,
                        split=distribution['positive_marker_count'] > 1,
                        nonmajor_positive_pixels=distribution['positive_pixels']-distribution['dominant_pixels'])
            atoms.append(atom)
            for marker in distribution['positive_marker_ids']:
                membership[marker].append(aid)
    new_pairs = defaultdict(list)
    for marker, aids in membership.items():
        for a, b in combinations(aids, 2):
            left, right = atoms[a-1], atoms[b-1]
            if left['gt_id'] != right['gt_id'] and left['old_marker_id'] != right['old_marker_id']:
                new_pairs[(a, b)].append(marker)
    merged = []
    for (a, b), shared in sorted(new_pairs.items()):
        left, right = atoms[a-1], atoms[b-1]
        merged.append(dict(a_atom_id=a, b_atom_id=b, a_gt_id=left['gt_id'], b_gt_id=right['gt_id'],
                           a_old_marker=left['old_marker_id'], b_old_marker=right['old_marker_id'],
                           new_marker_ids=sorted(shared),
                           a_shared_pixels=sum(left['marker_counts'][str(v)] for v in shared),
                           b_shared_pixels=sum(right['marker_counts'][str(v)] for v in shared)))
    splits = sum(a['split'] for a in atoms)
    lost = sum(a['zero_pixels'] for a in atoms)
    return dict(format='boundary_core_marker_atoms_v1',
                core_identity={k:cores['summary'][k] for k in ('native_shape', 'core_map_sha256', 'component_map_sha256')},
                before_marker_sha256=_sha(before), after_marker_sha256=_sha(after),
                atom_definition='GT ID / retained core component / previous positive marker',
                atoms=atoms, new_merged_atom_pairs=merged,
                summary=dict(atom_count=len(atoms), old_positive_pixels=sum(a['pixels'] for a in atoms),
                             atom_splits=splits, zero_pixels_from_old_positive=lost,
                             nonmajor_pixels=sum(a['nonmajor_positive_pixels'] for a in atoms),
                             new_merged_atom_pairs=len(merged), old_zero_core_pixels=old_zero_pixels,
                             old_zero_now_positive_pixels=old_zero_now_positive,
                             atom_partition_preserved=not splits and not lost and not merged),
                no_significance_threshold=True,
                limitation='Atom preservation is a conservative stability gate, not proof the old markers are correct.')
