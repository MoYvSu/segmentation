# -*- coding: utf-8 -*-
"""正式原生图的铁素体 CPU 诊断核心；不是赛题打分器，也不改变预测。

输入必须来自允许的有标签训练源的完整 processed newGT（含 filled）。
一次交叉计数表派生匹配和贡献；不为每个实例反复扫描整张原图。
相同 GT 被多个预测覆盖只描述为贡献关系，不能一律称为内部误切。
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


FORMAT = 'ferrite_native_diagnostic_v1'
RULES = dict(minimum_iou=.5, significant_pixels=50, significant_fraction=.1,
             near_boundary_radius_native=4., boundary_offset_near_fraction=.8,
             gt_contact_connectivity=4, unknown_censor_connectivity=8,
             ff_merge_requires_both_gt_and_pred_significance=True,
             selection='safe whole objects only; strict-contained before same-parent; then ascending GT/pred IDs')
KINDS = ('ff_merge', 'internal_split', 'neighbor_leakage', 'fp_mix',
         'vote_error', 'boundary_offset', 'safe_area_bias')


def _labels(value, name):
    value = np.asarray(value)
    if value.ndim != 2 or min(value.shape, default=0) <= 0:
        raise ValueError(name + ' must be a nonempty two-dimensional grid')
    if not np.issubdtype(value.dtype, np.integer) or value.dtype == np.bool_:
        raise ValueError(name + ' must contain integer instance IDs')
    if value.min() < 0 or value.max() > 65535:
        raise ValueError(name + ' IDs must stay in 0..65535')
    return value


def _classes(mapping, labels, name):
    if not hasattr(mapping, 'items'):
        raise ValueError(name + ' must be an ID -> 0/1 mapping')
    result = {}
    for key, phase in mapping.items():
        try:
            value = int(key)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name + ' has invalid ID') from exc
        if isinstance(key, (float, np.floating)) and key != value:
            raise ValueError(name + ' has noninteger ID')
        if value <= 0 or value > 65535 or value in result:
            raise ValueError(name + ' has invalid or duplicate normalized ID')
        if isinstance(phase, (bool, np.bool_)) or not isinstance(phase, (int, np.integer)) or int(phase) not in (0, 1):
            raise ValueError(name + ' values must be integers 0=P or 1=F')
        result[value] = int(phase)
    actual = {int(v) for v in np.unique(labels) if v > 0}
    if set(result) != actual:
        raise ValueError(name + ' must exactly cover all nonzero instance IDs')
    return result


def _unsafe_ids(labels, known):
    safe = cv2.erode(known.astype(np.uint8), np.ones((3, 3), np.uint8),
                    borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return {int(v) for v in np.unique(labels[~safe]) if v > 0}


def _adjacency(gt, known):
    """实际四邻接触边的数量；不跨 unknown 推断两实例相邻。"""
    encoded = []
    for a, b, va, vb in ((gt[:, :-1], gt[:, 1:], known[:, :-1], known[:, 1:]),
                          (gt[:-1], gt[1:], known[:-1], known[1:])):
        keep = va & vb & (a != b)
        lo, hi = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
        encoded.append(lo.astype(np.int64) * 65536 + hi)
    values, counts = np.unique(np.concatenate(encoded), return_counts=True)
    return {(int(v // 65536), int(v % 65536)): int(n) for v, n in zip(values, counts)}


def _deep_mask(gt, known):
    """距离任何真实实例边/unknown 大于4原生像素；仅作空间位置描述。"""
    edge = np.zeros(gt.shape, bool)
    horizontal = gt[:, :-1] != gt[:, 1:]
    vertical = gt[:-1] != gt[1:]
    edge[:, :-1] |= horizontal; edge[:, 1:] |= horizontal
    edge[:-1] |= vertical; edge[1:] |= vertical
    interior = known & ~edge
    interior[[0, -1], :] = False; interior[:, [0, -1]] = False
    distance = cv2.distanceTransform(interior.astype(np.uint8), cv2.DIST_L2,
                                     cv2.DIST_MASK_PRECISE)
    return known & (distance > RULES['near_boundary_radius_native'])


def _pairs(g, p, matrix):
    """复现既有 class_independent_pairs：先合格匹配数，再IoU和，类别不入分配。"""
    ga, pa = matrix.sum(1), matrix.sum(0)
    indices = np.flatnonzero(p > 0)
    if not len(g) or not len(indices):
        return [], ga, pa
    intersection = matrix[:, indices]
    iou = intersection / np.maximum(1, ga[:, None] + pa[None, indices] - intersection)
    eligible = iou >= RULES['minimum_iou']
    score = eligible.astype(float) * (min(iou.shape) + 1) + np.where(eligible, iou, 0)
    left, right = linear_sum_assignment(-score)
    pairs = [dict(gt=int(g[i]), pred=int(p[indices[j]]), iou=float(iou[i, j]),
                  gt_known_pixels=int(ga[i]), pred_known_pixels=int(pa[indices[j]]))
             for i, j in zip(left, right) if eligible[i, j]]
    return pairs, ga, pa


def _area_summary(rows):
    if not rows:
        return dict(n=0, mean_absolute_relative_error=None,
                    median_signed_relative_error=None, paired_gt_pixels=0,
                    paired_pred_pixels=0, paired_signed_relative_error=None)
    errors = np.asarray([r['area_relative_error'] for r in rows], dtype=float)
    ga, pa = sum(r['area'] for r in rows), sum(r['matched_pred_total_pixels'] for r in rows)
    return dict(n=len(rows), mean_absolute_relative_error=float(np.abs(errors).mean()),
                median_signed_relative_error=float(np.median(errors)),
                paired_gt_pixels=ga, paired_pred_pixels=pa,
                paired_signed_relative_error=(pa - ga) / ga)


def analyze_ferrite_native(gt, pred, gt_classes, pred_classes, valid_content=None):
    """单张完整原图的纯CPU分析；不读取文件、不改输入、不推断测试GT。

    返回全部已知域几何匹配、F实例/有关预测的可核查贡献、保守候选与面积。
    ``contained``要求该预测在所有已知域只覆盖同一GT；有邻区贡献另列泄漏。
    两个contained且进入可信内部的显著片段才列内部误切候选。
    """
    gt, pred = _labels(gt, 'gt'), _labels(pred, 'pred')
    if gt.shape != pred.shape:
        raise ValueError('gt and pred must share the exact native pixel grid')
    gc, pc = _classes(gt_classes, gt, 'gt_classes'), _classes(pred_classes, pred, 'pred_classes')
    content = np.ones(gt.shape, bool) if valid_content is None else np.asarray(valid_content)
    if content.dtype != np.bool_ or content.shape != gt.shape:
        raise ValueError('valid_content must be a same-grid boolean array')
    known = content & (gt > 0)
    bad_g, bad_p = _unsafe_ids(gt, known), _unsafe_ids(pred, known)
    g, gi = np.unique(gt[known], return_inverse=True)
    p, pi = np.unique(pred[known], return_inverse=True)
    matrix = np.bincount(gi * len(p) + pi, minlength=len(g) * len(p)).reshape(len(g), len(p))
    deep = _deep_mask(gt, known)
    deep_matrix = np.bincount((gi * len(p) + pi)[deep[known]],
                              minlength=len(g) * len(p)).reshape(len(g), len(p))
    pairs, ga, pa = _pairs(g, p, matrix)
    gt_index, pred_index = {int(v): i for i, v in enumerate(g)}, {int(v): i for i, v in enumerate(p)}
    all_pred, total_pred = np.unique(pred, return_counts=True)
    pred_total = {int(v): int(n) for v, n in zip(all_pred, total_pred) if v > 0}
    adjacency = _adjacency(gt, known)
    pair_map = {r['gt']: r for r in pairs}
    significant_g = matrix >= np.maximum(RULES['significant_pixels'], RULES['significant_fraction'] * ga[:, None])
    significant_p = matrix >= np.maximum(RULES['significant_pixels'], RULES['significant_fraction'] * pa[None, :])
    if 0 in pred_index:
        significant_g[:, pred_index[0]] = False; significant_p[:, pred_index[0]] = False
    dominant_pred = {int(v): int(p[int(matrix[i].argmax())]) for i, v in enumerate(g)} if len(p) else {}
    pred_rows, pred_lookup, ff_rows, fp_rows, cooccupancy = [], {}, [], [], []
    for j, value in enumerate(p):
        if value == 0 or not any(gc[int(g[i])] == 1 for i in np.flatnonzero(matrix[:, j])):
            continue
        value = int(value)
        overlap_ids = np.flatnonzero(matrix[:, j])
        parent = int(g[int(matrix[:, j].argmax())])
        contributions = [dict(gt=int(g[i]), phase=gc[int(g[i])], pixels=int(matrix[i, j]),
            gt_fraction=float(matrix[i, j] / ga[i]), pred_fraction=float(matrix[i, j] / pa[j]),
            deep_pixels=int(deep_matrix[i, j]), significant_for_gt=bool(significant_g[i, j]),
            significant_for_pred=bool(significant_p[i, j])) for i in overlap_ids]
        supported = [int(g[i]) for i in np.flatnonzero(significant_p[:, j])]
        ferrite = [v for v in supported if gc[v] == 1]
        cooccupied_pairs = [list(q) for q in combinations(ferrite, 2) if tuple(q) in adjacency]
        merge_ferrite = [v for v in ferrite if significant_g[gt_index[v], j]]
        ff_pairs = [list(q) for q in combinations(merge_ferrite, 2) if tuple(q) in adjacency]
        f_pixels = int(sum(r['pixels'] for r in contributions if r['phase'] == 1))
        p_pixels = int(pa[j]) - f_pixels
        mixed = min(f_pixels, p_pixels) >= max(RULES['significant_pixels'], RULES['significant_fraction'] * int(pa[j]))
        censored = value in bad_p or any(v in bad_g for v in supported)
        row = dict(pred=value, phase=pc[value], total_pixels=pred_total[value],
                   known_pixels=int(pa[j]), unknown_or_invalid_pixels=pred_total[value]-int(pa[j]),
                   censored=value in bad_p, dominant_gt=parent, dominant_gt_phase=gc[parent],
                   contributions=contributions, ff_pairs=ff_pairs, fp_mixed_candidate=bool(mixed),
                   F_known_pixels=f_pixels, P_known_pixels=p_pixels,
                   vote_inconsistent_with_known_phase_majority=pc[value] != int(f_pixels > p_pixels))
        pred_rows.append(row); pred_lookup[value] = row
        if len(ferrite) >= 2:
            cooccupancy.append(dict(pred=value, gt_ids=ferrite, adjacent_gt_pairs=cooccupied_pairs,
                contact_edges=[dict(gt_pair=q, edges=adjacency[tuple(q)]) for q in cooccupied_pairs],
                censored=censored, whole_grain_cooccupancy=all(dominant_pred[v] == value for v in ferrite),
                supporting_contributions=[r for r in contributions if r['gt'] in ferrite]))
        if ff_pairs:
            ff_rows.append(dict(pred=value, gt_ids=merge_ferrite, adjacent_gt_pairs=ff_pairs,
                contact_edges=[dict(gt_pair=q, edges=adjacency[tuple(q)]) for q in ff_pairs],
                censored=value in bad_p or any(v in bad_g for v in merge_ferrite),
                whole_grain_cooccupancy=all(dominant_pred[v] == value for v in merge_ferrite),
                supporting_contributions=[r for r in contributions if r['gt'] in merge_ferrite]))
        if mixed:
            involved = [r['gt'] for r in contributions
                        if r['significant_for_pred'] or r['pixels'] >= RULES['significant_pixels']]
            fp_rows.append(dict(pred=value, gt_ids=involved, F_pixels=f_pixels, P_pixels=p_pixels,
                predicted_phase=pc[value], censored=value in bad_p or any(v in bad_g for v in involved),
                contributions=contributions))
    gt_rows = []
    for i, value in enumerate(g):
        value = int(value)
        if gc[value] != 1:
            continue
        contributors = np.flatnonzero((matrix[i] > 0) & (p > 0))
        ids = [int(p[j]) for j in contributors]
        sig = [int(p[j]) for j in contributors if significant_g[i, j]]
        contained = [v for v in sig if pred_lookup[v]['known_pixels'] == int(matrix[i, pred_index[v]])]
        leaked = [v for v in ids if pred_lookup[v]['known_pixels'] > int(matrix[i, pred_index[v]])]
        contained_deep = [v for v in contained if deep_matrix[i, pred_index[v]] > 0]
        same_parent_deep = [v for v in sig if pred_lookup[v]['dominant_gt'] == value
                            and deep_matrix[i, pred_index[v]] > 0]
        foreign = [v for v in ids if pred_lookup[v]['dominant_gt'] != value]
        significant_foreign = [v for v in foreign if significant_g[i, pred_index[v]]]
        match = pair_map.get(value)
        matched = match['pred'] if match else None
        j = pred_index[matched] if match else None
        geom_split = len(sig) >= 2
        matched_support = [int(g[k]) for k in np.flatnonzero(significant_p[:, j])] if match else []
        geom_merge = len(matched_support) >= 2
        safe = value not in bad_g and matched is not None and matched not in bad_p
        same_phase = match is not None and pc[matched] == 1
        area_eligible = safe and same_phase and not geom_split and not geom_merge
        missed = int(ga[i] - matrix[i, j]) if match else None
        extra = int(pa[j] - matrix[i, j]) if match else None
        missed_deep = int(deep_matrix[i].sum() - deep_matrix[i, j]) if match else None
        extra_deep = int(deep_matrix[:, j].sum() - deep_matrix[i, j]) if match else None
        mismatch = (missed + extra) if match else None
        deep_mismatch = (missed_deep + extra_deep) if match else None
        near_fraction = (mismatch-deep_mismatch)/mismatch if mismatch else None
        near_candidate = (match is not None and not geom_split and not geom_merge
            and mismatch >= RULES['significant_pixels'] and near_fraction >= RULES['boundary_offset_near_fraction'])
        censored_relation = value in bad_g or any(v in bad_p for v in sig)
        gt_rows.append(dict(gt=value, phase=1, area=int(ga[i]), censored=value in bad_g,
            matched_pred=matched, matched_iou=match['iou'] if match else None,
            predicted_phase=pc[matched] if match else None,
            significant_pred_ids=sig, contained_significant_pred_ids=contained,
            neighbor_leakage_pred_ids=leaked, foreign_dominant_pred_ids=foreign,
            significant_foreign_pred_ids=significant_foreign,
            foreign_dominant_pixels=int(sum(matrix[i, pred_index[v]] for v in foreign)),
            foreign_dominant_deep_pixels=int(sum(deep_matrix[i, pred_index[v]] for v in foreign)),
            deep_gt_pixels=int(deep_matrix[i].sum()),
            internal_split_candidate=len(contained_deep) >= 2,
            same_parent_significant_deep_pred_ids=same_parent_deep,
            same_parent_internal_partition_candidate=len(same_parent_deep) >= 2,
            interior_partition_candidate=bool(sum(deep_matrix[i, pred_index[v]] > 0 for v in sig) >= 2),
            mixed_partition_and_leakage_candidate=geom_split and bool(leaked),
            any_crossGT_overlap_candidate=bool(leaked),
            foreign_neighbor_overlap_candidate=bool(foreign),
            boundary_leakage_candidate=bool(significant_foreign),
            significant_foreign_neighbor_invasion_candidate=bool(significant_foreign),
            topology_censored=censored_relation,
            vote_wrong=match is not None and pc[matched] == 0,
            F_pixels_predicted_P=int(sum(matrix[i, k] for k, v in enumerate(p) if v > 0 and pc[int(v)] == 0)),
            unassigned_known_pixels=int(matrix[i, pred_index[0]]) if 0 in pred_index else 0,
            significant_geometry_split=geom_split, matched_pred_significant_geometry_merge=geom_merge,
            matched_pred_total_pixels=pred_total[matched] if match else None,
            matched_pred_known_pixels=int(pa[j]) if match else None,
            missed_pixels=missed, extra_known_pixels=extra,
            missed_deep_pixels=missed_deep, extra_deep_pixels=extra_deep,
            near_boundary_mismatch_fraction=near_fraction,
            boundary_offset_candidate=bool(near_candidate),
            safe_one_to_one_area_eligible=bool(area_eligible),
            area_relative_error=(pred_total[matched]-int(ga[i]))/int(ga[i]) if area_eligible else None,
            contributions=[dict(pred=int(p[k]), predicted_phase=pc[int(p[k])],
                pixels=int(matrix[i, k]), deep_pixels=int(deep_matrix[i, k]),
                significant=bool(significant_g[i, k]),
                other_GT_known_pixels=int(pa[k]-matrix[i, k]),
                purity_in_known_domain=float(matrix[i, k]/pa[k]),
                dominant_gt=pred_lookup[int(p[k])]['dominant_gt'],
                pred_censored=int(p[k]) in bad_p,
                category='contained_fragment' if pa[k] == matrix[i, k] else 'neighbor_leakage')
                for k in contributors]))
    area_rows = [r for r in gt_rows if r['safe_one_to_one_area_eligible']]
    mechanisms = dict(ff_merge_candidates=len(ff_rows),
        safe_whole_grain_ff_merge=sum(not r['censored'] and r['whole_grain_cooccupancy'] for r in ff_rows),
        internal_split_candidates=sum(r['internal_split_candidate'] for r in gt_rows),
        safe_internal_split_candidates=sum(r['internal_split_candidate'] and not r['topology_censored'] for r in gt_rows),
        same_parent_internal_partition_candidates=sum(r['same_parent_internal_partition_candidate'] for r in gt_rows),
        safe_same_parent_internal_partition_candidates=sum(r['same_parent_internal_partition_candidate']
            and not r['topology_censored'] for r in gt_rows),
        mixed_partition_and_leakage_candidates=sum(r['mixed_partition_and_leakage_candidate'] for r in gt_rows),
        any_crossGT_overlap_gt=sum(r['any_crossGT_overlap_candidate'] for r in gt_rows),
        foreign_neighbor_overlap_gt=sum(r['foreign_neighbor_overlap_candidate'] for r in gt_rows),
        significant_foreign_neighbor_invasion_gt=sum(r['significant_foreign_neighbor_invasion_candidate'] for r in gt_rows),
        fp_mixed_pred=len(fp_rows), safe_fp_mixed_pred=sum(not r['censored'] for r in fp_rows),
        F_matched_vote_wrong=sum(r['vote_wrong'] for r in gt_rows),
        boundary_offset_candidates=sum(r['boundary_offset_candidate'] for r in gt_rows))
    return dict(format=FORMAT, rules=dict(RULES), shape=list(gt.shape),
        summary=dict(counts=dict(known_pixels=int(known.sum()), unknown_or_invalid_pixels=int((~known).sum()),
            F_gt=len(gt_rows), safe_F_gt=sum(not r['censored'] for r in gt_rows),
            geometry_matched_F=sum(r['matched_pred'] is not None for r in gt_rows),
            unmatched_F_gt=[r['gt'] for r in gt_rows if r['matched_pred'] is None],
            censored_F_gt=[r['gt'] for r in gt_rows if r['censored']],
            F_pixels_predicted_P=sum(r['F_pixels_predicted_P'] for r in gt_rows)),
            mechanism_counts=mechanisms, safe_complete_F_area=_area_summary(area_rows)),
        pairs=pairs, gt_rows=gt_rows, pred_rows=pred_rows, ff_merge_rows=ff_rows,
        ff_cooccupancy_rows=cooccupancy, fp_mixed_rows=fp_rows,
        censored_gt_ids=sorted(bad_g), censored_pred_ids=sorted(bad_p),
        caveats=['Seen labelled training-image mechanism diagnostics; not independent validation or official mIoU/area.',
            'All positive processed GT including filled is used; GT0/invalid ignored, never counted as false.',
            'Whole GT/pred objects touching unknown, invalid content or frame are censored separately.',
            '50px/10% marks significant contributions; multiple IDs alone never establish internal oversegmentation.',
            'Contained fragments and neighbor leakage are separate; near-boundary4px is descriptive, not a correctness oracle.',
            'Any cross-GT overlap is broad audit only; neighbor-invasion cases require a foreign dominant GT and significant contribution to this F GT.',
            'Strict containment is a non-exhaustive lower-bound description; same-parent deep fragments tolerate measured neighbor spill but remain partition candidates.',
            'Safe area uses same-class F one-to-one matches without significant split/merge; unmatched/censored excluded explicitly.',
            'No global predicted mean vs incomplete GT mean, expected count fitting, new thresholds or deployment changes.'])


def select_ferrite_cases(report, max_per_kind=1):
    """固定机制顺序、GT/pred ID升序；只取整对象可信案例，不按观感挑图。"""
    if isinstance(max_per_kind, bool) or not isinstance(max_per_kind, int) or max_per_kind < 0:
        raise ValueError('max_per_kind must be a nonnegative integer')
    candidates = {k: [] for k in KINDS}
    pred_lookup = {q['pred']: q for q in report['pred_rows']}
    for r in report['ff_merge_rows']:
        if not r['censored'] and r['whole_grain_cooccupancy']:
            candidates['ff_merge'].append(dict(kind='ff_merge', gt_ids=r['gt_ids'], pred_ids=[r['pred']]))
    for r in report['fp_mixed_rows']:
        if not r['censored']:
            candidates['fp_mix'].append(dict(kind='fp_mix', gt_ids=r['gt_ids'], pred_ids=[r['pred']]))
    for r in report['gt_rows']:
        if r['censored']:
            continue
        contributors = {q['pred']: q for q in r['contributions']}
        if r['internal_split_candidate'] and not r['topology_censored']:
            candidates['internal_split'].append(dict(kind='internal_split', gt_ids=[r['gt']],
                pred_ids=r['contained_significant_pred_ids'], evidence_rule='strict_contained_deep_fragments'))
        elif r['same_parent_internal_partition_candidate'] and not r['topology_censored']:
            candidates['internal_split'].append(dict(kind='internal_split', gt_ids=[r['gt']],
                pred_ids=r['same_parent_significant_deep_pred_ids'],
                evidence_rule='same_parent_candidate_not_proven_split'))
        leakage = [v for v in r['significant_foreign_pred_ids'] if not contributors[v]['pred_censored']
                   and pred_lookup[v]['dominant_gt'] not in report['censored_gt_ids']]
        if leakage:
            candidates['neighbor_leakage'].append(dict(kind='neighbor_leakage', gt_ids=[r['gt']],
                pred_ids=leakage, evidence_rule='significant_foreign_dominant_GT_contribution'))
        safe_match = r['matched_pred'] is not None and r['matched_pred'] not in report['censored_pred_ids']
        if safe_match and r['vote_wrong']:
            candidates['vote_error'].append(dict(kind='vote_error', gt_ids=[r['gt']], pred_ids=[r['matched_pred']]))
        if safe_match and r['boundary_offset_candidate']:
            candidates['boundary_offset'].append(dict(kind='boundary_offset', gt_ids=[r['gt']], pred_ids=[r['matched_pred']]))
        if r['safe_one_to_one_area_eligible'] and r['area_relative_error'] != 0:
            candidates['safe_area_bias'].append(dict(kind='safe_area_bias', gt_ids=[r['gt']], pred_ids=[r['matched_pred']]))
    selected = []
    for kind in KINDS:
        selected.extend(sorted(candidates[kind], key=lambda r: (
            r.get('evidence_rule') == 'same_parent_candidate_not_proven_split',
            tuple(r['gt_ids']), tuple(r['pred_ids'])))[:max_per_kind])
    return selected


def _color_panel(rgb, labels, gt, known):
    """同GT固定颜色，预测按最大重叠父GT着色，黑线保留真实预测实例边。"""
    palette = np.random.default_rng(20261002).integers(50, 230, (int(gt.max())+1, 3), dtype=np.uint8)
    palette[0] = 80
    lookup = np.zeros(int(labels.max())+1, np.int64)
    if known.any():
        g, gi = np.unique(gt[known], return_inverse=True)
        p, pi = np.unique(labels[known], return_inverse=True)
        counts = np.bincount(gi*len(p)+pi, minlength=len(g)*len(p)).reshape(len(g), len(p))
        keep = p > 0
        lookup[p[keep]] = g[counts[:, keep].argmax(0)]
    color = palette[lookup[labels]]
    panel = np.rint(.6*rgb + .4*color).astype(np.uint8)
    edge = np.zeros(labels.shape, bool)
    edge[1:] |= labels[1:] != labels[:-1]; edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    panel[edge] = 0; panel[~known] = 90
    return panel


def render_ferrite_cases(folder, stem, rgb, gt, pred, report, cases, boundary=None, markers=None):
    """仅渲染协调器固定选择的案例；所有GT来自调用方允许的训练源。"""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (*gt.shape, 3):
        raise ValueError('Native RGB uint8 grid required for rendering')
    known = gt > 0
    panels = [('Original training RGB', rgb), ('Processed newGT', _color_panel(rgb, gt, gt, known)),
              ('Best R1 native1024 final', _color_panel(rgb, pred, gt, known))]
    if boundary is not None:
        boundary = np.asarray(boundary)
        if boundary.shape != gt.shape or not np.isfinite(boundary).all() or boundary.min() < 0 or boundary.max() > 1:
            raise ValueError('Actual continuous native boundary must stay in 0..1')
        heat = cv2.applyColorMap(np.rint(boundary*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB); heat[~known] = 90
        panels.append(('Actual continuous boundary', heat))
    if markers is not None:
        if np.asarray(markers).shape != gt.shape:
            raise ValueError('Actual markers must share native grid')
        panels.append(('Actual watershed markers', _color_panel(rgb, np.maximum(markers, 0), gt, known)))
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    metadata = []
    for index, case in enumerate(cases):
        focus = np.isin(gt, case['gt_ids'])
        if not focus.any():
            raise ValueError('Selected case has no GT pixels')
        y, x = np.nonzero(focus); h, w = gt.shape
        box = (max(0, int(y.min())-32), max(0, int(x.min())-32),
               min(h, int(y.max())+33), min(w, int(x.max())+33))
        strips = []
        for title, panel in panels:
            body = panel[box[0]:box[2], box[1]:box[3]]
            body = cv2.resize(body, (360, max(1, round(360*body.shape[0]/body.shape[1]))), interpolation=cv2.INTER_AREA)
            body = cv2.copyMakeBorder(body, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(body, title, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,0,0), 1, cv2.LINE_AA)
            strips.append(body)
        name = f'{stem}_{case["kind"]}_{index:02d}.png'
        if not cv2.imwrite(str(folder/name), cv2.cvtColor(np.concatenate(strips, 1), cv2.COLOR_RGB2BGR)):
            raise RuntimeError('Ferrite gallery write failed')
        metadata.append(dict(**case, file=name, native_box_yxyx=list(box),
            scope='seen original labelled training RGB; full-image native1024 final, crop is display only'))
    return metadata


def selftest():
    """CPU合成门禁：匹配、内部碎片/邻区泄漏、空域、整对象censor及面积。"""
    gt = np.full((60, 60), 2, np.uint16); gt[20:40, 20:40] = 1
    pred = gt.copy()
    good = analyze_ferrite_native(gt, pred, {1:1, 2:0}, {1:1, 2:0})
    assert good['summary']['safe_complete_F_area']['n'] == 1
    assert good['gt_rows'][0]['area_relative_error'] == 0
    pred[20:40, 30:40] = 3
    split = analyze_ferrite_native(gt, pred, {1:1, 2:0}, {1:1, 2:0, 3:1})
    assert split['gt_rows'][0]['internal_split_candidate']
    assert not split['gt_rows'][0]['neighbor_leakage_pred_ids']
    pred = gt.copy(); pred[20:40, 36:40] = 2
    leak = analyze_ferrite_native(gt, pred, {1:1, 2:0}, {1:1, 2:0})
    assert not leak['gt_rows'][0]['internal_split_candidate']
    assert leak['gt_rows'][0]['boundary_leakage_candidate']
    pred = gt.copy(); pred[20:40, 38:40] = 0
    shrink = analyze_ferrite_native(gt, pred, {1:1, 2:0}, {1:1, 2:0})
    assert abs(shrink['gt_rows'][0]['area_relative_error']+.1) < 1e-12
    empty = np.zeros((5, 7), np.uint16)
    blank = analyze_ferrite_native(empty, empty, {}, {})
    assert blank['summary']['safe_complete_F_area']['n'] == 0
    assert blank['summary']['safe_complete_F_area']['mean_absolute_relative_error'] is None
    for report in (good, split, leak, shrink, blank):
        json.dumps(report, ensure_ascii=False, allow_nan=False)
    return dict(passed=True, format=FORMAT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if not args.selftest:
        parser.error('This CPU array core is called by the frozen native runner; standalone only --selftest')
    print(json.dumps(selftest(), ensure_ascii=False))
