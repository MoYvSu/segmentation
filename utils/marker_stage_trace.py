# -*- coding: utf-8 -*-
"""只读追踪整图种子屏障阶段；不把固定端点关系当作整粒修复。

stage_maps 直接接受 tools.semantic_marker_diagnostic.connectivity_maps 的
有序返回值。标签必须先在整幅原尺寸图上计算，不能先裁图再生成标签。
本模块仅在已知目标 GT 内重算连通作诊断，不修改部署或推断未知 GT。
"""
from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer, not bool or float')
    return int(value)


def _labels(value, name, shape=None):
    value = np.asarray(value)
    if (value.ndim != 2 or not value.size
            or not np.issubdtype(value.dtype, np.integer) or np.any(value < 0)):
        raise ValueError(f'{name} must be a nonempty nonnegative integer HW map')
    if shape is not None and value.shape != shape:
        raise ValueError(f'{name} must retain the full GT image shape')
    return value


def _relation(values, direction):
    positive = [v > 0 for v in values]
    connected = bool(values[0] == values[1]) if all(positive) else None
    goal = None if connected is None else (not connected if direction == 'negative' else connected)
    return dict(endpoint_labels=list(values), endpoints_positive=positive,
                connected=connected, endpoint_relation_goal=goal)


def _contributions(labels, mask, deep, gid):
    values, counts = np.unique(labels[mask], return_counts=True)
    pairs = sorted(zip(values.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0]))
    positive = [(label, count) for label, count in pairs if label > 0]
    dominant = int(positive[0][0]) if positive else 0
    known = int(mask.sum())
    deep_count = int(deep.sum())
    contributions = [dict(label=int(label), pixels=int(count),
                          deep_pixels=int((deep & (labels == label)).sum()))
                     for label, count in pairs]
    return dict(gt=gid, known_pixels=known, deep_pixels=deep_count,
                dominant_label=dominant,
                dominant_fraction=float((mask & (labels == dominant)).sum() / known) if dominant else 0.,
                dominant_deep_fraction=float((deep & (labels == dominant)).sum() / max(1, deep_count)) if dominant else 0.,
                significant_label_ids=[int(label) for label, count in positive if count >= max(50, .1 * known)],
                zero_pixels=int((mask & (labels == 0)).sum()),
                contributions=contributions)


def _known_relation(labels, allowed, local_points, full_values, direction, domain):
    """在原正标签内重算 CC；最终实例图的不同 ID 不得被 free>0 误合并。"""
    keys = [[0, 0], [0, 0]]
    for value in sorted(set(full_values) - {0}):
        free = (allowed & (labels == value)).astype(np.uint8)
        _, components = cv2.connectedComponents(free, connectivity=8)
        for index, (endpoint_value, point) in enumerate(zip(full_values, local_points)):
            if endpoint_value == value:
                keys[index] = [value, int(components[point])]
    positive = [value > 0 and component > 0 for value, component in keys]
    connected = bool(keys[0] == keys[1]) if all(positive) else None
    return dict(endpoint_component_keys=keys, endpoints_positive=positive,
                connected=connected,
                endpoint_relation_goal=None if connected is None else
                    (not connected if direction == 'negative' else connected),
                zero_pixels=int((allowed & (labels == 0)).sum()),
                definition=f'8CC within each original positive stage label and {domain}; diagnostics only')


def trace_case_stages(stage_maps, marker_labels, instances, gt, case, *, watershed=None):
    """返回有序 JSON 记录；输入均不改写，阶段标签不是跨阶段可比较的 ID。

    GT/阶段/marker 为非负整数 HW；instances 必须为最终契约 uint16 HW。
    case 的两个 native points 必须属于依次指定的 GT（positive 仅一个 GT）。
    完整已知域贡献包含 label=0 和全部小碎片。深部使用与 case_quality 相同
    的 9×9 全一核、常数零边界腐蚀；censored 只保留提示，不删除已知域统计。
    """
    gt = _labels(gt, 'GT')
    marker_labels = _labels(marker_labels, 'marker_labels', gt.shape)
    instances = _labels(instances, 'instances', gt.shape)
    if watershed is not None:
        watershed = _labels(watershed, 'watershed', gt.shape)
    if instances.dtype != np.uint16:
        raise ValueError('instances must use the final single-channel uint16 contract')
    if not isinstance(case, Mapping) or case.get('direction') not in ('negative', 'positive'):
        raise ValueError('case.direction must be negative or positive')
    direction = case['direction']
    groups = case.get('gt_ids')
    if not isinstance(groups, (list, tuple)) or len(groups) != (2 if direction == 'negative' else 1):
        raise ValueError('negative requires two GT IDs; positive requires one GT ID')
    groups = [_integer(gid, 'GT ID') for gid in groups]
    if any(gid <= 0 or not np.any(gt == gid) for gid in groups) or len(set(groups)) != len(groups):
        raise ValueError('GT IDs must be distinct positive IDs present in GT')
    points = case.get('points')
    if not isinstance(points, (list, tuple)) or len(points) != 2:
        raise ValueError('case.points must contain exactly two native [y,x] coordinates')
    normalized = []
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError('Each point must contain integer [y,x]')
        y, x = [_integer(v, 'Point coordinate') for v in point]
        if not (0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]):
            raise ValueError('Native point lies outside the full image')
        if int(gt[y, x]) != groups[min(index, len(groups) - 1)]:
            raise ValueError('Native point does not lie in its specified target GT')
        normalized.append((y, x))
    if normalized[0] == normalized[1]:
        raise ValueError('Native points must be distinct')
    if 'censored' in case and not isinstance(case['censored'], (bool, np.bool_)):
        raise ValueError('case.censored must be bool when supplied')
    if not isinstance(stage_maps, Mapping) or not stage_maps:
        raise ValueError('stage_maps must be a nonempty ordered mapping from connectivity_maps')
    validated = []
    for name, value in stage_maps.items():
        if not isinstance(name, str) or not name or name in ('marker_labels', 'watershed', 'instances'):
            raise ValueError('Connectivity stage names must be nonempty and not reserved')
        if not isinstance(value, Mapping) or 'labels' not in value:
            raise ValueError('Each stage must contain the full-image labels map')
        labels = _labels(value['labels'], f'stage {name}', gt.shape)
        if 'component_count' in value and _integer(value['component_count'], 'component_count') < 0:
            raise ValueError('component_count must be nonnegative')
        validated.append((name, labels, 'unfiltered_barrier_components'))
    validated.append(('marker_labels', marker_labels, 'actual_area_filtered_seeds'))
    if watershed is not None:
        validated.append(('watershed', watershed, 'actual_watershed_partition'))
    validated.append(('instances', instances, 'final_instance_partition'))

    target = np.isin(gt, groups)
    known_all = gt > 0
    ys, xs = np.nonzero(target)
    y0, x0, y1, x1 = int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
    crop = (slice(y0, y1), slice(x0, x1))
    local_gt = gt[crop]
    allowed = target[crop]
    local_points = [(y - y0, x - x0) for y, x in normalized]
    masks = []
    for gid in groups:
        mask = local_gt == gid
        deep = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        masks.append((gid, mask, deep))
    rows = []
    for name, labels, kind in validated:
        values = [int(labels[point]) for point in normalized]
        local_labels = labels[crop]
        gt_rows = [_contributions(local_labels, mask, deep, gid) for gid, mask, deep in masks]
        rows.append(dict(name=name, kind=kind, **_relation(values, direction),
            zero_pixels=int(np.count_nonzero(labels == 0)),
            zero_pixels_known_target_domain=int(np.count_nonzero(allowed & (local_labels == 0))),
            known_all=_known_relation(labels, known_all, normalized, values, direction,
                                     'all known GT>0; third GT allowed, unknown excluded'),
            known_target_domain=_known_relation(local_labels, allowed, local_points, values, direction,
                                               'target GT union; unknown/third GT excluded'),
            gt_rows=gt_rows,
            dominant_labels_separate=len(gt_rows) == 2 and all(row['dominant_label'] > 0 for row in gt_rows)
                and gt_rows[0]['dominant_label'] != gt_rows[1]['dominant_label'],
            all_dominant_deep_fraction_at_least_95=all(row['dominant_deep_fraction'] >= .95 for row in gt_rows),
            labels_are_final_instance_ids=name == 'instances',
            labels_are_actual_seed_ids=name == 'marker_labels',
            labels_are_actual_watershed_ids=name == 'watershed',
            not_actual_watershed_barrier=True))
    return dict(format='marker_stage_trace_v1', direction=direction, gt_ids=groups,
                points=[list(point) for point in normalized], shape=list(gt.shape),
                censored=bool(case.get('censored', True)), known_domain_only=True,
                stage_order=[row['name'] for row in rows], stages=rows,
                known_target_bbox=[y0, x0, y1, x1],
                interpretation='Endpoint connectivity is not whole-grain correctness; labels are stage-local; no unknown GT inferred')


def classify_stage_failure(trace):
    """只报告相邻阶段的关系变化；不把 zero 当分开，不宣称物理边界错误。"""
    if not isinstance(trace, Mapping) or trace.get('format') != 'marker_stage_trace_v1':
        raise ValueError('Expected marker_stage_trace_v1')
    direction = trace.get('direction')
    stages = trace.get('stages')
    if direction not in ('negative', 'positive') or not isinstance(stages, list) or not stages:
        raise ValueError('Trace must contain direction and ordered stages')
    result = dict(direction=direction, reopened=False, reopened_full=False,
                  reopened_known_all=False, reopened_known_target_domain=False, full_transitions=[],
                  known_all_transitions=[],
                  known_target_domain_transitions=[], endpoint_evidence_only=True,
                  physical_error_conclusion=False)
    for domain, field in [('full', 'full_transitions'), ('known_all', 'known_all_transitions'),
                          ('known_target_domain', 'known_target_domain_transitions')]:
        for before, after in zip(stages, stages[1:]):
            a = before if domain == 'full' else before.get(domain)
            b = after if domain == 'full' else after.get(domain)
            if not isinstance(a, Mapping) or not isinstance(b, Mapping):
                raise ValueError('Stage is missing a connectivity domain')
            for relation in (a, b):
                if ((relation.get('connected') is not None and type(relation.get('connected')) is not bool)
                        or not isinstance(relation.get('endpoints_positive'), list)
                        or len(relation['endpoints_positive']) != 2
                        or any(type(v) is not bool for v in relation['endpoints_positive'])):
                    raise ValueError('Invalid stage endpoint relation')
            if not all(a['endpoints_positive']) or not all(b['endpoints_positive']):
                continue
            connected_before, connected_after = a['connected'], b['connected']
            if connected_before is None or connected_after is None or connected_before == connected_after:
                continue
            transition = 'merge' if connected_after else 'split'
            reopened = direction == 'negative' and transition == 'merge'
            result[field].append(dict(before=before['name'], after=after['name'],
                transition=transition, reopened=reopened,
                dominant_labels_separate_before=before['dominant_labels_separate'],
                dominant_labels_separate_after=after['dominant_labels_separate'],
                deep_95_before=before['all_dominant_deep_fraction_at_least_95'],
                deep_95_after=after['all_dominant_deep_fraction_at_least_95']))
            if reopened:
                result['reopened_' + domain] = True
    # 主 reopened 要求已知目标域的可信通路；full 单独报告，避免 unknown 绕行混淆。
    result['reopened'] = result['reopened_known_target_domain']
    return result
