# -*- coding: utf-8 -*-
"""固定处理后GT的局部关系干预与真实部署阶段追踪；纯CPU，不训练。

GT>0即当前处理后已知域（含filled）；unknown/padding不作答案。
本工具只改所选合法affinity关系，不能作为上界或独立错误计数。
"""
from __future__ import annotations

from itertools import combinations
from numbers import Integral
from unittest.mock import patch

import cv2
import numpy as np
import torch

from tools.affinity_connectivity_views import decode_native
from tools.semantic_marker_diagnostic import marker_stages
from utils.affinity_connectivity import _line_points
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS, _edge_slices
from utils.post_process import _boundary_skeleton_belt

STAGE_ORDER = ('threshold', 'reconstruction', 'bridge', 'skeleton', 'dilation',
               'sealed_core', 'markers', 'watershed', 'final')
ARMS = ('r1', 'tail')


def _ids(value, name):
    array = np.asarray(value)
    if (array.ndim != 2 or not array.size or not np.issubdtype(array.dtype, np.integer)
            or array.dtype == np.bool_ or np.any(array < 0)):
        raise ValueError(name + ' must be a nonempty nonnegative integer HW map')
    return array


def _native_tensor(value, name):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if (value.device.type != 'cpu' or value.ndim != 4 or tuple(value.shape[:2]) != (1, 1)
            or min(value.shape) < 1 or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())):
        raise ValueError(name + ' must be finite CPU floating [1,1,H,W]')
    return value.detach().clone().float()


def decode_capture(semantic, boundary, config):
    """用真实decode_native捕获watershed；marker复刻必须与实际输入逐值相同。"""
    semantic, boundary = _native_tensor(semantic, 'semantic'), _native_tensor(boundary, 'boundary')
    if semantic.shape != boundary.shape or bool(((boundary < 0) | (boundary > 1)).any()):
        raise ValueError('Native semantic/boundary shapes or probability range differ')
    if config['inference'].get('marker_partition_restore', {}).get('enabled', False):
        raise ValueError('Current probe requires disabled marker_partition_restore')
    original, captures = cv2.watershed, []

    def capture(image, markers):
        before = markers.copy()
        result = original(image, markers)
        captures.append(dict(markers=before, watershed=np.maximum(result, 0).copy()))
        return result

    with patch('utils.post_process.cv2.watershed', side_effect=capture):
        instances, classes = decode_native(semantic, boundary, config)
    stages = marker_stages(boundary[0, 0].numpy(), config)
    if len(captures) > 1:
        raise AssertionError('Unexpected multiple watershed calls')
    if captures:
        actual = captures[0]['markers']
        watershed = captures[0]['watershed']
    else:
        if stages['marker_count'] != 0:
            raise AssertionError('A nonempty marker set skipped actual watershed')
        actual = np.zeros(instances.shape, np.int32)
        watershed = actual.copy()
    if not np.array_equal(actual, stages['marker_labels']):
        raise AssertionError('marker_stages differs from actual cv2.watershed input')
    if (instances.dtype != np.uint16 or int(instances.max()) > 65535
            or set(map(int, classes)) != set(map(int, np.unique(instances[instances > 0])))):
        raise AssertionError('Final uint16/instance class contract differs')
    return dict(instances=instances, classes=classes, stages=stages,
                actual_markers=actual, watershed=watershed,
                capture_audit=dict(marker_exact=True, actual_watershed_calls=len(captures),
                    empty_markers=not bool(stages['marker_count']), cpu_only=True,
                    final_dtype='uint16', maximum_id=int(instances.max())))


def representative(mask, bounds=None):
    """距离图的首个最大值为固定内部点；不在裁块上判断整图连通性。"""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2:
        raise ValueError('Representative mask must be HW')
    if bounds is None:
        yy, xx = np.nonzero(mask)
        if not len(yy):
            return None
        bounds = (slice(int(yy.min()), int(yy.max()) + 1),
                  slice(int(xx.min()), int(xx.max()) + 1))
    region = mask[bounds]
    if not region.any():
        return None
    distance = cv2.distanceTransform(np.pad(region.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return [int(y + bounds[0].start), int(x + bounds[1].start)]


def _unsafe_ids(labels, known):
    safe = cv2.erode(known.astype(np.uint8), np.ones((3, 3), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return {int(i) for i in np.unique(labels[~safe]) if i > 0}


def _point_relation(labels, points):
    values = [int(labels[y, x]) for y, x in points]
    return 'blocked' if min(values) <= 0 else 'same' if values[0] == values[1] else 'different'


def _case_shape(case):
    kind, values = case.get('kind'), case.get('gt_ids')
    if kind not in ('split', 'merge') or not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError('Case requires kind and two GT IDs')
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in values):
        raise ValueError('Case GT IDs must be positive integers')
    a, b = map(int, values)
    if (kind == 'split') != (a == b):
        raise ValueError('Split requires identical GT IDs; merge requires distinct GT IDs')
    return kind, a, b


def select_cases(gt, gtclasses, predictions, diagnostics):
    """从R1/TAIL保守拆分/粘连并集选择固定两点；优先用R1参考碎片。

    predictions[arm]可为整数图或含instances的dict；diagnostics[arm]
    使用diagnose_partition返回值。每条记录只是固定点对，不能累计为独立错误。
    """
    gt = _ids(gt, 'gt'); known = gt > 0
    classes = {int(k): int(v) for k, v in gtclasses.items()}
    if set(np.unique(gt[known]).tolist()) - set(classes) or any(v not in (0, 1) for v in classes.values()):
        raise ValueError('All known newGT IDs require phase 0 or 1')
    maps, errors = {}, {}
    for arm in ARMS:
        item = predictions[arm]
        maps[arm] = _ids(item['instances'] if isinstance(item, dict) else item, arm)
        if maps[arm].shape != gt.shape:
            raise ValueError('Prediction/newGT shape differs')
        conservative = diagnostics[arm]['conservative']
        split = {int(g) for g in conservative['split_gt_ids']}
        merged = {tuple(sorted(map(int, pair))) for pair in conservative['merged_gt_pairs']}
        errors[arm] = dict(split=split, merge=merged)
    selected = sorted(set.union(*(errors[a]['split'] for a in ARMS)))
    keys = [('split', (g, g)) for g in selected]
    keys += [('merge', pair) for pair in sorted(set.union(*(errors[a]['merge'] for a in ARMS)))]
    unsafe_gt = _unsafe_ids(gt, known)
    cases = []
    for kind, ids in keys:
        reference = next(a for a in ARMS if (ids[0] in errors[a]['split'] if kind == 'split'
                                             else ids in errors[a]['merge']))
        pred, masks = maps[reference], []
        if kind == 'split':
            mask = gt == ids[0]; area = int(mask.sum())
            values, counts = np.unique(pred[mask], return_counts=True)
            pieces = sorted([(int(n), int(p)) for p, n in zip(values, counts)
                             if p > 0 and n >= max(50, .1 * area)], key=lambda r: (-r[0], r[1]))
            if len(pieces) < 2:
                raise ValueError('Conservative split lacks two actual significant fragments')
            pids = [p for _, p in pieces[:2]]
            masks = [mask & (pred == p) for p in pids]
        else:
            merged = diagnostics[reference]['conservative']['merged_predictions']
            pids = sorted(int(row['pred']) for row in merged if set(ids).issubset(row['gt_ids']))
            if not pids:
                raise ValueError('Conservative GT pair lacks a shared actual prediction')
            pid = min(pids, key=lambda p: (-int((known & (pred == p) & np.isin(gt, ids)).sum()), p))
            pids = [pid, pid]
            masks = [(gt == g) & (pred == pid) for g in ids]
        points = [representative(m) for m in masks]
        if any(p is None for p in points):
            raise ValueError('Conservative relation lacks known endpoint support')
        bad_pred = _unsafe_ids(pred, known)
        censor = dict(gt_unknown_or_frame=any(g in unsafe_gt for g in ids),
                      reference_prediction_unknown_or_frame=any(p in bad_pred for p in pids))
        if any(censor.values()):
            raise ValueError('Provided conservative relation includes a censored whole object')
        reference_relation = _point_relation(pred, points)
        desired_reference = 'different' if kind == 'split' else 'same'
        if reference_relation != desired_reference:
            raise ValueError('Reference endpoints do not reproduce the selected error')
        cases.append(dict(case_id=kind+'_gt_'+'_'.join(map(str, ids)), kind=kind, gt_ids=list(ids),
            gt_phases=[classes[g] for g in ids], points=points, reference_arm=reference,
            reference_prediction_ids=pids, reference_fragment_pixels=[int(m.sum()) for m in masks],
            shared_prediction=pids[0] if kind == 'merge' else None,
            desired_relation='same' if kind == 'split' else 'different', censor=censor,
            baseline_relations={a: _point_relation(maps[a], points) for a in ARMS},
            error_present_in=[a for a in ARMS if (ids[0] in errors[a]['split'] if kind == 'split' else ids in errors[a]['merge'])],
            support='full processed newGT>0 including filled', unit='fixed_endpoint_relation_not_independent_error'))
    return cases


def correct_local_relations(logits, grid_gt, valid, case, channels='short', *, partition=None):
    """仅冻结输出干预；返回新logits、所选关系mask、计数，不原地修改。

    valid必须是像素valid_content，HW/BHW/B1HW；edge_valid只编码端点，
    不足以保证中间像素有效。主split传固定参考partition，限定其两个碎片。
    partition=None的整GT内部版本只供合成检查，明确标为非主实验。
    """
    if (not torch.is_tensor(logits) or logits.device.type != 'cpu' or logits.ndim != 4
            or tuple(logits.shape[:2]) != (1, 8) or min(logits.shape) < 1
            or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())):
        raise ValueError('logits must be finite CPU floating [1,8,H,W]')
    gt = torch.as_tensor(grid_gt)
    if gt.ndim == 3 and gt.shape[0] == 1:
        gt = gt[0]
    if (gt.device.type != 'cpu' or gt.ndim != 2 or tuple(gt.shape) != tuple(logits.shape[-2:])
            or gt.dtype == torch.bool or gt.is_floating_point() or gt.is_complex()):
        raise ValueError('grid_gt must be a matching nonnegative integer CPU HW map')
    gt = gt.to(torch.int64)
    if bool((gt < 0).any()):
        raise ValueError('grid_gt must be nonnegative')
    content = torch.as_tensor(valid)
    if content.ndim == 4 and tuple(content.shape[:2]) == (1, 1):
        content = content[0, 0]
    elif content.ndim == 3 and content.shape[0] == 1:
        content = content[0]
    if content.device.type != 'cpu' or content.dtype != torch.bool or content.shape != gt.shape:
        raise ValueError('valid must be matching bool pixel valid_content, not edge_valid')
    if channels not in ('short', 'all'):
        raise ValueError('channels must be short or all')
    kind, a, b = _case_shape(case)
    pred = None
    if partition is not None:
        pred = torch.as_tensor(partition)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if (pred.device.type != 'cpu' or pred.shape != gt.shape or pred.dtype == torch.bool
                or pred.is_floating_point() or pred.is_complex()):
            raise ValueError('partition must be a matching integer CPU HW map')
        pred = pred.to(torch.int64)
        if bool((pred < 0).any()):
            raise ValueError('partition must be nonnegative')
    if kind == 'split' and pred is not None:
        fragments = case.get('reference_prediction_ids')
        if (not isinstance(fragments, (list, tuple)) or len(fragments) != 2
                or any(isinstance(p, bool) or not isinstance(p, Integral) or p <= 0 for p in fragments)
                or fragments[0] == fragments[1]):
            raise ValueError('Split reference requires two distinct positive prediction IDs')
        p, q = map(int, fragments)
    chosen = torch.zeros_like(logits, dtype=torch.bool)
    counts = dict(kind=kind, channels=channels, gt_ids=[a, b], selected_edges=0,
        changed_edges=0, original_endpoint_edges=0, path_filtered_edges=0,
        third_known_GT_path_edges=0, per_channel=[], no_contact=False,
        selection_scope=('split_reference_fragments' if pred is not None else 'split_all_instance_nonprimary')
                        if kind == 'split' else 'merge_direct_GT_pair',
        input_mutated=False, output_intervention_only=True, is_upper_bound=False)
    height, width = gt.shape
    known = content & (gt > 0)
    for c, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        if height <= abs(dy) or width <= abs(dx):
            counts['per_channel'].append(0); continue
        source, dest = _edge_slices(height, width, dy, dx)
        left, right = gt[source], gt[dest]
        eligible = known[source] & known[dest]
        if kind == 'split':
            relevant = (left == a) & (right == a)
            if pred is not None:
                lp, rp = pred[source], pred[dest]
                relevant &= ((lp == p) & (rp == q)) | ((lp == q) & (rp == p))
        else:
            relevant = ((left == a) & (right == b)) | ((left == b) & (right == a))
        endpoint = eligible & relevant
        y0, y1, x0, x1 = source[0].start, source[0].stop, source[1].start, source[1].stop
        entire = endpoint.clone(); third = torch.zeros_like(endpoint)
        for sy, sx in _line_points(dy, dx):
            path_gt = gt[y0+sy:y1+sy, x0+sx:x1+sx]
            entire &= known[y0+sy:y1+sy, x0+sx:x1+sx]
            third |= (path_gt > 0) & (path_gt != a) & (path_gt != b)
        enabled = channels == 'all' or c < 4
        take = entire if enabled else torch.zeros_like(entire)
        chosen[0, c, source[0], source[1]] = take
        count = int(take.sum()); counts['per_channel'].append(count)
        if enabled:
            counts['original_endpoint_edges'] += int(endpoint.sum())
            counts['path_filtered_edges'] += int((endpoint & ~entire).sum())
            counts['third_known_GT_path_edges'] += int((take & third).sum())
    replacement = logits.new_tensor(16. if kind == 'split' else -16.)
    corrected = torch.where(chosen, replacement, logits)
    counts['selected_edges'] = int(chosen.sum())
    counts['changed_edges'] = int((chosen & (corrected != logits)).sum())
    counts['no_contact'] = counts['selected_edges'] == 0
    return corrected, chosen, counts


def trace_relations(decoded, gt, cases):
    """原尺寸全图连通性：同时报告实际全域与仅processed GT已知域通路。"""
    gt = _ids(gt, 'gt'); known = gt > 0
    stages, watershed, final = decoded['stages'], decoded['watershed'], decoded['instances']
    if final.shape != gt.shape or watershed.shape != gt.shape:
        raise ValueError('Decoded/newGT shape differs')
    for case in cases:
        _case_shape(case)
        points = case.get('points')
        if (not isinstance(points, (list, tuple)) or len(points) != 2
                or any(len(p) != 2 or any(isinstance(i, bool) or not isinstance(i, Integral) for i in p)
                       or not (0 <= p[0] < gt.shape[0] and 0 <= p[1] < gt.shape[1]) for p in points)):
            raise ValueError('Case points require two in-bounds integer [y,x] positions')
        if any(not known[y, x] or int(gt[y, x]) != int(case['gt_ids'][i])
               for i, (y, x) in enumerate(points)):
            raise ValueError('Endpoint is outside corresponding processed newGT')
    marker = stages['marker_boundary_mask']
    bridge, dilation = stages['resolved']['bridge_width'], stages['resolved']['watershed_dilate_width']
    bridged = marker.copy()
    if bridge:
        bridged = cv2.dilate(bridged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*bridge+1, 2*bridge+1)))
    skeleton, _ = _boundary_skeleton_belt(marker, bridge, 0)
    belt = skeleton
    if dilation:
        belt = cv2.dilate(skeleton, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation+1, 2*dilation+1)))
    barriers = dict(threshold=stages['high_mask'], reconstruction=marker, bridge=bridged,
                    skeleton=skeleton, dilation=belt, sealed_core=stages['marker_belt'])
    result = {}
    for name, barrier in barriers.items():
        free = barrier == 0
        _, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
        full = [_point_relation(labels, c['points']) for c in cases]
        _, labels = cv2.connectedComponents((free & known).astype(np.uint8), connectivity=8)
        only = [_point_relation(labels, c['points']) for c in cases]
        result[name] = dict(full=full, known_only=only,
            unknown_route_dependent=[a == 'same' and b != 'same' for a, b in zip(full, only)])
    for name, labels in [('markers', stages['marker_labels']), ('watershed', watershed), ('final', final)]:
        full, only = [], []
        for case in cases:
            points = case['points']; relation = _point_relation(labels, points); full.append(relation)
            if relation == 'same':
                mask = (labels == int(labels[tuple(points[0])])) & known
                _, component = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
                only.append(_point_relation(component, points))
            else:
                only.append(relation)
        result[name] = dict(full=full, known_only=only,
            unknown_route_dependent=[a == 'same' and b != 'same' for a, b in zip(full, only)])
    rows = []
    for i, case in enumerate(cases):
        desired = 'same' if case['kind'] == 'split' else 'different'
        rows.append(dict(case_id=case.get('case_id'), kind=case['kind'], gt_ids=case['gt_ids'], points=case['points'],
            desired_relation=desired, final_relation=result['final']['full'][i],
            final_desired=result['final']['full'][i] == desired,
            first_full_desired_stage=next((n for n in STAGE_ORDER if result[n]['full'][i] == desired), None),
            stage_sequence=[result[n]['full'][i] for n in STAGE_ORDER],
            known_stage_sequence=[result[n]['known_only'][i] for n in STAGE_ORDER]))
    return dict(order=list(STAGE_ORDER), stages=result, cases=rows,
        support='full processed newGT>0 including filled; unknown/frame censor in case selection',
        unit='fixed_endpoint_relation_not_independent_error', connectivity=8,
        scope='frozen-output intervention, not training, not an upper bound')
