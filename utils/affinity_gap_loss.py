# -*- coding: utf-8 -*-
"""连续affinity边界的静态种子完整分隔；不生成新GT或改变部署读出。"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from skimage.morphology import skeletonize

from utils.boundary_core_supervision import build_trusted_cores
from utils.boundary_region_cut import select_region_cut

FORMAT = 'affinity_gap_cut_v1'
ARRAY_KEYS = ('ids', 'original', 'valid', 'core_ids', 'seed_ids', 'allowed')
DEFAULTS = dict(core_radius=4, min_core_pixels=64, gt_margin=12., input_margin=4.,
                min_atom_pixels=64, threshold=.65, target_margin=.70,
                max_pair_attempts=4, capacity_scale=1000, native_roi_size=256,
                seed_policy='legacy_skeleton', seed_low_threshold=.45)


def settings(values=None):
    values = {} if values is None else dict(values)
    if set(values) - set(DEFAULTS):
        raise ValueError('Unknown affinity gap settings')
    out = {**DEFAULTS, **values}
    for key in ('core_radius', 'min_core_pixels', 'min_atom_pixels', 'max_pair_attempts',
                'capacity_scale', 'native_roi_size'):
        if type(out[key]) is not int or out[key] < (0 if key == 'core_radius' else 1):
            raise ValueError(key + ' requires an integer')
    for key in ('gt_margin', 'input_margin', 'threshold', 'target_margin', 'seed_low_threshold'):
        if isinstance(out[key], bool) or not math.isfinite(float(out[key])):
            raise ValueError(key + ' requires a finite number')
        out[key] = float(out[key])
    if not 0 < out['threshold'] < out['target_margin'] < 1 or min(out['gt_margin'], out['input_margin']) < 0:
        raise ValueError('Invalid margins')
    if out['seed_policy'] not in ('legacy_skeleton', 'continuous_low'):
        raise ValueError('Unknown seed_policy')
    if not 0 <= out['seed_low_threshold'] <= 1:
        raise ValueError('seed_low_threshold must lie in [0,1]')
    return out


def distance_to(mask):
    mask = np.asarray(mask, bool)
    return (cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            if mask.any() else np.full(mask.shape, np.inf, np.float32))


def native_gap_geometry(ids, original, target, valid, reference_boundary, values=None):
    """在全原生图建立静态种子；必须在裁窗/投影/D4之前调用。

    target/valid由原load_processed_binary生成。reference_boundary只能是固定N1，
    不得用当前候选输出重选种子。filled不能生成core或可训练割，unknown保持开放节点。
    legacy_skeleton保留v1的dS>4资格；continuous_low改为固定B<=seed_low_threshold，
    两者共享原标core、dGT和valid，后者不再要求距单像素骨架超过4px。
    """
    cfg = settings(values)
    ids = np.asarray(ids)
    original, target, valid = [np.asarray(x, bool) for x in (original, target, valid)]
    p = np.asarray(reference_boundary)
    if (ids.ndim != 2 or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids > 65535)
            or any(x.shape != ids.shape for x in (original, target, valid, p))
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or np.any(original & (ids == 0)) or np.any(valid & (ids == 0))):
        raise ValueError('Invalid native geometry or N1 probability')
    core = build_trusted_cores(ids, original, radius=cfg['core_radius'],
                              min_component_pixels=cfg['min_core_pixels'])['core_map']
    skeleton = skeletonize(p >= .5)
    dg = distance_to(target)
    shared_eligible = (core > 0) & valid & (dg > cfg['gt_margin'])
    if cfg['seed_policy'] == 'legacy_skeleton':
        eligible = shared_eligible & (distance_to(skeleton) > cfg['input_margin'])
    else:
        # 只在固定原N1上建立一次。当前候选输出不能裁掉静态seed中的高点。
        eligible = shared_eligible & (p <= cfg['seed_low_threshold'])
    allowed = original & valid & (dg <= cfg['gt_margin'])
    out = dict(ids=np.ascontiguousarray(ids, dtype=np.int64), original=original,
               valid=valid, core_ids=np.ascontiguousarray(core, dtype=np.int64),
               seed_ids=np.where(eligible, core, 0).astype(np.int64), allowed=allowed,
               scale=1., seed_policy=cfg['seed_policy'], seed_low_threshold=cfg['seed_low_threshold'],
               reference_boundary_sha256=hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest(),
               reference_skeleton_sha256=hashlib.sha256(skeleton.tobytes()).hexdigest())
    if np.any((out['seed_ids'] > 0) & allowed):
        raise RuntimeError('Static gap seeds overlap the editable corridor')
    return out


def _check_geometry(probability, geometry):
    p = np.asarray(probability)
    if p.ndim != 2 or not p.size or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Boundary must be finite HW probability')
    out = {}
    for key in ARRAY_KEYS:
        a = np.asarray(geometry[key])
        if a.shape != p.shape:
            raise ValueError('Geometry and boundary shapes differ: ' + key)
        if key in ('ids', 'core_ids', 'seed_ids'):
            if a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a > 65535):
                raise ValueError('Invalid integer geometry: ' + key)
            out[key] = a.astype(np.int64, copy=False)
        else:
            if not np.isin(a, [0, 1]).all():
                raise ValueError('Binary geometry required: ' + key)
            out[key] = a.astype(bool)
    if (np.any(out['allowed'] & ~(out['original'] & out['valid']))
            or np.any((out['seed_ids'] > 0) & ~out['valid'])
            or np.any((out['seed_ids'] > 0) & (out['seed_ids'] != out['ids']))
            or np.any((out['seed_ids'] > 0) & out['allowed'])):
        raise ValueError('Unknown/filled/seed geometry contract violated')
    return p.astype(np.float32, copy=False), out


def adjacent_pairs(ids, original, valid):
    pairs = set()
    for a, b, known in ((ids[1:], ids[:-1], original[1:] & original[:-1] & valid[1:] & valid[:-1]),
                        (ids[:, 1:], ids[:, :-1], original[:, 1:] & original[:, :-1] & valid[:, 1:] & valid[:, :-1])):
        selected = known & (a > 0) & (b > 0) & (a != b)
        if selected.any():
            pairs.update(tuple(map(int, row)) for row in
                         np.unique(np.sort(np.stack([a[selected], b[selected]], 1), axis=1), axis=0))
    return sorted(pairs)


def _atoms(seed_ids, minimum):
    result = []
    for gid in sorted(map(int, np.unique(seed_ids))):
        if gid == 0:
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats((seed_ids == gid).astype(np.uint8), connectivity=8)
        local = []
        for component in range(1, count):
            pixels = int(stats[component, cv2.CC_STAT_AREA])
            if pixels >= minimum:
                local.append(dict(gt_id=gid, component=component, pixels=pixels,
                                  mask=labels == component))
        result.extend(sorted(local, key=lambda a: (-a['pixels'], a['component'])))
    return result


def _certificate(p, cut, source, sink, threshold):
    wall = (p >= threshold) | cut
    if np.any(wall & (source | sink)):
        return False
    _, labels = cv2.connectedComponents((~wall).astype(np.uint8), connectivity=8)
    left, right = set(map(int, np.unique(labels[source]))) - {0}, set(map(int, np.unique(labels[sink]))) - {0}
    outside = set(map(int, np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))) - {0}
    return bool(left and right and not (left & right)
                and not (left & outside and right & outside))


def select_gap_cuts(probability, geometry, values=None, *, seed=0, gt_pair=None):
    """输入同一ROI，输出最多一对完整cut及静态seed；选取停止梯度。

    结果cut/source_seed/sink_seed都是与输入同shape的bool数组；audit不含大数组。
    外部节点始终开启。当前墙挡住任一静态seed点会拒绝，绝不删seed改善证书。
    """
    started = time.perf_counter(); cfg = settings(values)
    if (geometry.get('seed_policy', cfg['seed_policy']) != cfg['seed_policy']
            or geometry.get('seed_low_threshold', cfg['seed_low_threshold']) != cfg['seed_low_threshold']):
        raise ValueError('Static seed policy metadata differs from selector settings')
    p, g = _check_geometry(probability, geometry)
    empty = np.zeros(p.shape, bool)
    atoms = _atoms(g['seed_ids'], cfg['min_atom_pixels'])
    adjacent = set(adjacent_pairs(g['ids'], g['original'], g['valid']))
    if gt_pair is not None:
        pair = tuple(sorted(map(int, gt_pair)))
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError('Different GT pair required')
        adjacent &= {pair}
    candidates = [(i, j) for i in range(len(atoms)) for j in range(i + 1, len(atoms))
                  if tuple(sorted((atoms[i]['gt_id'], atoms[j]['gt_id']))) in adjacent]
    rng = np.random.default_rng(int(seed))
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    reasons, attempts = Counter(), []
    result = dict(cut=empty.copy(), source_seed=empty.copy(), sink_seed=empty.copy())
    selected_pair = None
    for first, second in candidates[:cfg['max_pair_attempts']]:
        a, b = atoms[first], atoms[second]
        row = dict(gt_pair=sorted([a['gt_id'], b['gt_id']]), source_component=a['component'],
                   sink_component=b['component'], source_pixels=a['pixels'], sink_pixels=b['pixels'])
        if ((p[a['mask']] >= cfg['threshold']).any() or (p[b['mask']] >= cfg['threshold']).any()):
            row['reason'] = 'static_seed_blocked_by_current_wall'; reasons[row['reason']] += 1
            attempts.append(row); continue
        cut, audit = select_region_cut(p, g['allowed'], a['mask'], b['mask'],
            threshold=cfg['threshold'], capacity_scale=cfg['capacity_scale'], exterior=True)
        row.update(selector=audit, reason=audit['reason'])
        valid = bool(audit['certificate_valid']) and _certificate(p, cut, a['mask'], b['mask'], cfg['threshold'])
        row['independent_certificate'] = valid
        attempts.append(row); reasons[row['reason']] += 1
        if not cut.any() or not valid:
            continue
        result.update(cut=cut, source_seed=a['mask'], sink_seed=b['mask'])
        selected_pair = row['gt_pair']; break
    if not candidates:
        reasons['no_adjacent_static_atom_pair'] += 1
    result['audit'] = dict(format=FORMAT, settings=cfg, eligible_pairs=len(candidates),
        atom_count=len(atoms), attempts=attempts, reasons=dict(reasons),
        selected_pairs=int(result['cut'].any()), selected_pair=selected_pair,
        cut_pixels=int(result['cut'].sum()), exterior=True, selection_stopped_gradient=True,
        static_seed_source='fixed_original_N1', seed_policy=cfg['seed_policy'],
        seed_low_threshold=cfg['seed_low_threshold'],
        seed_low_threshold_used=cfg['seed_policy']=='continuous_low',
        static_seed_metadata_verified='seed_policy' in geometry and 'seed_low_threshold' in geometry,
        elapsed_seconds=time.perf_counter()-started)
    return result


def cut_margin_loss(boundary, cut, *, margin=.70):
    """只补至margin，达到即停；空割保持可微零，不奖励泛亮。"""
    if not torch.is_tensor(boundary) or boundary.ndim != 2 or not boundary.is_floating_point():
        raise ValueError('Floating HW boundary required')
    mask = torch.as_tensor(cut, dtype=torch.bool, device=boundary.device)
    if mask.shape != boundary.shape or not 0 < float(margin) < 1:
        raise ValueError('Cut shape or margin invalid')
    return F.relu(float(margin) - boundary[mask]).mean() if bool(mask.any()) else boundary.sum() * 0.


def reference_roi(reference_boundary, geometry, values=None):
    """只由固定N1/GT确定一个弱界ROI，不用当前候选重新挑简单病例。"""
    cfg = settings(values)
    ref, g = _check_geometry(reference_boundary, geometry)
    scale = float(geometry.get('scale', 1.))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive native-to-view scale required')
    side = max(8, int(round(cfg['native_roi_size'] * scale)))
    h, w = ref.shape; rh, rw = min(side, h), min(side, w)
    candidates = []
    seed_ids = set(map(int, np.unique(g['seed_ids']))) - {0}
    for first, second in adjacent_pairs(g['ids'], g['original'], g['valid']):
        if not {first, second}.issubset(seed_ids):
            continue
        a = g['ids'] == first; b = g['ids'] == second
        contact = (a & (cv2.dilate(b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
                   & g['valid'] & g['original'])
        yy, xx = np.nonzero(contact)
        if not len(yy):
            continue
        index = int(np.argmin(ref[yy, xx]))
        cy, cx = int(yy[index]), int(xx[index])
        y, x = max(0, min(h-rh, cy-rh//2)), max(0, min(w-rw, cx-rw//2))
        candidates.append((float(ref[cy, cx]), first, second, (y, x, y+rh, x+rw)))
    if not candidates:
        return None
    _, first, second, box = min(candidates)
    return dict(box=box, gt_pair=(first, second), scale=scale)


def continuous_gap_loss(boundary, geometry, reference_boundary, values=None, *, seed=0, enabled=True):
    """训练视图内的连续B辅助项；最终整图拼接分隔仍需独立验收。"""
    cfg = settings(values)
    if not torch.is_tensor(boundary) or boundary.ndim != 2:
        raise ValueError('A single floating HW view is required')
    if not enabled:
        return boundary.sum()*0., dict(enabled=False, selected_pairs=0, cut_pixels=0,
            seed_policy=cfg['seed_policy'],seed_low_threshold=cfg['seed_low_threshold']), np.zeros(tuple(boundary.shape), bool)
    p = boundary.detach().float().cpu().numpy()
    _check_geometry(p, geometry)
    roi = reference_roi(reference_boundary, geometry, cfg)
    full = np.zeros(p.shape, bool)
    if roi is None:
        return boundary.sum()*0., dict(enabled=True, selected_pairs=0, cut_pixels=0,
            seed_policy=cfg['seed_policy'],seed_low_threshold=cfg['seed_low_threshold'],
            reasons={'no_reference_adjacent_seed_pair':1}), full
    y,x,y1,x1 = roi['box']; sl = np.s_[y:y1,x:x1]
    local = {key:np.asarray(geometry[key])[sl] for key in ARRAY_KEYS}
    for key in ('seed_policy', 'seed_low_threshold'):
        if key in geometry:
            local[key] = geometry[key]
    converted = {**cfg, 'min_atom_pixels': max(1, int(math.ceil(cfg['min_atom_pixels'] * roi['scale']**2)))}
    selected = select_gap_cuts(p[sl], local, converted, seed=seed, gt_pair=roi['gt_pair'])
    full[sl] = selected['cut']
    loss = cut_margin_loss(boundary, full, margin=cfg['target_margin'])
    audit = {**selected['audit'], 'enabled':True, 'view_roi':list(roi['box']),
             'native_roi_size':cfg['native_roi_size'], 'native_to_view_scale':roi['scale'],
             'full_view_certificate_claimed':False, 'loss':float(loss.detach())}
    return loss, audit, full
