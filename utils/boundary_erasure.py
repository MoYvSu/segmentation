# -*- coding: utf-8 -*-
"""只损伤已有支持短界面的在线擦断；不生成或修改任何监督目标。

资格由处理后GT、原标来源与冻结输入确定，不看待训练模型预测。全图只缓存
稀疏候选及必要的GT几何；每draw只构造裁窗大小的请求mask，不复制整张T8。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

from utils.boundary_core_supervision import build_trusted_cores


VERSION = 'native_supported_v1'
DEFAULTS = dict(enabled=True, version=VERSION, probability=.5, count_range=[1, 2],
                lengths=[16, 24, 32, 40], normal_half_width=8, max_attempts=64)


def erasure_settings(settings=None):
    """缺省关闭；关闭时不向旧数据身份或样本加入字段。"""
    if settings is None:
        return {'enabled': False}
    if not isinstance(settings, dict) or set(settings) - set(DEFAULTS):
        raise ValueError('erasure contains unsupported settings')
    enabled = settings.get('enabled', False)
    if type(enabled) is not bool:
        raise ValueError('erasure.enabled must be boolean')
    if not enabled:
        return {'enabled': False}
    cfg = {**DEFAULTS, **settings}
    if cfg['version'] != VERSION:
        raise ValueError('Unsupported erasure.version')
    p = cfg['probability']
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('erasure.probability must lie in [0,1]')
    for name in ('normal_half_width', 'max_attempts'):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError('erasure.' + name + ' must be a positive integer')
    counts = cfg['count_range']
    if (not isinstance(counts, (tuple, list)) or len(counts) != 2
            or any(type(v) is not int or v < 1 for v in counts) or counts[0] > counts[1]):
        raise ValueError('erasure.count_range must be an ascending positive pair')
    lengths = cfg['lengths']
    if (not isinstance(lengths, (tuple, list)) or not lengths
            or any(type(v) is not int or v < 8 for v in lengths) or len(set(lengths)) != len(lengths)):
        raise ValueError('erasure.lengths must be unique native integer lengths >=8')
    # 这一版有意只支持穿过ellipse8的固定法向宽度，不偷偷变成另一项扫参。
    if cfg['normal_half_width'] != 8:
        raise ValueError('native_supported_v1 requires normal_half_width=8')
    cfg.update(probability=float(p), count_range=list(counts), lengths=list(lengths))
    return cfg


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def draw_seed(seed, epoch, draw, source):
    """独立随机流；既不消费全局numpy/torch流，也不消费源窗/D4的计划流。"""
    payload = [VERSION, int(seed), int(epoch), int(draw), str(source)]
    return int.from_bytes(hashlib.sha256(json.dumps(payload).encode()).digest()[:8], 'little')


def prepare_erasure_source(gt, original_covered, target, valid, input_skeleton):
    """一次性准备全图候选。core只佐证ID来源，不把腐蚀后的core当界面。"""
    gt, original = np.asarray(gt), np.asarray(original_covered, dtype=bool)
    shape = gt.shape
    arrays = [np.asarray(a, dtype=bool) for a in (target, valid, input_skeleton)]
    if (gt.ndim != 2 or gt.dtype.kind not in 'iu' or not gt.size or int(gt.max()) > 65535
            or original.shape != shape or any(a.shape != shape for a in arrays)
            or np.any(original & (gt == 0))):
        raise ValueError('Erasure needs matching native processed GT/provenance/boolean fields')
    target, valid, input_skeleton = arrays
    cores = build_trusted_cores(gt, original, radius=4, min_component_pixels=64)
    core_ids = sorted(row['gt_id'] for row in cores['gt_stats'] if row['core_pixels'])
    skeleton = np.ascontiguousarray(skeletonize(target), dtype=bool)
    neighbors = np.ones((3, 3), np.int16)
    neighbors[1, 1] = 0
    degree = cv2.filter2D(skeleton.astype(np.int16), cv2.CV_16S, neighbors,
                          borderType=cv2.BORDER_CONSTANT)
    supported = cv2.dilate(input_skeleton.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    safe = cv2.erode(valid.astype(np.uint8), np.ones((17, 17), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    simple = skeleton & (degree == 2)
    candidates = np.ascontiguousarray(np.argwhere(simple & safe & supported), dtype=np.int32)
    summary = dict(format='boundary_erasure_source_v1', shape=list(shape),
        target_positive_points=int((target & valid).sum()), skeleton_points=int(skeleton.sum()),
        simple_points=int(simple.sum()), full_mask_safe_points=int((simple & safe).sum()),
        supported_candidate_count=len(candidates), candidates_sha256=array_sha(candidates),
        gt_skeleton_sha256=array_sha(skeleton),
        target_sha256=array_sha(target), valid_sha256=array_sha(valid), gt_sha256=array_sha(gt),
        original_covered_sha256=array_sha(original), input_skeleton_sha256=array_sha(input_skeleton),
        trusted_core_summary=cores['summary'], trusted_core_ids=core_ids,
        candidate_definition='GT skeleton degree2; entire normal8 neighborhood valid; original N1 skeleton radius2 support',
        provenance_policy='A: whole erasure context original-covered; B: two adjacent processed IDs each have original radius4 core >=64px',
        filled_is_exact_truth=False, model_quality_filter=False)
    summary['qualification_sha256'] = _canonical_sha(summary)
    return dict(gt=gt, original=original, valid=valid, skeleton=skeleton,
                input_skeleton=input_skeleton, candidates=candidates,
                core_ids=frozenset(core_ids), summary=summary)


def _neighbors(skeleton, point):
    y, x = map(int, point)
    y0, x0 = max(0, y-1), max(0, x-1)
    points = np.argwhere(skeleton[y0:y+2, x0:x+2]) + [y0, x0]
    return [tuple(map(int, p)) for p in points if tuple(p) != (y, x)]


def _trace(skeleton, center, reach):
    """只接受两侧无分叉简单链；曲率过大随后拒绝，不臆造直线GT。"""
    adjacent = _neighbors(skeleton, center)
    if len(adjacent) != 2:
        return None
    halves = []
    for first in adjacent:
        chain, previous, current = [], tuple(center), first
        distance = 0.
        while distance < reach:
            if current == tuple(center) or current in chain:
                return None
            distance += math.dist(previous, current)
            chain.append(current)
            options = _neighbors(skeleton, current)
            if len(options) != 2:
                return None
            options = [p for p in options if p != previous]
            if len(options) != 1:
                return None
            previous, current = current, options[0]
        halves.append(chain)
    return np.asarray(halves[0][::-1] + [tuple(center)] + halves[1], np.int32)


def _proposal(prepared, boundary, center, length, box, crop_margin, occupied):
    """在小ROI定义原生矩形请求mask；只返回稀疏局部数据。"""
    half, width, end_guard = length / 2., 8, 6
    chain = _trace(prepared['skeleton'], center, half + end_guard)
    if chain is None:
        return None, 'branch_or_endpoint'
    y, x = map(int, center)
    delta = chain.astype(np.float64) - [y, x]
    _, vectors = np.linalg.eigh(delta.T @ delta)
    tangent = vectors[:, -1]
    normal = np.array([-tangent[1], tangent[0]])
    longitudinal, transverse = delta @ tangent, delta @ normal
    if np.max(np.abs(transverse)) > 1.5 or longitudinal.min() > -half-end_guard+2 or longitudinal.max() < half+end_guard-2:
        return None, 'curved_or_short'
    radius = int(math.ceil(half + width + end_guard + 3))
    y0, y1 = max(0, y-radius), min(boundary.shape[0], y+radius+1)
    x0, x1 = max(0, x-radius), min(boundary.shape[1], x+radius+1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    along = (yy-y)*tangent[0] + (xx-x)*tangent[1]
    across = (yy-y)*normal[0] + (xx-x)*normal[1]
    mask = (np.abs(along) <= half) & (np.abs(across) <= width)
    context = (np.abs(along) <= half+end_guard) & (np.abs(across) <= width)
    top, left, height, breadth = map(int, box)
    allowed = ((yy >= top+crop_margin) & (yy < top+height-crop_margin)
               & (xx >= left+crop_margin) & (xx < left+breadth-crop_margin))
    if np.any(context & (~allowed | ~prepared['valid'][y0:y1, x0:x1])):
        return None, 'unknown_frame_crop_rim_or_padding'
    ids = np.unique(prepared['gt'][y0:y1, x0:x1][context])
    if len(ids) != 2 or np.any(ids == 0):
        return None, 'third_interface_or_unknown'
    # GT和原N1骨架都不能在法向两侧另有一条线/支路。矩形穿过T8整带，
    # 而不是只擦掉GT中心若干像素、留下可绕行的厚边肩部。
    for name in ('skeleton', 'input_skeleton'):
        local = prepared[name][y0:y1, x0:x1]
        if np.any(local & context & (np.abs(across) > 2.)):
            return None, 'other_GT_or_input_skeleton_in_corridor'
    raw_skeleton = prepared['input_skeleton'][y0:y1, x0:x1]
    supported = cv2.dilate(raw_skeleton.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    cy, cx = chain[:, 0]-y0, chain[:, 1]-x0
    if not supported[cy, cx].all():
        return None, 'existing_gap_or_unsupported_endpoint'
    original = prepared['original'][y0:y1, x0:x1]
    provenance = 'A' if original[context].all() else 'B'
    if provenance == 'B' and not set(map(int, ids)) <= prepared['core_ids']:
        return None, 'missing_two_original_trusted_cores'
    my, mx = np.where(mask)
    ey, ex = np.where(context)
    if occupied[ey+y0-top, ex+x0-left].any():
        return None, 'overlapping_erasure'
    deleted = mask & boundary[y0:y1, x0:x1]
    if not deleted.any():
        return None, 'no_input_pixels'
    # 未擦两端必须实际有T8支撑。此处只检查输入，不用G结果筛选。
    for sign in (-1, 1):
        endpoint = context & (sign*along >= half+2)
        if not np.any(boundary[y0:y1, x0:x1] & endpoint):
            return None, 'unsupported_endpoint'
    record = dict(provenance=provenance, center_native_yx=[y, x],
        length_native=int(length), normal_half_width_native=width,
        tangent_native_yx=tangent.tolist(), native_bbox_yxyx=[y0, x0, y1, x1],
        gt_ids=list(map(int, ids)), chain_native_yx=chain.tolist(),
        requested_mask_pixels=int(mask.sum()), deleted_input_pixels=int(deleted.sum()),
        support_reference='original whole-source N1 skeleton, radius2 L-infinity',
        whole_T8_after_erasure_renormalized=False)
    return (my+y0-top, mx+x0-left, ey+y0-top, ex+x0-left, record), None


def make_erasure_mask(prepared, boundary, *, box, crop_margin, settings, seed):
    """返回未旋转原生crop请求mask与漏斗；缺资格原样返回，绝不换源。"""
    cfg = erasure_settings(settings)
    if not cfg['enabled']:
        raise ValueError('make_erasure_mask requires explicitly enabled erasure')
    top, left, height, width = map(int, box)
    if (min(top, left) < 0 or min(height, width) < 1 or
            top+height > boundary.shape[0] or left+width > boundary.shape[1]):
        raise ValueError('Erasure crop lies outside native source')
    mask = np.zeros((height, width), bool)
    occupied = np.zeros_like(mask)
    rng = np.random.default_rng(int(seed))
    requested = bool(rng.random() < cfg['probability'])
    metadata = dict(version=VERSION, seed=int(seed), requested=requested, applied=False,
        reason='probability_skip', requested_count=0, applied_count=0,
        deleted_input_pixels=0, artificial_gap_gt_points=0, segments=[], funnel={},
        mask_definition='requested native rectangle; actual deleted = clean_T8 & erasure_mask',
        coordinate_scope='whole native source, then crop/D4; mask never enters padding',
        input_only=True, target_and_valid_unchanged=True)
    if not requested:
        return mask, metadata
    metadata['requested_count'] = int(rng.integers(cfg['count_range'][0], cfg['count_range'][1]+1))
    points = prepared['candidates']
    inside = ((points[:, 0] >= top+crop_margin) & (points[:, 0] < top+height-crop_margin)
              & (points[:, 1] >= left+crop_margin) & (points[:, 1] < left+width-crop_margin))
    candidates = points[inside]
    funnel = Counter(source_candidates=len(points), crop_candidates=len(candidates))
    if len(candidates):
        for _ in range(cfg['max_attempts']):
            funnel['attempts'] += 1
            point = candidates[int(rng.integers(len(candidates)))]
            length = cfg['lengths'][int(rng.integers(len(cfg['lengths'])))]
            proposal, reason = _proposal(prepared, boundary, point, length, box, crop_margin, occupied)
            if proposal is None:
                funnel[reason] += 1
                continue
            yy, xx, ey, ex, record = proposal
            mask[yy, xx] = True
            occupied[ey, ex] = True
            metadata['segments'].append(record)
            funnel['accepted'] += 1
            funnel['accepted_'+record['provenance']] += 1
            if len(metadata['segments']) == metadata['requested_count']:
                break
    metadata.update(applied=bool(mask.any()), applied_count=len(metadata['segments']),
        reason='applied' if mask.any() else ('no_crop_candidates' if not len(candidates) else 'attempts_exhausted'),
        deleted_input_pixels=int((mask & boundary[top:top+height, left:left+width]).sum()),
        funnel=dict(funnel))
    return mask, metadata


def transform_erasure_mask(mask, crop_size, rotation=0, flip=False):
    """同步D4但用零填充，确保padding不是人工缺口或输入损伤。"""
    view = np.rot90(mask, int(rotation) % 4)
    if flip:
        view = view[:, ::-1]
    if max(view.shape) > crop_size:
        raise ValueError('Erasure mask exceeds crop size')
    return np.pad(np.ascontiguousarray(view, dtype=bool),
                  ((0, crop_size-view.shape[0]), (0, crop_size-view.shape[1])), constant_values=False)
