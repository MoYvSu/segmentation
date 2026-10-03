# -*- coding: utf-8 -*-
"""可信原覆盖内部的固定响应带增益与外围/教师保护。

原完整八通道GT合同保持不变；额外项逐原生像素检查，不跨未知、filled
或第三GT。响应带只由固定原头的实际融合网格选位置，监督答案仍来自GT。
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real

import cv2
import numpy as np
import torch

from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS, _edge_slices
from utils.affinity_loss import balanced_affinity_loss, build_affinity_targets_torch
from utils.affinity_retain_loss import (balanced_affinity_retention_kl,
                                        build_reliable_retention_mask)
from utils.affinity_structure import interior_mask


def _sha(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(str(value.shape).encode('ascii'))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _counts(targets, mask):
    target = targets.detach().cpu().numpy() if torch.is_tensor(targets) else np.asarray(targets)
    selected = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    channels = []
    for channel in range(8):
        positive = int(np.count_nonzero(selected[0, channel] & (target[0, channel] == 1)))
        negative = int(np.count_nonzero(selected[0, channel] & (target[0, channel] == 0)))
        channels.append(dict(channel=channel, positive_edges=positive, negative_edges=negative,
                             selected_edges=positive+negative))
    return dict(selected_edges=int(np.count_nonzero(selected)),
                positive_edges=sum(r['positive_edges'] for r in channels),
                negative_edges=sum(r['negative_edges'] for r in channels),
                active_channels=sum(r['selected_edges'] > 0 for r in channels),
                per_channel=channels, mask_sha256=_sha(selected))


def _native_masks(tile, gt, original, ring):
    """全部八关系及目标4px内侧周圈；原生路径只允许两个端点GT。"""
    from tools.probe_affinity_bottleneck import project_native_support

    nodes, _, yy, xx = project_native_support(tile, gt, original)
    grid_gt = tile['grid_gt']
    height, width = grid_gt.shape
    trusted = np.zeros((1, 8, height, width), bool)
    outer = np.zeros_like(trusted)
    ring_nodes = nodes & ring[yy, xx]
    for channel, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        if height <= abs(dy) or width <= abs(dx):
            continue
        source, dest = _edge_slices(height, width, dy, dx)
        sy, sx = np.nonzero(nodes[source] & nodes[dest])
        sy, sx = sy+source[0].start, sx+source[1].start
        ty, tx = sy+dy, sx+dx
        if not len(sy):
            continue
        p0y, p0x, p1y, p1x = yy[sy, sx], xx[sy, sx], yy[ty, tx], xx[ty, tx]
        ga, gb = grid_gt[sy, sx], grid_gt[ty, tx]
        steps = np.maximum(np.abs(p1y-p0y), np.abs(p1x-p0x))
        legal = np.ones(len(sy), bool)
        for step in range(int(steps.max(initial=0))+1):
            fraction = np.minimum(step, steps)/np.maximum(1, steps)
            py = np.rint(p0y+fraction*(p1y-p0y)).astype(np.int32)
            px = np.rint(p0x+fraction*(p1x-p0x)).astype(np.int32)
            labels = gt[py, px]
            legal &= original[py, px] & ((labels == ga) | (labels == gb)) & (labels > 0)
        trusted[0, channel, sy[legal], sx[legal]] = True
        selected = legal & (ring_nodes[sy, sx] | ring_nodes[ty, tx])
        outer[0, channel, sy[selected], sx[selected]] = True
    return trusted, outer, nodes, yy, xx


@torch.no_grad()
def build_interior_supervision(tile, gt, original, teacher_logits, *, gid,
                               margin=4, low=.45, confidence=.9):
    """返回 ``(masks, audit)``；masks均在teacher设备，audit可直接写JSON。

    masks: targets/full_legal/interior/local/outer/retention/native_legal 为
    [1,8,H,W]；band为[H,W]。完整GT按既有nearest/valid规则构造，不另改。
    local仅四短程，至少一端落入固定原融合grid>=low的3x3膨胀内部带。
    outer含同/异GT八通道，至少一端在目标GT内部靠边margin原生像素周圈。
    """
    from tools.probe_affinity_interior import positive_relation_mask
    from tools.probe_affinity_bottleneck import project_native_support

    gt, original = np.asarray(gt), np.asarray(original)
    if (gt.ndim != 2 or not gt.size or not np.issubdtype(gt.dtype, np.integer)
            or gt.dtype == np.bool_ or np.any(gt < 0)):
        raise ValueError('GT must be nonempty nonnegative integer native HW')
    if (original.shape != gt.shape or original.dtype != np.bool_
            or np.any(original & (gt == 0))):
        raise ValueError('original must be bool original-covered known-GT native HW')
    if (isinstance(margin, (bool, np.bool_)) or not isinstance(margin, Integral)
            or margin < 0 or isinstance(gid, (bool, np.bool_))
            or not isinstance(gid, Integral) or gid <= 0):
        raise ValueError('gid and margin must be positive/nonnegative integers')
    if (isinstance(low, (bool, np.bool_)) or not isinstance(low, Real)
            or not math.isfinite(low) or not 0 <= low <= 1):
        raise ValueError('low must be a finite probability')
    grid_gt, valid = np.asarray(tile['grid_gt']), np.asarray(tile['valid'])
    if (grid_gt.ndim != 2 or not np.issubdtype(grid_gt.dtype, np.integer)
            or grid_gt.dtype == np.bool_ or np.any(grid_gt < 0)
            or valid.shape != grid_gt.shape or valid.dtype != np.bool_):
        raise ValueError('tile grid GT and bool valid must retain matching HW')
    box = tile['box']
    if (len(box) != 4 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in box)
            or not (0 <= box[0] < box[2] <= gt.shape[0] and 0 <= box[1] < box[3] <= gt.shape[1])):
        raise ValueError('tile box must be integer native bounds within GT')
    expected_shape = (1, 8, *grid_gt.shape)
    if (not torch.is_tensor(teacher_logits) or tuple(teacher_logits.shape) != expected_shape
            or teacher_logits.dtype != torch.float32 or not torch.isfinite(teacher_logits).all()):
        raise ValueError('teacher_logits must be fixed finite FP32 [1,8,H,W]')
    fused = np.asarray(tile['grid'])
    if (fused.shape != (1, 1, *grid_gt.shape) or fused.dtype != np.float32
            or not np.isfinite(fused).all() or np.any((fused < 0) | (fused > 1))):
        raise ValueError('tile grid must be captured actual FP32 fused [1,1,H,W] probability')
    # 不重新在CPU融合teacher，不根据更新后的student动态选带。
    response = fused[0, 0] >= low
    dilated = cv2.dilate(response.astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    proposal = interior_mask(gt, original, int(gid), margin=int(margin))
    interior_np, interior_audit = positive_relation_mask(
        tile, gt, original, channels='short', gid=int(gid), margin=int(margin))
    interior_nodes, _, _, _ = project_native_support(tile, gt, proposal['strict_mask'])
    band_np = dilated & interior_nodes & (grid_gt == gid)
    local_np = np.zeros_like(interior_np)
    for channel, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS[:4]):
        if grid_gt.shape[0] <= abs(dy) or grid_gt.shape[1] <= abs(dx):
            continue
        source, dest = _edge_slices(*grid_gt.shape, dy, dx)
        local_np[0, channel][source] = (interior_np[0, channel][source]
                                      & (band_np[source] | band_np[dest]))
    ring = (gt == gid) & ~proposal['mask'] & original
    native_np, outer_np, _, _, _ = _native_masks(tile, gt, original, ring)
    device = teacher_logits.device
    target, legal = build_affinity_targets_torch(
        torch.as_tensor(grid_gt.copy(), device=device, dtype=torch.long)[None],
        torch.as_tensor(valid.copy(), device=device)[None, None])
    masks = dict(targets=target, full_legal=legal,
                 interior=torch.as_tensor(interior_np.copy(), device=device),
                 local=torch.as_tensor(local_np.copy(), device=device),
                 outer=torch.as_tensor(outer_np.copy(), device=device),
                 native_legal=torch.as_tensor(native_np.copy(), device=device),
                 band=torch.as_tensor(band_np.copy(), device=device))
    retained, retention_selection = build_reliable_retention_mask(
        teacher_logits, target, legal, masks['band'], confidence=confidence,
        instance_map=grid_gt, valid_content=valid)
    masks['retention'] = retained & masks['native_legal']
    for name in ('interior', 'local', 'outer', 'native_legal', 'retention'):
        if torch.any(masks[name] & ~legal):
            raise AssertionError(name+' selected relations outside original complete GT legal mask')
    if (torch.any(masks['local'] & ~masks['interior']) or masks['local'][:, 4:].any()
            or masks['interior'][:, 4:].any() or torch.any(target[masks['local']] != 1)):
        raise AssertionError('Local increment must remain same-GT interior short4')
    audit = dict(format='affinity_interior_supervision_v1', gid=int(gid),
                 complete_GT_recipe='unchanged build_affinity_targets_torch(grid_GT,valid)',
                 masks={name:_counts(target, masks[name]) for name in
                        ('full_legal', 'interior', 'local', 'outer', 'native_legal', 'retention')},
                 interior=interior_audit,
                 band=dict(low=float(low), dilation_kernel=[3, 3],
                           captured_fused_grid_sha256=_sha(fused),
                           teacher_logits_sha256=_sha(teacher_logits),
                           raw_response_pixels=int(response.sum()), dilated_response_pixels=int(dilated.sum()),
                           trusted_interior_band_pixels=int(band_np.sum()), band_sha256=_sha(band_np),
                           fixed_selection=True, recomputed_teacher_fusion=False,
                           rule='actual captured fused grid>=low, 3x3 dilation, trusted interior nodes only; at least one endpoint'),
                 outer=dict(margin_native_pixels=int(margin), native_ring_pixels=int(ring.sum()),
                            native_ring_sha256=_sha(ring), rule='target GT inner perimeter ring; at least one endpoint',
                            complete_outer_closure_proven=False),
                 retention_selection=retention_selection,
                 retention_excluded_native_path_edges=int((retained & ~masks['native_legal']).sum()),
                 native_trust=dict(nearest_GT_equals_actual_interpolation_center=True,
                                   both_endpoints_and_all_native_path_original_covered=True,
                                   path_may_only_contain_two_endpoint_GT_ids=True,
                                   unknown_filled_third_GT_excluded=True),
                 extra_masks_subset_full_legal=True,
                 caveat='Fixed old response chooses positions only, never pseudo-labels; trustworthy processed GT mechanism, not physical boundary verification')
    return masks, audit


def interior_objective(logits, teacher_logits, supervision, *, local_weight,
                       loss_options=None):
    """共同full1+outer0.5+KL1；A local0，B local0.25。返回总损失/活梯度分项/统计。"""
    if isinstance(local_weight, bool) or local_weight not in (0., .25):
        raise ValueError('Matched experiment requires local_weight 0 or .25')
    options = {} if loss_options is None else dict(loss_options)
    target = supervision['targets']
    terms, statistics = {}, {}
    for name, key in (('full', 'full_legal'), ('outer', 'outer'), ('local', 'local')):
        terms[name], statistics[name] = balanced_affinity_loss(logits, target, supervision[key], **options)
    terms['retention'], statistics['retention'] = balanced_affinity_retention_kl(
        logits, teacher_logits, target, supervision['retention'])
    total = terms['full']+.5*terms['outer']+terms['retention']+float(local_weight)*terms['local']
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite interior objective')
    return total, terms, statistics
