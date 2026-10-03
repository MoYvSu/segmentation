# -*- coding: utf-8 -*-
"""外围负关系保留的processed-known支持扩展；独立冻结诊断，不改旧损失。"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
import torch

from utils.affinity_interior_loss import _native_masks
from utils.affinity_retain_loss import build_reliable_retention_mask
from utils.affinity_spatial_trace import native_grid_footprint


def array_sha(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(str(value.shape).encode('ascii'))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def processed_native_support(tile, gt, original):
    """返回(original-path, known-path)两个numpy bool[1,8,H,W]及审计。

    原坐标、最近邻GT与实际插值中心一致条件、整条原生采样路径和端点GT
    约束全部复用旧_native_masks；唯一替换是支持域original→GT>0。
    不声称filled等于原标真实边界，只检查processed-known路径机制。
    """
    gt, original = np.asarray(gt), np.asarray(original)
    if (gt.ndim != 2 or not gt.size or not np.issubdtype(gt.dtype, np.integer)
            or gt.dtype == np.bool_ or np.any(gt < 0)):
        raise ValueError('gt must be a nonempty nonnegative integer HW map')
    if original.dtype != np.bool_ or original.shape != gt.shape or np.any(original & (gt == 0)):
        raise ValueError('original must be bool known-GT coverage matching native GT')
    # 空足迹只核对冻结tile geometry/box/valid，不进行任何插值autograd。
    native_grid_footprint(tile, np.zeros(gt.shape, bool))
    y, x, y1, x1 = tile['box']
    from utils.offset_letterbox import letterbox_instance_geometry
    expected, _, _ = letterbox_instance_geometry(
        gt[y:y1, x:x1], tile['geometry']['input_size'], tile['geometry']['output_grid'])
    if not np.array_equal(expected, tile['grid_gt']):
        raise ValueError('tile grid_GT differs from actual nearest native GT letterbox')
    ring = np.zeros(gt.shape, bool)
    original_path, _, _, _, _ = _native_masks(tile, gt, original, ring)
    known_path, _, _, _, _ = _native_masks(tile, gt, gt > 0, ring)
    if np.any(original_path & ~known_path):
        raise AssertionError('original-supported paths must be a subset of processed-known paths')
    audit = dict(format='processed_native_support_v1',
                 original_path_edges=int(original_path.sum()), known_path_edges=int(known_path.sum()),
                 extra_known_path_edges=int((known_path & ~original_path).sum()),
                 original_path_sha256=array_sha(original_path), known_path_sha256=array_sha(known_path),
                 original_subset_known=True, unknown_padding_third_GT_paths_excluded=True,
                 nearest_GT_equals_actual_interpolation_center=True,
                 path_sampling='unchanged existing _native_masks',
                 only_changed_native_support='original_covered -> processed_GT>0',
                 filled_support_is_processed_GT_mechanism_only=True)
    return original_path, known_path, audit


@torch.no_grad()
def build_processed_negative_retention(tile, gt, original, teacher_logits, supervision, *, confidence=.9):
    """返回 ``(masks, audit)``，mask为teacher设备上的冻结bool张量。

    masks含old、added、expanded。old必须逐值复现既有retention；added
    仅GT负关系、教师负置信严格>confidence、双方在原固定band外、grid
    与完整native采样路径known且无第三GT/padding，并确有非original支持。
    expanded=old|added保留全部旧正/负关系。没有训练或KL权重/归一化改动。
    """
    if (isinstance(confidence, bool) or not isinstance(confidence, Real)
            or not math.isfinite(confidence) or not .5 < confidence < 1):
        raise ValueError('confidence must be finite and within (.5,1)')
    grid_shape = tuple(np.asarray(tile['grid_gt']).shape)
    shape = (1, 8, *grid_shape)
    if (not torch.is_tensor(teacher_logits) or teacher_logits.shape != shape
            or teacher_logits.dtype != torch.float32 or not torch.isfinite(teacher_logits).all()
            or teacher_logits.requires_grad or teacher_logits.grad is not None):
        raise ValueError('teacher_logits must be frozen finite FP32 [1,8,H,W]')
    if not isinstance(supervision, Mapping):
        raise ValueError('supervision must be the unchanged frozen original dictionary')
    for name in ('targets', 'full_legal', 'band', 'retention', 'native_legal'):
        if name not in supervision or not torch.is_tensor(supervision[name]):
            raise ValueError('missing original supervision tensor: '+name)
        value = supervision[name]
        expected_shape = grid_shape if name == 'band' else shape
        if value.shape != expected_shape or value.device != teacher_logits.device or value.requires_grad:
            raise ValueError('original supervision shape/device/frozen contract differs: '+name)
        if name != 'targets' and value.dtype != torch.bool:
            raise ValueError('original supervision mask must be bool: '+name)
    targets, legal = supervision['targets'], supervision['full_legal']
    original_path, known_path, path_audit = processed_native_support(tile, gt, original)
    original_tensor = torch.as_tensor(original_path, device=teacher_logits.device)
    known_tensor = torch.as_tensor(known_path, device=teacher_logits.device)
    if not torch.equal(original_tensor, supervision['native_legal']):
        raise ValueError('supplied original native_legal differs from the frozen path contract')
    retained, retention_audit = build_reliable_retention_mask(
        teacher_logits, targets, legal, supervision['band'], confidence=confidence,
        instance_map=tile['grid_gt'], valid_content=tile['valid'])
    old = retained & original_tensor
    if not torch.equal(old, supervision['retention']):
        raise ValueError('supplied old retention is not exact under the frozen teacher/band/GT contract')
    probability = teacher_logits.detach().sigmoid()
    strict_negative = (targets == 0) & (probability < 1.-float(confidence))
    added = retained & strict_negative & known_tensor & ~original_tensor & ~old
    expanded = old | added
    if (torch.any(added & (targets != 0)) or torch.any(added & ~legal)
            or torch.any(added & ~known_tensor) or torch.any(old & ~expanded)):
        raise AssertionError('processed-known retention expansion violated its frozen negative-only contract')
    per_channel = [dict(channel=c, old_edges=int(old[0, c].sum()),
                        added_negative_edges=int(added[0, c].sum()),
                        expanded_edges=int(expanded[0, c].sum())) for c in range(8)]
    audit = dict(format='processed_negative_retention_v1', confidence=float(confidence),
                 old_edges=int(old.sum()), added_negative_edges=int(added.sum()),
                 expanded_edges=int(expanded.sum()), per_channel=per_channel,
                 old_sha256=array_sha(old), added_sha256=array_sha(added),
                 expanded_sha256=array_sha(expanded), teacher_sha256=array_sha(teacher_logits),
                 band_sha256=array_sha(supervision['band']), path=path_audit,
                 original_reliable_selection=retention_audit,
                 old_retention_exact=True, old_subset_expanded=True, newly_added_negative_only=True,
                 added_teacher_confidence_rule='1-sigmoid(logit)>confidence (strict)',
                 fixed_band_preserved=True, whole_grid_and_native_paths_checked=True,
                 unknown_padding_third_GT_excluded=True, no_model_forward=True,
                 no_training=True, no_optimizer=True, input_mutated=False,
                 caveat='Filled path support follows processed GT; mask eligibility alone does not prove physical boundary or training gain')
    return dict(old=old.clone(), added=added, expanded=expanded), audit
