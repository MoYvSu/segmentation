# -*- coding: utf-8 -*-
"""冻结结构反事实的支持因素与既有local监督筛选。

这里只做集合选择，不改GT、不重定义原覆盖可信路径、不改变任何捕获张量。
原生插值足迹复用现有实际标量算子；含filled的完整界面仅作乐观机制对照。
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import torch

from utils.affinity_spatial_trace import edge_roi_mask, native_grid_footprint
from utils.affinity_structure import interface_mask, mask_provenance


def _sha(value):
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(str(value.shape).encode('ascii'))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def interface_support_factor(gt, original, pair, radius=2):
    """返回 ``(masks, audit)``，四个原生bool HW mask。

    strict复用双方接触端点原覆盖的界面支持；extra_original是完整界面内
    尚未被strict选中的原覆盖像素；filled是完整界面内所有非原覆盖像素。
    strict_plus_original=full&original；strict_plus_filled=strict|filled；
    completed=full。两种增量严格互斥，两增量合并才等于完整界面。
    原覆盖增量也不声称可信训练可行性；这里只拆解已有界面的机制支持。
    """
    proposal = interface_mask(gt, original, pair, radius=radius)
    gt, original = np.asarray(gt), np.asarray(original)
    full, strict = proposal['mask'], proposal['strict_mask']
    extra_original = full & original & ~strict
    filled = full & ~original
    masks = dict(strict=strict.copy(), strict_plus_original=full & original,
                 strict_plus_filled=strict | filled, completed=full.copy())
    checks = dict(
        strict_subset_completed=not np.any(strict & ~full),
        strict_original_only=not np.any(strict & ~original),
        extra_original_and_filled_disjoint=not np.any(extra_original & filled),
        all_three_factors_disjoint=not np.any((strict & extra_original)
                                            | (strict & filled) | (extra_original & filled)),
        strict_plus_original_exact=np.array_equal(masks['strict_plus_original'],
                                                 strict | extra_original),
        strict_plus_filled_exact=np.array_equal(masks['strict_plus_filled'], strict | filled),
        both_augmented_intersection_strict=np.array_equal(
            masks['strict_plus_original'] & masks['strict_plus_filled'], strict),
        both_augmented_union_completed=np.array_equal(
            masks['strict_plus_original'] | masks['strict_plus_filled'], full),
        completed_exact_three_factor_union=np.array_equal(full, strict | extra_original | filled),
        unknown_never_selected=not np.any(full & (gt == 0)))
    if not all(checks.values()):
        raise AssertionError('interface support factors violate their exact set contract')
    audit = dict(
        format='interface_support_factor_v1', gt_ids=list(proposal['gt_ids']),
        radius=proposal['radius'], radius_unit='native_pixels',
        contact_count=proposal['contact_count'], strict_contact_count=proposal['strict_contact_count'],
        masks={name: dict(provenance=mask_provenance(gt, original, mask),
                          selected_pixels=int(mask.sum()), mask_sha256=_sha(mask))
               for name, mask in masks.items()},
        factors={name: dict(provenance=mask_provenance(gt, original, mask),
                            selected_pixels=int(mask.sum()), mask_sha256=_sha(mask))
                 for name, mask in [('strict', strict), ('extra_original', extra_original),
                                    ('filled', filled)]},
        set_checks=checks, input_mutated=False,
        no_new_trust_definition=True, mechanism_counterfactual_only=True,
        completed_including_filled_is_optimistic_mechanism_only=True,
        caveat='Original/fill factorization does not certify physical boundaries or training feasibility')
    return masks, audit


def _local_numpy(supervision, shape):
    if not isinstance(supervision, Mapping) or 'local' not in supervision:
        raise ValueError('supervision must contain the existing frozen local mask')
    local = supervision['local']
    if torch.is_tensor(local):
        if local.dtype != torch.bool or local.requires_grad or local.layout != torch.strided:
            raise ValueError('supervision local must be a frozen strided torch bool mask')
        device, source_type = str(local.device), 'torch_bool_detached_copy'
        result = local.detach().cpu().numpy().copy()
    else:
        if not isinstance(local, np.ndarray) or local.dtype != np.bool_:
            raise ValueError('supervision local must be numpy bool or frozen torch bool')
        device, source_type = 'cpu', 'numpy_bool_copy'
        result = local.copy()
    if result.shape != shape:
        raise ValueError('supervision local must match [1,8,gridH,gridW] exactly')
    if result[:, 4:].any():
        raise ValueError('existing local mask must contain short4 channels only')
    return result, dict(device=device, type=source_type, frozen=True,
                       mask_sha256=_sha(result), selected_edges=int(result.sum()))


def footprint_local_masks(tile, gt, original, roi, supervision, gid, device='cpu'):
    """返回 ``(selected_numpy_bool_1x8xHxW, audit)``，只过滤现有local。

    roi为整幅原生bool HW结构区域，不得包含unknown GT。先复用实际
    非零插值足迹，再筛双方grid_GT均为gid且至少一端属于足迹的候选，
    最后与原supervision['local']相交。local须为冻结bool、形状相同且仅
    short4；不会根据roi或原覆盖来源新增可信边，不重新推断native路径。
    """
    roi_provenance = mask_provenance(gt, original, roi)
    footprint, footprint_audit = native_grid_footprint(tile, roi, device=device)
    raw, raw_audit = edge_roi_mask(tile, gt, original, footprint, [gid], 'positive')
    local, local_audit = _local_numpy(supervision, raw.shape)
    selected = raw & local
    if selected[:, 4:].any() or np.any(selected & ~local) or np.any(selected & ~raw):
        raise AssertionError('footprint filtering must remain a short4 subset of existing local')
    per_channel = [dict(channel=channel, raw_candidate_edges=int(raw[0, channel].sum()),
                        existing_local_edges=int(local[0, channel].sum()),
                        selected_local_edges=int(selected[0, channel].sum()))
                   for channel in range(8)]
    audit = dict(
        format='footprint_local_masks_v1', gid=int(gid), roi_provenance=roi_provenance,
        roi_sha256=_sha(roi), footprint=footprint_audit, raw_candidates=raw_audit,
        existing_local=local_audit, selected_local_edges=int(selected.sum()),
        raw_candidate_edges=int(raw.sum()),
        raw_candidates_excluded_by_existing_local=int((raw & ~local).sum()),
        existing_local_outside_target_roi_edges=int((local & ~raw).sum()),
        selected_mask_shape=list(selected.shape), selected_mask_sha256=_sha(selected),
        per_channel=per_channel,
        selection_rule='actual interpolation footprint and same-GT endpoints intersect existing supervision local',
        short4_only=True, subset_existing_local=True, no_new_trust_definition=True,
        native_trust_source="unchanged supervision['local']; no native path recalculation",
        input_mutated=False, no_model_or_parameter_gradients=True, no_optimizer=True,
        caveat='Selected edge trust is inherited from supplied local mask; ROI provenance describes pixels, not edge trust')
    return selected, audit
