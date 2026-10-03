# -*- coding: utf-8 -*-
"""原生标量边界的完整结构反事实；仅用于机制诊断。

本模块没有模型、训练、分水岭或测试标签逻辑。完整processed GT版本包含
filled，只是乐观机制对照；strict版本需要原标注覆盖支持，也不是训练
可行性或独立准确率证明。所有尺寸及radius/margin均指原图像素。
"""
from __future__ import annotations

from numbers import Integral, Real

import cv2
import numpy as np


def _gt_and_coverage(gt, original_covered):
    gt = np.asarray(gt)
    coverage = np.asarray(original_covered)
    if (gt.ndim != 2 or not gt.size or gt.dtype == np.bool_
            or not np.issubdtype(gt.dtype, np.integer) or np.any(gt < 0)):
        raise ValueError('gt must be a nonempty nonnegative integer native HW map')
    if coverage.shape != gt.shape or coverage.dtype != np.bool_:
        raise ValueError('original_covered must be a matching bool native HW map')
    if np.any(coverage & (gt == 0)):
        raise ValueError('original_covered cannot include unknown GT=0 pixels')
    return gt, coverage


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return int(value)


def _mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError('mask must be a matching bool native HW map')
    return mask


def _component_count(mask):
    return int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)


def mask_provenance(gt, original_covered, mask):
    """统计所选像素的原标覆盖/filled来源，未知域选择必须失败。

返回纯JSON计数。original_covered是已核对来源的布尔覆盖，不在这里推断；
filled定义为processed GT>0且不属于original_covered。输入保持不变。
"""
    gt, coverage = _gt_and_coverage(gt, original_covered)
    mask = _mask(mask, gt.shape)
    unknown = int(np.count_nonzero(mask & (gt == 0)))
    if unknown:
        raise ValueError('Structure mask must not select unknown GT=0 pixels')
    original = int(np.count_nonzero(mask & coverage))
    filled = int(np.count_nonzero(mask & (gt > 0) & ~coverage))
    return dict(masked_pixels=int(np.count_nonzero(mask)),
                original_covered_pixels=original, filled_pixels=filled,
                unknown_pixels=unknown, known_pixels=original + filled,
                format='affinity_structure_provenance_v1',
                coverage_definition='explicit original annotation coverage; not inferred',
                filled_definition='processed GT>0 outside original annotation coverage',
                full_GT_is_optimistic_mechanism_only=True)


def interface_mask(gt, original_covered, pair, *, radius=2):
    """两个GT的全部4邻直接接触双端，等半径椭圆膨胀后限于该GT并集。

返回mask、strict_mask、contact_mask、strict_contact_mask及来源统计。
contact_count/strict_contact_count计的是无重复水平/竖直接触边，不是像素。
strict接触边要求双端original_covered；其膨胀结果再次限制原标覆盖。
完整版本含filled，仅为乐观机制对照，不得直接当可信训练监督。
"""
    gt, coverage = _gt_and_coverage(gt, original_covered)
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError('pair must contain two distinct positive GT IDs')
    groups = [_integer(gid, 'GT ID', 1) for gid in pair]
    if groups[0] == groups[1] or any(not np.any(gt == gid) for gid in groups):
        raise ValueError('pair must contain two distinct positive GT IDs present in gt')
    radius = _integer(radius, 'radius')
    full_contacts = np.zeros(gt.shape, np.bool_)
    strict_contacts = np.zeros(gt.shape, np.bool_)
    count = 0
    strict_count = 0
    # 只遍历右/下两个方向；反向边不得重复计数，斜角不构成接触。
    for source, destination in [
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ]:
        left, right = gt[source], gt[destination]
        contact = ((left == groups[0]) & (right == groups[1])) | (
            (left == groups[1]) & (right == groups[0]))
        strict = contact & coverage[source] & coverage[destination]
        full_contacts[source] |= contact
        full_contacts[destination] |= contact
        strict_contacts[source] |= strict
        strict_contacts[destination] |= strict
        count += int(np.count_nonzero(contact))
        strict_count += int(np.count_nonzero(strict))
    full = full_contacts.copy()
    strict = strict_contacts.copy()
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        full = cv2.dilate(full.astype(np.uint8), kernel,
                          borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        strict = cv2.dilate(strict.astype(np.uint8), kernel,
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    target = np.isin(gt, groups)
    full &= target
    strict &= target & coverage
    return dict(mask=full, strict_mask=strict, contact_mask=full_contacts,
        strict_contact_mask=strict_contacts, contact_count=count,
        strict_contact_count=strict_count, gt_ids=groups, radius=radius,
        connectivity='4-neighbor contact, both endpoints selected',
        radius_unit='native_pixels',
        provenance=mask_provenance(gt, coverage, full),
        strict_provenance=mask_provenance(gt, coverage, strict),
        full_components=_component_count(full), strict_components=_component_count(strict),
        full_GT_is_optimistic_mechanism_only=True,
        strict_scope='both contact endpoints original-covered; dilated mask original-covered only')


def interior_mask(gt, original_covered, gid, *, margin=4):
    """单GT全部内部：方形全1核腐蚀，图外固定为0，strict再限原标覆盖。

margin4即9×9腐蚀；margin0允许整个GT。filled孔不会被strict补齐；返回
组件数及排除filled数量作为不连续提示，不推断物理组织边界。
"""
    gt, coverage = _gt_and_coverage(gt, original_covered)
    gid = _integer(gid, 'GT ID', 1)
    if not np.any(gt == gid):
        raise ValueError('GT ID must be present in gt')
    margin = _integer(margin, 'margin')
    full = gt == gid
    if margin:
        kernel = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
        full = cv2.erode(full.astype(np.uint8), kernel,
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    else:
        full = full.copy()
    strict = full & coverage
    return dict(mask=full, strict_mask=strict, gt_id=gid, margin=margin,
        margin_unit='native_pixels', erosion='square all-ones kernel, constant-zero border',
        provenance=mask_provenance(gt, coverage, full),
        strict_provenance=mask_provenance(gt, coverage, strict),
        full_components=_component_count(full), strict_components=_component_count(strict),
        strict_excluded_filled_pixels=int(np.count_nonzero(full & ~coverage)),
        full_GT_is_optimistic_mechanism_only=True,
        strict_scope='eroded processed GT intersect original-covered; holes retained')


def apply_structure(boundary, mask, value, *, gt, original_covered):
    """返回(新float32边界, audit)，mask外逐值不变，任何GT0修改均拒绝。

boundary必须为非空float32 native HW、有限且在[0,1]内；mask必须bool。
GT与原标覆盖显式传入以核对未知域和来源。value=1可加强完整界面，
value=0可削弱完整内部屏障；任意合法value仍只表示标量反事实。
"""
    boundary = np.asarray(boundary)
    if (boundary.ndim != 2 or not boundary.size or boundary.dtype != np.float32
            or not np.all(np.isfinite(boundary)) or np.any((boundary < 0) | (boundary > 1))):
        raise ValueError('boundary must be a nonempty finite float32 native HW map in [0,1]')
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('value must be a finite real scalar in [0,1]')
    gt, coverage = _gt_and_coverage(gt, original_covered)
    if gt.shape != boundary.shape:
        raise ValueError('GT/coverage and boundary must retain identical native HW dimensions')
    mask = _mask(mask, boundary.shape)
    provenance = mask_provenance(gt, coverage, mask)
    result = boundary.copy()
    result[mask] = np.float32(value)
    changed = mask & (result != boundary)
    if not np.array_equal(result[~mask], boundary[~mask]):
        raise AssertionError('Structure counterfactual changed values outside mask')
    audit = dict(format='affinity_structure_apply_v1', value=float(np.float32(value)),
        masked_pixels=provenance['masked_pixels'], changed_pixels=int(np.count_nonzero(changed)),
        provenance=provenance, changed_provenance=mask_provenance(gt, coverage, changed),
        outside_mask_exact=True, input_mutated=False, output_dtype='float32',
        native_shape=list(boundary.shape),
        mechanism_counterfactual_only=True, is_training_feasibility_proof=False)
    return result, audit
