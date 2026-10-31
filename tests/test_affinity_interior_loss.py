# -*- coding: utf-8 -*-
"""原生可信域、固定响应带与共同外围保留损失的虚构CPU检查。"""
import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from utils.affinity_interior_loss import build_interior_supervision, interior_objective
from utils.affinity_loss import balanced_affinity_loss, build_affinity_targets_torch
from utils.affinity_retain_loss import balanced_affinity_retention_kl
from utils.offset_letterbox import letterbox_instance_geometry


def fixture(gt, original=None, grid=None):
    original = np.ones(gt.shape, bool) if original is None else original
    grid = gt.shape[0] if grid is None else grid
    labels, valid, geometry = letterbox_instance_geometry(gt, 1024, grid)
    targets, _ = build_affinity_targets_torch(torch.from_numpy(labels.copy()).long()[None],
                                            torch.from_numpy(valid.copy())[None, None])
    teacher = torch.where(targets == 1, torch.tensor(9.), torch.tensor(-9.))
    fused = np.zeros((1, 1, grid, grid), np.float32)
    fused[0, 0, grid//2, grid//2] = .7
    tile = dict(box=[0, 0, *gt.shape], grid_gt=labels, valid=valid,
                geometry=geometry.to_dict(), grid=fused)
    return tile, original, teacher


def test_captured_fixed_band_short_local_and_common_complete_gt_unchanged():
    gt = np.ones((25, 25), np.uint16)
    gt[:, 19:] = 2
    tile, original, teacher = fixture(gt)
    old_tile, old_gt, old_original, old_teacher = deepcopy(tile), gt.copy(), original.copy(), teacher.clone()
    masks, audit = build_interior_supervision(tile, gt, original, teacher, gid=1)
    expected, legal = build_affinity_targets_torch(torch.from_numpy(tile['grid_gt']).long()[None],
                                                torch.from_numpy(tile['valid'])[None, None])
    assert torch.equal(masks['targets'], expected) and torch.equal(masks['full_legal'], legal)
    assert masks['interior'].any() and masks['local'].any()
    assert not masks['interior'][:, 4:].any() and not masks['local'][:, 4:].any()
    assert not (masks['local'] & ~masks['interior']).any()
    assert torch.all(masks['targets'][masks['local']] == 1)
    # actual fused snapshot选带，不以teacher重融合或更新后置信度重选。
    assert masks['band'].sum() == 9 and masks['band'][11:14, 11:14].all()
    assert not masks['retention'][0, 0, 12, 10]  # source带外，destination进入带也必须排除。
    changed_teacher = torch.zeros_like(teacher)
    again, other_audit = build_interior_supervision(tile, gt, original, changed_teacher, gid=1)
    assert torch.equal(again['band'], masks['band']) and torch.equal(again['local'], masks['local'])
    assert not again['retention'].any()
    assert other_audit['band']['captured_fused_grid_sha256'] == audit['band']['captured_fused_grid_sha256']
    assert audit['masks']['outer']['negative_edges'] > 0
    assert audit['masks']['outer']['positive_edges'] > 0
    assert audit['masks']['retention']['active_channels'] == 8
    json.dumps(audit, allow_nan=False)
    for key in ('grid_gt', 'valid', 'grid'):
        assert np.array_equal(tile[key], old_tile[key])
    assert np.array_equal(gt, old_gt) and np.array_equal(original, old_original)
    assert torch.equal(teacher, old_teacher)


@pytest.mark.parametrize('obstacle', ['unknown', 'filled', 'third_GT'])
def test_thin_native_obstacle_rejects_short_long_outer_and_retention(obstacle):
    gt = np.ones((41, 41), np.uint16)
    original = np.ones(gt.shape, bool)
    if obstacle == 'unknown':
        gt[:, 6] = 0
        original[:, 6] = False
    elif obstacle == 'filled':
        original[:, 6] = False
    else:
        gt[:, 6] = 3
    tile, original, teacher = fixture(gt, original, grid=11)
    masks, audit = build_interior_supervision(tile, gt, original, teacher, gid=1, margin=4)
    assert tile['grid_gt'][5, 1] == tile['grid_gt'][5, 2] == 1
    # 两端known nearest-GT合法，但实际中心native4->8跨薄障碍。
    assert masks['full_legal'][0, 0, 5, 1]
    for name in ('native_legal', 'interior', 'local', 'outer', 'retention'):
        assert not masks[name][0, 0, 5, 1]
        assert not masks[name][0, 4, 5, 0]  # native0->8；起点在目标内部4px周圈。
        assert not masks[name][0, 6, 5, 0]  # native0->16。
        assert not (masks[name] & ~masks['full_legal']).any()
    assert masks['native_legal'][0, 0, 5, 7]
    assert audit['native_trust']['unknown_filled_third_GT_excluded']


def test_outer_keeps_trusted_cross_gt_but_padding_and_native_frame_protected():
    gt = np.ones((17, 25), np.uint16)
    gt[:, 16:] = 2
    tile, original, teacher = fixture(gt, grid=25)
    masks, audit = build_interior_supervision(tile, gt, original, teacher, gid=1)
    assert audit['masks']['outer']['negative_edges'] > 0
    assert audit['masks']['outer']['active_channels'] == 8
    valid = torch.from_numpy(tile['valid'])
    for name in ('full_legal', 'interior', 'local', 'outer', 'native_legal', 'retention'):
        assert not masks[name][0, :, ~valid].any()
    # 图外按未知处理，GT内部保护圈亦包括原生图像边缘，不能在边缘增益。
    assert not masks['interior'][0, :, 0].any()
    assert not masks['interior'][0, :, :, 0].any()
    assert audit['outer']['complete_outer_closure_proven'] is False


def test_selected_extra_loss_gradient_outside_masks_is_zero_and_empty_backpropagates():
    gt = np.ones((25, 25), np.uint16)
    gt[:, 19:] = 2
    tile, original, teacher = fixture(gt)
    masks, _ = build_interior_supervision(tile, gt, original, teacher, gid=1)
    student = (teacher+.3).requires_grad_(True)
    extra = torch.zeros_like(masks['local'])
    for name in ('local', 'outer'):
        loss, _ = balanced_affinity_loss(student, masks['targets'], masks[name])
        loss.backward(retain_graph=True)
        assert not student.grad[~masks[name]].count_nonzero()
        assert student.grad[masks[name]].count_nonzero()
        student.grad = None
        extra |= masks[name]
    loss, _ = balanced_affinity_retention_kl(student, teacher, masks['targets'], masks['retention'])
    loss.backward()
    assert not student.grad[~masks['retention']].count_nonzero()
    assert student.grad[masks['retention']].count_nonzero()
    # 空掩码仍接在实际student上，可安全反传零；不引入常量断图。
    empty = {key:torch.zeros_like(value) if key in ('full_legal','local','outer','retention') else value
             for key, value in masks.items()}
    student = torch.randn_like(teacher, requires_grad=True)
    loss, terms, _ = interior_objective(student, teacher, empty, local_weight=.25)
    assert loss.item() == 0 and all(value.item() == 0 for value in terms.values())
    loss.backward()
    assert not student.grad.count_nonzero()


def test_matched_objective_unique_local_increment_and_legal_gradient_contract():
    gt = np.ones((25, 25), np.uint16)
    gt[:, 19:] = 2
    tile, original, teacher = fixture(gt)
    masks, _ = build_interior_supervision(tile, gt, original, teacher, gid=1)
    student = (teacher*.3).requires_grad_(True)
    control, a_terms, _ = interior_objective(student, teacher, masks, local_weight=0)
    treatment, b_terms, _ = interior_objective(student, teacher, masks, local_weight=.25)
    assert torch.allclose(treatment-control, .25*b_terms['local'], atol=1e-7)
    assert torch.equal(control, a_terms['full']+.5*a_terms['outer']+a_terms['retention'])
    treatment.backward()
    assert not student.grad[~masks['full_legal']].count_nonzero()
    assert student.grad[masks['local']].count_nonzero()
    with pytest.raises(ValueError, match='local_weight'):
        interior_objective(student, teacher, masks, local_weight=1)


def test_wrong_actual_fused_contract_and_unknown_claimed_original_are_rejected():
    gt = np.ones((13, 13), np.uint16)
    tile, original, teacher = fixture(gt)
    tile['grid'] = tile['grid'].astype(np.float64)
    with pytest.raises(ValueError, match='captured actual FP32'):
        build_interior_supervision(tile, gt, original, teacher, gid=1)
    tile, original, teacher = fixture(gt)
    gt[2, 2] = 0
    with pytest.raises(ValueError, match='original-covered'):
        build_interior_supervision(tile, gt, original, teacher, gid=1)
