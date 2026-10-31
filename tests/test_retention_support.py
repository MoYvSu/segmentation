# -*- coding: utf-8 -*-
"""虚构CPU关系验证processed-known负保留扩展，不训练。"""
import copy
import math

import numpy as np
import pytest
import torch

from utils.affinity_interior_loss import _native_masks
from utils.affinity_loss import build_affinity_targets_torch
from utils.affinity_retain_loss import build_reliable_retention_mask
from utils.offset_letterbox import letterbox_instance_geometry
from utils.retention_support import build_processed_negative_retention, processed_native_support


def _case(*, pad=False, third=False, unknown=False):
    gt = np.ones((3 if pad else 6, 6), np.int32)
    gt[:, 3:] = 2
    if third:
        gt[:, 2] = 3
    if unknown:
        gt[:, 2] = 0
    original = gt > 0
    original[:, 3] = False
    grid_gt, valid, geom = letterbox_instance_geometry(gt, 6, 6)
    tile = dict(box=[0, 0, *gt.shape], grid_gt=grid_gt, valid=valid,
                geometry=geom.to_dict(), pad_h=6-geom.resized_height,
                pad_w=6-geom.resized_width)
    targets, legal = build_affinity_targets_torch(torch.from_numpy(grid_gt)[None],
                                                 torch.from_numpy(valid)[None, None])
    teacher = torch.where(targets == 1, 3., -3.).float()
    band = torch.zeros((6, 6), dtype=torch.bool)
    native, _, _, _, _ = _native_masks(tile, gt, original, np.zeros(gt.shape, bool))
    native = torch.from_numpy(native)
    retained, _ = build_reliable_retention_mask(teacher, targets, legal, band,
                                               instance_map=grid_gt, valid_content=valid)
    supervision = dict(targets=targets, full_legal=legal, band=band,
                       native_legal=native, retention=retained & native)
    return tile, gt, original, teacher, supervision


def test_old_retention_preserved_new_only_high_confidence_negative_known_paths():
    tile, gt, original, teacher, sup = _case()
    masks, audit = build_processed_negative_retention(tile, gt, original, teacher, sup)
    assert masks['added'].any()
    assert torch.equal(masks['old'], sup['retention'])
    assert not (masks['old'] & ~masks['expanded']).any()
    assert torch.equal(masks['expanded'], masks['old'] | masks['added'])
    assert torch.all(sup['targets'][masks['added']] == 0)
    assert torch.all(1-teacher.sigmoid()[masks['added']] > .9)
    assert not (masks['added'] & sup['native_legal']).any()
    assert audit['old_retention_exact'] and audit['newly_added_negative_only']
    assert audit['added_negative_edges'] == masks['added'].sum()
    assert all(not value.requires_grad for value in masks.values())


@pytest.mark.parametrize('pad,third,unknown', [(True, False, False), (False, True, False),
                                             (False, False, True)])
def test_grid_native_unknown_padding_and_third_GT_paths_excluded(pad, third, unknown):
    tile, gt, original, teacher, sup = _case(pad=pad, third=third, unknown=unknown)
    masks, _ = build_processed_negative_retention(tile, gt, original, teacher, sup)
    if pad:
        assert not masks['added'][:, :, 3:].any()
    if third or unknown:
        # +4从GT1到GT2越过第2列第三GT/unknown，必须拒绝。
        assert not masks['added'][0, 6, 1, 0]


def test_correction_band_and_low_confidence_never_added():
    tile, gt, original, teacher, sup = _case()
    teacher[0, 0, 1, 2] = -.5  # 原负边但低置信。
    sup['band'][2, 2] = True
    retained, _ = build_reliable_retention_mask(teacher, sup['targets'], sup['full_legal'], sup['band'],
                                               instance_map=tile['grid_gt'], valid_content=tile['valid'])
    sup['retention'] = retained & sup['native_legal']
    masks, _ = build_processed_negative_retention(tile, gt, original, teacher, sup)
    assert not masks['added'][0, 0, 1, 2]
    assert not masks['added'][0, 0, 2, 2]


def test_confidence_boundary_uses_strict_added_rule_and_keeps_old_exact():
    tile, gt, original, teacher, sup = _case()
    teacher[0, 0, 1, 2] = math.log(.1/.9)
    retained, _ = build_reliable_retention_mask(teacher, sup['targets'], sup['full_legal'], sup['band'],
                                               instance_map=tile['grid_gt'], valid_content=tile['valid'])
    sup['retention'] = retained & sup['native_legal']
    masks, _ = build_processed_negative_retention(tile, gt, original, teacher, sup)
    expected = bool(teacher.sigmoid()[0, 0, 1, 2] < 1.-.9)
    assert bool(masks['added'][0, 0, 1, 2]) == expected
    assert torch.equal(masks['old'], sup['retention'])


def test_no_filled_means_no_added_and_inputs_and_teacher_grads_unchanged():
    tile, gt, original, teacher, sup = _case()
    original[:] = gt > 0
    native, _, _, _, _ = _native_masks(tile, gt, original, np.zeros(gt.shape, bool))
    sup['native_legal'] = torch.from_numpy(native)
    retained, _ = build_reliable_retention_mask(teacher, sup['targets'], sup['full_legal'], sup['band'],
                                               instance_map=tile['grid_gt'], valid_content=tile['valid'])
    sup['retention'] = retained & sup['native_legal']
    before_tile, before_sup = copy.deepcopy(tile), copy.deepcopy(sup)
    before_gt, before_original, before_teacher = gt.copy(), original.copy(), teacher.clone()
    masks, _ = build_processed_negative_retention(tile, gt, original, teacher, sup)
    assert not masks['added'].any() and torch.equal(masks['expanded'], sup['retention'])
    np.testing.assert_array_equal(gt, before_gt)
    np.testing.assert_array_equal(original, before_original)
    for key in ('grid_gt', 'valid'):
        np.testing.assert_array_equal(tile[key], before_tile[key])
    for key in sup:
        assert torch.equal(sup[key], before_sup[key])
    assert torch.equal(teacher, before_teacher) and teacher.grad is None


@pytest.mark.parametrize('mutation', ['teacher_grad', 'teacher_nonfinite', 'old_tamper',
                                     'native_tamper', 'wrong_grid_GT', 'padding_tamper'])
def test_frozen_contract_mismatches_rejected(mutation):
    tile, gt, original, teacher, sup = _case()
    if mutation == 'teacher_grad':
        teacher.requires_grad_(True)
    elif mutation == 'teacher_nonfinite':
        teacher[0, 0, 0, 0] = float('nan')
    elif mutation == 'old_tamper':
        sup['retention'][0, 0, 0, 0] ^= True
    elif mutation == 'native_tamper':
        sup['native_legal'][0, 0, 0, 0] ^= True
    elif mutation == 'wrong_grid_GT':
        tile['grid_gt'][0, 0] = 9
    else:
        tile['pad_h'] = 1
    with pytest.raises(ValueError):
        build_processed_negative_retention(tile, gt, original, teacher, sup)


def test_processed_support_original_subset_known_and_no_old_loss_mutation():
    tile, gt, original, _, _ = _case()
    old, known, audit = processed_native_support(tile, gt, original)
    assert not (old & ~known).any() and (known & ~old).any()
    assert audit['only_changed_native_support'] == 'original_covered -> processed_GT>0'
