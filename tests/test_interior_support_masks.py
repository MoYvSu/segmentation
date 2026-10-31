# -*- coding: utf-8 -*-
"""纯虚构支持因素与冻结local选择合同。"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from utils.affinity_spatial_trace import edge_roi_mask, native_grid_footprint
from utils.affinity_structure import interface_mask
from utils.interior_support_masks import interface_support_factor, footprint_local_masks
from utils.offset_letterbox import letterbox_instance_geometry


def _interface_case():
    gt = np.array([
        [0, 67, 67, 68, 68, 9],
        [0, 67, 67, 68, 68, 9],
        [0, 67, 67, 68, 68, 9],
        [0, 67, 67, 68, 68, 9],
        [0, 67, 67, 68, 68, 9],
        [0, 67, 67, 68, 68, 9],
        [0, 67, 67, 68, 68, 9],
        [0, 67, 67, 68, 68, 9],
    ], np.int32)
    original = gt > 0
    original[2:, 3] = False  # 前两条接触边有双端覆盖，后续接触边含filled。
    original[7, 1] = False
    return gt, original


@pytest.mark.parametrize('radius', [0, 1, 2, 4])
def test_exact_two_factor_sets_and_provenance(radius):
    gt, original = _interface_case()
    proposal = interface_mask(gt, original, [67, 68], radius=radius)
    full, strict = proposal['mask'], proposal['strict_mask']
    masks, audit = interface_support_factor(gt, original, [67, 68], radius)
    assert set(masks) == {'strict', 'strict_plus_original', 'strict_plus_filled', 'completed'}
    extra_original = full & original & ~strict
    filled = full & ~original
    np.testing.assert_array_equal(masks['strict'], strict)
    np.testing.assert_array_equal(masks['strict_plus_original'], full & original)
    np.testing.assert_array_equal(masks['strict_plus_filled'], strict | filled)
    np.testing.assert_array_equal(masks['completed'], full)
    np.testing.assert_array_equal(masks['strict_plus_original'] & masks['strict_plus_filled'], strict)
    np.testing.assert_array_equal(masks['strict_plus_original'] | masks['strict_plus_filled'], full)
    assert not (extra_original & filled).any()
    assert all(audit['set_checks'].values())
    assert audit['factors']['extra_original']['selected_pixels'] == extra_original.sum()
    assert audit['factors']['filled']['selected_pixels'] == filled.sum()
    assert extra_original.any() and filled.any()
    for name, mask in masks.items():
        assert mask.dtype == bool and not mask[gt == 0].any() and not mask[gt == 9].any()
        provenance = audit['masks'][name]['provenance']
        assert provenance['masked_pixels'] == mask.sum()
        assert provenance['original_covered_pixels'] == (mask & original).sum()
        assert provenance['filled_pixels'] == (mask & ~original).sum()
        assert provenance['unknown_pixels'] == 0


@pytest.mark.parametrize('coverage', ['all_original', 'all_filled', 'no_contact'])
def test_degenerate_factor_sets_are_exact(coverage):
    gt = np.tile(np.array([67, 67, 68, 68], np.int32), (4, 1))
    original = np.ones(gt.shape, bool)
    if coverage == 'all_filled':
        original[:] = False
    elif coverage == 'no_contact':
        gt[:, 2] = 0
        original = gt > 0
    masks, audit = interface_support_factor(gt, original, [67, 68])
    assert all(audit['set_checks'].values())
    if coverage == 'all_original':
        for mask in masks.values():
            np.testing.assert_array_equal(mask, masks['completed'])
    elif coverage == 'all_filled':
        assert not masks['strict'].any() and not masks['strict_plus_original'].any()
        np.testing.assert_array_equal(masks['strict_plus_filled'], masks['completed'])
    else:
        assert not any(mask.any() for mask in masks.values())


def test_factor_inputs_not_mutated_and_output_arrays_independent():
    gt, original = _interface_case()
    before_gt, before_original = gt.copy(), original.copy()
    masks, _ = interface_support_factor(gt, original, [67, 68])
    completed = masks['completed'].copy()
    masks['strict'][:] = False
    masks['strict_plus_original'][:] = False
    masks['strict_plus_filled'][:] = False
    np.testing.assert_array_equal(masks['completed'], completed)
    np.testing.assert_array_equal(gt, before_gt)
    np.testing.assert_array_equal(original, before_original)


def _local_case():
    gt = np.full((7, 7), 5, np.int32)
    gt[0, 0] = 0
    gt[-1, -1] = 9
    original = gt > 0
    original[3, 3] = False
    grid_gt, valid, geometry = letterbox_instance_geometry(gt, 8, 4)
    tile = dict(box=[0, 0, 7, 7], grid_gt=grid_gt, valid=valid,
                geometry=geometry.to_dict(), pad_h=0, pad_w=0,
                logits=torch.ones((1, 8, 4, 4), requires_grad=True),
                features=[torch.ones((1, 2, 2, 2), requires_grad=True)])
    roi = np.zeros(gt.shape, bool)
    roi[3, 3] = True
    local = np.zeros((1, 8, 4, 4), bool)
    local[:, :4] = True
    local[0, 0, 1, 1] = False  # 原可信监督的洞不得被ROI补回。
    return tile, gt, original, roi, local


@pytest.mark.parametrize('torch_local', [False, True])
def test_actual_footprint_relation_then_only_existing_local_intersection(torch_local):
    tile, gt, original, roi, local = _local_case()
    supplied = torch.from_numpy(local.copy()) if torch_local else local
    selected, audit = footprint_local_masks(tile, gt, original, roi, {'local': supplied}, 5)
    footprint, _ = native_grid_footprint(tile, roi)
    raw, _ = edge_roi_mask(tile, gt, original, footprint, [5], 'positive')
    np.testing.assert_array_equal(selected, raw & local)
    assert selected.any() and selected.dtype == bool and selected.shape == local.shape
    assert not selected[:, 4:].any() and not (selected & ~local).any()
    assert not selected[0, 0, 1, 1]
    assert audit['selected_local_edges'] == selected.sum()
    assert sum(row['selected_local_edges'] for row in audit['per_channel']) == selected.sum()
    assert audit['roi_provenance']['filled_pixels'] == 1
    assert audit['roi_provenance']['original_covered_pixels'] == 0
    assert audit['no_new_trust_definition'] and audit['subset_existing_local']
    assert audit['footprint']['interpolation_device'] == 'cpu'
    assert audit['raw_candidates']['requires_existing_supervision_intersection']


def test_footprint_filter_does_not_mutate_capture_or_supervision_or_gradients():
    tile, gt, original, roi, local = _local_case()
    torch_local = torch.from_numpy(local.copy())
    before_tile = copy.deepcopy(tile)
    arrays = [value.copy() for value in (gt, original, roi, local)]
    with torch.no_grad():
        footprint_local_masks(tile, gt, original, roi, {'local': torch_local}, 5)
        assert not torch.is_grad_enabled()
    for value, before in zip((gt, original, roi, local), arrays):
        np.testing.assert_array_equal(value, before)
    np.testing.assert_array_equal(torch_local.numpy(), local)
    for key in ('grid_gt', 'valid'):
        np.testing.assert_array_equal(tile[key], before_tile[key])
    assert torch.equal(tile['logits'], before_tile['logits']) and tile['logits'].grad is None
    assert torch.equal(tile['features'][0], before_tile['features'][0])
    assert tile['features'][0].grad is None


def test_empty_roi_and_empty_existing_local_preserve_exact_empty_selection():
    tile, gt, original, roi, local = _local_case()
    selected, audit = footprint_local_masks(tile, gt, original, roi*False, {'local': local}, 5)
    assert not selected.any() and audit['selected_local_edges'] == 0
    selected, audit = footprint_local_masks(tile, gt, original, roi, {'local': local*False}, 5)
    assert not selected.any() and audit['selected_local_edges'] == 0


@pytest.mark.parametrize('mutation', [
    'roi_unknown', 'roi_dtype', 'roi_shape', 'missing_local', 'local_numpy_float',
    'local_tensor_float', 'local_shape', 'local_long_channel',
])
def test_invalid_roi_or_existing_local_contract_rejected(mutation):
    tile, gt, original, roi, local = _local_case()
    supervision = {'local': local}
    if mutation == 'roi_unknown':
        roi[0, 0] = True
    elif mutation == 'roi_dtype':
        roi = roi.astype(np.uint8)
    elif mutation == 'roi_shape':
        roi = roi[:-1]
    elif mutation == 'missing_local':
        supervision = {}
    elif mutation == 'local_numpy_float':
        supervision['local'] = local.astype(np.float32)
    elif mutation == 'local_tensor_float':
        supervision['local'] = torch.from_numpy(local).float().requires_grad_(True)
    elif mutation == 'local_shape':
        supervision['local'] = local[0]
    elif mutation == 'local_long_channel':
        local[0, 7, 1, 1] = True
    with pytest.raises(ValueError):
        footprint_local_masks(tile, gt, original, roi, supervision, 5)


@pytest.mark.parametrize('mutation', ['gt_float', 'original_dtype', 'original_unknown', 'radius_bool'])
def test_interface_factor_reuses_existing_input_contract(mutation):
    gt, original = _interface_case()
    radius = 2
    if mutation == 'gt_float':
        gt = gt.astype(np.float32)
    elif mutation == 'original_dtype':
        original = original.astype(np.uint8)
    elif mutation == 'original_unknown':
        original[0, 0] = True
    else:
        radius = True
    with pytest.raises(ValueError):
        interface_support_factor(gt, original, [67, 68], radius)
