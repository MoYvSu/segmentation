# -*- coding: utf-8 -*-
"""虚构CPU图的实际插值支持、端点关系与输入隔离合同。"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from utils.affinity_deployment import crop_letterbox_output
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
from utils.affinity_spatial_trace import edge_roi_mask, native_grid_footprint
from utils.offset_letterbox import letterbox_instance_geometry


def _tile(gt, *, box=None, input_size=8, grid=4):
    box = [0, 0, *gt.shape] if box is None else box
    y, x, y1, x1 = box
    labels, valid, meta = letterbox_instance_geometry(
        gt[y:y1, x:x1], input_size=input_size, output_grid=grid)
    return dict(box=list(box), geometry=meta.to_dict(), grid_gt=labels, valid=valid,
                pad_h=input_size-meta.resized_height,
                pad_w=input_size-meta.resized_width)


def _basis_support(tile, mask):
    """独立前向证明：逐格单位脉冲，只收非零贡献，避免只镜像反向实现。"""
    y, x, y1, x1 = tile['box']
    selected = mask[y:y1, x:x1]
    shape = tile['grid_gt'].shape
    expected = np.zeros(shape, bool)
    for gy in range(shape[0]):
        for gx in range(shape[1]):
            basis = torch.zeros((1, 1, *shape), dtype=torch.float32)
            basis[0, 0, gy, gx] = 1
            native = crop_letterbox_output(
                basis, tile['geometry']['input_size'], tile['pad_h'], tile['pad_w'],
                (y1-y, x1-x))[0, 0].numpy()
            expected[gy, gx] = bool(np.any(native[selected] > 0))
    return expected


@pytest.mark.parametrize('native_shape,grid', [
    ((5, 5), 3), ((7, 7), 4), ((3, 7), 4), ((1, 7), 4),
    ((7, 1), 4), ((1, 1), 4), ((2, 17), 4), ((17, 2), 4),
    ((1, 100), 2),
])
def test_support_matches_actual_forward_basis_and_torch_gradient(native_shape, grid):
    gt = np.ones(native_shape, np.int32)
    tile = _tile(gt, grid=grid)
    mask = np.zeros(native_shape, bool)
    mask[0, 0] = True
    mask[native_shape[0]//2, native_shape[1]//2] = True
    mask[-1, -1] = True
    footprint, audit = native_grid_footprint(tile, mask)
    np.testing.assert_array_equal(footprint, _basis_support(tile, mask))
    # 另以实际torch算子的梯度支持核对裁内容后的align_corners路径。
    h, w = tile['geometry']['content_height'], tile['geometry']['content_width']
    source = torch.zeros((1, 1, grid, grid), requires_grad=True)
    output = F.interpolate(source[:, :, :h, :w], size=native_shape,
                           mode='bilinear', align_corners=True)
    derivative, = torch.autograd.grad(output, source,
                                     grad_outputs=torch.from_numpy(mask).float()[None, None])
    np.testing.assert_array_equal(footprint, derivative[0, 0].numpy() > 0)
    assert not (footprint & ~tile['valid']).any()
    assert audit['no_model_or_parameter_gradients'] and audit['no_optimizer']
    assert not audit['model_forward_replayed'] and audit['interpolation_device'] == 'cpu'
    assert not audit['nearest_mapping_used']


def test_integer_coordinates_do_not_include_zero_weight_neighbours():
    gt = np.ones((7, 7), np.int32)
    tile = _tile(gt, grid=4)
    mask = np.zeros(gt.shape, bool)
    mask[2, 2] = True  # align_corners: 2*(4-1)/(7-1)=1，恰为一个格。
    footprint, _ = native_grid_footprint(tile, mask)
    expected = np.zeros((4, 4), bool)
    expected[1, 1] = True
    np.testing.assert_array_equal(footprint, expected)
    mask[:] = False
    mask[1, 1] = True  # 0.5×0.5四个真正正权重点。
    footprint, _ = native_grid_footprint(tile, mask)
    expected[:] = False
    expected[:2, :2] = True
    np.testing.assert_array_equal(footprint, expected)


def test_global_mask_cropped_to_box_and_padding_excluded():
    gt = np.ones((8, 12), np.int32)
    tile = _tile(gt, box=[2, 3, 5, 10], grid=4)
    mask = np.zeros(gt.shape, bool)
    mask[0, 0] = True  # 同一源图的其它窗关注点不贡献本窗。
    mask[3, 4] = True
    footprint, audit = native_grid_footprint(tile, mask)
    np.testing.assert_array_equal(footprint, _basis_support(tile, mask))
    assert footprint.sum() == 4
    assert audit['native_selected_pixels_in_tile'] == 1
    assert audit['native_selected_pixels_outside_tile'] == 1
    assert audit['content_shape'] == [2, 4]
    assert not footprint[2:].any()


def test_empty_mask_exact_and_no_input_or_capture_gradient_change():
    gt = np.ones((7, 7), np.int32)
    tile = _tile(gt)
    tile['logits'] = torch.randn((1, 8, 4, 4), requires_grad=True)
    tile['features'] = [torch.randn((1, 2, 2, 2), requires_grad=True)]
    before = copy.deepcopy(tile)
    mask = np.zeros(gt.shape, bool)
    footprint, audit = native_grid_footprint(tile, mask)
    assert not footprint.any() and audit['contributing_grid_nodes'] == 0
    mask[3, 3] = True
    mask_before = mask.copy()
    native_grid_footprint(tile, mask)
    np.testing.assert_array_equal(mask, mask_before)
    for key in ('grid_gt', 'valid'):
        np.testing.assert_array_equal(tile[key], before[key])
    for key in ('box', 'geometry', 'pad_h', 'pad_w'):
        assert tile[key] == before[key]
    assert torch.equal(tile['logits'], before['logits'])
    assert tile['logits'].grad is None
    assert torch.equal(tile['features'][0], before['features'][0])
    assert tile['features'][0].grad is None


@pytest.mark.parametrize('context', ['no_grad', 'inference_mode'])
def test_local_autograd_restores_outer_context(context):
    gt = np.ones((7, 7), np.int32)
    tile = _tile(gt)
    mask = np.ones(gt.shape, bool)
    manager = torch.no_grad() if context == 'no_grad' else torch.inference_mode()
    with manager:
        before = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        footprint, _ = native_grid_footprint(tile, mask)
        assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == before
    assert footprint.all()


@pytest.mark.parametrize('mutation', [
    'missing_geometry', 'bad_geometry', 'bad_content', 'missing_pad', 'bad_pad',
    'bad_box_dtype', 'empty_box', 'bad_grid_dtype', 'bad_grid_shape',
    'bad_valid_dtype', 'bad_valid_content', 'nonzero_padding',
])
def test_captured_geometry_contract_rejects_invalid_inputs(mutation):
    gt = np.ones((3, 7), np.int32)
    tile = _tile(gt)
    if mutation == 'missing_geometry':
        del tile['geometry']
    elif mutation == 'bad_geometry':
        tile['geometry']['original_height'] += 1
    elif mutation == 'bad_content':
        tile['geometry']['content_height'] += 1
    elif mutation == 'missing_pad':
        del tile['pad_h']
    elif mutation == 'bad_pad':
        tile['pad_h'] += 1
    elif mutation == 'bad_box_dtype':
        tile['box'][0] = False
    elif mutation == 'empty_box':
        tile['box'][2] = 0
    elif mutation == 'bad_grid_dtype':
        tile['grid_gt'] = tile['grid_gt'].astype(np.float32)
    elif mutation == 'bad_grid_shape':
        tile['grid_gt'] = tile['grid_gt'][:-1]
    elif mutation == 'bad_valid_dtype':
        tile['valid'] = tile['valid'].astype(np.uint8)
    elif mutation == 'bad_valid_content':
        tile['valid'][0, 0] = False
    elif mutation == 'nonzero_padding':
        tile['grid_gt'][-1, 0] = 1
    with pytest.raises(ValueError):
        native_grid_footprint(tile, np.ones(gt.shape, bool))


@pytest.mark.parametrize('mask', [
    np.ones((7, 7), np.uint8), np.ones((1, 7, 7), bool), np.zeros((0, 7), bool),
    np.ones((6, 7), bool),
])
def test_native_mask_dtype_shape_and_box_rejected(mask):
    tile = _tile(np.ones((7, 7), np.int32))
    with pytest.raises(ValueError):
        native_grid_footprint(tile, mask)


def _manual_edges(tile, footprint, ids, direction):
    gt = tile['grid_gt']
    valid = tile['valid']
    expected = np.zeros((1, 8, *gt.shape), bool)
    for channel, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        for y in range(gt.shape[0]):
            for x in range(gt.shape[1]):
                ty, tx = y+dy, x+dx
                if not (0 <= ty < gt.shape[0] and 0 <= tx < gt.shape[1]):
                    continue
                if not (valid[y, x] and valid[ty, tx]
                        and (footprint[y, x] or footprint[ty, tx])):
                    continue
                left, right = gt[y, x], gt[ty, tx]
                if direction == 'positive':
                    relation = left == right == ids[0]
                else:
                    relation = {int(left), int(right)} == set(ids)
                expected[0, channel, y, x] = relation
    return expected


@pytest.mark.parametrize('direction,ids', [('negative', (67, 68)), ('positive', (5,))])
def test_all_eight_channel_actual_endpoints_and_either_roi_endpoint(direction, ids):
    gt = np.array([
        [67, 67, 68, 68, 9, 9],
        [67, 5, 5, 68, 9, 9],
        [0, 5, 5, 68, 9, 9],
        [0, 5, 5, 68, 68, 68],
        [67, 67, 67, 68, 68, 68],
        [67, 67, 67, 68, 68, 68],
    ], np.int32)
    tile = _tile(gt, input_size=6, grid=6)
    footprint = np.zeros(gt.shape, bool)
    footprint[1, 2] = True
    footprint[4, 2] = True
    footprint[4, 1] = True
    footprint[0, 4] = True  # 第三GT足迹节点不能冒充目标关系。
    footprint[2, 0] = True  # 未知GT节点不能成为合法端点。
    original = gt > 0
    raw, audit = edge_roi_mask(tile, gt, original, footprint, ids, direction)
    np.testing.assert_array_equal(raw, _manual_edges(tile, footprint, ids, direction))
    assert raw.dtype == bool and raw.shape == (1, 8, 6, 6)
    assert raw.any()
    assert sum(row['raw_candidate_edges'] for row in audit['per_channel']) == raw.sum()
    assert audit['footprint_unknown_GT_nodes'] == 1
    assert audit['requires_existing_supervision_intersection']
    assert not audit['native_path_trust_applied']
    assert not audit['original_coverage_projected_to_grid']
    if direction == 'positive':
        assert raw[0, 0, 1, 1]  # 只有目标端点(1,2)在ROI中，来源不在。
    else:
        assert raw[0, 0, 4, 2] and raw[0, 6, 4, 1]  # 短/长关系分别看端点。
        assert not raw[0, 6, 4, 2]  # +4已超图，不得用夹紧端点伪造关系。


def test_original_is_not_a_new_trust_projection_and_candidates_are_not_valid_edges():
    gt = np.full((6, 6), 5, np.int32)
    tile = _tile(gt, input_size=6, grid=6)
    footprint = np.zeros(gt.shape, bool)
    footprint[2, 2] = True
    covered = np.ones(gt.shape, bool)
    filled = np.zeros(gt.shape, bool)
    a, audit_a = edge_roi_mask(tile, gt, covered, footprint, [5], 'positive')
    b, audit_b = edge_roi_mask(tile, gt, filled, footprint, [5], 'positive')
    np.testing.assert_array_equal(a, b)
    assert audit_a['native_tile_original_covered_pixels'] == 36
    assert audit_b['native_tile_filled_known_pixels'] == 36
    assert audit_a['footprint_nodes'] == 1
    assert audit_a['raw_candidate_edges'] > audit_a['footprint_nodes']


def test_padded_grid_and_empty_footprint_do_not_make_candidate_edges():
    gt = np.full((2, 10), 5, np.int32)
    tile = _tile(gt, grid=4)
    original = np.ones(gt.shape, bool)
    raw, audit = edge_roi_mask(tile, gt, original, np.zeros((4, 4), bool), [5], 'positive')
    assert not raw.any() and audit['raw_candidate_edges'] == 0
    raw, _ = edge_roi_mask(tile, gt, original, tile['valid'].copy(), [5], 'positive')
    assert not raw[:, :, 1:].any()  # 一行有效格，纵向关系也不得进入padding。
    assert not raw[0, [1, 2, 3, 5, 7]].any()  # 所有纵/斜关系无第二个有效端点。
    assert raw[0, 4].any()  # 横向+2关系仍按两个有效端点选中。
    np.testing.assert_array_equal(raw, _manual_edges(tile, tile['valid'], [5], 'positive'))


@pytest.mark.parametrize('mutation', [
    'gt_float', 'original_uint8', 'original_shape', 'original_unknown',
    'footprint_uint8', 'footprint_shape', 'footprint_padding', 'grid_gt_mismatch',
    'invalid_direction', 'positive_two_ids', 'negative_one_id', 'duplicate_ids',
    'boolean_id', 'unknown_id',
])
def test_raw_edge_input_contract_rejections(mutation):
    gt = np.full((3, 7), 5, np.int32)
    gt[0, 0] = 0
    tile = _tile(gt)
    original = gt > 0
    footprint = tile['valid'].copy()
    ids, direction = [5], 'positive'
    if mutation == 'gt_float':
        gt = gt.astype(np.float32)
    elif mutation == 'original_uint8':
        original = original.astype(np.uint8)
    elif mutation == 'original_shape':
        original = original[:-1]
    elif mutation == 'original_unknown':
        original[0, 0] = True
    elif mutation == 'footprint_uint8':
        footprint = footprint.astype(np.uint8)
    elif mutation == 'footprint_shape':
        footprint = footprint[:-1]
    elif mutation == 'footprint_padding':
        footprint[-1, -1] = True
    elif mutation == 'grid_gt_mismatch':
        tile['grid_gt'][0, 1] = 7
    elif mutation == 'invalid_direction':
        direction = 'same'
    elif mutation == 'positive_two_ids':
        ids = [5, 7]
    elif mutation == 'negative_one_id':
        direction = 'negative'
    elif mutation == 'duplicate_ids':
        ids, direction = [5, 5], 'negative'
    elif mutation == 'boolean_id':
        ids = [True]
    elif mutation == 'unknown_id':
        ids = [7]
    with pytest.raises(ValueError):
        edge_roi_mask(tile, gt, original, footprint, ids, direction)


def test_raw_edge_selection_does_not_mutate_any_array():
    gt = np.full((6, 6), 5, np.int32)
    tile = _tile(gt, input_size=6, grid=6)
    original = np.ones(gt.shape, bool)
    footprint = np.ones((6, 6), bool)
    before = [value.copy() for value in (gt, original, footprint, tile['grid_gt'], tile['valid'])]
    edge_roi_mask(tile, gt, original, footprint, [5], 'positive')
    for value, prior in zip((gt, original, footprint, tile['grid_gt'], tile['valid']), before):
        np.testing.assert_array_equal(value, prior)
