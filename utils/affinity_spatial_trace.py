# -*- coding: utf-8 -*-
"""原生关注区到实际插值网格的足迹，以及逐通道关系候选。

足迹只回答哪些标量网格位置以非零系数参与指定原生像素的双线性插值。
它不等同于网络感受野或可信监督。关系候选必须另与既有可信边监督相交，
不能把被关注的节点数解释为可训练边数，也不能用最近邻代替插值支持。
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np
import torch

from utils.affinity_deployment import crop_letterbox_output
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS, _edge_slices
from utils.offset_letterbox import (
    geometry_letterbox_metadata,
    letterbox_instance_geometry,
)


def _sha(array):
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(str(value.shape).encode('ascii'))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _integer(value, name, *, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < minimum):
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _bool_hw(value, name):
    if (not isinstance(value, np.ndarray) or value.ndim != 2 or not value.size
            or value.dtype != np.bool_):
        raise ValueError(f'{name} must be a nonempty numpy bool HW map')
    return value


def _gt_hw(value, name):
    if (not isinstance(value, np.ndarray) or value.ndim != 2 or not value.size
            or not np.issubdtype(value.dtype, np.integer) or value.dtype == np.bool_
            or np.any(value < 0) or int(value.max()) > np.iinfo(np.int32).max):
        raise ValueError(f'{name} must be a nonempty nonnegative int32-range integer HW map')
    return value


def _tile_geometry(tile):
    if not isinstance(tile, Mapping):
        raise ValueError('tile must be a captured tile mapping')
    try:
        box_values = tile['box']
        geometry = tile['geometry']
        grid_gt = _gt_hw(tile['grid_gt'], 'tile grid_gt')
        valid = _bool_hw(tile['valid'], 'tile valid')
    except KeyError as exc:
        raise ValueError(f'missing captured tile field: {exc.args[0]}') from exc
    if (not isinstance(box_values, (Sequence, np.ndarray))
            or isinstance(box_values, (str, bytes))
            or (isinstance(box_values, np.ndarray) and box_values.ndim != 1)
            or len(box_values) != 4):
        raise ValueError('tile box must contain four integer native bounds')
    y, x, y1, x1 = [_integer(value, 'tile box') for value in box_values]
    if y1 <= y or x1 <= x:
        raise ValueError('tile box must have positive native height and width')
    if not isinstance(geometry, Mapping):
        raise ValueError('tile geometry must be a captured metadata mapping')
    keys = ('original_height', 'original_width', 'input_size', 'output_grid',
            'resized_height', 'resized_width', 'content_height', 'content_width')
    try:
        metadata = {key: _integer(geometry[key], 'geometry '+key,
                                 minimum=0 if key.startswith('resized_') else 1)
                    for key in keys}
        pad_h = _integer(tile['pad_h'], 'tile pad_h')
        pad_w = _integer(tile['pad_w'], 'tile pad_w')
    except KeyError as exc:
        raise ValueError(f'missing captured geometry/pad field: {exc.args[0]}') from exc
    expected = geometry_letterbox_metadata(
        (y1-y, x1-x), metadata['input_size'], metadata['output_grid']).to_dict()
    if metadata != expected:
        raise ValueError('tile geometry differs from actual box/letterbox arithmetic')
    if (pad_h != metadata['input_size']-metadata['resized_height']
            or pad_w != metadata['input_size']-metadata['resized_width']):
        raise ValueError('tile padding differs from actual letterbox geometry')
    grid_shape = (metadata['output_grid'], metadata['output_grid'])
    if grid_gt.shape != grid_shape or valid.shape != grid_shape:
        raise ValueError('tile grid_gt/valid shape differs from captured output grid')
    expected_valid = np.zeros(grid_shape, bool)
    expected_valid[:metadata['content_height'], :metadata['content_width']] = True
    if not np.array_equal(valid, expected_valid):
        raise ValueError('tile valid must be the exact actual crop-content rectangle')
    if np.any(grid_gt[~valid] != 0):
        raise ValueError('tile grid_gt must retain unknown zero in padded positions')
    return (y, x, y1, x1), metadata, grid_gt, valid, pad_h, pad_w


def native_grid_footprint(tile, native_mask, *, device='cpu'):
    """返回 ``(grid_bool_HW, audit)``；输入为整幅原生 ``numpy bool HW``。

    先按tile.box裁关注区，再反向合并实际crop_letterbox_output的非零
    插值支持。临时全零FP32标量格使用线性插值的输入梯度：关注像素给正1，
    因所有插值系数非负，梯度>0恰为非零支持并集，零权重邻点不计入。
    没有模型/参数梯度，没有优化器；局部使用autograd并恢复外层上下文。
    默认仅CPU算子足迹；显式传device可在对应设备计算，不代表网络前向复现。
    既有capture/logits/features都不进入图。仅临时格和窗口矩阵，不存系数矩阵。
    """
    native_mask = _bool_hw(native_mask, 'native_mask')
    box, metadata, _, valid, pad_h, pad_w = _tile_geometry(tile)
    y, x, y1, x1 = box
    if y1 > native_mask.shape[0] or x1 > native_mask.shape[1]:
        raise ValueError('tile box must be contained in full native_mask')
    try:
        interpolation_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError('device must be a valid torch device') from exc
    if interpolation_device.type not in ('cpu', 'cuda'):
        raise ValueError('interpolation footprint supports CPU or explicit CUDA only')
    selected = native_mask[y:y1, x:x1]
    footprint = np.zeros(valid.shape, bool)
    selected_count = int(selected.sum())
    if selected_count:
        # 关闭可能来自外层的inference_mode，且只对全新标量格启用输入梯度。
        with torch.inference_mode(False), torch.enable_grad():
            dummy = torch.zeros((1, 1, *valid.shape), dtype=torch.float32,
                                device=interpolation_device, requires_grad=True)
            native = crop_letterbox_output(
                dummy, metadata['input_size'], pad_h, pad_w, (y1-y, x1-x))
            weights = torch.as_tensor(selected.copy(), dtype=torch.float32,
                                      device=interpolation_device)[None, None]
            gradient, = torch.autograd.grad(native, dummy, grad_outputs=weights)
            if (not bool(torch.isfinite(gradient).all())
                    or bool((gradient < 0).any())):
                raise RuntimeError('actual interpolation support derivative must be finite nonnegative')
            footprint = (gradient[0, 0] > 0).detach().cpu().numpy().copy()
    if np.any(footprint & ~valid):
        raise AssertionError('crop interpolation must never use padded grid positions')
    audit = dict(
        format='affinity_native_grid_footprint_v1', box=list(box),
        native_mask_shape=list(native_mask.shape), native_tile_shape=[y1-y, x1-x],
        grid_shape=list(valid.shape), content_shape=[metadata['content_height'],
                                                  metadata['content_width']],
        input_size=metadata['input_size'], pad_h=pad_h, pad_w=pad_w,
        operator='actual crop_letterbox_output: content crop then bilinear align_corners=True',
        support_rule='union of strictly positive interpolation coefficients for selected native pixels',
        computation='temporary scalar interpolation input autograd, not network autograd',
        interpolation_device=str(interpolation_device), interpolation_dtype='FP32',
        no_model_or_parameter_gradients=True, no_optimizer=True,
        model_forward_replayed=False, nearest_mapping_used=False,
        native_selected_pixels_in_tile=selected_count,
        native_selected_pixels_outside_tile=int(native_mask.sum())-selected_count,
        contributing_grid_nodes=int(footprint.sum()), padding_grid_nodes=0,
        native_mask_sha256=_sha(native_mask), footprint_sha256=_sha(footprint),
        coefficient_matrix_saved=False,
        caveat='Scalar interpolation support only; not network receptive field or trustworthy edge supervision')
    return footprint, audit


def edge_roi_mask(tile, gt, original, node_footprint, gt_ids, direction):
    """返回 ``(raw_candidates_bool_1x8xHxW, audit)``，不生成可信训练mask。

    direction='negative'须给两个不同正GT ID，选择跨这两个ID的关系；
    direction='positive'须给一个正GT ID，选择其内部同ID关系。每通道位置
    以DEFAULT_AFFINITY_OFFSETS的两个grid_GT端点为准，且至少一端在足迹内。
    两端均须在有效格内且GT已知。original只验证/记录原覆盖来源，不被
    最近邻投影成新的信任条件；必须另与现有完整GT/native可信监督相交。
    """
    gt = _gt_hw(gt, 'gt')
    original = _bool_hw(original, 'original')
    if original.shape != gt.shape or np.any(original & (gt == 0)):
        raise ValueError('original must match native GT shape and cover known GT only')
    box, metadata, grid_gt, valid, _, _ = _tile_geometry(tile)
    y, x, y1, x1 = box
    if y1 > gt.shape[0] or x1 > gt.shape[1]:
        raise ValueError('tile box must be contained in native GT/original')
    node_footprint = _bool_hw(node_footprint, 'node_footprint')
    if node_footprint.shape != valid.shape or np.any(node_footprint & ~valid):
        raise ValueError('node_footprint must match grid shape and exclude padding')
    expected_grid, expected_valid, _ = letterbox_instance_geometry(
        gt[y:y1, x:x1], metadata['input_size'], metadata['output_grid'])
    if (not np.array_equal(grid_gt, expected_grid)
            or not np.array_equal(valid, expected_valid)):
        raise ValueError('tile grid_gt must be the actual nearest letterbox of the specified native GT')
    if not isinstance(direction, str) or direction not in ('positive', 'negative'):
        raise ValueError('direction must be positive or negative')
    if (not isinstance(gt_ids, (Sequence, np.ndarray))
            or isinstance(gt_ids, (str, bytes))
            or (isinstance(gt_ids, np.ndarray) and gt_ids.ndim != 1)):
        raise ValueError('gt_ids must be a sequence of positive integer IDs')
    ids = tuple(_integer(value, 'gt_ids', minimum=1) for value in gt_ids)
    required = 1 if direction == 'positive' else 2
    if len(ids) != required or len(set(ids)) != required:
        raise ValueError(f'{direction} requires {required} distinct positive GT IDs')
    if any(not np.any(gt == gid) for gid in ids):
        raise ValueError('every selected GT ID must exist in the specified native GT')
    height, width = valid.shape
    raw = np.zeros((1, len(DEFAULT_AFFINITY_OFFSETS), height, width), bool)
    per_channel = []
    for channel, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        relation_count = spatial_count = 0
        if height > abs(dy) and width > abs(dx):
            source, dest = _edge_slices(height, width, dy, dx)
            left, right = grid_gt[source], grid_gt[dest]
            known_endpoints = valid[source] & valid[dest] & (left > 0) & (right > 0)
            spatial = known_endpoints & (node_footprint[source] | node_footprint[dest])
            if direction == 'positive':
                relation = (left == ids[0]) & (right == ids[0])
            else:
                relation = (((left == ids[0]) & (right == ids[1]))
                            | ((left == ids[1]) & (right == ids[0])))
            raw[0, channel][source] = spatial & relation
            relation_count = int((known_endpoints & relation).sum())
            spatial_count = int(spatial.sum())
        per_channel.append(dict(channel=channel, offset=[dy, dx],
                                known_endpoint_spatial_edges=spatial_count,
                                target_relation_edges_in_valid_grid=relation_count,
                                raw_candidate_edges=int(raw[0, channel].sum())))
    target_nodes = np.isin(grid_gt, ids)
    tile_gt, tile_original = gt[y:y1, x:x1], original[y:y1, x:x1]
    audit = dict(
        format='affinity_edge_roi_candidates_v1', box=list(box), direction=direction,
        gt_ids=list(ids), grid_shape=[height, width], raw_mask_shape=list(raw.shape),
        affinity_target=1 if direction == 'positive' else 0,
        rule='both actual grid_GT endpoints match target relation; at least one endpoint in interpolation footprint',
        footprint_nodes=int(node_footprint.sum()),
        footprint_unknown_GT_nodes=int((node_footprint & (grid_gt == 0)).sum()),
        footprint_other_GT_nodes=int((node_footprint & (grid_gt > 0) & ~target_nodes).sum()),
        footprint_target_GT_nodes=int((node_footprint & target_nodes).sum()),
        raw_candidate_edges=int(raw.sum()), per_channel=per_channel,
        native_tile_original_covered_pixels=int(tile_original.sum()),
        native_tile_filled_known_pixels=int(((tile_gt > 0) & ~tile_original).sum()),
        native_tile_unknown_pixels=int((tile_gt == 0).sum()),
        grid_GT_actual_nearest_verified=True, native_path_trust_applied=False,
        original_coverage_projected_to_grid=False,
        requires_existing_supervision_intersection=True,
        footprint_sha256=_sha(node_footprint), raw_mask_sha256=_sha(raw),
        caveat='Raw spatial relation candidates only, not trustworthy or valid training-edge counts')
    return raw, audit
