# -*- coding: utf-8 -*-
"""透明捕获正式窗口的冻结encoder特征，并真实重放独立affinity头。

特征仅留在内存，不序列化；device/storage/stride保持实际forward输入。
真实数据必须使用原CUDA sam2_env；CPU仅服务虚构合同测试。
"""
from __future__ import annotations

import hashlib
import inspect
import json
from unittest.mock import patch

import numpy as np
import torch

from tools.affinity_bottleneck_capture import capture_native_tiles
from train_backend_adaptation import fusion_options
from utils.affinity_fusion import affinity_boundary_probability
from utils.patch_diagnostic import BlendMap, tile_boxes


def _same_device(actual, expected):
    return actual.type == expected.type and (expected.index is None or actual.index == expected.index)


def _tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _features_metadata(features, device, channels=None):
    if not isinstance(features, (list, tuple)) or not features:
        raise RuntimeError('Actual encoder features must be a nonempty list/tuple of tensors')
    if channels is not None and len(features) != len(channels):
        raise RuntimeError('Actual feature count differs from affinity FPN interface')
    records = []
    for index, value in enumerate(features):
        if (not torch.is_tensor(value) or value.ndim != 4 or value.shape[0] != 1
                or min(value.shape[1:]) < 1
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise RuntimeError('Actual frozen features must be finite FP32 singleton BCHW tensors')
        if channels is not None and int(value.shape[1]) != int(channels[index]):
            raise RuntimeError('Actual feature channels differ from affinity FPN interface')
        if value.requires_grad:
            raise RuntimeError('Captured encoder feature must already have requires_grad=False')
        if not _same_device(value.device, device):
            raise RuntimeError('Actual feature device differs from original tile device')
        records.append(dict(shape=list(value.shape), stride=list(value.stride()),
            dtype=str(value.dtype), device=str(value.device), storage_offset=int(value.storage_offset()),
            requires_grad=False, sha256=_tensor_sha(value)))
    return records


def _metadata_sha(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _head_channels(head):
    fpn = getattr(head, 'geometry_fpn', None)
    lateral = getattr(fpn, 'lateral_convs', None)
    if lateral is None:
        return None
    channels = [int(layer.in_channels) for layer in lateral]
    if not channels or int(getattr(fpn, 'num_stages', len(channels))) != len(channels):
        raise RuntimeError('Affinity FPN stage-count/channel interface is inconsistent')
    return channels


def validate_frozen_tile_features(tile, head=None):
    """核对feature数值及shape/device/stride/冻结状态，返回同一metadata SHA。

features_sha256同时包含来源布局，不等于tensor_digest仅数值的哈希。
head可选，仅用于按真实FPN接口额外检查；不会执行任何forward。
"""
    if not isinstance(tile, dict) or not torch.is_tensor(tile.get('logits')):
        raise ValueError('Actual tile cache must include original-device logits')
    channels = _head_channels(head) if head is not None else None
    records = _features_metadata(tile.get('features'), tile['logits'].device, channels)
    digest = _metadata_sha(records)
    if records != tile.get('features_metadata') or digest != tile.get('features_sha256'):
        raise RuntimeError('Frozen feature cache differs from actual capture')
    return digest


def capture_frozen_features(views, model, restorer, path, config, gt, device, seed):
    """返回原rgb/captured/tiles/audit，tile追加features及features_metadata。

仅实际boundary_grid调用内激活pre-hook，整图语义/其他head调用不捕获。
    对原函数签名明确命名的image Tensor核对[1,3,1024,1024]；非Tensor或
    不明确的签名不猜位置。特征数/通道按实际affinity FPN接口核对，空间
    shape原样记录，不写死SAM2级数。所有输入已冻结，不以detach掩盖未冻结。
"""
    requested = torch.device(device)
    channels = _head_channels(model.affinity_decoder)
    for root in (model, getattr(model, 'encoder', None), model.affinity_decoder, restorer):
        if root is not None and callable(getattr(root, 'parameters', None)):
            if any(parameter.requires_grad for parameter in root.parameters()):
                raise RuntimeError('Original model/restorer parameters must be frozen before capture')
    original_grid = views.boundary_grid
    try:
        grid_signature = inspect.signature(original_grid)
    except (TypeError, ValueError):
        grid_signature = None
    pending = {}
    observed = []

    def observe_input(_module, inputs):
        if not pending.get('active'):
            return None
        if 'features' in pending:
            raise RuntimeError('More than one affinity head input inside an actual tile')
        if not isinstance(inputs, tuple) or len(inputs) != 1:
            raise RuntimeError('Actual affinity head must receive one positional feature list')
        records = _features_metadata(inputs[0], requested, channels)
        # detach保留原storage/stride；pre-hook不替换任何输入或输出。
        pending['features'] = [value.detach() for value in inputs[0]]
        pending['features_metadata'] = records
        return None

    def observe_grid(*args, **kwargs):
        if pending:
            raise RuntimeError('Nested actual boundary_grid feature capture is unsupported')
        image = None
        if grid_signature is not None:
            image = grid_signature.bind(*args, **kwargs).arguments.get('image')
        checked = torch.is_tensor(image)
        if checked and (tuple(image.shape) != (1, 3, 1024, 1024)
                or image.dtype != torch.float32 or not bool(torch.isfinite(image).all())
                or image.requires_grad or not _same_device(image.device, requested)):
            raise RuntimeError('Actual tile RGB must be frozen finite FP32 [1,3,1024,1024] on original device')
        pending['active'] = True
        try:
            grid = original_grid(*args, **kwargs)
            if 'features' not in pending:
                raise RuntimeError('Actual tile did not call the affinity feature pre-hook')
            record = dict(features=pending['features'], features_metadata=pending['features_metadata'],
                          feature_input_tensor_checked=checked)
            if _features_metadata(record['features'], requested, channels) != record['features_metadata']:
                raise RuntimeError('Actual head mutated its frozen encoder input')
            record['features_sha256'] = _metadata_sha(record['features_metadata'])
            observed.append(record)
            return grid
        finally:
            pending.clear()

    hook = model.affinity_decoder.register_forward_pre_hook(observe_input)
    try:
        with patch.object(views, 'boundary_grid', side_effect=observe_grid):
            rgb, captured, tiles, audit = capture_native_tiles(
                views, model, restorer, path, config, gt, device, seed)
    finally:
        hook.remove()
    if pending or len(observed) != len(tiles):
        raise RuntimeError('Actual tile features and original captured tile count differ')
    for tile, record in zip(tiles, observed):
        tile.update(record)
    audit = dict(audit, format='affinity_feature_capture_v1',
        frozen_encoder_features_exact=True, single_head_input_per_tile=True,
        head_input_scope='actual native boundary_grid only; whole-image inputs ignored',
        feature_shapes='actual singleton BCHW shapes; no assumed SAM2 stage count',
        affinity_FPN_input_channels=channels,
        actual_feature_counts=[len(record['features']) for record in observed],
        explicit_image_tensor_checks=sum(record['feature_input_tensor_checked'] for record in observed),
        no_feature_serialization=True, no_feature_storage_or_stride_conversion=True,
        feature_capture_tile_count=len(observed))
    return rgb, captured, tiles, audit


def replay_head_boundary(views, tiles, head, config, shape, *, empty_exact=False):
    """返回(native HW float32边界, audit)，每个tile都执行真实head/fuse/crop。

head须eval；其参数可训练，本函数no_grad不清除或改写梯度。empty_exact
严格对照原logits/stride、grid/native和原顺序完整Blend；不复用old native。
特征哈希前后检查防止候选head污染冻结cache。
"""
    if head.training:
        raise ValueError('Head boundary replay requires eval mode')
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in shape):
        raise ValueError('Native shape must contain two positive integers')
    if config['affinity_native']['overlap'] != .25:
        raise ValueError('Head replay requires original native1024 overlap .25')
    boxes = tile_boxes(tuple(shape), 1024, .25)
    if not isinstance(tiles, (list, tuple)) or len(tiles) != len(boxes):
        raise ValueError('Complete original-order tile feature cache required')
    mode, options = fusion_options(config)
    blend = BlendMap(tuple(shape))
    old_blend = BlendMap(tuple(shape)) if empty_exact else None
    rows = []
    with torch.no_grad():
        for index, (tile, expected_box) in enumerate(zip(tiles, boxes)):
            if tile.get('index') != index or list(tile.get('box', ())) != list(expected_box):
                raise RuntimeError('Head replay changed original tile order or boxes')
            old_logits = tile['logits']
            if (not torch.is_tensor(old_logits) or tuple(old_logits.shape) != (1, 8, 512, 512)
                    or old_logits.dtype != torch.float32 or old_logits.requires_grad
                    or not bool(torch.isfinite(old_logits).all())):
                raise RuntimeError('Original cached tile logits contract differs')
            features = tile['features']
            validate_frozen_tile_features(tile, head)
            records = tile['features_metadata']
            result = head(features)
            if not isinstance(result, dict) or 'affinity_logits' not in result:
                raise RuntimeError('Replayed head must return affinity_logits')
            logits = result['affinity_logits']
            if (not torch.is_tensor(logits) or tuple(logits.shape) != (1, 8, 512, 512)
                    or logits.dtype != torch.float32 or logits.device != old_logits.device
                    or not bool(torch.isfinite(logits).all())):
                raise RuntimeError('Replayed tile logits must be finite FP32 on original device and shape')
            if _features_metadata(features, old_logits.device) != records:
                raise RuntimeError('Replayed head mutated frozen feature cache')
            grid = affinity_boundary_probability(logits, mode=mode, **options)
            y, x, y1, x1 = expected_box
            native = views.crop_letterbox_output(grid, 1024, tile['pad_h'], tile['pad_w'], (y1-y, x1-x))
            if (not torch.is_tensor(native) or tuple(native.shape) != (1, 1, y1-y, x1-x)
                    or native.dtype != torch.float32 or not bool(torch.isfinite(native).all())):
                raise RuntimeError('Replayed native tile crop contract differs')
            actual_native = native.detach().cpu().numpy()
            exact = dict(logits=bool(torch.equal(logits, old_logits)),
                logits_stride=list(logits.stride()) == list(old_logits.stride()),
                grid=bool(np.array_equal(grid.detach().cpu().numpy(), tile['grid'])),
                native=bool(np.array_equal(actual_native, tile['native'])))
            if empty_exact and not all(exact.values()):
                raise RuntimeError('Actual empty head replay logits/grid/native is not exact at tile ' + str(index))
            blend.add(expected_box, actual_native[0, 0])
            if old_blend is not None:
                old_blend.add(expected_box, tile['native'][0, 0])
            rows.append(dict(index=index, box=list(expected_box), features_sha256=tile['features_sha256'],
                actual_head_recomputed=True, actual_fusion_recomputed=True, actual_native_recomputed=True,
                reused_old_native=False, output_logits_shape=list(logits.shape),
                output_logits_stride=list(logits.stride()), device=str(logits.device),
                empty_checks=exact))
    boundary, blend_audit = blend.finish()
    blend_audit['tiles'] = len(tiles)
    whole_exact = None
    if old_blend is not None:
        baseline, old_audit = old_blend.finish()
        old_audit['tiles'] = len(tiles)
        whole_exact = bool(np.array_equal(boundary, baseline) and blend_audit == old_audit)
        if not whole_exact:
            raise RuntimeError('Actual empty head replay full original-order native BlendMap is not exact')
    return boundary, dict(format='affinity_head_boundary_replay_v1', tiles=rows,
        tile_count=len(tiles), blend=blend_audit, output_shape=list(shape), dtype='float32',
        all_tiles_actual_head_fusion_crop=True, no_old_native_reuse=True,
        frozen_features_unchanged=True, empty_exact_requested=bool(empty_exact),
        whole_baseline_exact=whole_exact, no_model_encoder_or_restorer_forward=True)
