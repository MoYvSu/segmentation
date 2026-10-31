# -*- coding: utf-8 -*-
"""双输入NAF的有界原生分窗；不加载后端、RGB或生产checkpoint。"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from models.boundary_dual import INPUT_MODE
from tools.boundary_restore_readout import _evaluation, _weights, tiles
from utils.boundary_input import input_normalization_settings, normalize_boundary_input


FORMAT = 'boundary_dual_tiled_probability_v1'
MAX_NATIVE_PIXELS = 6_000_000
MAX_PADDED_PIXELS = 8_000_000
MAX_TILES = 4096
MAX_BATCH_ELEMENTS = 16_777_216


def _sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def _validate(boundary, model, settings, use_continuous):
    # 必须先检查头/shape；逐元素finite检查与任何padding均在尺寸门槛之后。
    if (not isinstance(boundary, np.ndarray) or boundary.dtype != np.float32
            or boundary.ndim != 2 or not boundary.size):
        raise ValueError('Dual boundary must be nonempty native float32 HW probability')
    if boundary.size > MAX_NATIVE_PIXELS:
        raise ValueError('Dual boundary exceeds bounded 6M native pixels')
    if not isinstance(settings, dict) or type(use_continuous) is not bool:
        raise ValueError('Dual settings must be dict and use_continuous must be bool')
    if (getattr(model, 'input_mode', None) != INPUT_MODE
            or settings.get('input_mode', INPUT_MODE) != INPUT_MODE):
        raise ValueError('Dual tiled input_mode must match boundary_continuous model')
    threshold = settings.get('input_threshold', .5)
    if type(threshold) not in (int, float) or float(threshold) != .5:
        raise ValueError('Dual binary input threshold must remain 0.5')
    if settings.get('iterations', 1) != 1 or settings.get('revert_unconnected', False):
        raise ValueError('Dual NAF uses one pass without post-restoration revert')
    normalization = input_normalization_settings(settings.get('input_normalization'))
    if normalization != dict(mode='skeleton_dilate', kernel_size=8):
        raise ValueError('Dual input requires whole-native T8 skeleton_dilate/kernel8')
    size = settings.get('tile_size', 256)
    overlap = settings.get('overlap', 64)
    pad = settings.get('reflect_pad', 32)
    batch_size = settings.get('batch_size', 16)
    if (type(size) is not int or size < 1 or type(overlap) is not int
            or not 0 <= overlap < size or type(pad) is not int or pad < 0
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError('Invalid dual native tile geometry or batch size')
    shape = tuple(int(v + 2 * pad) for v in boundary.shape)
    if shape[0] * shape[1] > MAX_PADDED_PIXELS:
        raise ValueError('Dual reflect padding exceeds bounded 8M pixels')
    if batch_size * 2 * size * size > MAX_BATCH_ELEMENTS:
        raise ValueError('Dual tile batch exceeds bounded 16M elements')

    def count(length):
        if length <= size:
            return 1
        distance, stride = length - size, size - overlap
        return distance // stride + 1 + int(distance % stride != 0)

    if count(shape[0]) * count(shape[1]) > MAX_TILES:
        raise ValueError('Dual native tiling exceeds bounded 4096 windows')
    if not np.isfinite(boundary).all() or np.any((boundary < 0) | (boundary > 1)):
        raise ValueError('Dual boundary must be finite within [0,1]')
    return size, overlap, pad, batch_size


def predict_boundary_dual_tiled(boundary, model, settings, *, device='cpu', use_continuous=True):
    """整图生成一次T8，同源连续B同步reflect后分窗，按旧顺序拼接概率。

    ``use_continuous=False``只关闭新分支，仍运行NAF，并不返回原N1热图。
    一次只持有一个源图和一个小批；不缓存所有窗口输入/输出，不运行WS。
    """
    size, overlap, pad, batch_size = _validate(boundary, model, settings, use_continuous)
    before_sha = _sha(boundary)
    binary, normalization = normalize_boundary_input(boundary, threshold=.5,
        settings=settings['input_normalization'])
    source = binary.astype(np.float32)
    padded = np.pad(source, pad, mode='reflect') if pad else source
    padded_continuous = np.pad(boundary, pad, mode='reflect') if pad else boundary
    boxes = tiles(padded.shape, size, overlap)
    total = np.zeros(padded.shape, np.float32)
    weights = np.zeros_like(total)
    target = torch.device(device)
    with _evaluation(model), torch.autocast(device_type=target.type, enabled=False):
        for start in range(0, len(boxes), batch_size):
            chunk = boxes[start:start + batch_size]
            inputs = []
            for y, x, y1, x1 in chunk:
                a = padded[y:y1, x:x1]
                b = padded_continuous[y:y1, x:x1]
                h, w = a.shape
                if (h, w) != (size, size):
                    a = np.pad(a, ((0, size-h), (0, size-w)), mode='reflect')
                    b = np.pad(b, ((0, size-h), (0, size-w)), mode='reflect')
                inputs.append(np.stack((a, b), axis=0))
            tensor = torch.from_numpy(np.stack(inputs).copy()).to(target)
            prediction = model(tensor, use_continuous=use_continuous)
            if (not torch.is_tensor(prediction) or not prediction.is_floating_point()
                    or tuple(prediction.shape) != (len(chunk), 1, size, size)
                    or not bool(torch.isfinite(prediction).all())
                    or bool(((prediction < 0) | (prediction > 1)).any())):
                raise RuntimeError('Dual NAF output must be finite B1HW probability in [0,1]')
            arrays = prediction.detach().float().cpu().numpy()[:, 0]
            for output, (y, x, y1, x1) in zip(arrays, chunk):
                h, w = y1-y, x1-x
                weight = _weights(h, w)
                total[y:y1, x:x1] += output[:h, :w] * weight
                weights[y:y1, x:x1] += weight
            # 明确只持有当前批，避免下一批创建时叠加上一批的设备张量。
            del prediction, arrays, tensor, inputs
    if not np.all(weights > 0):
        raise RuntimeError('Dual NAF tiling missed native pixels')
    result = total / weights
    if pad:
        result = result[pad:-pad, pad:-pad]
    result = np.clip(result, 0, 1).astype(np.float32)
    if _sha(boundary) != before_sha:
        raise RuntimeError('Dual NAF tiling changed frozen continuous N1 input')
    return result, dict(format=FORMAT, bypass=False, input_mode=INPUT_MODE,
        input_order=['binary_boundary_T8', 'continuous_N1_boundary'], input_threshold=.5,
        use_continuous=use_continuous, continuous_input_binarized=False,
        input_boundary_sha256=before_sha, input_continuous_sha256=before_sha,
        input_normalization=normalization, output_sha256=_sha(result),
        native_shape=list(boundary.shape), tiles=len(boxes), tile_size=size,
        overlap=overlap, reflect_pad=pad, batch_size=batch_size,
        iterations=1, revert_unconnected=False, native_resize=False,
        stitching='direct_probability_weighted_mean',
        absolute_change_mean=float(np.abs(result-boundary).mean()))
