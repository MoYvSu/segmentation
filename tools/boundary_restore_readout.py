# -*- coding: utf-8 -*-
"""只替换原生边界概率的残差补全；固定语义与原 reflect 解码/最终投票。

拼接有符号残差，而不是反复混合原边界。strength=0 或全零残差直接返回
原数组，保证关闭补全时不会因为浮点加权产生新边界或实例变化。
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib

import numpy as np
import torch


def _probability(value, name):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 2
            or not value.size or not np.isfinite(value).all()
            or np.any((value < 0) | (value > 1))):
        raise ValueError(f'{name} must be a nonempty native float32 HW probability in [0,1]')
    return value


def tiles(shape, tile_size=512, overlap=128):
    """按行返回 (y,x,y1,x1)，overlap 的单位是像素。"""
    if (len(shape) != 2 or any(type(v) not in (int, np.int32, np.int64) or v < 1 for v in shape)
            or type(tile_size) is not int or tile_size < 1
            or type(overlap) is not int or not 0 <= overlap < tile_size):
        raise ValueError('Invalid native tile geometry')
    def starts(length):
        if length <= tile_size:
            return [0]
        values = list(range(0, length-tile_size+1, tile_size-overlap))
        if values[-1] != length-tile_size:
            values.append(length-tile_size)
        return values
    height, width = shape
    return [(y, x, min(height, y+tile_size), min(width, x+tile_size))
            for y in starts(height) for x in starts(width)]


def _weights(height, width):
    wy = np.maximum(.05, 1-np.abs(np.linspace(-1, 1, height, dtype=np.float32)))
    wx = np.maximum(.05, 1-np.abs(np.linspace(-1, 1, width, dtype=np.float32)))
    return wy[:, None]*wx[None, :]


@contextmanager
def _evaluation(model):
    # monitor 可能在训练循环内调用；结束时恢复每层原有 train/eval 标志。
    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for module, training in flags:
            module.training = training


def predict_residual_tiled(boundary, model, *, device='cpu', tile_size=512,
                           overlap=128, reflect_pad=32, strength=1.0):
    """返回 (native_boundary, audit)，不运行任何上游模型或改变 decoder。

模型必须显式提供 residual(x) 或 predict_residual(x)，输出有符号 B1HW。
不能把模型 forward 返回的修复概率误当残差。全图外缘先 reflect，再分窗；
不足一窗时继续 reflect 到固定窗口尺寸，填充部分不进入有效输出。
"""
    boundary = _probability(boundary, 'boundary')
    if not np.isfinite(strength) or not 0 <= float(strength) <= 1:
        raise ValueError('strength must lie in [0,1]')
    if type(reflect_pad) is not int or reflect_pad < 0:
        raise ValueError('reflect_pad must be a nonnegative integer')
    boxes = tiles((boundary.shape[0]+2*reflect_pad, boundary.shape[1]+2*reflect_pad), tile_size, overlap)
    audit = dict(strength=float(strength), tile_size=tile_size, overlap=overlap,
                 reflect_pad=reflect_pad, tiles=0, bypass=False, all_zero_residual=False,
                 boundary_input_sha256=hashlib.sha256(boundary.tobytes()).hexdigest())
    if float(strength) == 0:
        audit.update(bypass=True, boundary_exact=True, reason='strength_zero')
        return boundary, audit
    residual_method = getattr(model, 'residual', None)
    if not callable(residual_method):
        residual_method = getattr(model, 'predict_residual', None)
    if not callable(residual_method):
        raise ValueError('Model must expose an explicit residual method; forward probabilities are forbidden')
    padded = np.pad(boundary, reflect_pad, mode='reflect') if reflect_pad else boundary
    total = np.zeros(padded.shape, np.float32)
    weight = np.zeros(padded.shape, np.float32)
    local_zero = True
    target = torch.device(device)
    with _evaluation(model), torch.autocast(device_type=target.type, enabled=False):
        for box in boxes:
            y, x, y1, x1 = box
            height, width = y1-y, x1-x
            value = padded[y:y1, x:x1]
            if height != tile_size or width != tile_size:
                value = np.pad(value, ((0, tile_size-height), (0, tile_size-width)), mode='reflect')
            image = torch.from_numpy(np.ascontiguousarray(value))[None, None].to(target)
            residual = residual_method(image)
            if (not isinstance(residual, torch.Tensor) or tuple(residual.shape) != tuple(image.shape)
                    or not torch.is_floating_point(residual) or not bool(torch.isfinite(residual).all())):
                raise ValueError('Residual must be a finite signed tensor matching input B1HW')
            delta = residual.detach().float().cpu()[0, 0, :height, :width].numpy()
            local_zero = local_zero and not bool(np.any(delta))
            blend = _weights(height, width)
            total[y:y1, x:x1] += delta*blend
            weight[y:y1, x:x1] += blend
            audit['tiles'] += 1
    if not np.all(weight > 0):
        raise RuntimeError('Tiled residual left uncovered pixels')
    merged = total/weight
    if reflect_pad:
        merged = merged[reflect_pad:-reflect_pad, reflect_pad:-reflect_pad]
    if not np.any(merged):
        audit.update(all_zero_residual=local_zero, boundary_exact=True, reason='zero_merged_residual',
                     residual_min=0., residual_max=0., residual_abs_mean=0.)
        return boundary, audit
    restored = np.clip(boundary + np.float32(strength)*merged, 0., 1.).astype(np.float32, copy=False)
    audit.update(boundary_exact=bool(np.array_equal(restored, boundary)), reason='residual_applied',
                 residual_min=float(merged.min()), residual_max=float(merged.max()),
                 residual_abs_mean=float(np.abs(merged).mean()),
                 boundary_output_sha256=hashlib.sha256(restored.tobytes()).hexdigest())
    return restored, audit


def evaluate_frozen_readout(semantic_logits, raw_probability, boundary, config, *,
                            decode_fn=None, vote_fn=None, reference=None):
    """拼接连续残差后统一使用原 decoder；geometry 语义和最终 raw 投票分别冻结。

semantic_logits 必须已在原生全图尺度，不在这里缩放或重新估计。reference
    可传 {'instances': uint16_HW, 'classes': dict}，用于控制组强制核对最终
    实例 ID 和类别。调用方先调用 predict_residual_tiled，再统一解码全图。
"""
    boundary = _probability(boundary, 'boundary')
    raw_probability = _probability(raw_probability, 'raw_probability')
    semantic = torch.as_tensor(semantic_logits)
    if semantic.ndim == 2:
        semantic = semantic[None, None]
    if (tuple(semantic.shape) != (1, 1, *boundary.shape) or raw_probability.shape != boundary.shape
            or semantic.dtype != torch.float32 or not bool(torch.isfinite(semantic).all())):
        raise ValueError('Frozen semantic logits and raw probability must match native boundary shape in float32')
    if decode_fn is None:
        from tools.infer_semifinal import decode_reflect
        decode_fn = decode_reflect
    if vote_fn is None:
        from tools.semantic_crossover import revote_instances
        vote_fn = revote_instances
    # 原部署解码在 CPU，语义参与 watershed 的输入保持原值。
    ids, _ = decode_fn(semantic.detach().cpu(), torch.from_numpy(boundary)[None, None], config)
    classes, scores = vote_fn(ids, raw_probability)
    if (not isinstance(ids, np.ndarray) or ids.dtype != np.uint16 or ids.shape != boundary.shape
            or int(ids.max(initial=0)) > 65535
            or {str(int(i)) for i in np.unique(ids) if i} != set(classes)
            or any(type(v) is not int or v not in (0, 1) for v in classes.values())):
        raise RuntimeError('Frozen decoder final uint16/ID/class contract failed')
    audit = dict(native_shape=list(boundary.shape), decoder='frozen_reflect32',
                 vote='fixed_raw_probability_mean',
                 boundary_sha256=hashlib.sha256(boundary.tobytes()).hexdigest(),
                 semantic_sha256=hashlib.sha256(semantic.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                 raw_probability_sha256=hashlib.sha256(raw_probability.tobytes()).hexdigest())
    if reference is not None:
        if (not np.array_equal(ids, reference['instances']) or classes != reference['classes']):
            raise RuntimeError('Unchanged boundary failed exact final instance/class compatibility')
        audit['reference_final_exact'] = True
    return ids, classes, audit
