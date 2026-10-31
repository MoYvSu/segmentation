# -*- coding: utf-8 -*-
"""单通道边界的整数平移一致性，不改变原GT或推理读出。

正dy/dx将内容向下/右移动，图外只用reflect扩展，不使用roll跨边相接。
训练将移位输出逆对齐，在共同内域分别平均晶界带/可信内部的概率L1。
常量输出也会一致，因此本量只能与原恢复监督和完整读出共同验收。
"""
from __future__ import annotations

import hashlib
import math
from numbers import Real

import torch
from torch.nn import functional as F


VERSION = 'integer_shift_v1'


def shift_consistency_settings(value):
    """缺省关闭；开启配方显式写入训练identity，不能暗中改变旧实验。"""
    if value is None:
        return {'enabled': False}
    if not isinstance(value, dict) or type(value.get('enabled')) is not bool:
        raise ValueError('shift_consistency needs an explicit boolean enabled')
    allowed = {'enabled', 'version', 'weight', 'margin', 'band_radius', 'shift_seed'}
    if set(value) - allowed:
        raise ValueError('Unknown shift_consistency fields')
    if not value['enabled']:
        return {'enabled': False}
    result = dict(enabled=True, version=VERSION, weight=.1, margin=32,
                  band_radius=8, shift_seed=20261029)
    result.update(value)
    if result['version'] != VERSION:
        raise ValueError('Unsupported shift_consistency version')
    weight = result['weight']
    if (not isinstance(weight, Real) or isinstance(weight, bool)
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError('Enabled consistency weight must be finite and positive')
    for key, minimum in [('margin', 1), ('band_radius', 1), ('shift_seed', 0)]:
        if type(result[key]) is not int or result[key] < minimum:
            raise ValueError(key + ' must be an integer >= ' + str(minimum))
    result['weight'] = float(weight)
    return result


def shift_for_step(settings, step):
    """独立无状态随机流：同一步续训得同一方向，不读取Python/NumPy/Torch RNG。"""
    cfg = shift_consistency_settings(settings)
    if not cfg['enabled'] or type(step) is not int or step < 0:
        raise ValueError('Enabled shift settings and nonnegative zero-based step required')
    key = (VERSION + ':' + str(cfg['shift_seed']) + ':' + str(step)).encode('ascii')
    index = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 4
    return ((1, 0), (-1, 0), (0, 1), (0, -1))[index]


def shift_integer_reflect(value, dy, dx):
    """整数平移同形BCHW；不插值，输出位置q读取输入q-(dy,dx)。"""
    if (not torch.is_tensor(value) or value.ndim != 4 or min(value.shape) < 1
            or not value.is_floating_point()):
        raise ValueError('Shift input must be a nonempty floating BCHW tensor')
    if type(dy) is not int or type(dx) is not int:
        raise ValueError('Integer translation required')
    h, w = value.shape[-2:]
    py, px = abs(dy), abs(dx)
    if py >= h or px >= w:
        raise ValueError('Reflect translation must be smaller than the image')
    if py == px == 0:
        return value
    padded = F.pad(value, (px, px, py, py), mode='reflect')
    top, left = py - dy, px - dx
    return padded[..., top:top + h, left:left + w]


@torch.no_grad()
def build_shift_masks(prepared, settings, dy=0, dx=0):
    """掩码在原坐标系；两次前向均离图框margin，GT band与可信内部互斥。"""
    cfg = shift_consistency_settings(settings)
    if not cfg['enabled'] or type(dy) is not int or type(dx) is not int:
        raise ValueError('Enabled settings and integer shifts required')
    positive = prepared['positive']
    trusted, negative = prepared['trusted'], prepared['negative']
    if (not torch.is_tensor(positive) or positive.ndim != 4 or positive.shape[1] != 1
            or any(not torch.is_tensor(a) or a.shape != positive.shape or a.dtype != torch.bool
                   or a.device != positive.device for a in [positive, trusted, negative])):
        raise ValueError('Prepared positive/trusted/negative must be aligned bool B1HW')
    h, w = positive.shape[-2:]
    margin, radius = cfg['margin'], cfg['band_radius']
    inner = torch.zeros_like(positive)
    # Both the original pixel and its translated partner must remain in the central field.
    y0, y1 = max(margin, margin - dy), min(h - margin, h - margin - dy)
    x0, x1 = max(margin, margin - dx), min(w - margin, w - margin - dx)
    if y1 > y0 and x1 > x0:
        inner[..., y0:y1, x0:x1] = True
    band = F.max_pool2d(positive.float(), 2 * radius + 1, stride=1, padding=radius) > 0
    band &= trusted & inner
    interior = negative & ~band & inner
    return dict(band=band, interior=interior, inner=inner)


def shift_consistency_loss(base_probability, aligned_probability, masks, *, denominators=None):
    """全批分母可用于分块梯度累积；缺失一类记0，不放大另一类。"""
    if (not torch.is_tensor(base_probability) or not base_probability.is_floating_point()
            or base_probability.shape != aligned_probability.shape or base_probability.ndim != 4
            or base_probability.shape[1] != 1 or base_probability.device != aligned_probability.device
            or not bool(torch.isfinite(base_probability).all())
            or not bool(torch.isfinite(aligned_probability).all())):
        raise ValueError('Consistency probabilities must be aligned finite B1HW tensors')
    zero = (base_probability.sum() + aligned_probability.sum()) * 0
    losses, counts = {}, {}
    difference = (base_probability - aligned_probability).abs()
    for name in ['band', 'interior']:
        chosen = masks[name]
        if chosen.shape != base_probability.shape or chosen.dtype != torch.bool or chosen.device != difference.device:
            raise ValueError('Consistency mask must be aligned bool B1HW')
        count = int(chosen.sum())
        denominator = count if denominators is None else denominators[name]
        if type(denominator) is not int or denominator < count:
            raise ValueError('Global consistency denominators cannot be smaller than local counts')
        losses[name] = difference[chosen].sum() / denominator if count else zero
        counts[name] = count
    loss = .5 * (losses['band'] + losses['interior'])
    return loss, dict(band=losses['band'], interior=losses['interior'],
                     band_pixels=counts['band'], interior_pixels=counts['interior'])


@torch.no_grad()
def translation_probe(generator, input_map, target, valid, *, settings,
                      shifts=((1, 0), (0, 1), (16, 0), (0, 16)), base_probability=None):
    """固定已见视图的几何稳定性诊断；不把低一致性误差当正确分割。"""
    from utils.boundary_rgb_training import build_tolerant_targets
    cfg = shift_consistency_settings(settings)
    if not cfg['enabled'] or input_map.ndim != 4 or input_map.shape[1] != 1:
        raise ValueError('Translation probe requires enabled single-channel input')
    shifts = tuple(tuple(pair) for pair in shifts)
    if not shifts or any(len(pair) != 2 or any(type(v) is not int for v in pair) for pair in shifts):
        raise ValueError('Translation probe needs nonempty integer dy/dx pairs')
    flags = [(module, module.training) for module in generator.modules()]
    try:
        generator.eval()
        prepared = build_tolerant_targets(target, valid)
        # Use one common coordinate set for 1px and 16px comparisons. Otherwise the
        # larger shift silently changes the metric's denominator toward an easier centre.
        probe_cfg = dict(cfg, margin=cfg['margin'] + max(abs(v) for pair in shifts for v in pair))
        masks = build_shift_masks(prepared, probe_cfg)
        base = generator(input_map) if base_probability is None else base_probability
        rows = []
        for dy, dx in shifts:
            moved = generator(shift_integer_reflect(input_map, dy, dx))
            aligned = shift_integer_reflect(moved, -dy, -dx)
            loss, audit = shift_consistency_loss(base, aligned, masks)
            row = dict(dy=dy, dx=dx, loss=float(loss), band_pixels=audit['band_pixels'],
                       interior_pixels=audit['interior_pixels'],
                       band_mae=float(audit['band']), interior_mae=float(audit['interior']))
            for name in ['band', 'interior']:
                selected = masks[name]
                row[name + '_above065_flip_pixels'] = int(((base > .65) != (aligned > .65))[selected].sum())
            rows.append(row)
        return dict(complete=True, version=VERSION, no_training=True,
                    input_mode='boundary', margin=cfg['margin'], band_radius=cfg['band_radius'],
                    probe_common_margin=probe_cfg['margin'], common_coordinates_for_all_shifts=True,
                    interpretation='Aligned probability stability only; constant predictions can also be stable',
                    rows=rows)
    finally:
        for module, training in flags:
            module.training = training
