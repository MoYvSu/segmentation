# -*- coding: utf-8 -*-
"""语义光照诊断的纯亮度变换；不改变标签、空间尺寸或随机数流。"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def apply_illumination(image, kind, value, field=None, clamp=True):
    """对 BCHW RGB 加偏移、平滑局部偏移，或仅对亮度作 gamma。

    RGB 通道使用相同增量，截断前保留通道差。gamma 同时改变明暗对比，
    不应把 gamma 的诊断结果称为纯亮度平移。关闭条件返回原对象。
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("expected BCHW RGB")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("illumination value must be finite")
    if kind in ("offset", "local") and value == 0:
        return image
    if kind == "gamma" and value == 1:
        return image
    if kind == "offset":
        changed = image + value
    elif kind == "local":
        if field is None or field.shape != (image.shape[0], 1, *image.shape[-2:]):
            raise ValueError("local illumination requires a B1HW field")
        changed = image + value * field
    elif kind == "gamma":
        if value <= 0:
            raise ValueError("gamma must be positive")
        luminance = (image * image.new_tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1, keepdim=True)
        changed = image + luminance.clamp(0, 1).pow(value) - luminance
    else:
        raise ValueError(f"unknown illumination kind: {kind}")
    return changed.clamp(0, 1) if clamp else changed


def _reflected_coordinates(length, start, end, device):
    """将画布坐标映射回有效矩形，支持窄长图及翻转后位于右下的内容。"""
    size = end - start
    if size < 1:
        raise ValueError("empty content extent")
    positions = torch.arange(length, device=device) - start
    if size == 1:
        return torch.zeros(length, device=device, dtype=torch.float32)
    folded = positions.remainder(2 * (size - 1))
    folded = torch.minimum(folded, 2 * (size - 1) - folded)
    return folded.float() / (size - 1)


def fixed_smooth_field(image, content_shape=None, content_mask=None, angle=0.0):
    """有效矩形内的缓变光照场，padding 使用相同 reflect 规则。

    默认从左亮到右暗。angle 是弧度，用于改变光照方向。
    shape 用于未几何增强的左上内容；mask 支持翻转/旋转后的内容位置。
    """
    batch, _, height, width = image.shape
    if content_shape is not None and content_mask is not None:
        raise ValueError("use content_shape or content_mask, not both")
    if content_mask is not None:
        mask = content_mask.to(device=image.device, dtype=torch.float32)
        if mask.ndim == 3:
            mask = mask[:, None]
        mask = F.interpolate(mask, size=(height, width), mode="nearest") > .5
        if mask.shape != (batch, 1, height, width):
            raise ValueError("content mask must be B1HW")
    fields = []
    c, s = math.cos(float(angle)), math.sin(float(angle))
    norm = max(abs(c) + abs(s), 1e-8)
    for index in range(batch):
        if content_mask is not None:
            ys, xs = torch.where(mask[index, 0])
            if not len(ys):
                raise ValueError("empty content mask")
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
        else:
            y0 = x0 = 0
            y1, x1 = (height, width) if content_shape is None else map(int, content_shape)
            if not (0 < y1 <= height and 0 < x1 <= width):
                raise ValueError("content extent is outside image")
        x = _reflected_coordinates(width, x0, x1, image.device)
        y = _reflected_coordinates(height, y0, y1, image.device)
        field = (c * torch.cos(math.pi * x)[None, :] +
                 s * torch.cos(math.pi * y)[:, None]) / norm
        fields.append(field)
    return torch.stack(fields)[:, None].to(image.dtype)


def training_illumination(image, config, seed, content_mask=None):
    """D5a 后的可重放亮度增强；只读取图像和几何有效域，不读取标签。

    独立随机流不推进 dropout、数据退化或 D5a 的随机数。整幅 RGB 采用
    相同增量，局部场延续已有平滑场；截断预算按有效像素而非通道计数。
    若饱和像素使本次偏移无法满足预算，允许退化为精确恒等并如实记录。
    """
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("illumination config must be a mapping")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("illumination.enabled must be boolean")
    values = {}
    for key, default in (("probability", .5), ("local_probability", .5),
                         ("max_offset", .10), ("max_clipped_fraction", .01)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"illumination.{key} must be a finite number in [0,1]")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"illumination.{key} must be a finite number in [0,1]")
        values[key] = value
    stats = {"kind": "identity", "requested_offset": 0.0, "actual_offset": 0.0,
             "angle": 0.0, "scale": 0.0, "clipped_fraction": 0.0, "applied": False}
    if not enabled or values["probability"] == 0 or values["max_offset"] == 0:
        return image, stats
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("illumination seed must be a nonnegative integer")
    # 固定专用 stream；不能改用全局 np.random 或 torch.rand，否则会扰动其它消融变量。
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20260926]))
    if rng.random() >= values["probability"]:
        return image, stats
    if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
        raise ValueError("expected floating BCHW RGB")
    if not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
        raise ValueError("illumination input must be finite and within [0,1]")
    if content_mask is None:
        valid = torch.ones_like(image[:, :1], dtype=torch.bool)
    else:
        mask = content_mask.to(device=image.device, dtype=torch.float32)
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.ndim != 4 or mask.shape[:2] != (image.shape[0], 1):
            raise ValueError("content mask must be B1HW or BHW")
        if not bool(torch.isfinite(mask).all()):
            raise ValueError("content mask must be finite")
        valid = F.interpolate(mask, size=image.shape[-2:], mode="nearest") > .5
        if not bool(valid.flatten(1).any(1).all()):
            raise ValueError("content mask must include content for every image")
    kind = "local" if rng.random() < values["local_probability"] else "offset"
    requested = float(rng.uniform(-values["max_offset"], values["max_offset"]))
    angle = float(rng.uniform(0, 2 * math.pi)) if kind == "local" else 0.0
    field = fixed_smooth_field(image, content_mask=valid, angle=angle) if kind == "local" else None
    delta = requested * field if field is not None else requested
    valid_count = int(valid.sum())

    def clipped_fraction(scale):
        candidate = image + scale * delta
        clipped = ((candidate < 0) | (candidate > 1)).any(dim=1, keepdim=True)
        return int((clipped & valid).sum()) / valid_count

    scale = 1.0
    fraction = clipped_fraction(scale)
    if fraction > values["max_clipped_fraction"]:
        low, high = 0.0, 1.0
        for _ in range(12):
            midpoint = (low + high) / 2
            if clipped_fraction(midpoint) <= values["max_clipped_fraction"]:
                low = midpoint
            else:
                high = midpoint
        scale = low
        fraction = clipped_fraction(scale)
    actual = requested * scale
    stats.update(kind=kind, requested_offset=requested, actual_offset=actual,
                 angle=angle, scale=scale, clipped_fraction=fraction, applied=actual != 0)
    if actual == 0:
        return image, stats
    return (image + scale * delta).clamp(0, 1), stats
