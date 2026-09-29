# -*- coding: utf-8 -*-
"""在整图重建完成后同步裁取语义视图，不改变灰度先验或全局随机流。"""
from __future__ import annotations

import math
from random import Random

import torch
from torch.nn import functional as F


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _content_box(image, valid_content):
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[:2] != (1, 3):
        raise ValueError('Expected one BCHW RGB image')
    if not image.is_floating_point():
        raise ValueError('RGB image must be floating point')
    if not torch.is_tensor(valid_content) or valid_content.ndim not in (3, 4):
        raise ValueError('valid_content must be B1HW or BHW')
    mask = valid_content[:, None] if valid_content.ndim == 3 else valid_content
    if mask.shape[:2] != (1, 1):
        raise ValueError('Expected one valid-content mask')
    if not bool(torch.isfinite(mask).all()) or bool(((mask != 0) & (mask != 1)).any()):
        raise ValueError('valid_content must be binary')
    mask = F.interpolate(mask.float(), image.shape[-2:], mode='nearest')[0, 0].bool()
    occupied = mask.nonzero()
    if not len(occupied):
        raise ValueError('Content mask is empty')
    top, left = occupied.min(dim=0).values.tolist()
    bottom, right = (occupied.max(dim=0).values + 1).tolist()
    if not bool(mask[top:bottom, left:right].all()):
        raise ValueError('valid_content must describe the rectangular image content, not GT coverage')
    return [top, left, bottom, right]


def choose_scale_view(image, valid_content, config, seed, force_mode=None):
    """返回可写入 JSON 的整图/局部选择；局部窗口完全位于有效内容内。

    默认一半保留整图，一半把 512 方窗放大到原输入尺寸。内容不足时保留整图，
    不缩小窗口或混入 reflect padding。seed 使用独立 Python 随机流。
    调用方应优先传入完整尺寸的内容 mask；低分辨率 mask 按原网格扩展。
    """
    if config is None:
        config = {'enabled': False}
    if not isinstance(config, dict):
        raise ValueError('Scale config must be a mapping')
    enabled = config.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('enabled must be boolean')
    probability = config.get('local_probability', .5)
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError('local_probability must be finite and within [0,1]')
    crop_size = _integer(config.get('crop_size', 512), 'crop_size', 1)
    alignment = _integer(config.get('alignment', 4), 'alignment', 1)
    seed = _integer(seed, 'seed')
    if crop_size % alignment:
        raise ValueError('crop_size must be a multiple of alignment')
    if force_mode not in (None, 'full', 'local'):
        raise ValueError('force_mode must be full or local')
    content = _content_box(image, valid_content)
    height, width = image.shape[-2:]
    if height % alignment or width % alignment:
        raise ValueError('Image dimensions must be multiples of alignment')
    spec = {'mode': 'full', 'image_size': [height, width],
            'crop': [0, 0, height, width], 'seed': seed, 'content_box': content,
            'reason': 'full_view'}
    if not enabled:
        spec['reason'] = 'disabled'
        return spec
    rng = Random(seed)
    local = force_mode == 'local' if force_mode is not None else rng.random() < probability
    if not local:
        return spec
    top, left, bottom, right = content
    first_y = (top + alignment - 1) // alignment * alignment
    first_x = (left + alignment - 1) // alignment * alignment
    last_y = (bottom - crop_size) // alignment * alignment
    last_x = (right - crop_size) // alignment * alignment
    if first_y > last_y or first_x > last_x:
        spec['reason'] = 'content_smaller_than_crop'
        return spec
    y = rng.randrange(first_y, last_y + 1, alignment)
    x = rng.randrange(first_x, last_x + 1, alignment)
    spec.update(mode='local', crop=[y, x, crop_size, crop_size], reason='local_view')
    return spec


def _crop_coordinates(value, spec):
    if not torch.is_tensor(value) or value.ndim not in (2, 3, 4):
        raise ValueError('Target must be HW, BHW or BCHW')
    height, width = value.shape[-2:]
    source_height, source_width = spec['image_size']
    top, left, crop_height, crop_width = spec['crop']
    if min(source_height, source_width, crop_height, crop_width, height, width) <= 0:
        raise ValueError('Empty view dimensions')
    if not (0 <= top < top + crop_height <= source_height
            and 0 <= left < left + crop_width <= source_width):
        raise ValueError('Crop exceeds image bounds')
    numerators = (top * height, left * width, crop_height * height, crop_width * width)
    denominators = (source_height, source_width, source_height, source_width)
    if any(n % d for n, d in zip(numerators, denominators)):
        raise ValueError('Crop coordinates must align exactly with the target grid')
    return [n // d for n, d in zip(numerators, denominators)]


def apply_scale_image(image, spec):
    """对已完成 D5a 的 RGB 视图裁切；全图分支直接返回原张量。"""
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[:2] != (1, 3):
        raise ValueError('Expected one BCHW RGB image')
    if list(image.shape[-2:]) != spec['image_size']:
        raise ValueError('Image and scale specification differ')
    if not image.is_floating_point():
        raise ValueError('RGB image must be floating point')
    if spec['mode'] == 'full':
        return image
    top, left, height, width = _crop_coordinates(image, spec)
    return F.interpolate(image[..., top:top + height, left:left + width],
                         size=spec['image_size'], mode='bilinear', align_corners=False)


def apply_scale_target(value, spec, mode='nearest'):
    """同步变换标签/先验；默认最近邻，不插值实例 ID、ignore 或有效 mask。

    支持不同目标网格（本轮 1024/512/256）。先验必须在原整图上计算，
    再用此函数同时变换 target 与 mask，不能在裁块后重新计算灰度先验。
    """
    if mode not in ('nearest', 'bilinear'):
        raise ValueError('Target interpolation must be nearest or bilinear')
    if spec['mode'] == 'full':
        return value
    top, left, height, width = _crop_coordinates(value, spec)
    cropped = value[..., top:top + height, left:left + width]
    output_height, output_width = value.shape[-2:]
    if mode == 'nearest':
        # 整数索引直接复制，避免 int64 ID 经 float 转换损失精度。
        yi = torch.arange(output_height, device=value.device) * height // output_height
        xi = torch.arange(output_width, device=value.device) * width // output_width
        return cropped.index_select(-2, yi).index_select(-1, xi)
    if not value.is_floating_point():
        raise ValueError('Bilinear target interpolation requires floating point')
    leading = cropped.shape[:-2]
    resized = F.interpolate(cropped.reshape(-1, 1, height, width),
                            size=(output_height, output_width), mode='bilinear',
                            align_corners=False)
    return resized.reshape(*leading, output_height, output_width)


def apply_scale_batch(batch, spec):
    """复制 batch，仅同步本轮语义需要的字段；保留 affinity 字段及元数据。"""
    result = dict(batch)
    for key in ('image', 'semantic_target', 'semantic_boundary', 'semantic_instance_map',
                'semantic_valid_content', 'valid_content'):
        if key in batch:
            result[key] = (apply_scale_image(batch[key], spec) if key == 'image'
                           else apply_scale_target(batch[key], spec))
    return result
