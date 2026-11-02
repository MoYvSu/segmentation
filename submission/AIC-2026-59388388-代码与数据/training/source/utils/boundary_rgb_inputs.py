# -*- coding: utf-8 -*-
"""只替换边界通道，保持训练/monitor的RGB条件完全一致。"""
from __future__ import annotations

import torch


def model_input_from_boundary(boundary, rgb=None, *, input_mode):
    """CHW或BCHW的T组成同坐标模型输入；旧单通道返回原tensor。"""
    if (not torch.is_tensor(boundary) or boundary.ndim not in (3, 4)
            or boundary.shape[-3] != 1 or min(boundary.shape) < 1
            or not boundary.is_floating_point()
            or not bool(torch.isfinite(boundary).all())
            or bool(((boundary != 0) & (boundary != 1)).any())):
        raise ValueError('Boundary input must be finite binary 1HW or B1HW tensor')
    if input_mode == 'boundary':
        return boundary
    if input_mode != 'rgb_boundary':
        raise ValueError('Unknown boundary RGB input_mode')
    if (not torch.is_tensor(rgb) or rgb.ndim != boundary.ndim or rgb.shape[-3] != 3
            or rgb.shape[:-3] != boundary.shape[:-3] or rgb.shape[-2:] != boundary.shape[-2:]
            or rgb.device != boundary.device or rgb.dtype != boundary.dtype
            or not bool(torch.isfinite(rgb).all()) or bool(((rgb < 0) | (rgb > 1)).any())):
        raise ValueError('RGB must be aligned finite 3HW or B3HW in [0,1], same dtype/device as boundary')
    return torch.cat((rgb, boundary), dim=-3)


def clean_model_input(sample_or_batch, *, input_mode):
    """干净/损伤对照只换T，禁止把四通道模型误喂成B1。"""
    boundary = sample_or_batch['clean_input_boundary']
    if input_mode == 'boundary':
        return boundary
    return model_input_from_boundary(boundary, sample_or_batch.get('rgb'), input_mode=input_mode)
