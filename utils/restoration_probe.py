# -*- coding: utf-8 -*-
"""首步恢复的显式噪声策略；默认行为不变，可用于隔离局部零噪声消融。"""
from __future__ import annotations

import torch

from utils.restoration_noise import restoration_noise


@torch.no_grad()
def first_raw(restorer, image, noise, *, audit=None):
    """保留裁剪前 x0，正负噪声平均必须在最后一次范围裁剪之前完成。"""
    if noise.shape != image.shape or noise.device != image.device or noise.dtype != torch.float32:
        raise ValueError('Noise must match the float32 image shape/device')
    with torch.autocast(device_type=image.device.type, enabled=False):
        condition = image.float()
        step = torch.full((condition.shape[0],), int(restorer.num_steps),
                          dtype=torch.long, device=condition.device)
        state = restorer.q_sample(condition, condition, step, noise=noise)
        if audit is not None:
            audit.update(q_sample_calls=1, step=int(restorer.num_steps),
                         dtype=str(condition.dtype), noise_nonzero=int(torch.count_nonzero(noise)),
                         initial_equals_condition=bool(torch.equal(state, condition)),
                         initial_max_abs=float((state-condition).abs().max()))
        return restorer.denoise(state, condition, step).float()


def finish(image, raw):
    # 原 restore_first 的 strength=1 算术顺序也属于对齐契约。
    condition = image.float()
    return (condition + 1.0 * (raw - condition)).clamp(0.0, 1.0)


@torch.no_grad()
def restore_probe(restorer, image, seed, mode='single', *, positive_raw=None, audit=None):
    if mode not in ('single', 'zero', 'pair'):
        raise ValueError('Expected single/zero/pair')
    if audit is not None and mode == 'pair':
        raise ValueError('Single-state audit is limited to single/zero')
    noise = (torch.zeros_like(image, dtype=torch.float32) if mode == 'zero'
             else restoration_noise(image.shape, int(seed), image.device))
    if mode == 'pair':
        positive = first_raw(restorer, image, noise) if positive_raw is None else positive_raw
        if positive.shape != image.shape or positive.device != image.device:
            raise ValueError('Cached positive x0 differs from image')
        raw = (positive + first_raw(restorer, image, -noise)) * .5
    else:
        if positive_raw is not None:
            raise ValueError('Cached positive x0 is only allowed for pair')
        raw = first_raw(restorer, image, noise) if audit is None else first_raw(restorer, image, noise, audit=audit)
    return finish(image, raw), raw
