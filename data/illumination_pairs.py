# -*- coding: utf-8 -*-
"""全部允许训练图的在线光照逆变换；配对两端保持完全相同的模糊。"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.mim_dataset import list_images
from data.rgb_restoration_dataset import read_rgb, prepare_rgb, degrade_rgb


def light_pair(target, rng, cfg, identity=False):
    """只在真实内容上合成；返回实际截断率，绝不按测试图统计拟合目标。"""
    if identity:
        return target.copy(), dict(gamma=1., log_exposure=0., field_strength=0., clipped_fraction=0.)
    gamma = float(np.exp(rng.uniform(*np.log(cfg['gamma']))))
    log_exposure = float(rng.uniform(*np.log(cfg['exposure'])))
    strength = float(rng.uniform(0, cfg['local_log_strength'])) if rng.random() < .5 else 0.
    grid = int(rng.choice([2, 4, 8]))
    coarse = rng.uniform(-1, 1, (grid, grid)).astype(np.float32)
    field = cv2.resize(coarse, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
    light = np.sum(target * np.array([.2126, .7152, .0722], np.float32), axis=-1)
    # 超出截断上限时收缩整次扰动；日志记录实际参数而非只报名义范围。
    factor = 1.
    for _ in range(24):
        effective_gamma = gamma ** factor
        scale = np.exp(factor * (log_exposure + strength * field)
                       + (effective_gamma - 1) * np.log(np.maximum(light, 1e-5)))
        changed = target * scale[..., None]
        fraction = float(np.any((changed < 0) | (changed > 1), axis=-1).mean())
        if fraction <= cfg['max_clipped_fraction']:
            break
        factor *= .7
    if fraction > cfg['max_clipped_fraction']:
        changed, factor, fraction, effective_gamma = target.copy(), 0., 0., 1.
    return np.clip(changed, 0, 1).astype(np.float32), dict(
        gamma=effective_gamma, log_exposure=log_exposure * factor,
        field_strength=strength * factor, clipped_fraction=fraction)


class IlluminationPairs(Dataset):
    def __init__(self, data_dir, cfg, degradation):
        self.samples = [Path(p) for p in list_images(data_dir)]
        if len(self.samples) != cfg['expected_sources']:
            raise ValueError(f'Expected all {cfg["expected_sources"]} training images, found {len(self.samples)}')
        self.cfg, self.degradation, self.epoch = cfg, degradation, 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cfg = self.cfg
        rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], self.epoch, index]))
        image, valid = prepare_rgb(read_rgb(str(self.samples[index])), cfg['image_size'])
        h, w = int(valid[:, 0].sum()), int(valid[0].sum())
        target = image[:h, :w]
        blurred = bool(rng.random() < cfg['blur_probability'])
        if blurred:
            target = degrade_rgb(target, rng, 'blur', self.degradation)
        identity = bool(rng.random() < cfg['identity_probability'])
        inp, info = light_pair(target, rng, cfg['augmentation'], identity)
        # 配对只差光照；不把清晰图作为模糊输入的校光目标。
        pair = [cv2.copyMakeBorder(a, 0, cfg['image_size'] - h, 0, cfg['image_size'] - w,
                                  cv2.BORDER_REFLECT) for a in (inp, target)]
        return dict(input=torch.from_numpy(np.ascontiguousarray(pair[0].transpose(2, 0, 1))),
                    target=torch.from_numpy(np.ascontiguousarray(pair[1].transpose(2, 0, 1))),
                    valid=torch.from_numpy(valid[None]), source=self.samples[index].stem,
                    identity=identity, blurred=blurred, **info)
