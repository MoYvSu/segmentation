# -*- coding: utf-8 -*-
"""原图裁块→原生像素退化→同步放到网络尺寸；随机流与训练视野独立。"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from data.rgb_restoration_dataset import (
    RGBRestorationDataset, PROFILES, read_rgb, local_reconstruction_mask,
)
from data.rgb_spatial_blur import apply_spatial_blur, apply_sigma_map_blur, make_endpoint_sigma_map
from utils.config import project_path


def stream(seed, epoch, index, salt):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, index, salt]))


def crop_native(rgb, size, fractions):
    """只裁原图，不先缩整图；不足尺寸反射填充并保留有效区。"""
    h, w = rgb.shape[:2]
    y, x = [min(max(0, edge-size), int(float(u)*(max(0, edge-size)+1)))
            for edge, u in zip((h, w), fractions)]
    crop = rgb[y:min(y+size, h), x:min(x+size, w)]
    vh, vw = crop.shape[:2]
    valid = np.zeros((size, size), np.float32)
    valid[:vh, :vw] = 1
    crop = cv2.copyMakeBorder(crop, 0, size-vh, 0, size-vw, cv2.BORDER_REFLECT)
    return crop.astype(np.float32)/255, valid, (y, x, vh, vw)


class NativeRestorationDataset(RGBRestorationDataset):
    def __init__(self, config, arm):
        cfg, opt = config['rgb_restoration'], config['restore_native']
        if cfg['split_policy'] != 'all_train' or cfg.get('holdout_manifest'):
            raise ValueError('Native restoration requires all training sources with no holdout')
        self.opt, self.arm = copy.deepcopy(opt), arm
        self.sizes = list(opt['arms'][arm])
        if not self.sizes or any(type(v) != int or v < 4 for v in self.sizes):
            raise ValueError('Invalid native sizes')
        super().__init__(project_path(config, cfg['data_dir']), split='train',
            image_size=opt['input_size'], crop_size=opt['input_size'],
            degradation=copy.deepcopy(cfg['degradation']), seed=cfg['seed'],
            masked_pretraining=copy.deepcopy(cfg.get('masked_pretraining')),
            training_noise=cfg.get('training_noise'), split_policy='all_train')
        if cfg.get('training_noise') is not None or list(self.probabilities) != [.2, .8, 0., 0.]:
            raise ValueError('This experiment preserves D5a identity/blur recipe without added lighting/noise')
        if len(self.samples) % len(self.sizes):
            raise ValueError('Sources must divide evenly for exact balanced view assignment')
        self.set_epoch(0)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        rng = stream(self.opt['view_seed'], self.epoch, 0, 71)
        labels = np.tile(np.asarray(self.sizes), len(self.samples)//len(self.sizes))
        self.view_sizes = rng.permutation(labels)

    def item(self, index, *, size=None, kind=None, downsample=None, local_task=True, augment=True):
        size = int(self.view_sizes[index] if size is None else size)
        rng = stream(self.seed, self.epoch, index, 0)
        profile = str(rng.choice(PROFILES, p=self.probabilities))
        sigma = float(rng.uniform(*self.degradation['blur_sigma']))
        spatial_selected = bool(stream(self.seed, self.epoch, index, 6).random()
                                < self.degradation['spatial_blur']['probability'])
        endpoint_cfg = self.degradation['spatial_blur']['endpoint_transition']
        endpoint_selected = bool(stream(self.seed, self.epoch, index, 7).random() < endpoint_cfg['probability'])
        sampled_kind = ('identity' if profile == 'identity' else 'endpoint' if spatial_selected and endpoint_selected
                        else 'spatial' if spatial_selected else 'uniform')
        kind = sampled_kind if kind is None else kind
        if kind not in ('identity', 'uniform', 'spatial', 'endpoint'):
            raise ValueError(kind)
        profile = 'identity' if kind == 'identity' else 'blur'
        ds_rng = stream(self.seed, self.epoch, index, 8)
        use_down = ds_rng.random() < self.opt['downsample_probability']
        half = ds_rng.random() < self.opt['exact_half_probability']
        scale = .5 if half else float(ds_rng.uniform(*self.opt['downsample_range']))
        scale = float(scale if use_down else 1.) if downsample is None else float(downsample)
        if profile == 'identity':
            scale, sigma = 1., 0.
        if not 0 < scale <= 1:
            raise ValueError('Downsample scale must be in (0,1]')

        fractions = stream(self.seed, self.epoch, index, 100).random(2)
        clean, valid, box = crop_native(read_rgb(self.samples[index]), size, fractions)
        y, x, vh, vw = box
        content = clean[:vh, :vw]
        sigma_map = None
        if kind == 'identity':
            degraded = content.copy()
        elif kind == 'uniform':
            degraded = cv2.GaussianBlur(content, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
        elif kind == 'spatial':
            degraded, sigma_map = apply_spatial_blur(content, sigma, self.degradation['spatial_blur'],
                stream(self.seed, self.epoch, index, 16), self.degradation['blur_sigma'])
        else:
            sigma_map = make_endpoint_sigma_map((vh, vw), (0, 0, vh, vw), endpoint_cfg,
                stream(self.seed, self.epoch, index, 17), self.degradation['blur_sigma'])
            degraded = apply_sigma_map_blur(content, sigma_map, self.degradation['blur_sigma'],
                                            self.degradation['spatial_blur']['levels'])
        if scale < 1:
            small_hw = (max(1, round(vw*scale)), max(1, round(vh*scale)))
            degraded = cv2.resize(cv2.resize(degraded, small_hw, interpolation=cv2.INTER_AREA),
                                  (vw, vh), interpolation=cv2.INTER_LINEAR)
        else:
            small_hw = (vw, vh)
        degraded = cv2.copyMakeBorder(degraded, 0, size-vh, 0, size-vw, cv2.BORDER_REFLECT)
        mask_fraction = 0.
        masked = bool(local_task and profile == 'blur' and self.masking_strength()
                      and stream(self.seed, self.epoch, index, 4).random() < self.masking_probability())
        if masked:
            # 低频损伤也先在原生像素上施加，保持所有空间退化的像素单位一致。
            degraded, mask = local_reconstruction_mask(degraded, valid,
                stream(self.seed, self.epoch, index, 2), self.masked_pretraining, self.masking_strength())
            mask_fraction = float(mask.sum()/max(1., valid.sum()))
        out_size = self.image_size
        zoom = out_size/size
        if size != out_size:
            clean, degraded = [cv2.resize(a, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
                                for a in (clean, degraded)]
            # 插值涉及任何填充像素即不监督；梯度损失还要求两个端点都有效。
            valid = (cv2.resize(valid, (out_size, out_size), interpolation=cv2.INTER_LINEAR) >= 1-1e-6).astype(np.float32)
        turns, flip = 0, False
        if augment:
            geom = stream(self.seed, self.epoch, index, 101)
            turns, flip = int(geom.integers(4)), bool(geom.integers(2))
            clean, degraded, valid = [np.rot90(a, turns)[:, ::-1] if flip else np.rot90(a, turns)
                                      for a in (clean, degraded, valid)]
        sigma_values = sigma_map if sigma_map is not None else np.asarray([sigma])
        native_min, native_max = float(sigma_values.min()), float(sigma_values.max())
        recipe = dict(source=Path(self.samples[index]).name, source_index=int(index), epoch=self.epoch,
            profile=profile, kind=kind, native_size=size, network_size=out_size, zoom=zoom,
            box=list(box), crop_fractions=fractions.tolist(), turns=turns, flip=flip,
            base_sigma_native=sigma, sigma_native_min=native_min, sigma_native_max=native_max,
            sigma_network_min=native_min*zoom, sigma_network_max=native_max*zoom,
            sigma_native_rms=float(np.sqrt(np.mean(sigma_values.astype(np.float64)**2))),
            weak_fraction=float(np.mean(sigma_values<=1.5)) if profile!='identity' else 0.,
            strong_fraction=float(np.mean(sigma_values>=4.)) if profile!='identity' else 0.,
            downsample_scale=scale, resampled_native_hw=[small_hw[1],small_hw[0]],
            local_task=masked, local_task_strength=self.masking_strength() if masked else 0.,
            local_task_pixel_units='native', local_task_coverage=mask_fraction,
            valid_pixels=int(valid.sum()), target_is_interpolated=size<out_size,
            sigma_scope='nominal Gaussian only; excludes resampling and local SSL lowpass')
        return dict(input=torch.from_numpy(np.ascontiguousarray(degraded.transpose(2,0,1))),
                    target=torch.from_numpy(np.ascontiguousarray(clean.transpose(2,0,1))),
                    valid=torch.from_numpy(np.ascontiguousarray(valid[None])),
                    source=recipe['source'], profile=profile,
                    recipe=json.dumps(recipe, ensure_ascii=False, separators=(',', ':')))

    def __getitem__(self, index):
        return self.item(index)
