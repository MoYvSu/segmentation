# -*- coding: utf-8 -*-
"""固定训练源的全图/局部同区域过程图；不用于验证或选择 checkpoint。"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from models.backend_adaptation import restore_first
from models.semantic_lora import semantic_features


FERRITE_RGB = np.array([244, 190, 64], dtype=np.uint8)
PEARLITE_RGB = np.array([70, 125, 226], dtype=np.uint8)
IGNORE_RGB = np.array([128, 128, 128], dtype=np.uint8)


def centre_roi(valid_content, image_size, crop_size=512):
    """有效内容中的中央方形窗口，坐标与边长按 4 像素对齐。"""
    height, width = map(int, image_size)
    mask = valid_content.reshape(1, 1, *valid_content.shape[-2:]).float()
    mask = F.interpolate(mask, size=(height, width), mode='nearest')[0, 0] > .5
    yy, xx = torch.where(mask)
    if not len(yy):
        raise ValueError('Scale monitor requires nonempty valid image content')
    top, left, bottom, right = int(yy.min()), int(xx.min()), int(yy.max()) + 1, int(xx.max()) + 1
    aligned_top, aligned_left = (top + 3) // 4 * 4, (left + 3) // 4 * 4
    side = min(int(crop_size), bottom - aligned_top, right - aligned_left) // 4 * 4
    if side < 4:
        raise ValueError('Scale monitor content is too small for an aligned crop')
    y = max(aligned_top, ((top + bottom - side) // 2) // 4 * 4)
    x = max(aligned_left, ((left + right - side) // 2) // 4 * 4)
    return y, x, side, side


def semantic_rgb(target, valid):
    """固定色表；无标签/无效域单独显示为灰，不能伪装成任一类别。"""
    values = np.asarray(target)
    accepted = np.asarray(valid, dtype=bool) & np.isfinite(values) & (values >= 0)
    result = np.broadcast_to(IGNORE_RGB, (*values.shape, 3)).copy()
    result[accepted & (values > .5)] = FERRITE_RGB
    result[accepted & (values <= .5)] = PEARLITE_RGB
    return result


def probability_rgb(probability):
    values = np.asarray(probability, dtype=np.float32)[..., None]
    return np.rint((1 - values) * PEARLITE_RGB + values * FERRITE_RGB).clip(0, 255).astype(np.uint8)


def _panel(image, title, *, nearest=False):
    resized = cv2.resize(image, (320, 320), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(resized, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(canvas, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _probability(model, image, size):
    logits = model.semantic_decoder(semantic_features(model, image.float()), image)
    return F.interpolate(logits.float(), size=size, mode='bilinear', align_corners=True).sigmoid()


@torch.no_grad()
def scale_monitor(model, restorer, dataset, config, output, epoch):
    """前两张训练源固定退化与 D5a 噪声；完整恢复一次，然后才裁窗口。

    full/local 概率及 GT 都显示同一个 ROI。训练时的 module 状态、dataset
    epoch 与随机流原样恢复，尤其不能在结束时递归 train() 冻结的 encoder。
    """
    device = next(model.parameters()).device
    scfg = config.get('semantic_scale', {})
    mcfg = scfg.get('monitor', {})
    count = min(int(mcfg.get('sources', 2)), len(dataset.base_dataset))
    if count < 1:
        raise ValueError('Scale monitor requires at least one training source')
    crop_size = int(scfg.get('crop_size', 512))
    fixed_epoch = int(mcfg.get('dataset_epoch', 1))
    fixed_seed = int(config['semantic_adaptation']['seed']) + 9000000
    destination = Path(output) / 'scale_monitor' / f'epoch_{epoch:03d}'
    destination.mkdir(parents=True, exist_ok=True)
    modules = list(dict.fromkeys([*model.modules(), *restorer.modules()]))
    module_modes = [(module, module.training) for module in modules]
    old_epoch = dataset.epoch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index or 0] if device.type == 'cuda' else []
    rows = []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            restorer.eval()
            dataset.set_epoch(fixed_epoch)
            for index in range(count):
                sample = dataset[index]
                image = sample['image'][None].to(device)
                height, width = image.shape[-2:]
                y, x, h, w = centre_roi(sample['valid_content'], (height, width), crop_size)
                seed = fixed_seed + index
                with torch.autocast(device_type=device.type, enabled=False):
                    restored = restore_first(restorer, image.float(), seed)
                    full = _probability(model, restored, (height, width))
                    roi = restored[..., y:y + h, x:x + w]
                    local_image = F.interpolate(roi, size=(height, width), mode='bilinear', align_corners=False)
                    local = _probability(model, local_image, (h, w))
                target = sample['semantic_target'].reshape(1, 1, *sample['semantic_target'].shape[-2:]).float()
                valid = sample['semantic_valid_content'].reshape(1, 1, *sample['semantic_valid_content'].shape[-2:]).float()
                target = F.interpolate(target, size=(height, width), mode='nearest')[0, 0, y:y + h, x:x + w].numpy()
                valid = F.interpolate(valid, size=(height, width), mode='nearest')[0, 0, y:y + h, x:x + w].numpy() > .5
                valid &= target >= 0
                full_roi = full[0, 0, y:y + h, x:x + w].cpu().numpy()
                local_roi = local[0, 0].cpu().numpy()
                restored_rgb = np.ascontiguousarray(np.rint(
                    restored[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
                cv2.rectangle(restored_rgb, (x, y), (x + w - 1, y + h - 1), (30, 245, 70), 3)
                name = Path(sample.get('name', sample.get('image_name', f'source_{index:02d}'))).stem
                panels = [
                    _panel(restored_rgb, f'{name} e{epoch}: D5a full + ROI'),
                    _panel(probability_rgb(full_roi), 'Full-view P(F), SAME ROI, 0..1'),
                    _panel(probability_rgb(local_roi), f'Local {height / h:.1f}x P(F), SAME ROI, 0..1'),
                    _panel(semantic_rgb(target, valid), 'GT: F gold / P blue / ignore grey', nearest=True),
                ]
                path = destination / f'{name}.png'
                if not cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR)):
                    raise IOError(f'Failed to write scale monitor: {path}')
                rows.append({
                    'source_index': index, 'image': name, 'dataset_epoch': fixed_epoch,
                    'restoration_seed': seed, 'roi_top_left_height_width': [y, x, h, w],
                    'full_input_size': [height, width], 'local_scale': height / h,
                    'valid_gt_pixels': int(valid.sum()), 'ignore_pixels': int((~valid).sum()),
                    'full_gt_disagreements': int((((full_roi > .5) != (target > .5)) & valid).sum()),
                    'local_gt_disagreements': int((((local_roi > .5) != (target > .5)) & valid).sum()),
                    'mean_absolute_probability_difference': float(np.abs(full_roi - local_roi).mean()),
                })
            report = {'epoch': int(epoch), 'purpose': 'fixed_training_process_monitor_not_validation_or_selection',
                      'single_restoration_before_crop': True, 'probability_colour_range': [0, 1], 'samples': rows}
            (destination / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            return report
    finally:
        dataset.set_epoch(old_epoch)
        for module, training in module_modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
