# -*- coding: utf-8 -*-
"""原 N0 自损伤配对增加同源 RGB；不改变目标、损伤、采样或损失支持域。"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np
import torch

from data.boundary_real import BoundaryRealDataset
from data.boundary_restore import collate_boundary_restore
from data.dino_semantic import canonical_sha


def native_rgb_crop(rgb, box, crop_size, rotation=0, flip=False):
    """按热图同一原生坐标变换；RGB 补 reflect，补齐区域仍由旧 valid 排除。"""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Context RGB must be native uint8 HWC with three channels')
    top, left, height, width = map(int, box)
    if (min(top, left) < 0 or min(height, width) < 1
            or top + height > rgb.shape[0] or left + width > rgb.shape[1]
            or max(height, width) > int(crop_size)):
        raise ValueError('Native RGB crop outside source or crop size')
    value = np.rot90(rgb[top:top+height, left:left+width], int(rotation) % 4)
    if flip:
        value = value[:, ::-1]
    value = np.ascontiguousarray(value)
    bottom, right = int(crop_size)-value.shape[0], int(crop_size)-value.shape[1]
    if bottom or right:
        mode = cv2.BORDER_REFLECT_101 if min(value.shape[:2]) > 1 else cv2.BORDER_REPLICATE
        value = cv2.copyMakeBorder(value, 0, bottom, 0, right, mode)
    return np.ascontiguousarray(value)


class BoundaryContextDataset(BoundaryRealDataset):
    """直接继承旧抽样器；RGB 不消耗随机流，也不参与 GT/损伤位置选择。"""
    def __init__(self, *args, **kwargs):
        # load_source 在每次旧 pair 生成时已解码 RGB，复用当次结果避免二次读盘。
        self._context_source = None
        self._context_rgb = None
        super().__init__(*args, **kwargs)
        parent_identity = canonical_sha(self.identity)
        self.identity.update(format='boundary_context_data_v1', data_mode='real_n0_rgb_context',
            parent_real_identity_sha256=parent_identity,
            context=dict(kind='original_source_RGB', channels=3, dtype='float32', scaling='uint8 / 255',
                source_binding='processed manifest image SHA == frozen N0 image SHA; no basename join',
                geometry='same native box, np.rot90 then horizontal flip; no resize',
                padding='bottom/right reflect101, singleton replicate; old valid excludes padding',
                restoration_applied=False, augmentation_beyond_shared_geometry=False,
                sampling_rng_consumed=False))

    def load_source(self, source):
        loaded = super().load_source(source)
        self._context_source, self._context_rgb = source, loaded[0]
        return loaded

    def __getitem__(self, index):
        pair = super().__getitem__(index)
        source = pair['source']
        if self._context_source != source or self._context_rgb is None:
            raise RuntimeError('Context must come from the exact source loaded for this pair')
        rgb = self._context_rgb
        record = self.by_source[source]
        if list(rgb.shape[:2]) != record['shape']:
            raise ValueError('RGB source shape differs from the N0-bound processed source')
        geometry = pair['meta']['geometry']
        crop = native_rgb_crop(rgb, geometry['box'], self.crop_size,
                               geometry['rotation'], geometry['flip'])
        pair['context_rgb'] = torch.from_numpy(crop.copy()).permute(2, 0, 1).float().div_(255.)
        pair['meta']['context'] = dict(source=source, image_sha256=record['image_sha256'],
            crop_rgb_uint8_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
            geometry=dict(geometry), scaling='uint8 / 255', restoration_applied=False)
        return pair


def collate_boundary_context(samples):
    batch = collate_boundary_restore(samples)
    batch['context_rgb'] = torch.stack([sample['context_rgb'] for sample in samples])
    return batch
