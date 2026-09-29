# -*- coding: utf-8 -*-
"""沿用既有几何数据流，仅为人工图增加独立的原标可信域。"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.direct_dual_head_dataset import _spatial_transform
from train_backend_adaptation import build_datasets as _build_backend_datasets
from utils.offset_letterbox import geometry_letterbox_metadata


def resize_trusted(original_covered, *, input_size=1024, output_grid=512):
    """原标覆盖保守缩到 affinity 网格；混入任何补缝/unknown 的格点不可信。

    使用与实例目标相同的 letterbox 尺寸，再以 AREA 统计格点覆盖率。
    返回单样本 [1,G,G] bool；padding 始终为 False。
    """
    covered = np.asarray(original_covered)
    if covered.ndim != 2 or not covered.size:
        raise ValueError('original_covered must be a nonempty HW mask')
    if not np.isfinite(covered).all() or not np.all((covered == 0) | (covered == 1)):
        raise ValueError('original_covered must be binary')
    metadata = geometry_letterbox_metadata(covered.shape, input_size, output_grid)
    fraction = cv2.resize(covered.astype(np.float32),
                          (metadata.content_width, metadata.content_height),
                          interpolation=cv2.INTER_AREA)
    trusted = np.zeros((int(output_grid), int(output_grid)), dtype=bool)
    trusted[:metadata.content_height, :metadata.content_width] = fraction >= 1.0 - 1e-6
    return torch.from_numpy(trusted)[None]


def undo_spatial(tensor, hflip, vflip, k):
    """反转既有翻转后再 rot90 的空间变换，保持通道和数值不变。"""
    if not torch.is_tensor(tensor) or tensor.ndim < 2:
        raise ValueError('Spatial inverse requires a tensor with at least two dimensions')
    result = torch.rot90(tensor, -int(k) % 4, dims=(-2, -1))
    if vflip:
        result = result.flip(-2)
    if hflip:
        result = result.flip(-1)
    return result.contiguous()


class TrustedManualDataset(Dataset):
    """包装配对增强人工流；不重新采样或改写旧图像、BCE目标和回执。"""

    def __init__(self, paired_dataset):
        self.paired_dataset = paired_dataset
        self.base_dataset = paired_dataset.base_dataset
        if self.base_dataset.completed_gt_dir is None:
            raise ValueError('Connectivity trust requires completed GT with original_covered')
        self._trusted_cache = {}

    @property
    def epoch(self):
        return self.paired_dataset.epoch

    def set_epoch(self, epoch):
        self.paired_dataset.set_epoch(epoch)

    def __len__(self):
        return len(self.paired_dataset)

    def _canonical_trust(self, source_index):
        if source_index not in self._trusted_cache:
            path = self.base_dataset.samples[source_index]
            target = self.base_dataset.completed_gt_dir / (path.stem + '_gt.npz')
            with np.load(target, allow_pickle=False) as payload:
                if 'original_covered' not in payload:
                    raise ValueError(f'Missing original_covered for topology trust: {target}')
                covered = payload['original_covered']
                if covered.shape != payload['instance_map'].shape:
                    raise ValueError(f'Original-covered and instance map shapes differ: {target}')
                trusted = resize_trusted(covered, input_size=self.base_dataset.image_size,
                                         output_grid=self.base_dataset.affinity_grid)
                native_shape = tuple(map(int, covered.shape))
            self._trusted_cache[source_index] = trusted, native_shape
        return self._trusted_cache[source_index]

    def __getitem__(self, index):
        sample = self.paired_dataset[index]
        trusted, native_shape = self._canonical_trust(int(sample['source_index']))
        aligned = _spatial_transform(trusted, sample['horizontal_flip'],
                                     sample['vertical_flip'], sample['rotation_k'])
        valid = sample['affinity_valid_content'].bool()
        instances = sample['affinity_instance_map']
        if aligned.shape != valid.shape or aligned.shape[1:] != instances.shape:
            raise ValueError('Topology trust and affinity targets differ in shape')
        return {**sample,
                'trusted_pixels': aligned & valid & (instances > 0)[None],
                'native_shape': torch.tensor(native_shape, dtype=torch.int32),
                'source_kind': 'manual'}


def build_datasets(config):
    """复用全32人工+原64 SAM2核验；SAM2保持原BCE，不进入拓扑监督。"""
    manual, pseudo, metadata = _build_backend_datasets(config)
    manual = TrustedManualDataset(manual)
    return manual, pseudo, {**metadata, 'topology_supervision': {
        'source': 'manual_completed_gt.original_covered',
        'resize': 'area_fraction_at_least_1_minus_1e-6',
        'padding_and_zero_instance': 'ignore',
        'filled_pixels': 'excluded_from_topology_only',
        'sam2': 'original_bce_only',
        'native_shape': 'original_unaugmented_height_width',
        'original_bce_and_augmentation': 'unchanged',
    }}
