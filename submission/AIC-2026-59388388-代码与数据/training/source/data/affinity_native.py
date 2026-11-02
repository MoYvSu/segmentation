# -*- coding: utf-8 -*-
"""先裁原生RGB与同源实例，再沿用旧在线退化；不改变来源清单或未知域规则。"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.backend_adaptation import PairedDegradationDataset
from data.dataset import letterbox
from data.rgb_restoration_dataset import read_rgb
from data.semantic_targets import load_completed_semantic_source
from train_backend_adaptation import build_datasets as build_original_datasets
from utils.offset_letterbox import letterbox_instance_geometry


def choose_box(instances, size, rng, *, attempts=32, minimum_pairs=16):
    """均匀取窗，仅排除完全缺乏监督的窗口，不按类别、边界密度或模型表现选窗。"""
    h, w = instances.shape
    ch, cw = min(h, int(size)), min(w, int(size))
    if min(ch, cw, attempts, minimum_pairs) < 1:
        raise ValueError('Invalid crop settings')
    for attempt in range(int(attempts)):
        y, x = int(rng.integers(h - ch + 1)), int(rng.integers(w - cw + 1))
        known = instances[y:y + ch, x:x + cw] > 0
        pairs = int((known[1:] & known[:-1]).sum() + (known[:, 1:] & known[:, :-1]).sum())
        if pairs >= minimum_pairs:
            return (y, x, y + ch, x + cw), attempt
    raise RuntimeError('No supervised native crop; keep source and fail instead of silently skipping it')


def geometry_sample(image, ids, *, image_size, grid):
    """RGB反射padding、GT零padding；窗外端点和未覆盖像素由原affinity目标函数忽略。"""
    if image.shape[:2] != ids.shape:
        raise ValueError('RGB and canonical instances differ in shape')
    rgb, _, _, _ = letterbox(image, image_size)
    labels, valid, meta = letterbox_instance_geometry(ids, image_size, grid)
    image_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.
    instances = torch.from_numpy(labels.astype(np.int64))
    content = torch.from_numpy(valid)[None]
    return dict(image=image_tensor, clean_image=image_tensor.clone(),
                instance_map=instances, affinity_instance_map=instances,
                foreground=(instances > 0)[None], valid_content=content,
                affinity_valid_content=content, uncovered_boundary_source=torch.tensor(False),
                input_content_shape=torch.tensor([meta.resized_height, meta.resized_width], dtype=torch.int32),
                content_shape=torch.tensor([meta.content_height, meta.content_width], dtype=torch.int32))


class NativeViewDataset(Dataset):
    def __init__(self, base, kind, size, seed, options):
        if kind not in ('manual', 'pseudo'):
            raise ValueError(kind)
        self.base, self.kind, self.size, self.seed = base, kind, int(size), int(seed)
        self.options = options
        self.image_size = base.image_size
        self.grid = base.affinity_grid if kind == 'manual' else base.output_grid
        self.epoch, self.draw = 0, 0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, source_index):
        if self.kind == 'manual':
            path = self.base.samples[source_index]
            image = read_rgb(str(path))
            ids, _ = load_completed_semantic_source(
                self.base.completed_gt_dir / (path.stem + '_gt.npz'), image.shape[:2])
        else:
            row = self.base.rows[source_index]
            path = self.base._source_path(row)
            image = read_rgb(str(path))
            ids = self.base._load_instance_map(row)
        if image.shape[:2] != ids.shape:
            raise ValueError('Native source image and labels differ')
        # 独立流，不消耗旧退化、几何增强或全局随机数。
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.draw, 90]))
        box, attempts = choose_box(ids, self.size, rng, attempts=self.options['crop_attempts'],
                                   minimum_pairs=self.options['minimum_known_pairs'])
        y, x, y1, x1 = box
        sample = geometry_sample(image[y:y1, x:x1], ids[y:y1, x:x1],
                                 image_size=self.image_size, grid=self.grid)
        sample.update(name=path.name, image_name=path.name, image_path=str(path),
                      source_kind=self.kind, crop_box=torch.tensor(box, dtype=torch.int32),
                      native_source_shape=torch.tensor(ids.shape, dtype=torch.int32),
                      crop_attempt=attempts, native_crop_size=self.size)
        return sample


class NativeDegradationDataset(PairedDegradationDataset):
    _SPATIAL_KEYS = (*PairedDegradationDataset._SPATIAL_KEYS, 'clean_image')

    def __getitem__(self, index):
        self.base_dataset.epoch, self.base_dataset.draw = self.epoch, int(index)
        return super().__getitem__(index)


def build_datasets(config, size):
    """旧builder负责32人工源、64 SAM2清单授权与哈希检查；只替换原生读取视野。"""
    manual, pseudo, metadata = build_original_datasets(config)
    if int(size) not in config['affinity_native']['sizes']:
        raise ValueError('Unconfigured native size')
    result = []
    for kind, original in [('manual', manual), ('pseudo', pseudo)]:
        native = NativeViewDataset(original.base_dataset, kind, size, original.seed,
                                   config['affinity_native'])
        paired = NativeDegradationDataset(native, original.degradation, original.noise,
                                         original.seed, original.repeats, original.augment)
        result.append(paired)
    return *result, dict(parent_data=metadata, native_crop_size=int(size),
                         order='native_crop -> letterbox1024 -> existing_degradation -> D5a',
                         sample_policy='uniform_windows_with_nonempty_known_pairs',
                         artificial_crop_edges='outside-offset pairs ignored',
                         updates_per_epoch=metadata['updates_per_epoch'])
