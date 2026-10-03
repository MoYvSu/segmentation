# -*- coding: utf-8 -*-
"""复用两种已验证读图路径；用控制回执将相同更新映射到独立的50/50视野表。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.affinity_native import NativeViewDataset, NativeDegradationDataset
from data.rgb_restoration_dataset import read_rgb
from train_backend_adaptation import build_datasets as original_datasets
from utils.config import project_path


def view_plan(seed, epoch, count):
    if count < 2 or count % 2:
        raise ValueError('Mixed training requires an even update count')
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 7919]))
    return rng.permutation(np.repeat([0, 1024], count//2)).tolist()


def draw_views(receipts, kind, epoch, plan):
    rows = [r for r in receipts if r['epoch'] == epoch]
    if len(rows) != len(plan) or [r['step'] for r in rows] != list(range(1, len(plan)+1)):
        raise ValueError('Incomplete reference epoch')
    result = {}
    for row, size in zip(rows, plan):
        index = int(row[kind+'_draw']['draw_index'][0])
        if index in result:
            raise ValueError('Repeated reference draw')
        result[index] = size
    if set(result) != set(range(len(plan))):
        raise ValueError('Reference does not contain every draw')
    return result


class WholeViewDataset(Dataset):
    """原全图样本原样复用，只补充不参与loss的清晰图及坐标记录。"""
    def __init__(self, base, kind):
        self.base, self.kind = base, kind

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = dict(self.base[index])
        path = (self.base.samples[index] if self.kind == 'manual' else
                self.base._source_path(self.base.rows[index]))
        shape = read_rgb(str(path)).shape[:2]
        sample.update(clean_image=sample['image'].clone(), crop_box=torch.tensor([0,0,*shape]),
                      native_source_shape=torch.tensor(shape), crop_attempt=0, native_crop_size=0)
        return sample


class MixedViewDataset(Dataset):
    def __init__(self, original, kind, options, receipts):
        self.kind, self.options, self.receipts = kind, options, receipts
        self.whole = NativeDegradationDataset(WholeViewDataset(original.base_dataset, kind),
            original.degradation, original.noise, original.seed, original.repeats, original.augment)
        self.native = NativeDegradationDataset(NativeViewDataset(original.base_dataset, kind,
            1024, original.seed, options), original.degradation, original.noise,
            original.seed, original.repeats, original.augment)
        self.mapping = None

    def __len__(self):
        return len(self.whole)

    def set_epoch(self, epoch):
        plan = view_plan(self.options['mixed_view_seed'], epoch, len(self))
        self.mapping = draw_views(self.receipts, self.kind, epoch, plan)
        self.whole.set_epoch(epoch)
        self.native.set_epoch(epoch)

    def __getitem__(self, index):
        if self.mapping is None:
            raise RuntimeError('Set the epoch before reading mixed views')
        size = self.mapping[int(index)]
        sample = (self.whole if size == 0 else self.native)[index]
        sample['training_view'] = 'whole' if size == 0 else 'native1024'
        return sample


def build_datasets(config, size=1024):
    if size != 1024:
        raise ValueError('This experiment mixes whole and native1024 only')
    manual, pseudo, metadata = original_datasets(config)
    options = config['affinity_native']
    reference = Path(project_path(config, options['reused_control']))/'steps.jsonl'
    receipts = [json.loads(line) for line in reference.read_text(encoding='utf8').splitlines()]
    if len(receipts) != config['backend_adaptation']['epochs'] * metadata['updates_per_epoch']:
        raise ValueError('Incomplete existing control receipts')
    data = dict(parent_data=metadata, native_crop_size=1024,
        training_views=['whole', 'native1024'], per_epoch_view_updates={'whole':32,'native1024':32},
        view_seed=options['mixed_view_seed'], sample_policy='same sources, independent balanced update views',
        order='whole letterbox1024 OR native_crop1024 -> existing degradation -> frozen D5a',
        updates_per_epoch=metadata['updates_per_epoch'])
    return (MixedViewDataset(manual, 'manual', options, receipts),
            MixedViewDataset(pseudo, 'pseudo', options, receipts), data)
