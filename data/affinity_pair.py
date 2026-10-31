# -*- coding: utf-8 -*-
"""主mixed样本原样保留；互补视野有自己的裁窗、退化与几何随机流。"""
from __future__ import annotations

from copy import deepcopy

from torch.utils.data import Dataset

from data.affinity_mixed import WholeViewDataset, build_datasets as mixed_datasets
from data.affinity_native import NativeDegradationDataset, NativeViewDataset


def validate_weight(weight):
    if isinstance(weight, bool) or float(weight) not in (0.0, 0.25):
        raise ValueError('Only the authorized weight 0.25 or technical zero replay is supported')
    return float(weight)


class PairViewDataset(Dataset):
    """一个draw返回同源的main/auxiliary，alpha0时不读取或构造辅助路径。"""
    def __init__(self, main, auxiliary_seed, *, auxiliary_enabled=True):
        self.main, self.kind = main, main.kind
        self.auxiliary_seed = int(auxiliary_seed)
        self.auxiliary_enabled = bool(auxiliary_enabled)
        self.auxiliary_whole = self.auxiliary_native = None
        if self.auxiliary_seed == main.whole.seed:
            raise ValueError('Auxiliary RNG must be independent of the primary seed')
        if self.auxiliary_enabled:
            original = main.whole
            base = original.base_dataset.base
            common = dict(degradation=deepcopy(original.degradation), noise=deepcopy(original.noise),
                          seed=self.auxiliary_seed, repeats=original.repeats, augment=original.augment)
            self.auxiliary_whole = NativeDegradationDataset(WholeViewDataset(base, self.kind), **common)
            self.auxiliary_native = NativeDegradationDataset(
                NativeViewDataset(base, self.kind, 1024, self.auxiliary_seed, main.options), **common)

    def __len__(self):
        return len(self.main)

    def set_epoch(self, epoch):
        self.main.set_epoch(epoch)
        if self.auxiliary_enabled:
            self.auxiliary_whole.set_epoch(epoch)
            self.auxiliary_native.set_epoch(epoch)

    def __getitem__(self, index):
        primary = dict(self.main[int(index)])
        # 旧SAM2全图样本未记录image_path；仅补回执，不改变其像素或目标。
        base = self.main.whole.base_dataset.base
        source = int(primary['source_index'])
        path = str(base.samples[source] if self.kind == 'manual' else base._source_path(base.rows[source]))
        if primary.get('image_path', path) != path:
            raise RuntimeError('Primary path differs from canonical source')
        primary['image_path'] = path
        result = {'main': primary}
        if not self.auxiliary_enabled:
            return result
        if primary['training_view'] == 'whole':
            auxiliary = self.auxiliary_native[int(index)]
            auxiliary['training_view'] = 'native1024'
        elif primary['training_view'] == 'native1024':
            auxiliary = self.auxiliary_whole[int(index)]
            auxiliary['training_view'] = 'whole'
        else:
            raise RuntimeError('Unknown primary view')
        if auxiliary.get('image_path', path) != path:
            raise RuntimeError('Auxiliary path differs from canonical source')
        auxiliary['image_path'] = path
        for key in ('name', 'image_path', 'source_index', 'draw_index'):
            if primary[key] != auxiliary[key]:
                raise RuntimeError('Complementary branch changed source: ' + key)
        auxiliary['auxiliary_data_seed'] = self.auxiliary_seed
        result['auxiliary'] = auxiliary
        return result


def build_datasets(config, *, auxiliary_weight=None):
    options = config['affinity_pair']
    weight = validate_weight(options['auxiliary_weight'] if auxiliary_weight is None else auxiliary_weight)
    manual, pseudo, metadata = mixed_datasets(config, 1024)
    seed = int(options['auxiliary_seed'])
    return (PairViewDataset(manual, seed, auxiliary_enabled=weight > 0),
            PairViewDataset(pseudo, seed + 17, auxiliary_enabled=weight > 0),
            dict(parent_data=metadata, updates_per_epoch=metadata['updates_per_epoch'],
                 auxiliary_enabled=weight > 0, auxiliary_seed=seed,
                 main_policy='exact original mixed draw/view stream',
                 auxiliary_policy='same source and draw, complementary native1024/whole, independent RNG',
                 views_per_update=2 if weight > 0 else 1,
                 source_cohort_unchanged=True, unknown_and_outside_pairs_ignored=True))
