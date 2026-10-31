# -*- coding: utf-8 -*-
"""固定候选分隔的数据集：一次读入RAM，GT只作为标签，不作为模型输入。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CHANNEL_NAMES = ('R', 'G', 'B', 'N1', 'NAF')
MASK_NAMES = ('A', 'B', 'shared_seam', 'valid')
FORMAT = 'boundary_relation_data_v1'


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf8')).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('重复JSON键：' + key)
        result[key] = value
    return result


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=_unique_object)


class BoundaryRelationDataset(Dataset):
    """全部可用监督参加训练；monitor是已见样本诊断，不是留出验证。

    真实patch预先按原生像素裁取。训练时仅同步D4旋转/镜像，不在线重算
    N1、NAF、GT或候选，不让样本顺序改变增强，也不把GT覆盖mask传给网络。
    """

    def __init__(self, manifest_path, *, mode='rgb', augment=True, seed=20261030,
                 max_ram_gib=16., split='train'):
        if mode not in ('rgb', 'maps') or split not in ('train', 'monitor'):
            raise ValueError('不支持的输入模式或数据分组')
        if not np.isfinite(max_ram_gib) or max_ram_gib <= 0:
            raise ValueError('RAM预算必须为正数')
        self.manifest_path = Path(manifest_path).resolve()
        manifest_digest = file_sha256(self.manifest_path)
        self.manifest = _read(self.manifest_path)
        if self.manifest.get('format') != FORMAT or self.manifest.get('complete') is not True:
            raise ValueError('数据清单尚未完成或版本不匹配')
        self.patch_size = int(self.manifest.get('patch_size', 0))
        if self.patch_size != 256:
            raise ValueError('首轮只支持原生256方块，禁止隐式缩放')
        self.mode, self.augment, self.seed, self.split = mode, bool(augment), int(seed), split
        self.epoch = 0
        self._shards, self._ends, self._samples = [], [], []
        self.sample_ids, self.sources, all_labels = [], [], []
        seen = set()
        entries = self.manifest.get('shards' if split == 'train' else 'monitor_shards', [])
        if not isinstance(entries, list) or (split == 'train' and not entries):
            raise ValueError('训练分片清单为空或格式无效')
        self.ram_bytes = 0
        pin = []
        for entry in entries:
            relative = Path(entry['path'])
            path = (self.manifest_path.parent / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(self.manifest_path.parent):
                raise ValueError('数据分片路径越界')
            digest = file_sha256(path)
            if digest != entry['sha256']:
                raise ValueError('分片SHA不匹配：' + str(relative))
            count = int(entry['count'])
            if count <= 0:
                raise ValueError('非空分片应包含至少一个样本')
            h = self.patch_size
            anticipated = count * (3 + 4 + 4) * h * h + count
            if self.ram_bytes + anticipated > max_ram_gib * 1024**3:
                raise MemoryError('分片展开后超过RAM预算，不退回慢盘逐batch读取')
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {'rgb', 'maps', 'masks', 'labels'}:
                    raise ValueError('关系分片字段不匹配')
                arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
            shapes = dict(rgb=(count, 3, h, h), maps=(count, 2, h, h),
                          masks=(count, 4, h, h), labels=(count,))
            types = dict(rgb=np.uint8, maps=np.float16, masks=np.uint8, labels=np.uint8)
            for key, value in arrays.items():
                if value.shape != shapes[key] or value.dtype != types[key]:
                    raise ValueError('分片shape/dtype错误：' + key)
            if not np.isfinite(arrays['maps']).all() or np.any(arrays['maps'] < 0) or np.any(arrays['maps'] > 1):
                raise ValueError('边界图必须是有限的[0,1]概率')
            masks = arrays['masks']
            if np.any(masks > 1) or np.any(masks[:, :3] > masks[:, 3:4]):
                raise ValueError('A/B/缝mask必须是原图有效范围内的0/1值')
            if np.any(masks[:, 0] & masks[:, 1]):
                raise ValueError('两个预测区域不能重叠')
            if any(np.any(masks[:, channel].sum(axis=(1, 2)) == 0) for channel in (0, 1, 2, 3)):
                raise ValueError('每个patch须有A/B/共享分隔和有效原图像素')
            expected_labels = (0, 1) if split == 'train' else (255,)
            if not np.isin(arrays['labels'], expected_labels).all():
                raise ValueError('训练仅0/1；无GT monitor仅255，不得混用')
            metadata = entry.get('samples')
            if not isinstance(metadata, list) or len(metadata) != count:
                raise ValueError('逐patch回执缺失')
            for local, record in enumerate(metadata):
                sample_id = record['sample_id']
                source = record.get('source', entry['source'])
                if not isinstance(sample_id, str) or sample_id in seen or source != entry['source']:
                    raise ValueError('样本ID重复或来源不一致')
                if 'label' in record and int(record['label']) != int(arrays['labels'][local]):
                    raise ValueError('样本标签与元数据不一致')
                seen.add(sample_id)
                self.sample_ids.append(sample_id)
                self.sources.append(source)
                self._samples.append(dict(record, source=source))
            all_labels.append(arrays['labels'])
            self.ram_bytes += sum(value.nbytes for value in arrays.values())
            self._shards.append(arrays)
            self._ends.append(len(self._samples))
            pin.append(dict(path=relative.as_posix(), sha256=digest, count=count, source=entry['source']))
        self.labels = np.concatenate(all_labels) if all_labels else np.empty(0, np.uint8)
        if file_sha256(self.manifest_path) != manifest_digest:
            raise RuntimeError('数据预载期间清单发生变化')
        self.identity = dict(format='boundary_relation_loaded_v1',
                             manifest_sha256=manifest_digest,
                             split=split, patch_size=self.patch_size,
                             channels=list(CHANNEL_NAMES), masks=list(MASK_NAMES),
                             GT_is_never_a_feature=True, no_resize=True, shards=pin,
                             sample_count=len(self.sample_ids), sample_ids_sha256=canonical_sha256(self.sample_ids),
                             labels_sha256=hashlib.sha256(self.labels.tobytes()).hexdigest())
        # mode/epoch只属训练配方；两组相同数据的identity保持一致。
        self.identity_sha256 = canonical_sha256(self.identity)

    def __len__(self):
        return len(self.sample_ids)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('epoch须为非负整数')
        self.epoch = epoch

    def _locate(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(self):
            raise IndexError(index)
        index = int(index)
        shard = int(np.searchsorted(self._ends, index, side='right'))
        return shard, index - (self._ends[shard-1] if shard else 0)

    def raw(self, index):
        shard, local = self._locate(index)
        data = self._shards[shard]
        return dict(rgb=data['rgb'][local].copy(), maps=data['maps'][local].copy(),
                    masks=data['masks'][local].copy(), label=int(data['labels'][local]),
                    sample_id=self.sample_ids[index], source=self.sources[index], index=int(index),
                    meta=dict(self._samples[index]))

    def tensors(self, index, *, augment=None, mode=None):
        shard, local = self._locate(index)
        data = self._shards[shard]
        rgb = torch.from_numpy(data['rgb'][local]).float().div_(127.5).sub_(1.)
        maps = torch.from_numpy(data['maps'][local]).float()
        masks = torch.from_numpy(data['masks'][local]).float()
        selected_mode = self.mode if mode is None else mode
        if selected_mode not in ('maps', 'rgb'):
            raise ValueError('未知输入模式')
        if selected_mode == 'maps':
            rgb.zero_()
        x = torch.cat((rgb, maps), dim=0)
        code = 0
        do_augment = self.augment if augment is None else bool(augment)
        if do_augment:
            token = f'{self.seed}:{self.epoch}:{self.sample_ids[index]}'.encode('utf8')
            code = int.from_bytes(hashlib.sha256(token).digest()[:8], 'little') % 8
            x, masks = torch.rot90(x, code % 4, (-2, -1)), torch.rot90(masks, code % 4, (-2, -1))
            if code >= 4:
                x, masks = x.flip(-1), masks.flip(-1)
        return dict(input=x.contiguous(), masks=masks.contiguous(),
                    label=torch.tensor(float(data['labels'][local]), dtype=torch.float32),
                    sample_id=self.sample_ids[index], source=self.sources[index], index=int(index),
                    meta=dict(self._samples[index], geometry_D4=code))

    def __getitem__(self, index):
        return self.tensors(index)


def collate_boundary_relation(samples):
    if not samples:
        raise ValueError('空batch')
    result = {key: torch.stack([row[key] for row in samples]) for key in ('input', 'masks', 'label')}
    result.update({key: [row[key] for row in samples] for key in ('sample_id', 'source', 'index', 'meta')})
    return result
