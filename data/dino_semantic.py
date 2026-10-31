# -*- coding: utf-8 -*-
"""DINO 末端语义短训的数据契约：只读处理后新 GT，原生裁窗后等比补齐。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import letterbox
from data.semantic_targets import semantic_targets_from_instances
from models.dino_semantic import file_sha256
from utils.config import project_path
from utils.offset_letterbox import letterbox_instance_geometry


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_processed_source(image_path, gt_path, class_path=None):
    """返回原 RGB、处理后实例、类别字典及原标覆盖；所有未知像素仍无类别。"""
    gt_path = Path(gt_path)
    class_path = Path(class_path) if class_path else gt_path.with_name(gt_path.name.removesuffix('_gt.npz') + '_class.json')
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode RGB: {image_path}')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with np.load(gt_path, allow_pickle=False) as payload:
        required = {'instance_map', 'original_covered', 'filled', 'residual_unknown'}
        if not required.issubset(payload.files):
            raise ValueError('Processed new GT provenance keys required; old GT is not accepted')
        instances = payload['instance_map'].copy()
        masks = {k: payload[k].copy() for k in required - {'instance_map'}}
    if (instances.shape != rgb.shape[:2] or instances.dtype.kind not in 'iu'
            or instances.min() < 0 or instances.max() > 65535):
        raise ValueError('Processed GT must match the original RGB and uint16 instance range')
    for name, mask in masks.items():
        if mask.shape != instances.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError(f'Invalid processed GT mask: {name}')
    original = masks['original_covered'].astype(bool)
    if (np.any(original & (instances == 0))
            or not np.array_equal(masks['residual_unknown'].astype(bool), instances == 0)
            or not np.array_equal(masks['filled'].astype(bool), (instances > 0) & ~original)):
        raise ValueError('Processed GT provenance masks disagree')
    raw_classes = json.loads(class_path.read_text(encoding='utf-8'))
    if not isinstance(raw_classes, dict) or any(
        not re.fullmatch(r'[1-9][0-9]*', k) or type(v) is not int or v not in (0, 1)
        for k, v in raw_classes.items()
    ):
        raise ValueError('Canonical positive instance IDs and strict integer classes 0/1 required')
    classes = {int(k): v for k, v in raw_classes.items()}
    if not set(np.unique(instances)) - {0} <= set(classes):
        raise ValueError('Every known GT instance must have a class')
    if not np.any(instances > 0):
        raise ValueError('Processed source has no known semantic supervision')
    return rgb, instances.astype(np.int32), classes, original


def prepare_semantic_view(rgb, instances, classes, input_size=1024, box=None, rotation=0, flip=False):
    """先原生坐标裁取，再等比缩放；标签最近邻，RGB reflect，监督 padding 为 -1。"""
    if rgb.shape[:2] != instances.shape or instances.ndim != 2:
        raise ValueError('RGB and processed instance shapes must match')
    if box is None:
        box = [0, 0, rgb.shape[0], rgb.shape[1]]
    top, left, height, width = map(int, box)
    if min(height, width) < 1 or min(top, left) < 0 or top + height > rgb.shape[0] or left + width > rgb.shape[1]:
        raise ValueError('Native crop is outside source image')
    rgb = np.ascontiguousarray(np.rot90(rgb[top:top+height, left:left+width], int(rotation) % 4))
    ids = np.ascontiguousarray(np.rot90(instances[top:top+height, left:left+width], int(rotation) % 4))
    if flip:
        rgb, ids = rgb[:, ::-1].copy(), ids[:, ::-1].copy()
    image, _, _, _ = letterbox(rgb, int(input_size))
    grid, valid, geometry = letterbox_instance_geometry(ids, int(input_size), int(input_size))
    lookup = np.full(int(instances.max()) + 1, -1., np.float32)
    for key, value in classes.items():
        if 0 < key < len(lookup):
            lookup[key] = value
    semantic, boundary = semantic_targets_from_instances(grid, lookup)
    semantic[~valid | (grid == 0)] = -1.
    return dict(image=torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.,
                semantic_target=torch.from_numpy(semantic.copy())[None],
                semantic_boundary=torch.from_numpy(boundary.copy())[None],
                semantic_instance_map=torch.from_numpy(grid.astype(np.int64)),
                semantic_valid_content=torch.from_numpy(valid & (grid > 0))[None],
                geometry={**geometry.to_dict(), 'box': list(box), 'rotation': int(rotation) % 4,
                          'flip': bool(flip)})


def epoch_schedule(sources, epoch, seed, views_per_source=8):
    """每源整图/原生窗各半，源序与视图随机流固定，不依赖模型输出。"""
    if epoch < 1 or views_per_source < 2 or views_per_source % 2:
        raise ValueError('Positive epoch and an even views-per-source count required')
    if len(set(sources)) != len(sources) or not sources:
        raise ValueError('Unique nonempty source list required')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), 101]))
    plan = []
    for repeat in range(views_per_source):
        for source in rng.permutation(sorted(sources)):
            plan.append(dict(source=str(source), mode='whole' if repeat % 2 == 0 else 'native',
                             seed=int(rng.integers(0, 2**31 - 1)), repeat=repeat))
    return plan


class ProcessedSemanticDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.cfg = config['semantic_dino']
        self.size = int(self.cfg['input_size'])
        self.raw_dir = Path(project_path(config, self.cfg['raw_data_dir']))
        gt_dir = Path(project_path(config, self.cfg['completed_gt_dir']))
        paths = sorted(gt_dir.glob('*_gt.npz'))
        if len(paths) != int(self.cfg['expected_sources']):
            raise ValueError(f'Expected exactly {self.cfg["expected_sources"]} processed GT sources, got {len(paths)}')
        images = {}
        for path in sorted(self.raw_dir.iterdir()):
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
                images.setdefault(path.stem, []).append(path)
        self.records = []
        for gt in paths:
            name = gt.name.removesuffix('_gt.npz')
            if not re.fullmatch(r'train_[0-9]{3,4}', name) or len(images.get(name, [])) != 1:
                raise ValueError(f'Unique allowed training source missing or invalid: {name}')
            cp = gt.with_name(name + '_class.json')
            image = images[name][0]
            rgb, ids, classes, original = load_processed_source(image, gt, cp)
            self.records.append(dict(source=name, image=str(image.resolve()), gt=str(gt.resolve()),
                                     classes=str(cp.resolve()), image_sha256=file_sha256(image),
                                     gt_sha256=file_sha256(gt), classes_sha256=file_sha256(cp),
                                     shape=list(ids.shape), known_pixels=int((ids > 0).sum()),
                                     original_pixels=int(original.sum()),
                                     filled_pixels=int(((ids > 0) & ~original).sum())))
        self.sources = [r['source'] for r in self.records]
        self.by_source = {r['source']: r for r in self.records}
        identities = [{k: v for k, v in row.items() if k not in ('image', 'gt', 'classes')} for row in self.records]
        self.identity = dict(sources=identities, new_gt_manifest_sha256=canonical_sha(identities),
                             source_count=len(identities), test_training=False,
                             supervision='processed_known_including_filled; unknown/padding=-1',
                             old_labelme_targets_read=False)
        self.set_epoch(1)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.plan = epoch_schedule(self.sources, self.epoch, int(self.cfg['seed']),
                                   int(self.cfg.get('views_per_source', 8)))

    def load_source(self, stem):
        row = self.by_source[stem]
        return load_processed_source(row['image'], row['gt'], row['classes'])

    def verify_sources(self):
        """每轮结束复核不可变输入；不把训练过程中换掉的 GT 混入同一回执。"""
        for row in self.records:
            for key in ('image', 'gt', 'classes'):
                if file_sha256(row[key]) != row[key + '_sha256']:
                    raise RuntimeError(f'Training input changed: {row["source"]}/{key}')
        return self.identity['new_gt_manifest_sha256']

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        spec = self.plan[int(index)]
        rgb, ids, classes, _ = self.load_source(spec['source'])
        rng = np.random.default_rng(spec['seed'])
        box = None
        if spec['mode'] == 'native':
            # 只在原生像素裁窗；以已知像素为锚避免空监督，不按预测/类别筛选。
            side = min(int(self.cfg['native_crop_size']), *ids.shape)
            known = np.flatnonzero(ids > 0)
            y, x = divmod(int(known[int(rng.integers(len(known)))]), ids.shape[1])
            top = int(np.clip(y - int(rng.integers(side)), 0, ids.shape[0] - side))
            left = int(np.clip(x - int(rng.integers(side)), 0, ids.shape[1] - side))
            box = [top, left, side, side]
        sample = prepare_semantic_view(rgb, ids, classes, self.size, box=box,
                                       rotation=int(rng.integers(4)), flip=bool(rng.integers(2)))
        sample.update(source=spec['source'], mode=spec['mode'], view_seed=spec['seed'], draw=int(index))
        return sample

    def plan_receipt(self):
        return dict(epoch=self.epoch, views=len(self.plan), plan_sha256=canonical_sha(self.plan),
                    source_counts=dict(Counter(r['source'] for r in self.plan)),
                    view_counts=dict(Counter(r['mode'] for r in self.plan)))


def collate_semantic(samples):
    keys = ('image', 'semantic_target', 'semantic_boundary', 'semantic_instance_map', 'semantic_valid_content')
    return {**{k: torch.stack([s[k] for s in samples]) for k in keys},
            'views': [{k: s[k] for k in ('source', 'mode', 'view_seed', 'draw', 'geometry')} for s in samples]}
