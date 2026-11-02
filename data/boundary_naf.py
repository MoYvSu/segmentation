# -*- coding: utf-8 -*-
"""NAF二值配对：默认人工处理实例，显式教师实例保留独立来源。

Dual-CGB使用人工断裂GT恢复完整GT。原文没有给出全部断裂算法参数，
本文件将实际采用的原生像素断裂范围写入identity，不能称为逐项精确复现。
图像只检查来源SHA，不解码；目标和可靠域在初始化时预载，避免每步重复全图计算。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.boundary_restore import known_instance_boundary, reliable_region
from data.teacher_supervision import (
    is_teacher_supervision, load_teacher_instance_source, teacher_boundary_targets,
)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _axis_starts(length, crop_size, stride):
    """原生密集裁块显式覆盖末端；不足一窗的轴只有0，不缩放。"""
    last = max(0, int(length) - int(crop_size))
    starts = list(range(0, last + 1, int(stride)))
    if starts[-1] != last:
        starts.append(last)
    return starts


def source_boxes(shape, crop_size=256, stride=50):
    _positive_integer(crop_size, 'crop_size')
    _positive_integer(stride, 'stride')
    if len(shape) != 2 or min(map(int, shape)) < 1:
        raise ValueError('A positive source HW shape is required')
    height, width = map(int, shape)
    return [[top, left, min(crop_size, height), min(crop_size, width)]
            for top in _axis_starts(height, crop_size, stride)
            for left in _axis_starts(width, crop_size, stride)]


def build_training_plan(rows, crop_size=256, stride=50, augmentations=8,
                        seed=20261026, epoch=1, samples_per_epoch=None):
    """生成真实使用的有序计划；显式预算模式在源之间均衡，非伪称穷举。

    rows必须含source/shape，可用boxes字段限定实际有可靠像素的原生窗。
    不限制预算时遍历每源全部窗口×8增强；限制预算时各源次数最多差1，
    在源内随机遍历窗口/增强，只有该源候选用尽后才重复。所有随机流显式绑定epoch。
    """
    _positive_integer(crop_size, 'crop_size')
    _positive_integer(stride, 'stride')
    if augmentations not in (1, 8):
        raise ValueError('augmentations must be 1 or all 8 rotations/flips')
    if type(epoch) is not int or epoch < 0:
        raise ValueError('epoch must be a nonnegative integer')
    if samples_per_epoch is not None:
        _positive_integer(samples_per_epoch, 'samples_per_epoch')
    ordered = sorted(rows, key=lambda row: row['source'])
    sources = [row['source'] for row in ordered]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError('A nonempty unique training-source list is required')
    if any(not re.fullmatch(r'train_[0-9]{3,4}', source) for source in sources):
        raise ValueError('Only named training sources are permitted')
    boxes = {row['source']: row.get('boxes', source_boxes(row['shape'], crop_size, stride))
             for row in ordered}
    if any(not items for items in boxes.values()):
        raise ValueError('Every source needs at least one eligible native window')
    sizes = {source: len(boxes[source]) * augmentations for source in sources}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), epoch, 8191]))
    counts = dict(sizes)
    if samples_per_epoch is not None:
        base, remainder = divmod(samples_per_epoch, len(sources))
        extras = set(map(str, rng.permutation(sources)[:remainder]))
        counts = {source: base + int(source in extras) for source in sources}
    queues = {}
    for source in sources:
        remaining, selected = counts[source], []
        while remaining:
            order = rng.permutation(sizes[source])
            take = min(remaining, len(order))
            selected.extend(map(int, order[:take]))
            remaining -= take
        queues[source] = selected
    positions = Counter()
    plan = []
    while any(positions[source] < len(queues[source]) for source in sources):
        for source in map(str, rng.permutation(sources)):
            index = positions[source]
            if index >= len(queues[source]):
                continue
            flat = queues[source][index]
            box_index, augmentation = divmod(flat, augmentations)
            plan.append(dict(source=source, box=list(boxes[source][box_index]),
                             box_index=box_index, augmentation=augmentation,
                             rotation=augmentation % 4, flip=bool(augmentation // 4),
                             seed=int(rng.integers(0, 2**31 - 1)), source_draw=index))
            positions[source] += 1
    return plan


def load_processed_binary(gt_path, class_path, boundary_dilation=0, valid_margin=4, *,
                          teacher_boundary_policy=None):
    """直接读取新GT并核provenance；类别只校验合法性，不参与边界生成。"""
    _positive_integer(boundary_dilation, 'boundary_dilation', 0)
    _positive_integer(valid_margin, 'valid_margin')
    if is_teacher_supervision(gt_path):
        ids, _, known, unknown, provenance = load_teacher_instance_source(gt_path, class_path=class_path)
        policy = 'known_neighbors_v1' if teacher_boundary_policy is None else teacher_boundary_policy
        boundary, valid, geometry = teacher_boundary_targets(ids, valid_margin=valid_margin, policy=policy)
        if boundary_dilation:
            kernel = np.ones((2 * boundary_dilation + 1,) * 2, np.uint8)
            # 新明确分界版本只在原width的dilation=0配方使用，不悄悄扩大0监督域。
            if policy == 'explicit_separators_v1':
                raise ValueError('explicit_separators_v1 requires boundary_dilation=0')
            boundary = cv2.dilate(boundary.astype(np.uint8), kernel).astype(bool) & known
        if not valid.any():
            raise ValueError('Teacher source has no reliable known interior')
        positive = int((boundary & valid).sum())
        details = dict(provenance, **geometry)
        details.update(shape=list(ids.shape), teacher_boundary_policy=policy,
            raw_boundary_pixels=int(boundary.sum()), valid_pixels=int(valid.sum()),
            boundary_pixels=positive, valid_boundary_pixels=positive,
            valid_negative_pixels=int((valid & ~boundary).sum()),
            boundary_policy=policy,
            ID0=('instance/semantic/affinity endpoints ignore; explicit separators are geometry boundary; other zero ignore'
                 if policy == 'explicit_separators_v1' else 'ignore'))
        return boundary.astype(bool), valid, details
    if teacher_boundary_policy not in (None, 'known_neighbors_v1'):
        raise ValueError('Explicit teacher boundary policy cannot be applied to manual GT')
    with np.load(gt_path, allow_pickle=False) as payload:
        required = {'instance_map', 'original_covered', 'filled', 'residual_unknown'}
        if not required.issubset(payload.files):
            raise ValueError('Processed new GT provenance keys required; old GT is not accepted')
        ids = payload['instance_map'].copy()
        masks = {key: payload[key].copy() for key in required - {'instance_map'}}
    if ids.ndim != 2 or not ids.size or ids.dtype.kind not in 'iu' or ids.min() < 0 or ids.max() > 65535:
        raise ValueError('Processed GT must be an integer HW instance map in uint16 range')
    for key, mask in masks.items():
        if mask.shape != ids.shape or not np.isin(mask, (0, 1)).all():
            raise ValueError(f'Invalid processed GT provenance mask: {key}')
    original = masks['original_covered'].astype(bool)
    known = ids > 0
    if (np.any(original & ~known)
            or not np.array_equal(masks['residual_unknown'].astype(bool), ~known)
            or not np.array_equal(masks['filled'].astype(bool), known & ~original)):
        raise ValueError('Processed GT provenance masks disagree')
    raw_classes = json.loads(Path(class_path).read_text(encoding='utf-8'))
    if not isinstance(raw_classes, dict) or any(
        not isinstance(key, str) or not re.fullmatch(r'[1-9][0-9]*', key)
        or type(value) is not int or value not in (0, 1)
        for key, value in raw_classes.items()
    ):
        raise ValueError('Strict positive instance IDs and integer classes 0/1 required')
    if not set(map(int, np.unique(ids))) - {0} <= set(map(int, raw_classes)):
        raise ValueError('Every known instance must have a class')
    if not known.any():
        raise ValueError('Processed source has no known support')
    boundary = known_instance_boundary(ids)
    if boundary_dilation:
        kernel = np.ones((2 * boundary_dilation + 1,) * 2, np.uint8)
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel).astype(bool) & known
    valid = reliable_region(ids, valid_margin)
    if not valid.any():
        raise ValueError('Processed source has no reliable known interior')
    return boundary.astype(bool), valid, dict(shape=list(ids.shape), known_pixels=int(known.sum()),
        original_pixels=int(original.sum()), filled_pixels=int((known & ~original).sum()),
        unknown_pixels=int((~known).sum()), valid_pixels=int(valid.sum()),
        boundary_pixels=int((boundary & valid).sum()))


def _crop_pair(target, valid, box, crop_size, rotation=0, flip=False, crop_margin=4):
    top, left, height, width = map(int, box)
    if (min(top, left) < 0 or min(height, width) < 1 or max(height, width) > crop_size
            or top + height > target.shape[0] or left + width > target.shape[1]):
        raise ValueError('Native box lies outside source or exceeds crop_size')
    result = []
    for array in (target, valid):
        view = np.rot90(array[top:top + height, left:left + width], int(rotation) % 4)
        if flip:
            view = view[:, ::-1]
        # 全宽裁块可能仍是连续view；必须复制，避免清裁边时改写整图缓存。
        result.append(np.array(view, copy=True, order='C'))
    target_view, valid_view = result
    content_shape = list(target_view.shape)
    bottom, right = crop_size - target_view.shape[0], crop_size - target_view.shape[1]
    if bottom or right:
        mode = 'reflect' if min(target_view.shape) > 1 else 'edge'
        target_view = np.pad(target_view, ((0, bottom), (0, right)), mode=mode)
        valid_view = np.pad(valid_view, ((0, bottom), (0, right)), mode='constant', constant_values=False)
    margin = int(crop_margin)
    if margin:
        valid_view[:margin] = False
        valid_view[-margin:] = False
        valid_view[:, :margin] = False
        valid_view[:, -margin:] = False
    return target_view.astype(bool), valid_view.astype(bool), dict(box=list(map(int, box)),
        rotation=int(rotation) % 4, flip=bool(flip), content_shape=content_shape,
        padding_bottom=bottom, padding_right=right, resize_scale=1.0)


def _erase_binary_gaps(target, valid, rng, count_range, length_range, half_width_range):
    """沿局部GT切向画擦除椭圆；仅删除可靠域内真实边界，不制造假GT。"""
    candidates = np.argwhere(target & valid)
    erased = np.zeros(target.shape, bool)
    records = []
    if not len(candidates):
        return erased, records
    for _ in range(int(rng.integers(count_range[0], count_range[1] + 1))):
        y, x = map(int, candidates[int(rng.integers(len(candidates)))])
        top, bottom = max(0, y - 9), min(target.shape[0], y + 10)
        left, right = max(0, x - 9), min(target.shape[1], x + 10)
        points = np.argwhere(target[top:bottom, left:right]).astype(np.float64)
        angle = 0.0
        if len(points) > 2:
            covariance = np.cov(points[:, ::-1].T)
            if np.isfinite(covariance).all():
                _, vectors = np.linalg.eigh(covariance)
                vx, vy = vectors[:, -1]
                angle = float(np.degrees(np.arctan2(vy, vx)))
        length = int(rng.integers(length_range[0], length_range[1] + 1))
        across = int(rng.integers(half_width_range[0], half_width_range[1] + 1))
        field = np.zeros(target.shape, np.uint8)
        cv2.ellipse(field, (x, y), (int(math.ceil(length / 2)), across), angle,
                    0, 360, 1, thickness=-1)
        erased |= field.astype(bool) & target & valid
        records.append(dict(center_yx=[y, x], requested_length_native=length,
                            half_width_native=across, tangent_angle_degrees=angle))
    return erased, records


class BoundaryNAFDataset(Dataset):
    """GT二值线条→自动擦断→GT恢复；源内容缓存与每轮计划独立可复现。"""
    def __init__(self, rows_or_manifest, config=None, epoch=1):
        config = {} if config is None else dict(config)
        config = config.get('boundary_naf', config)
        cfg = dict(config.get('data', config))
        self.crop_size = _positive_integer(cfg.get('crop_size', 256), 'crop_size')
        self.stride = _positive_integer(cfg.get('stride', 50), 'stride')
        self.augmentations = cfg.get('augmentations', 8)
        self.seed = int(cfg.get('seed', config.get('seed', 20261026)))
        self.valid_margin = _positive_integer(cfg.get('valid_margin', 4), 'valid_margin')
        self.crop_margin = _positive_integer(cfg.get('crop_margin', self.valid_margin), 'crop_margin', 0)
        self.boundary_dilation = _positive_integer(cfg.get('boundary_dilation', 0), 'boundary_dilation', 0)
        self.samples_per_epoch = cfg.get('samples_per_epoch')
        self.identity_probability = float(cfg.get('identity_probability', 0.2))
        if not math.isfinite(self.identity_probability) or not 0 <= self.identity_probability <= 1:
            raise ValueError('identity_probability must lie in [0,1]')
        if self.crop_size <= 2 * self.crop_margin:
            raise ValueError('crop_size must exceed twice crop_margin')
        if self.valid_margin <= self.boundary_dilation:
            raise ValueError('valid_margin must exceed boundary_dilation')
        self.gap_count_range = self._range(cfg.get('gap_count_range', [1, 4]), 'gap_count_range')
        self.gap_length_range = self._range(cfg.get('gap_length_range', [8, 40]), 'gap_length_range')
        self.gap_half_width_range = self._range(cfg.get('gap_half_width_range', [3, 5]), 'gap_half_width_range')
        expected_sources = _positive_integer(cfg.get('expected_sources', config.get('expected_sources', 32)), 'expected_sources')
        self.manifest_path = None
        self.manifest_sha256 = None
        if isinstance(rows_or_manifest, (str, Path)):
            self.manifest_path = Path(rows_or_manifest).resolve()
            self.manifest_sha256 = file_sha256(self.manifest_path)
            document = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            rows = document.get('rows') if isinstance(document, dict) else None
            base = self.manifest_path.parent
        else:
            rows = rows_or_manifest
            base = Path(cfg.get('manifest_base', '.')).resolve()
        if not isinstance(rows, list) or not rows:
            raise ValueError('Processed training manifest rows are required')
        sources = [row.get('source') for row in rows if isinstance(row, dict)]
        if (len(sources) != len(rows) or any(not isinstance(source, str)
                or not re.fullmatch(r'train_[0-9]{3,4}', source) for source in sources)):
            raise ValueError('Only named training sources are permitted')
        if len(sources) != expected_sources or len(set(sources)) != len(sources):
            raise ValueError(f'Exactly {expected_sources} unique processed training sources required')
        self.records, self.cache = [], {}
        minimum = _positive_integer(cfg.get('min_valid_pixels', 64), 'min_valid_pixels')
        for row in rows:
            source = row['source']
            if not re.fullmatch(r'train_[0-9]{3,4}', source):
                raise ValueError('Only named training sources are permitted')
            record = dict(source=source)
            for key in ('image', 'gt', 'classes'):
                path = Path(row[key])
                path = path if path.is_absolute() else base / path
                expected = row.get('hashes', row).get(key + '_sha256')
                if not isinstance(expected, str) or len(expected) != 64:
                    raise ValueError('Every source file needs an explicit SHA256')
                actual = file_sha256(path)
                if actual != expected:
                    raise ValueError(f'Processed-source SHA256 differs: {source}/{key}')
                record[key], record[key + '_sha256'] = str(path.resolve()), actual
            target, valid, metadata = load_processed_binary(record['gt'], record['classes'],
                boundary_dilation=self.boundary_dilation, valid_margin=self.valid_margin,
                teacher_boundary_policy=cfg.get('teacher_boundary_policy'))
            if 'shape' in row and list(row['shape']) != list(target.shape):
                raise ValueError('Processed GT shape differs from checked source manifest')
            record.update(metadata)
            # 仅保存二值目标和可靠域；不保留原RGB或整型实例，降低缓存内存。
            self.cache[source] = (target, valid)
            boxes = source_boxes(target.shape, self.crop_size, self.stride)
            integral = cv2.integral(valid.astype(np.uint8), sdepth=cv2.CV_64F)
            eligible = []
            for box in boxes:
                top, left, height, width = box
                y0, x0 = top + self.crop_margin, left + self.crop_margin
                y1, x1 = top + height - self.crop_margin, left + width - self.crop_margin
                count = 0 if y1 <= y0 or x1 <= x0 else int(
                    integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])
                if count >= minimum:
                    eligible.append(box)
            if not eligible:
                raise ValueError('Source has no eligible reliable native windows: ' + source)
            record.update(boxes=eligible, dense_windows=len(boxes), eligible_windows=len(eligible))
            self.records.append(record)
        self.records.sort(key=lambda row: row['source'])
        self.sources = [row['source'] for row in self.records]
        if len(self.sources) != expected_sources or len(set(self.sources)) != len(self.sources):
            raise ValueError(f'Exactly {expected_sources} unique processed training sources required')
        self.by_source = {row['source']: row for row in self.records}
        self.dense_candidate_count = sum(len(row['boxes']) * self.augmentations for row in self.records)
        self.identity = dict(format='boundary_naf_data_v1', source_count=len(self.sources),
            sources=[{key: value for key, value in row.items() if key not in ('image', 'gt', 'classes', 'boxes')}
                     for row in self.records], manifest_sha256=self.manifest_sha256,
            crop_size_native=self.crop_size, stride_native=self.stride, augmentations=self.augmentations,
            boundary_representation='binary; known-known different instance ID; both interface sides',
            boundary_dilation_native=self.boundary_dilation, valid_margin_native=self.valid_margin,
            crop_margin_native=self.crop_margin, identity_probability=self.identity_probability,
            gap_count_range=self.gap_count_range, gap_length_native_range=self.gap_length_range,
            gap_half_width_native_range=self.gap_half_width_range, seed=self.seed,
            dense_eligible_augmented_candidates=self.dense_candidate_count,
            samples_per_epoch=self.samples_per_epoch, model_predictions_used=False,
            semantic_classes_used_for_boundary=False, rgb_decoded=False,
            test_training=False, independent_validation=False,
            policy='processed known including filled; unknown/frame/crop rim/padding ignored; native crop then D4 augmentation',
            limitation='Artificial gap parameters are explicit adaptations because paper does not provide full corruption implementation.')
        teacher_sources = [r['source'] for r in self.records if r.get('source_kind') == 'model_teacher']
        if teacher_sources:
            if len(teacher_sources) != len(self.sources):
                raise ValueError('NAF target cohort must not silently mix teacher and manual provenance')
            self.identity.update(target_source_kind='model_teacher', model_predictions_used=True,
                model_predictions_as_target=True, source_supervision_format='teacher_supervision_v1',
                teacher_boundary_policy=self.records[0]['teacher_boundary_policy'],
                target_valid_boundary_pixels=sum(r['boundary_pixels'] for r in self.records),
                target_valid_negative_pixels=sum(r['valid_negative_pixels'] for r in self.records),
                policy='teacher positive IDs supervised; ID0/unknown/frame/crop rim/padding ignored; no fabricated manual provenance')
            if self.identity['teacher_boundary_policy'] == 'explicit_separators_v1':
                self.identity.update(
                    boundary_representation='known-known different positive IDs both sides OR certified axis A-0-B separator center',
                    policy='ID0 instance/semantic/affinity endpoints ignored; certified single-pixel separators supervised as boundary; other zero/frame/crop/padding ignored',
                    target_separator_pixels=sum(r['separator_pixels'] for r in self.records),
                    target_geometry_known_pixels=sum(r['geometry_known_pixels'] for r in self.records))
        self.set_epoch(epoch)

    @staticmethod
    def _range(value, name):
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(type(item) is not int or item < 1 for item in value) or value[0] > value[1]):
            raise ValueError(f'{name} must be an ascending positive integer pair')
        return list(value)

    def set_epoch(self, epoch):
        self.plan = build_training_plan(self.records, self.crop_size, self.stride,
            self.augmentations, self.seed, epoch, self.samples_per_epoch)
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        draw = int(index)
        spec = self.plan[draw]
        target, valid = self.cache[spec['source']]
        target, valid, geometry = _crop_pair(target, valid, spec['box'], self.crop_size,
            spec['rotation'], spec['flip'], self.crop_margin)
        if not valid.any():
            raise RuntimeError('Prechecked window lost all valid support')
        rng = np.random.default_rng(spec['seed'])
        requested_identity = bool(rng.random() < self.identity_probability)
        gap, gaps = (np.zeros(target.shape, bool), []) if requested_identity else _erase_binary_gaps(
            target, valid, rng, self.gap_count_range, self.gap_length_range, self.gap_half_width_range)
        inputs = target.copy()
        inputs[gap] = False
        identity = not bool(gap.any())
        tensors = {key: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))[None]
                   for key, value in dict(input=inputs, target=target, valid=valid,
                       gap_mask=gap, fully_known_mask=valid).items()}
        metadata = dict(source=spec['source'], epoch=self.epoch, draw=draw,
            draw_seed=spec['seed'], source_draw=spec['source_draw'], augmentation=spec['augmentation'],
            geometry=geometry, mode='identity' if identity else 'gap',
            requested_mode='identity' if requested_identity else 'gap', gaps=gaps,
            valid_pixels=int(valid.sum()), boundary_pixels=int((target & valid).sum()),
            gap_pixels=int(gap.sum()), padding_supervised=False, resize_scale=1.0)
        return dict(**tensors, identity=torch.tensor(identity), source=spec['source'],
                    box=list(spec['box']), draw=draw, meta=metadata)

    def plan_receipt(self):
        return dict(epoch=self.epoch, samples=len(self.plan), plan_sha256=canonical_sha(self.plan),
            dense_eligible_augmented_candidates=self.dense_candidate_count,
            exhaustive=self.samples_per_epoch is None,
            source_counts=dict(Counter(item['source'] for item in self.plan)),
            augmentation_counts=dict(Counter(str(item['augmentation']) for item in self.plan)))

    def verify_sources(self):
        if self.manifest_path and file_sha256(self.manifest_path) != self.manifest_sha256:
            raise RuntimeError('Processed-GT manifest changed')
        for row in self.records:
            for key in ('image', 'gt', 'classes'):
                if file_sha256(row[key]) != row[key + '_sha256']:
                    raise RuntimeError(f'Processed source changed: {row["source"]}/{key}')
        return canonical_sha(self.identity)


def collate_boundary_naf(samples):
    keys = ('input', 'target', 'valid', 'gap_mask', 'fully_known_mask', 'identity')
    return {**{key: torch.stack([sample[key] for sample in samples]) for key in keys},
            'source': [sample['source'] for sample in samples],
            'box': [sample['box'] for sample in samples],
            'draw': [sample['draw'] for sample in samples],
            'meta': [sample['meta'] for sample in samples]}
