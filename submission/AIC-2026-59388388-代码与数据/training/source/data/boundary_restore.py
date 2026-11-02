# -*- coding: utf-8 -*-
"""处理后实例 GT 的原生尺度补边配对；不读取模型预测或测试标注。

这里只把相邻、均已知且实例 ID 不同的位置视为真边界。RGB 用于来源/尺寸
核验及预览，不参与制造边界；processed filled 与 original 一样属于已知区域。
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dino_semantic import canonical_sha, load_processed_source
from models.dino_semantic import file_sha256


def known_instance_boundary(instances):
    """四邻域已知—已知实例交界，标记交界两侧；零 ID 不形成负类/边界。"""
    ids = np.asarray(instances)
    if ids.ndim != 2 or ids.dtype.kind not in 'iu' or np.any(ids < 0):
        raise ValueError('Nonnegative integer HW instance IDs required')
    known = ids > 0
    boundary = np.zeros(ids.shape, bool)
    vertical = known[1:] & known[:-1] & (ids[1:] != ids[:-1])
    horizontal = known[:, 1:] & known[:, :-1] & (ids[:, 1:] != ids[:, :-1])
    boundary[1:] |= vertical
    boundary[:-1] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def reliable_region(instances, margin=8):
    """未知与裁窗外侧均视为不可监督区，向已知侧留出原生像素安全带。"""
    if type(margin) is not int or margin < 1:
        raise ValueError('A positive native-pixel safety margin is required')
    known = (np.asarray(instances) > 0).astype(np.uint8)
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
    return cv2.erode(known, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def render_soft_boundary(boundary, sigma=1.0):
    """按到真实离散交界的距离渲染软线；sigma 单位始终是原图像素。"""
    sigma = float(sigma)
    if not math.isfinite(sigma) or not .25 <= sigma <= 8:
        raise ValueError('Native boundary sigma must be finite and in [.25,8]')
    edge = np.asarray(boundary, bool)
    if not edge.any():
        return np.zeros(edge.shape, np.float32)
    distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    result = np.exp(-.5 * (distance / sigma) ** 2).astype(np.float32)
    result[distance > 3 * sigma] = 0
    return result


def native_crop(instances, box, crop_size, rotation=0, flip=False):
    """原生裁块后仅做90度转动/翻转；小于目标尺寸的源只补 unknown，不缩放。"""
    ids = np.asarray(instances)
    top, left, height, width = map(int, box)
    if (min(top, left) < 0 or min(height, width) < 1
            or top + height > ids.shape[0] or left + width > ids.shape[1]
            or max(height, width) > int(crop_size)):
        raise ValueError('Native crop is outside source or exceeds crop_size')
    cropped = np.rot90(ids[top:top+height, left:left+width], int(rotation) % 4)
    if flip:
        cropped = cropped[:, ::-1]
    result = np.zeros((int(crop_size), int(crop_size)), np.int32)
    result[:cropped.shape[0], :cropped.shape[1]] = cropped
    return result, dict(box=[top, left, height, width], rotation=int(rotation) % 4,
                        flip=bool(flip), content_shape=list(cropped.shape),
                        padding_bottom=int(crop_size)-cropped.shape[0],
                        padding_right=int(crop_size)-cropped.shape[1], resize_scale=1.0)


def _gap_field(boundary, valid, rng, target_sigma):
    candidates = np.argwhere(boundary & valid)
    erase = np.zeros(boundary.shape, np.float32)
    records = []
    if not len(candidates):
        return erase, records
    for _ in range(int(rng.integers(1, 5))):
        y, x = map(int, candidates[int(rng.integers(len(candidates)))])
        # 局部主方向只决定人工断口方向，不产生任何额外边界监督。
        y0, y1 = max(0, y-9), min(boundary.shape[0], y+10)
        x0, x1 = max(0, x-9), min(boundary.shape[1], x+10)
        points = np.argwhere(boundary[y0:y1, x0:x1]).astype(np.float64)
        angle = 0.
        if len(points) > 2:
            _, vectors = np.linalg.eigh(np.cov(points[:, ::-1].T))
            vx, vy = vectors[:, -1]
            angle = float(np.degrees(np.arctan2(vy, vx)))
        along = int(math.ceil(int(rng.integers(8, 41)) / 2))
        across = int(math.ceil(3 * target_sigma)) + int(rng.integers(0, 3))
        field = np.zeros(boundary.shape, np.float32)
        cv2.ellipse(field, (x, y), (along, across), angle, 0, 360, 1., thickness=-1)
        field = cv2.GaussianBlur(field, (3, 3), .55, borderType=cv2.BORDER_CONSTANT)
        depth = float(rng.uniform(.8, 1.0))
        erase = np.maximum(erase, field * depth)
        records.append(dict(center_yx=[y, x], length_along_native=2*along, radius_along_native=along,
                            radius_across_native=across, angle_degrees=angle, depth=depth))
    return erase * valid, records


def _false_line_field(target, valid, rng):
    interior = valid & (target < .01)
    candidates = np.argwhere(interior)
    field = np.zeros(target.shape, np.float32)
    records = []
    if not len(candidates):
        return field, records
    h, w = target.shape
    for _ in range(int(rng.integers(1, 5))):
        y, x = map(int, candidates[int(rng.integers(len(candidates)))])
        length = int(rng.integers(12, 65))
        angle = float(rng.uniform(0, 2 * np.pi))
        dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
        bend = float(rng.uniform(-8, 8))
        vertices = np.array([[x-dx, y-dy], [x-bend*math.sin(angle), y+bend*math.cos(angle)],
                             [x+dx, y+dy]], np.float64)
        vertices[:, 0] = np.clip(vertices[:, 0], 0, w-1)
        vertices[:, 1] = np.clip(vertices[:, 1], 0, h-1)
        vertices = np.rint(vertices).astype(np.int32)
        line = np.zeros(target.shape, np.float32)
        thickness = int(rng.integers(1, 3))
        strength = float(rng.uniform(.25, .65))
        cv2.polylines(line, [vertices], False, strength, thickness=thickness)
        line = cv2.GaussianBlur(line, (3, 3), .5, borderType=cv2.BORDER_CONSTANT)
        field = np.maximum(field, line)
        records.append(dict(vertices_xy=vertices.tolist(), thickness_native=thickness,
                            strength=strength))
    # 少量平滑斑点是额外负例，避免只有一种人造直线噪声。
    sparse = (rng.random(target.shape) < .0015).astype(np.float32)
    sparse *= rng.uniform(.08, .2, target.shape).astype(np.float32)
    sparse = cv2.GaussianBlur(sparse, (3, 3), .5, borderType=cv2.BORDER_CONSTANT)
    return np.maximum(field, sparse) * interior, records


def make_boundary_restore_pair(instances, rng, mode='gap', target_sigma=2., valid_margin=8):
    """由同一个已处理实例图构成 input/target；损伤掩码只覆盖可靠已知区。

    identity/gap/noise 是互斥类型。identity 完全不退化；另外两类在线改变
    软线宽和响应强度，再分别制造断口或假线。不得当成真实 affinity 分布。
    """
    if mode not in ('identity', 'gap', 'noise'):
        raise ValueError('Unknown synthetic corruption mode')
    if not isinstance(rng, np.random.Generator):
        raise TypeError('Use an explicit independent numpy Generator')
    ids = np.asarray(instances)
    boundary = known_instance_boundary(ids)
    known = ids > 0
    nominal_sigma = float(target_sigma)
    if not math.isfinite(nominal_sigma) or not .5 <= nominal_sigma <= 4:
        raise ValueError('Nominal native target sigma must be finite and in [.5,4]')
    target_sigma = nominal_sigma * float(rng.uniform(.75, 1.5))
    # 每对目标与输入保持相近宽度，主要学习修补/去假线，避免默认整体压窄。
    sigma = target_sigma if mode == 'identity' else target_sigma * float(rng.uniform(.85, 1.15))
    effective_margin = max(int(valid_margin), int(math.ceil(3 * max(target_sigma, sigma))) + 2)
    valid = reliable_region(ids, effective_margin)
    if not valid.any():
        raise ValueError('Crop has no reliable known interior')
    target = render_soft_boundary(boundary, target_sigma) * known
    requested_mode = mode
    if mode == 'gap' and not np.any(boundary & valid):
        mode = 'identity'
    if mode == 'identity':
        inputs = target.copy()
        gap = np.zeros(ids.shape, bool)
        noise = np.zeros(ids.shape, bool)
        sigma, strength, smoothing = float(target_sigma), 1., 0.
        gaps, lines = [], []
    else:
        strength = float(rng.uniform(.6, 1.))
        smoothing = float(rng.uniform(0., .35))
        inputs = render_soft_boundary(boundary, sigma) * strength
        if smoothing > .05:
            inputs = cv2.GaussianBlur(inputs, (5, 5), smoothing, borderType=cv2.BORDER_CONSTANT)
        inputs *= known
        gap, noise = np.zeros(ids.shape, bool), np.zeros(ids.shape, bool)
        gaps, lines = [], []
        if mode == 'gap':
            erased, gaps = _gap_field(boundary, valid, rng, target_sigma)
            removed = inputs * erased
            inputs -= removed
            gap = valid & (target >= .1) & (removed > .05)
        else:
            added, lines = _false_line_field(target, valid, rng)
            before = inputs.copy()
            inputs = np.maximum(inputs, added)
            noise = valid & (target < .01) & ((inputs-before) > .02)
    inputs = np.clip(inputs, 0, 1).astype(np.float32)
    target = np.clip(target, 0, 1).astype(np.float32)
    tensors = {key: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))[None]
               for key, value in dict(input=inputs, target=target, valid=valid,
                                      gap_mask=gap, noise_mask=noise).items()}
    counts = dict(total_pixels=int(ids.size), known_pixels=int(known.sum()), valid_pixels=int(valid.sum()),
                  true_boundary_pixels=int((boundary & valid).sum()), gap_pixels=int(gap.sum()),
                  noise_pixels=int(noise.sum()), unknown_pixels=int((~known).sum()))
    metadata = dict(requested_mode=requested_mode, mode=mode, native_pixel_units=True,
                    nominal_target_sigma_native=nominal_sigma,
                    target_sigma_native=float(target_sigma), input_sigma_native=sigma,
                    input_strength=strength, input_smoothing_sigma_native=smoothing,
                    minimum_valid_margin_native=int(valid_margin), valid_margin_native=effective_margin,
                    gaps=gaps, false_lines=lines,
                    coverage={**counts, 'valid_fraction':counts['valid_pixels']/counts['total_pixels'],
                              'gap_fraction_of_valid':counts['gap_pixels']/counts['valid_pixels'],
                              'noise_fraction_of_valid':counts['noise_pixels']/counts['valid_pixels']})
    return dict(**tensors, identity=torch.tensor(mode == 'identity'), meta=metadata)


class BoundaryRestoreDataset(Dataset):
    """全部32处理后源、每源等次数；视图和损伤由(epoch,draw_seed)确定。"""
    def __init__(self, manifest_path, crop_size=512, views_per_source=8, seed=20261025,
                 identity_probability=.2, gap_probability=.6, noise_probability=.2,
                 target_sigma=2., valid_margin=8, expected_sources=32):
        self.manifest_path = Path(manifest_path).resolve()
        self.crop_size, self.views_per_source = int(crop_size), int(views_per_source)
        self.seed, self.target_sigma, self.valid_margin = int(seed), float(target_sigma), int(valid_margin)
        probabilities = (float(identity_probability), float(gap_probability), float(noise_probability))
        if (not all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities)
                or not math.isclose(sum(probabilities), 1., abs_tol=1e-9)):
            raise ValueError('Mutually exclusive identity/gap/noise probabilities must sum to 1')
        if self.crop_size <= 2*self.valid_margin or self.views_per_source < 1 or self.valid_margin < 1:
            raise ValueError('Invalid native crop/known-region margin/view count')
        render_soft_boundary(np.zeros((2, 2), bool), self.target_sigma)
        self.probabilities = probabilities
        document = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if not isinstance(document, dict) or not isinstance(document.get('rows'), list):
            raise ValueError('Processed-GT source manifest with rows is required')
        self.records = []
        for row in document['rows']:
            source = row['source']
            if not re.fullmatch(r'train_[0-9]{3,4}', source):
                raise ValueError('Only named training sources are permitted')
            record = dict(source=source)
            for key in ('image', 'gt', 'classes'):
                path = Path(row[key])
                path = path if path.is_absolute() else self.manifest_path.parent/path
                expected = row.get('hashes', row).get(key+'_sha256')
                if not isinstance(expected, str) or len(expected) != 64:
                    raise ValueError('Each image/processed GT/class file requires an explicit SHA256')
                actual = file_sha256(path)
                if actual != expected:
                    raise ValueError(f'Processed-source SHA256 differs: {source}/{key}')
                record[key], record[key+'_sha256'] = str(path.resolve()), actual
            rgb, ids, _, original = load_processed_source(record['image'], record['gt'], record['classes'])
            record.update(shape=list(ids.shape), original_pixels=int(original.sum()),
                          filled_pixels=int(((ids > 0) & ~original).sum()),
                          unknown_pixels=int((ids == 0).sum()))
            self.records.append(record)
        self.records.sort(key=lambda row: row['source'])
        self.sources = [row['source'] for row in self.records]
        if len(self.sources) != int(expected_sources) or len(set(self.sources)) != len(self.sources):
            raise ValueError(f'Exactly {expected_sources} unique processed training sources required')
        self.by_source = {row['source']:row for row in self.records}
        self.identity = dict(format='boundary_restore_data_v1', manifest_sha256=file_sha256(self.manifest_path),
                             sources=[{k:v for k,v in row.items() if k not in ('image','gt','classes')}
                                      for row in self.records], source_count=len(self.sources),
                             crop_size_native=self.crop_size, views_per_source=self.views_per_source,
                             seed=self.seed, target_sigma_native=self.target_sigma,
                             valid_margin_native=self.valid_margin,
                             corruption_probabilities=dict(zip(('identity','gap','noise'), probabilities)),
                             corruption_types_mutually_exclusive=True, test_training=False,
                             target_sigma_factor_range=[.75,1.5], input_sigma_factor_of_target=[.85,1.15],
                             gap_length_along_native_range=[8,40],
                             semantic_classes_used_for_boundary=False, model_predictions_used=False,
                             known_policy='processed-positive IDs including filled; unknown/padding/crop rim ignored',
                             units='native pixels; no rescaling', independent_validation=False,
                             limitation='Synthetic soft GT rendering is a controlled prior, not measured affinity output distribution; synthetic-real domain gap remains.')
        self.set_epoch(1)

    def set_epoch(self, epoch):
        if int(epoch) < 1:
            raise ValueError('Training epoch must be positive')
        self.epoch = int(epoch)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 1703]))
        self.plan = []
        for repeat in range(self.views_per_source):
            for source in rng.permutation(self.sources):
                self.plan.append(dict(source=str(source), repeat=repeat,
                                      mode=str(rng.choice(('identity','gap','noise'), p=self.probabilities)),
                                      seed=int(rng.integers(0, 2**31-1))))

    def __len__(self):
        return len(self.plan)

    def load_source(self, source):
        record = self.by_source[source]
        return load_processed_source(record['image'], record['gt'], record['classes'])

    def __getitem__(self, index):
        spec = self.plan[int(index)]
        _, ids, _, original = self.load_source(spec['source'])
        rng = np.random.default_rng(spec['seed'])
        safe = reliable_region(ids, self.valid_margin)
        anchors = np.argwhere(known_instance_boundary(ids) & safe) if spec['mode']=='gap' else np.empty((0,2), int)
        if not len(anchors):
            anchors = np.argwhere(safe)
        if not len(anchors):
            raise ValueError('Source has no reliable native interior: '+spec['source'])
        height, width = min(self.crop_size, ids.shape[0]), min(self.crop_size, ids.shape[1])
        y, x = map(int, anchors[int(rng.integers(len(anchors)))])
        top = int(np.clip(y-int(rng.integers(self.valid_margin, height-self.valid_margin)), 0, ids.shape[0]-height))
        left = int(np.clip(x-int(rng.integers(self.valid_margin, width-self.valid_margin)), 0, ids.shape[1]-width))
        box = [top, left, height, width]
        rotation, flip = int(rng.integers(4)), bool(rng.integers(2))
        cropped, geometry = native_crop(ids, box, self.crop_size, rotation, flip)
        cropped_original, _ = native_crop(original.astype(np.int32), box, self.crop_size, rotation, flip)
        pair = make_boundary_restore_pair(cropped, rng, spec['mode'], self.target_sigma, self.valid_margin)
        pair['source'] = spec['source']
        pair['meta'].update(source=spec['source'], epoch=self.epoch, draw=int(index),
                            draw_seed=spec['seed'], repeat=spec['repeat'], geometry=geometry,
                            native_original_pixels=int((cropped_original>0).sum()),
                            native_filled_pixels=int(((cropped>0)&(cropped_original==0)).sum()))
        return pair

    def verify_sources(self):
        if file_sha256(self.manifest_path) != self.identity['manifest_sha256']:
            raise RuntimeError('Processed-GT manifest changed')
        for row in self.records:
            for key in ('image','gt','classes'):
                if file_sha256(row[key]) != row[key+'_sha256']:
                    raise RuntimeError(f'Processed source changed: {row["source"]}/{key}')
        return canonical_sha(self.identity)

    def plan_receipt(self):
        return dict(epoch=self.epoch, views=len(self.plan), plan_sha256=canonical_sha(self.plan),
                    source_counts=dict(Counter(row['source'] for row in self.plan)),
                    mode_counts=dict(Counter(row['mode'] for row in self.plan)),
                    modes_mutually_exclusive=True)


def collate_boundary_restore(samples):
    keys = ('input','target','valid','gap_mask','noise_mask','identity')
    return {**{key:torch.stack([sample[key] for sample in samples]) for key in keys},
            'source':[sample['source'] for sample in samples],
            'meta':[sample['meta'] for sample in samples]}
