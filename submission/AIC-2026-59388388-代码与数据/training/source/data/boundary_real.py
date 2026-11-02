# -*- coding: utf-8 -*-
"""固定真实 N0 热图的自损伤配对；GT 仅用于可靠区域和损伤位置。

目标不是 GT：它逐像素保留冻结 N0 的幅值、底噪和宽度，也保留教师本来的错误。
与旧数据类复用源顺序、裁窗/旋转/翻转随机序列，不重新渲染或整体校准热图。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

from data.boundary_restore import BoundaryRestoreDataset, known_instance_boundary, native_crop, reliable_region
from data.dino_semantic import canonical_sha
from models.dino_semantic import file_sha256


DEFAULT_REAL_RECIPE = dict(
    gap_count_range=[1, 4], gap_length_native_range=[8, 40],
    gap_radius_across_native_range=[8, 14], gap_depth_range=[.8, 1.], gap_soften_sigma=.55,
    floor_neighborhood_radius=48, floor_gt_distance=16, floor_quantile=.5,
    floor_min_samples=16, floor_fallback_quantile=.1, gap_min_delta=.05,
    noise_count_range=[1, 4], noise_length_native_range=[12, 64],
    noise_thickness_native_range=[1, 2], noise_strength_range=[.25, .65],
    noise_min_gt_distance=16, noise_max_original_response=.2, noise_min_delta=.02,
)


def validated_recipe(recipe=None):
    """保存实际完整配方，拒绝拼错的键；sigma 不再代表目标渲染宽度。"""
    supplied = {} if recipe is None else dict(recipe)
    unknown = set(supplied) - set(DEFAULT_REAL_RECIPE)
    if unknown:
        raise ValueError('Unknown real-heat recipe keys: ' + ','.join(sorted(unknown)))
    result = {**DEFAULT_REAL_RECIPE, **supplied}
    integer_ranges = ('gap_count_range', 'gap_length_native_range', 'gap_radius_across_native_range',
                      'noise_count_range', 'noise_length_native_range', 'noise_thickness_native_range')
    for key in integer_ranges:
        value = result[key]
        if (not isinstance(value, (tuple, list)) or len(value) != 2
                or any(type(item) is not int or item < 1 for item in value) or value[0] > value[1]):
            raise ValueError(f'{key} requires an ascending positive integer pair')
        result[key] = list(value)
    for key in ('gap_depth_range', 'noise_strength_range'):
        value = result[key]
        if (not isinstance(value, (tuple, list)) or len(value) != 2
                or not all(math.isfinite(float(item)) and 0 < float(item) <= 1 for item in value)
                or value[0] > value[1]):
            raise ValueError(f'{key} requires an ascending pair in (0,1]')
        result[key] = [float(item) for item in value]
    for key in ('floor_neighborhood_radius', 'floor_gt_distance', 'floor_min_samples', 'noise_min_gt_distance'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'{key} must be a positive native-pixel/sample count')
    for key in ('floor_quantile', 'floor_fallback_quantile', 'noise_max_original_response'):
        if not math.isfinite(float(result[key])) or not 0 <= float(result[key]) <= 1:
            raise ValueError(f'{key} must be in [0,1]')
        result[key] = float(result[key])
    for key in ('gap_soften_sigma', 'gap_min_delta', 'noise_min_delta'):
        if not math.isfinite(float(result[key])) or not 0 < float(result[key]) <= 1:
            raise ValueError(f'{key} must be in (0,1]')
        result[key] = float(result[key])
    return result


def _probability(value, shape=None):
    value = np.asarray(value)
    if (value.ndim != 2 or not value.size or value.dtype != np.float32
            or (shape is not None and value.shape != shape)
            or not np.isfinite(value).all() or np.any((value < 0) | (value > 1))):
        raise ValueError('Real heat must be native float32 HW in [0,1] matching processed source')
    return value


def native_heat_crop(heat, box, crop_size, rotation=0, flip=False):
    """连续概率原样裁取/90度转动/翻转；不用只支持整数实例的 native_crop。"""
    heat = _probability(heat)
    top, left, height, width = map(int, box)
    if (min(top, left) < 0 or min(height, width) < 1 or top+height > heat.shape[0]
            or left+width > heat.shape[1] or max(height, width) > int(crop_size)):
        raise ValueError('Native heat crop outside source or crop size')
    cropped = np.rot90(heat[top:top+height, left:left+width], int(rotation) % 4)
    if flip:
        cropped = cropped[:, ::-1]
    result = np.zeros((int(crop_size), int(crop_size)), np.float32)
    result[:cropped.shape[0], :cropped.shape[1]] = cropped
    return result


def _integer(rng, limits):
    return int(rng.integers(limits[0], limits[1]+1))


def _floor_at(target, valid, distance, y, x, recipe):
    deep = valid & (distance > recipe['floor_gt_distance'])
    radius = recipe['floor_neighborhood_radius']
    top, bottom = max(0, y-radius), min(target.shape[0], y+radius+1)
    left, right = max(0, x-radius), min(target.shape[1], x+radius+1)
    local = target[top:bottom, left:right][deep[top:bottom, left:right]]
    quantile = recipe['floor_quantile']
    if local.size >= recipe['floor_min_samples']:
        values, strategy = local, 'local_known_deep_interior'
    elif int(deep.sum()) >= recipe['floor_min_samples']:
        values, strategy = target[deep], 'crop_known_deep_interior'
    else:
        values, strategy = target[valid], 'crop_valid_lower_quantile'
        quantile = recipe['floor_fallback_quantile']
    return float(np.quantile(values, quantile)), dict(strategy=strategy, samples=int(values.size), quantile=quantile)


def _damage_gaps(target, boundary, valid, distance, rng, recipe):
    inputs = target.copy()
    candidates = np.argwhere(boundary & valid)
    details = []
    if not len(candidates):
        return inputs, np.zeros(target.shape, bool), details
    for _ in range(_integer(rng, recipe['gap_count_range'])):
        y, x = map(int, candidates[int(rng.integers(len(candidates)))])
        top, bottom = max(0, y-9), min(target.shape[0], y+10)
        left, right = max(0, x-9), min(target.shape[1], x+10)
        points = np.argwhere(boundary[top:bottom, left:right]).astype(np.float64)
        angle = 0.
        if len(points) > 2:
            _, vectors = np.linalg.eigh(np.cov(points[:, ::-1].T))
            vx, vy = vectors[:, -1]
            angle = float(np.degrees(np.arctan2(vy, vx)))
        along = int(math.ceil(_integer(rng, recipe['gap_length_native_range']) / 2))
        across = _integer(rng, recipe['gap_radius_across_native_range'])
        field = np.zeros(target.shape, np.float32)
        cv2.ellipse(field, (x, y), (along, across), angle, 0, 360, 1., thickness=-1)
        field = cv2.GaussianBlur(field, (3, 3), recipe['gap_soften_sigma'], borderType=cv2.BORDER_CONSTANT)
        depth = float(rng.uniform(*recipe['gap_depth_range']))
        field *= depth * valid
        floor, floor_meta = _floor_at(target, valid, distance, y, x, recipe)
        # min 保证 floor 不会抬高任何原始像素；多断口重叠也不低于同一floor。
        candidate = target - (target - np.minimum(target, np.float32(floor))) * field
        inputs = np.minimum(inputs, candidate)
        details.append(dict(center_yx=[y, x], length_along_native=2*along,
                            radius_along_native=along, radius_across_native=across,
                            angle_degrees=angle, depth=depth, floor=floor, floor_estimation=floor_meta))
    mask = valid & ((target-inputs) > recipe['gap_min_delta'])
    return inputs, mask, details


def _damage_noise(target, valid, distance, rng, recipe):
    # 仅生成新增假线；N0已经存在的内部响应完全留在目标和输入里。
    eligible = valid & (distance > recipe['noise_min_gt_distance']) & (target <= recipe['noise_max_original_response'])
    candidates = np.argwhere(eligible)
    added = np.zeros(target.shape, np.float32)
    details = []
    if not len(candidates):
        return target.copy(), np.zeros(target.shape, bool), details
    height, width = target.shape
    for _ in range(_integer(rng, recipe['noise_count_range'])):
        y, x = map(int, candidates[int(rng.integers(len(candidates)))])
        length = _integer(rng, recipe['noise_length_native_range'])
        angle = float(rng.uniform(0, 2*np.pi))
        dx, dy = math.cos(angle)*length/2, math.sin(angle)*length/2
        bend = float(rng.uniform(-8, 8))
        vertices = np.array([[x-dx, y-dy], [x-bend*math.sin(angle), y+bend*math.cos(angle)],
                             [x+dx, y+dy]], np.float64)
        vertices[:, 0] = np.clip(vertices[:, 0], 0, width-1)
        vertices[:, 1] = np.clip(vertices[:, 1], 0, height-1)
        vertices = np.rint(vertices).astype(np.int32)
        thickness = _integer(rng, recipe['noise_thickness_native_range'])
        strength = float(rng.uniform(*recipe['noise_strength_range']))
        line = np.zeros(target.shape, np.float32)
        cv2.polylines(line, [vertices], False, strength, thickness=thickness)
        line = cv2.GaussianBlur(line, (3, 3), .5, borderType=cv2.BORDER_CONSTANT)
        added = np.maximum(added, line)
        details.append(dict(vertices_xy=vertices.tolist(), length_native=length,
                            thickness_native=thickness, additive_strength=strength))
    inputs = np.clip(target + added*eligible, 0., 1.).astype(np.float32)
    return inputs, valid & ((inputs-target) > recipe['noise_min_delta']), details


def make_boundary_real_pair(instances, heat, rng, mode='gap', valid_margin=8, recipe=None):
    """原N0目标精确不变；unknown仅作为上下文，不监督、不损坏、不清零。"""
    if mode not in ('identity', 'gap', 'noise'):
        raise ValueError('Unknown real-heat corruption mode')
    if not isinstance(rng, np.random.Generator):
        raise TypeError('Use an explicit independent numpy Generator')
    recipe = validated_recipe(recipe)
    ids = np.asarray(instances)
    boundary = known_instance_boundary(ids)
    target = _probability(heat, ids.shape).copy()
    valid = reliable_region(ids, int(valid_margin))
    if not valid.any():
        raise ValueError('Crop has no reliable known interior')
    distance = distance_transform_edt(~boundary) if boundary.any() else np.full(ids.shape, np.inf)
    gap = np.zeros(ids.shape, bool); noise = gap.copy()
    gaps, lines = [], []
    requested_mode = mode
    if mode == 'gap':
        inputs, gap, gaps = _damage_gaps(target, boundary, valid, distance, rng, recipe)
    elif mode == 'noise':
        inputs, noise, lines = _damage_noise(target, valid, distance, rng, recipe)
    else:
        inputs = target.copy()
    # 没有真正超过损伤门槛的样本回退到精确identity，不留下无记录的轻微全图任务。
    fallback_reason = None
    if mode != 'identity' and not (gap.any() or noise.any()):
        inputs = target.copy(); fallback_reason = 'no_effective_local_damage'; mode = 'identity'
    delta = inputs-target
    metadata = dict(requested_mode=requested_mode, mode=mode, fallback_reason=fallback_reason,
                    native_pixel_units=True, target_kind='frozen_N0_teacher_response', target_exact=True,
                    global_strength_change=False, global_width_change=False, gaps=gaps, false_lines=lines,
                    recipe=recipe, coverage=dict(total_pixels=int(ids.size), known_pixels=int((ids>0).sum()),
                    valid_pixels=int(valid.sum()), true_boundary_pixels=int((boundary & valid).sum()),
                    gap_pixels=int(gap.sum()), noise_pixels=int(noise.sum()), unknown_pixels=int((ids==0).sum()),
                    changed_pixels=int(np.count_nonzero(delta)), changed_outside_valid=int(np.count_nonzero(delta[~valid])),
                    valid_fraction=float(valid.mean()), gap_fraction_of_valid=float(gap.sum()/valid.sum()),
                    noise_fraction_of_valid=float(noise.sum()/valid.sum())),
                    damage=dict(abs_max=float(np.abs(delta).max()), valid_abs_mean=float(np.abs(delta[valid]).mean())))
    arrays = dict(input=inputs, target=target, valid=valid, gap_mask=gap, noise_mask=noise)
    tensors = {key:torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))[None]
               for key,value in arrays.items()}
    return dict(**tensors, identity=torch.tensor(mode == 'identity'), meta=metadata)


class BoundaryRealDataset(BoundaryRestoreDataset):
    """复用旧实验计划，只替换固定热图目标和配套局部损伤。"""
    def __init__(self, manifest_path, *, real_manifest_path, recipe=None, crop_size=512,
                 views_per_source=8, seed=20261025, identity_probability=.2, gap_probability=.6,
                 noise_probability=.2, valid_margin=8, expected_sources=32):
        super().__init__(manifest_path, crop_size=crop_size, views_per_source=views_per_source,
                         seed=seed, identity_probability=identity_probability, gap_probability=gap_probability,
                         noise_probability=noise_probability, valid_margin=valid_margin, expected_sources=expected_sources)
        self.real_manifest_path = Path(real_manifest_path).resolve()
        self.recipe = validated_recipe(recipe)
        document = json.loads(self.real_manifest_path.read_text(encoding='utf-8'))
        if (document.get('format') != 'boundary_real_heatmaps_v1' or document.get('complete') is not True
                or not isinstance(document.get('files'), list)):
            raise ValueError('A complete frozen N0 boundary_real_heatmaps_v1 manifest is required')
        if document.get('processed_gt_manifest_sha256') != file_sha256(self.manifest_path):
            raise ValueError('Frozen N0 provenance references a different processed GT manifest')
        if (document.get('training') is not False or document.get('frozen_models_unchanged') is not True
                or document.get('sealed_source_unchanged') is not True
                or document.get('normalization') != 'none; original N0 float32 [0,1]'):
            raise ValueError('N0 capture must explicitly preserve frozen models/source and original unnormalized response')
        if any(not isinstance(document.get(key), dict) or not document[key] for key in ('config','model_states','runtime')):
            raise ValueError('Frozen N0 config/model_states/runtime provenance is required')
        states = document['model_states']
        protocol = document['config'].get('submission', {})
        noise = document['runtime'].get('restoration_noise', {})
        if (not all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) for value in states.values())
                or type(protocol.get('seed')) is not int or not isinstance(protocol.get('restoration_noise'), str)
                or not isinstance(noise, dict) or noise.get('name') != protocol.get('restoration_noise')):
            raise ValueError('N0 model hashes and matching runtime/config noise protocol are required')
        rows = document['files']
        if (len(rows) != len(self.sources) or len({row['source'] for row in rows}) != len(rows)
                or {row['source'] for row in rows} != set(self.sources)):
            raise ValueError('N0 heatmap sources must exactly match processed training sources')
        self._heat_cache = {}
        for row in rows:
            record = self.by_source[row['source']]
            if row.get('image_sha256') != record['image_sha256'] or row.get('shape') != record['shape']:
                raise ValueError('N0 image identity/shape differs from processed source: '+row['source'])
            path = Path(row['heat_path'])
            if not path.is_absolute():
                path = self.real_manifest_path.parent/path
            digest = file_sha256(path)
            if digest != row.get('heat_sha256'):
                raise ValueError('N0 heatmap SHA256 differs: '+row['source'])
            record.update(heat=str(path.resolve()), heat_sha256=digest)
        self.identity = dict(format='boundary_real_data_v1', data_mode='real_n0_heatmaps',
            manifest_sha256=file_sha256(self.manifest_path), real_manifest_sha256=file_sha256(self.real_manifest_path),
            n0_manifest_provenance={key:value for key,value in document.items() if key != 'files'},
            teacher_files=[{key:value for key,value in row.items() if key not in ('heat_path','map_path','source_path')}
                           for row in sorted(rows, key=lambda row:row['source'])],
            sources=[{key:value for key,value in row.items() if key not in ('image','gt','classes','heat')}
                     for row in self.records], source_count=len(self.sources), seed=self.seed,
            crop_size_native=self.crop_size, views_per_source=self.views_per_source, valid_margin_native=self.valid_margin,
            corruption_probabilities=dict(zip(('identity','gap','noise'), self.probabilities)),
            corruption_types_mutually_exclusive=True, recipe=self.recipe, target_kind='frozen_N0_teacher_response',
            fixed_corruption_operations=dict(gap_orientation_radius_native=9, blur_kernel=[3,3],
                noise_blur_sigma_native=.5, noise_bend_native_range=[-8,8],
                gap_angle='local GT boundary PCA', noise_composition='additive max-of-lines; no sparse global dots',
                background_floor='local known deep median, then crop deep, then valid lower quantile; never raises input'),
            target_global_remapping=False, target_gt_rendered=False, predictions_used=True, test_training=False,
            known_policy='processed-positive IDs including filled; GT only masks/corruption locations; unknown retains N0 context without supervision',
            units='native pixels; no rescaling', independent_validation=False,
            limitation='Teacher response is not GT; this task cannot correct boundaries already missing or wrong in the frozen teacher.')

    def load_heat(self, source):
        # 只读按源缓存避免每个draw重复解压；缓存不接触任何随机流。
        if source not in self._heat_cache:
            record = self.by_source[source]
            with np.load(record['heat'], allow_pickle=False) as payload:
                if payload.files != ['boundary']:
                    raise ValueError('N0 heatmap cache must contain only boundary')
                heat = _probability(payload['boundary'], tuple(record['shape'])).copy()
            heat.setflags(write=False)
            self._heat_cache[source] = heat
        return self._heat_cache[source]

    def __getitem__(self, index):
        # 与旧类到 geometry 为止保持同样的 RNG 消耗，保证同seed同epoch同crop。
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
        heat = native_heat_crop(self.load_heat(spec['source']), box, self.crop_size, rotation, flip)
        pair = make_boundary_real_pair(cropped, heat, rng, spec['mode'], self.valid_margin, self.recipe)
        pair['source'] = spec['source']
        pair['meta'].update(source=spec['source'], epoch=self.epoch, draw=int(index), draw_seed=spec['seed'],
            repeat=spec['repeat'], geometry=geometry, heat_sha256=self.by_source[spec['source']]['heat_sha256'],
            native_original_pixels=int((cropped_original>0).sum()),
            native_filled_pixels=int(((cropped>0)&(cropped_original==0)).sum()))
        return pair

    def verify_sources(self):
        super().verify_sources()
        if file_sha256(self.real_manifest_path) != self.identity['real_manifest_sha256']:
            raise RuntimeError('Frozen N0 manifest changed')
        for row in self.records:
            if file_sha256(row['heat']) != row['heat_sha256']:
                raise RuntimeError('Frozen N0 heatmap changed: '+row['source'])
        return canonical_sha(self.identity)
