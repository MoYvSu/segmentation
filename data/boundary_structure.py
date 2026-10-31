# -*- coding: utf-8 -*-
"""原生 GT 实例界面的可移动结构走廊；固定旧 RGB 自损伤配对与随机流。

几何候选不是像素级精确 GT：原标深部内核只提供实例归属，界面允许在
已知两实例并域内偏移。开放曲线只是局部连线代理，不宣称闭合实例分隔。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize

from data.boundary_context import BoundaryContextDataset, collate_boundary_context
from data.boundary_correct import _neighbours, _paths
from data.boundary_restore import native_crop
from data.dino_semantic import canonical_sha


DEFAULT_STRUCTURE_GEOMETRY = dict(corridor_radius=12, core_radius=16,
    min_core_pixels=64, min_path_points=8, tangent_half_window=4,
    closed_anchor_threshold=.65)
STRUCTURE_LOSS_KEYS = {'weight', 'response_margin', 'mass_weight',
                       'selection_change_cost', 'teacher_release_threshold'}


def validated_structure_geometry(recipe=None):
    """几何回执只绑定几何项，loss 项由损失模块验证，不静默吞掉错拼的键。"""
    supplied = {} if recipe is None else dict(recipe)
    unknown = set(supplied) - set(DEFAULT_STRUCTURE_GEOMETRY) - STRUCTURE_LOSS_KEYS
    if unknown:
        raise ValueError('Unknown boundary structure keys: ' + ','.join(sorted(unknown)))
    result = {**DEFAULT_STRUCTURE_GEOMETRY,
              **{k: v for k, v in supplied.items() if k in DEFAULT_STRUCTURE_GEOMETRY}}
    for key in set(DEFAULT_STRUCTURE_GEOMETRY) - {'closed_anchor_threshold'}:
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'{key} must be a positive native-pixel/sample integer')
    threshold = result['closed_anchor_threshold']
    if isinstance(threshold, bool) or not math.isfinite(float(threshold)) or not 0 < float(threshold) <= 1:
        raise ValueError('closed_anchor_threshold must be in (0,1]')
    result['closed_anchor_threshold'] = float(threshold)
    return result


def _array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def structure_geometry_identity(recipe=None):
    """把精确坐标规则和复用代码 SHA 一起封存，避免配方名相同而几何不同。"""
    cfg = validated_structure_geometry(recipe)
    dependencies = {}
    for module in ('boundary_structure.py', 'boundary_correct.py'):
        path = Path(__file__).with_name(module)
        dependencies['data/' + module] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(recipe=cfg, recipe_sha256=canonical_sha(cfg), builder_sources=dependencies,
        core_source='per-instance original_covered only; square erosion with constant-zero crop rim',
        interface_source='four-neighbor processed-known different IDs; no class or test GT',
        tracing='pairwise shared-interface skeleton restricted to old valid; remove degree>2 junction pixels, then complete deterministic nonbranching 8-neighbor chains; valid/rim truncation is explicitly open local surrogate',
        neighbors='axial connections suppress diagonal corner shortcuts as boundary_correct._neighbours',
        tangent='PCA over +/- tangent_half_window stations; cyclic for closed chains; deterministic direction sign',
        coordinates='integer offsets +/- corridor_radius along unit normal; numpy.rint to native yx; no interpolation',
        allowed='old valid & either pair ID & outside every original deep core; complete-path reachability pruning',
        closed_anchor='maximum clean N0 response among legal candidates; stable station/candidate tie; rotate to row0 and fix one candidate',
        open_path='partial local surrogate without two-sided strong anchors or global separation claim',
        unknown='zero IDs, padding and old invalid pixels never supervised',
        original_internal_boundaries='preserved; no same-ID connectivity or line clearing target')


def _transitions(previous, current):
    return np.max(np.abs(previous[:, None, :].astype(np.int64)
                         - current[None, :, :].astype(np.int64)), axis=-1) <= 1


def feasible_candidate_mask(coords_yx, allowed, closed=False, anchor_index=None):
    """只保留参与完整八邻接路径的候选；闭环必须回到同一个固定锚点。"""
    coords, valid = np.asarray(coords_yx), np.asarray(allowed)
    if (coords.ndim != 3 or coords.shape[-1] != 2 or coords.dtype.kind not in 'iu'
            or valid.shape != coords.shape[:2] or valid.dtype != np.bool_ or not coords.shape[0]):
        raise ValueError('Expected integer L K 2 coordinates and bool L K allowed mask')
    valid = valid.copy()
    if closed:
        if (type(anchor_index) is not int or not 0 <= anchor_index < coords.shape[1]
                or not valid[0, anchor_index]):
            return np.zeros_like(valid)
        valid[0] = False
        valid[0, anchor_index] = True
    elif anchor_index is not None:
        raise ValueError('Open paths cannot have a fixed closed anchor')
    forward = np.zeros_like(valid)
    forward[0] = valid[0]
    for row in range(1, len(coords)):
        transitions = _transitions(coords[row-1], coords[row])
        forward[row] = valid[row] & np.any(forward[row-1, :, None] & transitions, axis=0)
    backward = np.zeros_like(valid)
    if closed:
        backward[-1] = valid[-1] & (
            np.max(np.abs(coords[-1].astype(np.int64)
                          - coords[0, anchor_index].astype(np.int64)), axis=-1) <= 1)
    else:
        backward[-1] = valid[-1]
    for row in range(len(coords)-2, -1, -1):
        transitions = _transitions(coords[row], coords[row+1])
        backward[row] = valid[row] & np.any(transitions & backward[row+1, None, :], axis=1)
    result = forward & backward
    return result if np.all(result.any(axis=1)) else np.zeros_like(valid)


def _adjacent_pairs(ids):
    pairs = []
    for first, second in ((ids[:-1], ids[1:]), (ids[:, :-1], ids[:, 1:])):
        selected = (first > 0) & (second > 0) & (first != second)
        if selected.any():
            pairs.append(np.sort(np.stack((first[selected], second[selected]), axis=1), axis=1))
    return np.unique(np.concatenate(pairs, axis=0), axis=0) if pairs else np.empty((0, 2), np.int64)


def _pair_interface(ids, first, second):
    result = np.zeros(ids.shape, bool)
    a, b = ids[:-1], ids[1:]
    vertical = ((a == first) & (b == second)) | ((a == second) & (b == first))
    result[:-1] |= vertical
    result[1:] |= vertical
    a, b = ids[:, :-1], ids[:, 1:]
    horizontal = ((a == first) & (b == second)) | ((a == second) & (b == first))
    result[:, :-1] |= horizontal
    result[:, 1:] |= horizontal
    return result


def _normal_candidates(path, closed, radius, half):
    normals = []
    for row in range(len(path)):
        if closed:
            indexes = np.arange(row-half, row+half+1) % len(path)
        else:
            indexes = np.arange(max(0, row-half), min(len(path), row+half+1))
        neighbors = path[indexes].astype(np.float64)
        centered = neighbors - neighbors.mean(axis=0)
        _, vectors = np.linalg.eigh(centered.T @ centered)
        tangent = vectors[:, -1]
        direction = neighbors[-1] - neighbors[0]
        if np.dot(tangent, direction) < 0 or (np.dot(tangent, direction) == 0 and tangent[np.argmax(np.abs(tangent))] < 0):
            tangent = -tangent
        normals.append((-tangent[1], tangent[0]))
    offsets = np.arange(-radius, radius+1, dtype=np.int32)
    coords = np.rint(path[:, None, :].astype(np.float64)
                     + offsets[None, :, None] * np.asarray(normals)[:, None, :])
    return coords.astype(np.int32)


def build_boundary_structure(instances, original_known, target, valid, *, recipe=None, source=None):
    """只生成候选几何，不改 target/input，不读取预测语义，不消耗任何随机流。"""
    cfg = validated_structure_geometry(recipe)
    ids, original, heat, old_valid = map(np.asarray, (instances, original_known, target, valid))
    if ids.ndim != 2 or ids.dtype.kind not in 'iu' or np.any(ids < 0):
        raise ValueError('Structure instances must be a nonnegative integer HW map')
    if (original.shape != ids.shape or old_valid.shape != ids.shape
            or not np.isin(original, (0, 1)).all() or not np.isin(old_valid, (0, 1)).all()):
        raise ValueError('Original coverage and old valid must be aligned binary masks')
    if (heat.shape != ids.shape or heat.dtype != np.float32 or not np.isfinite(heat).all()
            or np.any((heat < 0) | (heat > 1))):
        raise ValueError('Clean N0 target must be aligned finite float32 [0,1]')
    original, old_valid = original.astype(bool), old_valid.astype(bool)
    if np.any(original & (ids == 0)) or np.any(old_valid & (ids == 0)):
        raise ValueError('Original coverage and old valid cannot include unknown zero IDs')
    json.dumps(source, allow_nan=False)
    kernel = np.ones((2*cfg['core_radius']+1, 2*cfg['core_radius']+1), np.uint8)
    cores = np.zeros(ids.shape, bool)
    core_sizes, eligible = {}, set()
    for ident in np.unique(ids[original]):
        ident = int(ident)
        if ident == 0:
            continue
        core = cv2.erode(((ids == ident) & original).astype(np.uint8), kernel,
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        cores |= core
        core_sizes[ident] = int(core.sum())
        if core_sizes[ident] >= cfg['min_core_pixels']:
            eligible.add(ident)
    pairs = _adjacent_pairs(ids)
    corridor = np.zeros(ids.shape, bool)
    paths, rejected, considered = [], Counter(), 0
    for pair in pairs:
        first, second = map(int, pair)
        if first not in eligible or second not in eligible:
            rejected['pair_without_two_original_deep_cores'] += 1
            continue
        pair_support = ((ids == first) | (ids == second)) & old_valid & ~cores
        skeleton = skeletonize(_pair_interface(ids, first, second) & old_valid)
        pixels = set(map(tuple, np.argwhere(skeleton).tolist()))
        junctions = [point for point in pixels if len(_neighbours(point, pixels)) > 2]
        if junctions:
            ys, xs = np.asarray(junctions).T
            skeleton[ys, xs] = False
        chains, invalid = _paths(skeleton)
        rejected['non_simple_components'] += invalid
        for chain_index, (centerline, closed) in enumerate(chains):
            considered += 1
            if len(centerline) < cfg['min_path_points']:
                rejected['chain_too_short'] += 1
                continue
            coords = _normal_candidates(centerline, closed, cfg['corridor_radius'], cfg['tangent_half_window'])
            inside = ((coords[..., 0] >= 0) & (coords[..., 0] < ids.shape[0])
                      & (coords[..., 1] >= 0) & (coords[..., 1] < ids.shape[1]))
            allowed = np.zeros(inside.shape, bool)
            allowed[inside] = pair_support[coords[..., 0][inside], coords[..., 1][inside]]
            anchor_index, anchor_yx, shift = None, None, 0
            if closed:
                score = np.full(allowed.shape, -1., np.float32)
                score[allowed] = heat[coords[..., 0][allowed], coords[..., 1][allowed]]
                flat = int(np.argmax(score))
                shift, anchor_index = map(int, np.unravel_index(flat, score.shape))
                if score[shift, anchor_index] < cfg['closed_anchor_threshold']:
                    rejected['closed_without_strong_clean_n0_anchor'] += 1
                    continue
                coords, allowed, centerline = (np.roll(value, -shift, axis=0)
                                               for value in (coords, allowed, centerline))
                anchor_yx = coords[0, anchor_index].tolist()
            allowed = feasible_candidate_mask(coords, allowed, bool(closed), anchor_index)
            if not allowed.any():
                rejected['no_complete_8_neighbor_path'] += 1
                continue
            corridor[coords[..., 0][allowed], coords[..., 1][allowed]] = True
            metadata = dict(pair_ids=[first, second], original_core_pixels=[core_sizes[first], core_sizes[second]],
                chain_index=chain_index, points=len(centerline), candidate_columns=coords.shape[1],
                candidate_pixels=int(allowed.sum()), closed=bool(closed), anchor_index=anchor_index,
                anchor_yx=anchor_yx, closed_rotation=shift, centerline_sha256=_array_sha(centerline),
                coords_sha256=_array_sha(coords), allowed_sha256=_array_sha(allowed),
                claim='closed local interface' if closed else 'open partial local surrogate; not full instance separation')
            paths.append(dict(coords_yx=np.ascontiguousarray(coords, dtype=np.int32),
                              allowed=np.ascontiguousarray(allowed, dtype=bool),
                              closed=bool(closed), anchor_index=anchor_index, meta=metadata))
    # 强旧响应不释放像素重建监督；合法路径仍可经过它以形成连续通路。
    free_mask = corridor & old_valid & (heat < .65)
    metadata = dict(source=source, recipe=cfg, geometry_recipe_sha256=canonical_sha(cfg),
        input_sha256=dict(instances=_array_sha(ids), original_known=_array_sha(original),
                          clean_n0=_array_sha(heat), old_valid=_array_sha(old_valid)),
        counts=dict(adjacent_pairs=len(pairs), eligible_original_core_instances=len(eligible),
            original_deep_core_pixels=int(cores.sum()), chains_considered=considered,
            paths_retained=len(paths), closed_paths=sum(path['closed'] for path in paths),
            open_paths=sum(not path['closed'] for path in paths),
            corridor_pixels=int(corridor.sum()), free_pixels=int(free_mask.sum())),
        rejected=dict(rejected), fallback_original_pair=not bool(paths),
        paths=[path['meta'] for path in paths],
        semantic_classes_used=False, test_gt_used=False, source_rng_consumed=False,
        partial_local_surrogate=True, exact_boundary_target=False,
        fixed_teacher_release_threshold=.65)
    json.dumps(metadata, allow_nan=False)
    return dict(paths=paths, corridor=corridor, free_mask=free_mask, cores=cores, meta=metadata)


class BoundaryStructureDataset(BoundaryContextDataset):
    """原配对逐位复用，只附加独立的 GT 几何候选；缓存不消费抽样随机数。"""
    def __init__(self, *args, structure_recipe=None, **kwargs):
        self._structure_source_cache = {}
        self.structure_recipe = validated_structure_geometry(structure_recipe)
        super().__init__(*args, **kwargs)
        parent = canonical_sha(self.identity)
        self.identity.update(format='boundary_structure_data_v1', data_mode='real_n0_rgb_structure',
            parent_context_identity_sha256=parent,
            structure_geometry=structure_geometry_identity(self.structure_recipe),
            structure_target='distinct original deep cores plus movable complete interface chains; no internal clearing',
            class_supervision_used=False, source_rng_consumed=False)

    def load_source(self, source):
        if source not in self._structure_source_cache:
            self._structure_source_cache[source] = super().load_source(source)
        loaded = self._structure_source_cache[source]
        # ContextDataset 依赖当次 source/rgb 绑定，命中缓存也要同步该只读引用。
        self._context_source, self._context_rgb = source, loaded[0]
        return loaded

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        _, ids, _, original = self._structure_source_cache[sample['source']]
        geometry = sample['meta']['geometry']
        cropped, _ = native_crop(ids, geometry['box'], self.crop_size,
                                 geometry['rotation'], geometry['flip'])
        cropped_original, _ = native_crop(original.astype(np.int32), geometry['box'], self.crop_size,
                                          geometry['rotation'], geometry['flip'])
        built = build_boundary_structure(cropped, cropped_original,
            sample['target'][0].numpy(), sample['valid'][0].numpy(),
            recipe=self.structure_recipe, source=sample['source'])
        sample['structure_paths'] = built['paths']
        for key, name in (('corridor', 'structure_corridor'), ('free_mask', 'structure_free_mask'),
                          ('cores', 'structure_cores')):
            sample[name] = torch.from_numpy(np.ascontiguousarray(built[key], dtype=bool))[None]
        sample['meta']['structure'] = built['meta']
        return sample


def collate_boundary_structure(samples):
    batch = collate_boundary_context(samples)
    for key in ('structure_corridor', 'structure_free_mask', 'structure_cores'):
        batch[key] = torch.stack([sample[key] for sample in samples])
    batch['structure_paths'] = [sample['structure_paths'] for sample in samples]
    return batch
