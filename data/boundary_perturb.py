# -*- coding: utf-8 -*-
"""固定前端受扰热图的NAF数据包装；GT、源图及原裁窗/D4计划不变。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
import torch

from data.boundary_naf import canonical_sha, file_sha256
from data.boundary_naf_real import LOCAL_RESTORATION, _full_input_crop
from data.boundary_rgb import BoundaryRGBDataset, collate_boundary_rgb
from utils.boundary_input import normalize_boundary_input


FORMAT = 'boundary_naf_perturbed_heatmaps_v1'
SELECTION_VERSION = 'source_epoch_draw_sha256_v1'


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sha(value, description):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(description + ' requires lowercase SHA256')
    return value


def perturb_settings(value):
    if not isinstance(value, dict) or set(value) - {'manifest', 'clean_probability', 'variant_seed'}:
        raise ValueError('input_perturbation requires manifest/clean_probability/variant_seed')
    manifest = value.get('manifest')
    probability = value.get('clean_probability', .5)
    seed = value.get('variant_seed', 20261030)
    if (not isinstance(manifest, (str, Path)) or not str(manifest)
            or type(probability) not in (int, float) or not np.isfinite(probability)
            or not 0 <= probability <= 1 or type(seed) is not int):
        raise ValueError('Invalid input_perturbation settings')
    return dict(manifest=str(manifest), clean_probability=float(probability), variant_seed=seed)


def choose_variant(variant_ids, *, clean_probability, seed, epoch, draw, source):
    """独立散列流，不读取或推进训练/裁窗/模型随机数。"""
    token = f'{SELECTION_VERSION}:{seed}:{epoch}:{draw}:{source}'.encode('utf8')
    digest = hashlib.sha256(token).digest()
    fraction = int.from_bytes(digest[:8], 'little') / 2**64
    return ('clean' if fraction < clean_probability else
            variant_ids[int.from_bytes(digest[8:16], 'little') % len(variant_ids)])


class BoundaryPerturbDataset(BoundaryRGBDataset):
    """一张源图仍只占一个采样位置；每次访问选择clean或已冻结受扰热图。"""

    def __init__(self, rows_or_manifest, config=None, epoch=1, *, real_input_manifest=None):
        outer = {} if config is None else copy.deepcopy(config)
        cfg = outer.get('boundary_naf', outer)
        settings = perturb_settings(cfg.get('input_perturbation'))
        if (cfg.get('input_mode') != 'boundary' or cfg.get('erasure', {}).get('enabled', False)
                or cfg.get('restored_rgb_manifest') is not None):
            raise ValueError('Perturbed-cache experiment only supports unchanged T-only input without erasure/RGB')
        super().__init__(rows_or_manifest, config=outer, epoch=epoch,
                         real_input_manifest=real_input_manifest)
        if self.input_normalization != {'mode': 'skeleton_dilate', 'kernel_size': 8}:
            raise ValueError('Perturbed-cache experiment requires whole-source skeleton_dilate kernel8')
        path = Path(settings['manifest'])
        if not path.is_absolute():
            if 'paths' in outer:
                from utils.config import project_path
                path = Path(project_path(outer, str(path)))
            else:
                path = (self.manifest_path.parent if self.manifest_path else Path.cwd()) / path
        self.perturb_manifest_path = path.resolve()
        self.perturb_manifest_sha256 = file_sha256(self.perturb_manifest_path)
        document = json.loads(self.perturb_manifest_path.read_text(encoding='utf8'))
        if (not isinstance(document, dict) or document.get('format') != FORMAT
                or document.get('complete') is not True or document.get('no_test_training') is not True
                or document.get('local_restoration') != LOCAL_RESTORATION
                or document.get('clean_real_input_manifest_sha256') != self.real_input_manifest_sha256
                or document.get('processed_gt_manifest_sha256') != self.manifest_sha256):
            raise ValueError('Perturbed heatmap manifest is incomplete or not bound to the clean sources/GT')
        for name, expected in self.identity['real_input_models'].items():
            value = document.get('models', {}).get(name)
            actual = value.get('sha256') if isinstance(value, dict) else value
            if actual != expected:
                raise ValueError('Perturbed heatmap frozen model differs: ' + name)
        variants = document.get('variants')
        if not isinstance(variants, list) or not variants:
            raise ValueError('At least one frozen perturbed variant is required')
        self.perturb_settings = settings
        self.variant_ids = []
        self.variant_boundaries, self.variant_raw_boundaries = {}, {}
        self.variant_records = []
        for variant in variants:
            vid = variant.get('id') if isinstance(variant, dict) else None
            if (not isinstance(vid, str) or re.fullmatch(r'[a-z][a-z0-9_]{0,31}', vid) is None
                    or vid == 'clean' or vid in self.variant_ids
                    or not isinstance(variant.get('augmentation'), dict)):
                raise ValueError('Variant ID/augmentation must be unique and explicit')
            rows = variant.get('rows')
            names = [r.get('source') if isinstance(r, dict) else None for r in rows] if isinstance(rows, list) else []
            if (len(names) != len(self.sources) or len(set(names)) != len(names)
                    or set(names) != set(self.sources)):
                raise ValueError('Every variant must match the unique training-source set exactly')
            self.variant_ids.append(vid)
            self.variant_boundaries[vid], self.variant_raw_boundaries[vid] = {}, {}
            for row in sorted(rows, key=lambda r: r['source']):
                source = row['source']; original = self.by_source[source]
                for key in ('image_sha256', 'gt_sha256', 'classes_sha256'):
                    if row.get(key) != original[key]:
                        raise ValueError('Perturbed heatmap source binding differs: '+source+'/'+key)
                heat = Path(row.get('heat_path', ''))
                if not row.get('heat_path'):
                    raise ValueError('Perturbed heat_path is required')
                if not heat.is_absolute():
                    heat = (self.perturb_manifest_path.parent / heat).resolve()
                    if not heat.is_relative_to(self.perturb_manifest_path.parent):
                        raise ValueError('Relative perturbed heat path escapes its manifest directory')
                expected = _sha(row.get('heat_sha256'), source+' heat')
                if file_sha256(heat) != expected:
                    raise ValueError('Perturbed heatmap file SHA256 differs: '+source)
                with np.load(heat, allow_pickle=False) as arrays:
                    if 'boundary' not in arrays.files:
                        raise ValueError('Perturbed heatmap needs a boundary array')
                    boundary = arrays['boundary'].copy()
                shape = self.cache[source][0].shape
                if list(row.get('shape', [])) != list(shape) or boundary.shape != shape:
                    raise ValueError('Perturbed heatmap native shape differs: '+source)
                representation = row.get('representation', 'float32_probability')
                if representation == 'thresholded_binary':
                    if (boundary.dtype not in (np.dtype('bool'), np.dtype('uint8'))
                            or not np.isin(boundary, (0, 1)).all() or type(row.get('threshold')) not in (int, float)
                            or row['threshold'] != .5):
                        raise ValueError('Binary cache requires bool/uint8 0/1 at threshold0.5')
                    continuous_sha = _sha(row.get('continuous_boundary_sha256'), source+' continuous B')
                    raw = np.ascontiguousarray(boundary, dtype=bool)
                    if array_sha(raw) != _sha(row.get('binary_sha256'), source+' binary T'):
                        raise ValueError('Binary cache array SHA256 differs: '+source)
                elif representation == 'float32_probability':
                    if (boundary.dtype != np.float32 or not np.isfinite(boundary).all()
                            or np.any((boundary < 0) | (boundary > 1))):
                        raise ValueError('Continuous cache must be finite float32 probability')
                    continuous_sha = array_sha(boundary)
                    raw = np.ascontiguousarray(boundary >= .5)
                else:
                    raise ValueError('Unknown perturbed heatmap representation')
                normalized, normalization = normalize_boundary_input(raw.astype(np.float32), settings=self.input_normalization)
                self.variant_boundaries[vid][source] = normalized
                self.variant_raw_boundaries[vid][source] = raw
                self.variant_records.append(dict(variant=vid, source=source, heat_path=str(heat.resolve()),
                    heat_sha256=expected, representation=representation, continuous_boundary_sha256=continuous_sha,
                    raw_binary_sha256=array_sha(raw), normalized_sha256=array_sha(normalized),
                    shape=list(shape), normalization=normalization))
        self.variant_ids.sort()
        records = [{k: v for k, v in r.items() if k != 'heat_path'} for r in self.variant_records]
        self.identity.update(format='boundary_perturbed_cache_data_v1',
            perturb_manifest_sha256=self.perturb_manifest_sha256,
            input_perturbation=dict(clean_probability=settings['clean_probability'], variant_seed=settings['variant_seed'],
                selection=SELECTION_VERSION, variant_ids=self.variant_ids),
            variant_sources=records, variant_sources_sha256=canonical_sha(records),
            no_new_sources_or_budget=True, crop_and_D4_plan_unchanged=True,
            perturbation_scope='RGB augmentation before frozen D5a/N1, cache boundary only; GT unchanged')

    def selected_variant(self, index):
        return choose_variant(self.variant_ids, clean_probability=self.perturb_settings['clean_probability'],
            seed=self.perturb_settings['variant_seed'], epoch=self.epoch, draw=int(index), source=self.plan[int(index)]['source'])

    def sample_variant(self, index, variant='clean'):
        if variant != 'clean' and variant not in self.variant_ids:
            raise ValueError('Unknown selected perturbation variant')
        sample = super().__getitem__(index)
        clean = sample['input_boundary'].clone()
        clean_gap = sample['gap_mask'].clone()
        if variant != 'clean':
            spec = self.plan[int(index)]; source = spec['source']; original_valid = self.cache[source][1]
            binary, geometry = _full_input_crop(self.variant_boundaries[variant][source], original_valid,
                spec, self.crop_size, self.crop_margin)
            raw, raw_geometry = _full_input_crop(self.variant_raw_boundaries[variant][source], original_valid,
                spec, self.crop_size, self.crop_margin)
            if geometry != sample['meta']['geometry'] or raw_geometry != geometry:
                raise RuntimeError('Perturbed input and original GT/D4 geometry differ')
            value = torch.from_numpy(np.ascontiguousarray(binary, dtype=np.float32))[None]
            sample.update(input=value, input_boundary=value,
                raw_input_boundary=torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32))[None])
            target, valid = sample['target'][0].numpy().astype(bool), sample['valid'][0].numpy().astype(bool)
            near = cv2.dilate(binary.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            near_raw = cv2.dilate(raw.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            sample['gap_mask'] = torch.from_numpy(np.ascontiguousarray(target & valid & ~near, dtype=np.float32))[None]
            sample['raw_gap_mask'] = torch.from_numpy(np.ascontiguousarray(target & valid & ~near_raw, dtype=np.float32))[None]
            sample['identity'] = torch.tensor(bool(np.array_equal(binary, target)))
        new_missing = sample['gap_mask'].bool() & ~clean_gap.bool()
        sample.update(clean_input_boundary=clean, clean_gap_mask=clean_gap,
                      new_missing_gt_mask=new_missing.float())
        sample['meta'].update(input_variant=variant, variant_selection=SELECTION_VERSION,
            perturb_manifest_sha256=self.perturb_manifest_sha256,
            gap_pixels=int(sample['gap_mask'].sum()), input_boundary_pixels=int(sample['input_boundary'].sum()),
            raw_input_boundary_pixels=int(sample['raw_input_boundary'].sum()),
            clean_gap_pixels=int(clean_gap.sum()), new_missing_gt_points=int(new_missing.sum()),
            old_missing_gt_points=int(clean_gap.sum()),
            variant_selected_by_GT=False, original_source_and_GT_unchanged=True,
            artificial_corruption=variant != 'clean', no_artificial_corruption=variant == 'clean')
        return sample

    def __getitem__(self, index):
        return self.sample_variant(index, self.selected_variant(index))

    def verify_sources(self):
        super().verify_sources()
        if file_sha256(self.perturb_manifest_path) != self.perturb_manifest_sha256:
            raise RuntimeError('Perturbed heatmap manifest changed')
        for row in self.variant_records:
            if file_sha256(row['heat_path']) != row['heat_sha256']:
                raise RuntimeError('Perturbed heatmap file changed: '+row['source'])
            vid, source = row['variant'], row['source']
            if (array_sha(self.variant_boundaries[vid][source]) != row['normalized_sha256']
                    or array_sha(self.variant_raw_boundaries[vid][source]) != row['raw_binary_sha256']):
                raise RuntimeError('In-memory perturbed input changed: '+source)
        return canonical_sha(self.identity)


def collate_boundary_perturb(samples):
    result = collate_boundary_rgb(samples)
    for key in ('clean_gap_mask', 'new_missing_gt_mask'):
        result[key] = torch.stack([sample[key] for sample in samples])
    return result
