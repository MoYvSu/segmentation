# -*- coding: utf-8 -*-
"""真实N1输入的区域分隔数据；保留旧配对，另给可靠内核与可改走廊。

实例身份来自处理后GT，内核只由原标覆盖生成。filled仅可提供界面位置提示，
不创造可靠种子；unknown、裁边与padding均不提供新的监督。这里不强制同GT
所有内核连通，训练器应从可靠低响应区域中选择单个连通原子作为两侧种子。
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

import cv2
import numpy as np
import torch

from data.boundary_naf import canonical_sha, collate_boundary_naf, file_sha256
from data.boundary_naf_real import BoundaryNAFRealDataset
from utils.boundary_core_supervision import build_trusted_cores


FORMAT = 'boundary_naf_region_data_v1'
REGION_KEYS = ('region_ids', 'region_original', 'region_core_ids', 'region_allowed',
               'region_keep_high', 'region_keep_low')


def _array_sha(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(str(value.shape).encode('ascii'))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _distance_value(value, name, minimum=0.):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f'{name} must be finite and >= {minimum}')
    return float(value)


def _region_recipe(cfg):
    region = cfg.get('region', {})
    if not isinstance(region, dict) or not isinstance(region.get('data', {}), dict):
        raise ValueError('boundary_naf.region.data must be an object')
    values = copy.deepcopy(region.get('data', {}))
    allowed_keys = {'core_radius', 'min_core_component_pixels', 'corridor_radius',
                    'keep_high_distance', 'keep_low_distance', 'keep_low_gt_margin'}
    if set(values) - allowed_keys:
        raise ValueError('Unknown region.data key: ' + ', '.join(sorted(set(values) - allowed_keys)))
    recipe = dict(
        core_radius=_integer(values.get('core_radius', 4), 'core_radius'),
        min_core_component_pixels=_integer(values.get('min_core_component_pixels', 64),
                                           'min_core_component_pixels', 1),
        corridor_radius=_distance_value(values.get('corridor_radius', 12), 'corridor_radius'),
        keep_high_distance=_distance_value(values.get('keep_high_distance', 4), 'keep_high_distance'),
        keep_low_distance=_distance_value(values.get('keep_low_distance', 4), 'keep_low_distance'))
    recipe['keep_low_gt_margin'] = _distance_value(
        values.get('keep_low_gt_margin', recipe['corridor_radius']), 'keep_low_gt_margin')
    if recipe['keep_low_gt_margin'] < recipe['corridor_radius']:
        raise ValueError('keep_low_gt_margin must be >= corridor_radius to avoid contradictory supervision')
    return recipe


def _distance_to(mask):
    """空边界没有零点，OpenCV的有限回退值不能被解释为有效距离。"""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2,
                                cv2.DIST_MASK_PRECISE)


def _native_view(array, spec, side):
    """与原配对同裁窗/D4/reflect，保留整数ID，不调用会转bool的_crop_pair。"""
    y, x, height, width = map(int, spec['box'])
    view = np.rot90(array[y:y + height, x:x + width], int(spec['rotation']) % 4)
    if spec['flip']:
        view = view[:, ::-1]
    view = np.array(view, copy=True, order='C')
    bottom, right = side - view.shape[0], side - view.shape[1]
    if bottom or right:
        mode = 'reflect' if min(view.shape) > 1 else 'edge'
        view = np.pad(view, ((0, bottom), (0, right)), mode=mode)
    return np.ascontiguousarray(view)


class BoundaryNAFRegionDataset(BoundaryNAFRealDataset):
    """同一真实配对/计划，追加区域监督字段；不产生人为擦断或新留出。"""

    def __init__(self, rows_or_manifest, config=None, epoch=1, *, real_input_manifest=None):
        outer = {} if config is None else dict(config)
        cfg = outer.get('boundary_naf', outer)
        self.region_recipe = _region_recipe(cfg)
        super().__init__(rows_or_manifest, config=config, epoch=epoch,
                         real_input_manifest=real_input_manifest)
        parent_format = self.identity['format']
        self.region_cache = {}
        self.region_records = []
        for row in self.records:
            source = row['source']
            if file_sha256(row['gt']) != row['gt_sha256']:
                raise ValueError('Processed GT changed before region cache construction: ' + source)
            with np.load(row['gt'], allow_pickle=False) as payload:
                ids = np.ascontiguousarray(payload['instance_map'], dtype=np.uint16)
                original = np.ascontiguousarray(payload['original_covered'], dtype=bool)
            # 父类已核GT的0..65535、provenance与SHA，内核仍复用项目原生实现。
            trusted = build_trusted_cores(ids, original,
                radius=self.region_recipe['core_radius'],
                min_component_pixels=self.region_recipe['min_core_component_pixels'])
            cores = np.ascontiguousarray(trusted['core_map'], dtype=np.uint16)
            target, valid = self.cache[source]
            skeleton = self.input_skeletons[source]
            distance_gt = _distance_to(target)
            distance_input = _distance_to(skeleton)
            allowed = (distance_gt <= self.region_recipe['corridor_radius']) & valid
            high = skeleton & (distance_gt <= self.region_recipe['keep_high_distance']) & valid
            low = ((cores > 0) & (distance_input > self.region_recipe['keep_low_distance'])
                   & (distance_gt > self.region_recipe['keep_low_gt_margin']) & valid)
            assert not np.any(allowed & low)
            arrays = dict(region_ids=ids, region_original=original,
                region_core_ids=cores, region_allowed=np.ascontiguousarray(allowed),
                region_keep_high=np.ascontiguousarray(high),
                region_keep_low=np.ascontiguousarray(low))
            self.region_cache[source] = arrays
            self.region_records.append(dict(source=source, gt_sha256=row['gt_sha256'],
                shape=list(ids.shape), array_sha256={key:_array_sha(value) for key,value in arrays.items()},
                trusted_cores=copy.deepcopy(trusted['summary']),
                allowed_pixels=int(allowed.sum()), keep_high_pixels=int(high.sum()),
                keep_low_pixels=int(low.sum())))
        self.identity.update(format=FORMAT, region_parent_format=parent_format,
            region_data=copy.deepcopy(self.region_recipe), region_sources=copy.deepcopy(self.region_records),
            region_core_implementation='utils.boundary_core_supervision.build_trusted_cores',
            region_core_implementation_sha256=file_sha256(Path(build_trusted_cores.__code__.co_filename)),
            region_policy='Original-only reliable core identities; GT corridor positions may use known filled; '
                          'no forced same-GT connectivity; keep-low excludes input lines and editable corridor; '
                          'no supervision outside parent valid or in padding.')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        spec = self.plan[int(index)]
        full = self.region_cache[spec['source']]
        valid = sample['valid'][0].numpy().astype(bool)
        arrays = {key:_native_view(value, spec, self.crop_size) for key,value in full.items()}
        # ids仅提供几何身份；反射padding可以带ID，但不得成为seed或监督。
        for key in REGION_KEYS[1:]:
            if key == 'region_core_ids':
                arrays[key] = np.where(valid, arrays[key], 0)
            else:
                arrays[key] = arrays[key] & valid
        for key, value in arrays.items():
            dtype = np.int64 if key in ('region_ids', 'region_core_ids') else bool
            sample[key] = torch.from_numpy(np.ascontiguousarray(value, dtype=dtype))[None]
        assert not torch.any(sample['region_allowed'] & sample['region_keep_low'])
        sample['meta']['region'] = dict(
            format=FORMAT, recipe=copy.deepcopy(self.region_recipe),
            reliable_core_pixels=int((sample['region_core_ids'] > 0).sum()),
            allowed_pixels=int(sample['region_allowed'].sum()),
            keep_high_pixels=int(sample['region_keep_high'].sum()),
            keep_low_pixels=int(sample['region_keep_low'].sum()),
            padding_supervised=False, same_GT_connectivity_forced=False)
        return sample

    def verify_sources(self):
        super().verify_sources()
        for row in self.region_records:
            arrays = self.region_cache[row['source']]
            for key, expected in row['array_sha256'].items():
                if _array_sha(arrays[key]) != expected:
                    raise RuntimeError('Region in-memory source changed: ' + row['source'] + '/' + key)
        return canonical_sha(self.identity)


def collate_boundary_naf_region(samples):
    """原NAF元数据合同不变，新增区域字段按batch堆叠。"""
    result = collate_boundary_naf(samples)
    result.update({key:torch.stack([sample[key] for sample in samples]) for key in REGION_KEYS})
    return result
