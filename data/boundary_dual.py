# -*- coding: utf-8 -*-
"""同源T8与连续N1双输入；GT、裁块、D4和旧诊断分母保持不变。"""
from __future__ import annotations

import copy
import hashlib
import zipfile

import numpy as np
import torch

from data.boundary_naf import canonical_sha
from data.boundary_rgb import BoundaryRGBDataset, collate_boundary_rgb


FORMAT = 'boundary_dual_data_v1'
INPUT_MODE = 'boundary_continuous'
INPUT_CONTRACT = 't8_continuous_v1'
NORMALIZATION = {'mode': 'skeleton_dilate', 'kernel_size': 8}
DEFAULT_CONTINUOUS_CACHE_MAX_BYTES = 1024 ** 3


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def continuous_heat_header(path, expected_shape):
    """先检查NPY头，再允许解压已封存的二维float32数组。"""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names.count('boundary.npy') != 1:
            raise ValueError('Continuous N1 archive needs exactly one boundary.npy')
        with archive.open('boundary.npy') as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError('Unsupported continuous N1 NPY version')
    if (tuple(shape) != tuple(expected_shape) or len(shape) != 2
            or dtype != np.dtype('float32') or fortran):
        raise ValueError('Continuous N1 must be native C-order float32 HW matching T8/GT')
    return dict(shape=list(shape), dtype='float32', bytes=int(np.prod(shape)) * 4,
                fortran_order=False, npy_version=list(version))


def crop_continuous_boundary(heat, box, crop_size, rotation=0, flip=False):
    """与旧T8相同的裁块/D4/右下reflect；不插值、不转bool、不乘valid。"""
    if (not isinstance(heat, np.ndarray) or heat.ndim != 2
            or heat.dtype != np.float32 or not heat.size):
        raise ValueError('Continuous boundary must be nonempty native float32 HW')
    top, left, height, width = map(int, box)
    if (min(top, left) < 0 or min(height, width) < 1 or max(height, width) > crop_size
            or top + height > heat.shape[0] or left + width > heat.shape[1]):
        raise ValueError('Continuous boundary crop lies outside native source')
    value = np.rot90(heat[top:top + height, left:left + width], int(rotation) % 4)
    if flip:
        value = value[:, ::-1]
    value = np.array(value, copy=True, order='C')
    bottom, right = crop_size - value.shape[0], crop_size - value.shape[1]
    if bottom or right:
        mode = 'reflect' if min(value.shape) > 1 else 'edge'
        value = np.pad(value, ((0, bottom), (0, right)), mode=mode)
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('Continuous boundary crop must be finite in [0,1]')
    return np.ascontiguousarray(value)


class BoundaryDualDataset(BoundaryRGBDataset):
    """复用旧单通道数据，仅增加已封存、未二值化的N1连续通道。

    RGB只保留给旧过程图显示，不进入模型。连续缓存先检查总字节预算，
    逐源解压；所有大数组只读，clone_for_epoch只重建独立采样计划。
    """

    def __init__(self, rows_or_manifest, config=None, epoch=1, *, real_input_manifest=None):
        outer = {} if config is None else copy.deepcopy(config)
        cfg = outer.get('boundary_naf', outer)
        if cfg.get('input_mode') != INPUT_MODE or cfg.get('input_contract') != INPUT_CONTRACT:
            raise ValueError('Dual input requires boundary_continuous and explicit t8_continuous_v1')
        if (cfg.get('input_normalization') != NORMALIZATION
                or cfg.get('erasure', {}).get('enabled', False)
                or cfg.get('ridge_supervision', {}).get('enabled', False)
                or cfg.get('restored_rgb_manifest') is not None
                or cfg.get('input_perturbation') is not None):
            raise ValueError('Dual input only adds continuous clean N1 to unchanged whole-source T8')
        maximum = cfg.get('continuous_cache_max_bytes', DEFAULT_CONTINUOUS_CACHE_MAX_BYTES)
        if type(maximum) is not int or maximum < 1:
            raise ValueError('continuous_cache_max_bytes must be a positive integer')
        cfg['input_mode'] = 'boundary'
        super().__init__(rows_or_manifest, config=outer, epoch=epoch,
                         real_input_manifest=real_input_manifest)
        self.base_identity = copy.deepcopy(self.identity)
        self.input_mode = INPUT_MODE
        headers = [continuous_heat_header(row['heat_path'], self.cache[row['source']][0].shape)
                   for row in self.real_input_records]
        required = sum(header['bytes'] for header in headers)
        if required > maximum:
            raise ValueError(f'Continuous N1 cache needs {required} bytes, exceeds {maximum}')
        self.continuous_boundaries, self.continuous_input_records = {}, []
        for row, header in zip(self.real_input_records, headers):
            with np.load(row['heat_path'], allow_pickle=False) as payload:
                heat = payload['boundary']
            if (heat.dtype != np.float32 or list(heat.shape) != header['shape']
                    or not heat.flags.c_contiguous or not np.isfinite(heat).all()
                    or np.any((heat < 0) | (heat > 1))):
                raise ValueError('Invalid continuous N1 content: ' + row['source'])
            if not np.array_equal(heat >= .5, self.raw_input_boundaries[row['source']]):
                raise ValueError('Continuous N1 and raw thresholded T must be the same frozen map')
            heat.setflags(write=False)
            self.continuous_boundaries[row['source']] = heat
            self.continuous_input_records.append(dict(source=row['source'],
                heat_sha256=row['heat_sha256'], continuous_array_sha256=array_sha(heat),
                **header))
        self.identity.update(format=FORMAT, parent_data_format=self.base_identity['format'],
            input_mode=INPUT_MODE, input_contract=INPUT_CONTRACT,
            input_channels=['binary_T8', 'continuous_N1_boundary'],
            input_representation='same frozen native N1: whole-source T8 plus unnormalized float32 boundary; shared crop/D4/reflect',
            continuous_input_sources=self.continuous_input_records,
            continuous_input_sources_sha256=canonical_sha(self.continuous_input_records),
            continuous_input_range=[0, 1], continuous_input_quantization='none',
            continuous_input_normalization='none', RGB_model_input=False,
            rgb_decoded=True, rgb_role='display only; not a model input',
            dataset_cache_strategy='single shared read-only source cache; independent epoch plans',
            continuous_cache_max_bytes=maximum, continuous_cache_bytes=required)
        self._freeze_source_arrays()
        self.cache_receipt = dict(continuous_bytes=required, continuous_max_bytes=maximum,
            numpy_source_cache_bytes=self._source_cache_bytes(), sources=len(self.sources),
            all_source_arrays_read_only=True, fixed_epoch_view_shares_cache=True)

    def _source_array_groups(self):
        return [self.cache, self.input_skeletons, self.input_boundaries,
                self.raw_input_boundaries, self.rgb_images, self.continuous_boundaries]

    def _freeze_source_arrays(self):
        for group in self._source_array_groups():
            for value in group.values():
                arrays = value if isinstance(value, tuple) else (value,)
                for array in arrays:
                    array.setflags(write=False)

    def _source_cache_bytes(self):
        unique = {}
        for group in self._source_array_groups():
            for value in group.values():
                arrays = value if isinstance(value, tuple) else (value,)
                unique.update({id(array): array.nbytes for array in arrays})
        return int(sum(unique.values()))

    def clone_for_epoch(self, epoch):
        """复用只读缓存；epoch、plan和可编辑身份字典独立，避免监控再载全32源。"""
        clone = copy.copy(self)
        clone.identity = copy.deepcopy(self.identity)
        clone.base_identity = copy.deepcopy(self.base_identity)
        clone.cache_receipt = copy.deepcopy(self.cache_receipt)
        clone.set_epoch(epoch)
        return clone

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        spec = self.plan[int(index)]
        continuous = crop_continuous_boundary(self.continuous_boundaries[spec['source']],
            spec['box'], self.crop_size, spec['rotation'], spec['flip'])
        tensor = torch.from_numpy(continuous)[None]
        sample['continuous_boundary'] = tensor
        sample['input'] = torch.cat([sample['input_boundary'], tensor], dim=0)
        sample['meta'].update(mode='real_T8_continuous_N1', requested_mode=INPUT_MODE,
            input_contract=INPUT_CONTRACT, input_order=list(self.identity['input_channels']),
            continuous_quantization='none', continuous_normalization='none',
            continuous_same_heat_sha256=next(row['heat_sha256'] for row in self.real_input_records
                if row['source'] == spec['source']), rgb_role='display_only')
        return sample

    def verify_sources(self):
        super().verify_sources()
        for row in self.continuous_input_records:
            array = self.continuous_boundaries[row['source']]
            if (array.dtype != np.float32 or array.flags.writeable
                    or array_sha(array) != row['continuous_array_sha256']):
                raise RuntimeError('In-memory continuous N1 changed: ' + row['source'])
        return canonical_sha(self.identity)


def collate_boundary_dual(samples):
    if not samples or not all('continuous_boundary' in sample for sample in samples):
        raise ValueError('Dual collate requires continuous N1 on every sample')
    result = collate_boundary_rgb(samples)
    result['continuous_boundary'] = torch.stack([sample['continuous_boundary'] for sample in samples])
    return result
