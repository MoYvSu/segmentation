# -*- coding: utf-8 -*-
"""同坐标原RGB＋真实N1未细化二值T，对照处理后新GT。

旧真实输入数据集只负责验证来源、GT及采样合同；本实验重新保留未细化T，
不使用其骨架作为模型输入。全图缓存后按原生坐标裁块，RGB/T/GT同步D4，
未知/填充由valid控制；默认不改输入，显式擦断只损伤已有支持短段，不改原文件。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
import torch

from data.boundary_naf import _crop_pair, canonical_sha, file_sha256
from data.boundary_naf_real import BoundaryNAFRealDataset, _full_input_crop
from data.boundary_restore import native_crop
from models.boundary_rgb import INPUT_MODES
from utils.boundary_input import input_normalization_settings, normalize_boundary_input
from utils.boundary_erasure import (
    array_sha as erasure_array_sha, draw_seed as erasure_draw_seed, erasure_settings,
    make_erasure_mask, prepare_erasure_source, transform_erasure_mask,
)
from utils.image_io import read_image


RESTORED_RGB_FORMAT = 'boundary_rgb_restored_v1'
RESTORED_RGB_COORDINATES = 'native1024_overlap025_letterbox_reflect_blend_before_crop_D4'


def _array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _restored_sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(label + ' must be a lowercase SHA256')
    return value


def load_restored_rgb_manifest(path):
    """加载独立训练/monitor清单；这里只校清单，不读取另一域的图像。"""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(manifest, dict) or manifest.get('format') != RESTORED_RGB_FORMAT
            or manifest.get('complete') is not True
            or manifest.get('coordinate_contract') != RESTORED_RGB_COORDINATES):
        raise ValueError('A complete native restored RGB manifest is required')
    restoration = manifest.get('restoration')
    fixed = dict(noise_mode='zero', steps=16, output='first', native_size=1024,
                 overlap=.25, cache_dtype='float32', quantization='none')
    if not isinstance(restoration, dict) or any(restoration.get(k) != v for k, v in fixed.items()):
        raise ValueError('Restored RGB requires fixed zero-noise first-output float32 N1 restoration')
    for key in ('model_file_sha256', 'model_state_sha256'):
        _restored_sha(restoration.get(key), 'Restored RGB ' + key)
    rows = manifest.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Restored RGB manifest needs nonempty source rows')
    records, domains = {}, set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid restored RGB row')
        name, domain = row.get('source'), row.get('domain')
        expression = r'train_[0-9]{3,4}' if domain == 'train' else r'test_[0-9]{3}'
        if (domain not in ('train', 'monitor_test') or not isinstance(name, str)
                or re.fullmatch(expression, name) is None or name in records):
            raise ValueError('Restored RGB rows must have unique named sources and explicit domains')
        shape = row.get('shape')
        if (not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
                or any(type(v) is not int or v < 1 for v in shape)):
            raise ValueError('Restored RGB row shape must be native HWC RGB')
        for key in ('image_sha256', 'original_rgb_array_sha256', 'restored_sha256', 'restored_array_sha256'):
            _restored_sha(row.get(key), name + '/' + key)
        stored_path = row.get('restored_path')
        if not isinstance(stored_path, str) or not stored_path:
            raise ValueError('Restored RGB row needs restored_path')
        restored_path = Path(stored_path)
        if not restored_path.is_absolute():
            restored_path = path.parent / restored_path
        restored_path = restored_path.resolve()
        if restored_path.suffix.lower() != '.npy':
            raise ValueError('Restored RGB cache must be lossless float32 NPY')
        records[name] = dict(row, restored_path=str(restored_path))
        domains.add(domain)
    if len(domains) != 1:
        raise ValueError('Training and monitor restored RGB must use separate manifests')
    return dict(path=str(path), sha256=file_sha256(path), manifest=manifest,
                records=records, domain=next(iter(domains)))


def restored_rgb_for_source(cache, source, raw_rgb, *, image_sha256=None):
    """核对同图原像素后读取连续RGB，不量化、不缩放、不读GT。"""
    if source not in cache['records']:
        raise ValueError('Restored RGB source missing: ' + source)
    row = cache['records'][source]
    if (not isinstance(raw_rgb, np.ndarray) or raw_rgb.dtype != np.uint8
            or raw_rgb.ndim != 3 or list(raw_rgb.shape) != row['shape']):
        raise ValueError('Original RGB must match restored RGB native uint8 HWC source: ' + source)
    if _array_sha(raw_rgb) != row['original_rgb_array_sha256']:
        raise ValueError('Original RGB array SHA256 differs: ' + source)
    if image_sha256 is not None and image_sha256 != row['image_sha256']:
        raise ValueError('Original RGB image file SHA256 differs: ' + source)
    path = Path(row['restored_path'])
    if file_sha256(path) != row['restored_sha256']:
        raise ValueError('Restored RGB file SHA256 differs: ' + source)
    restored = np.load(path, allow_pickle=False)
    if (not isinstance(restored, np.ndarray) or restored.dtype != np.float32
            or list(restored.shape) != row['shape'] or not np.isfinite(restored).all()
            or np.any((restored < 0) | (restored > 1))):
        raise ValueError('Restored RGB must be finite native float32 HWC in [0,1]: ' + source)
    restored = np.ascontiguousarray(restored)
    if _array_sha(restored) != row['restored_array_sha256']:
        raise ValueError('Restored RGB array SHA256 differs: ' + source)
    receipt = dict(source=source, domain=row['domain'], shape=list(restored.shape),
        path=str(path), file_sha256=row['restored_sha256'], array_sha256=row['restored_array_sha256'],
        image_sha256=row['image_sha256'], original_rgb_array_sha256=row['original_rgb_array_sha256'],
        manifest_path=cache['path'], manifest_sha256=cache['sha256'],
        coordinate_contract=cache['manifest']['coordinate_contract'],
        restoration=dict(cache['manifest']['restoration']), dtype='float32', range=[0, 1], quantization='none')
    return restored, receipt


def crop_native_rgb(rgb, box, crop_size, rotation=0, flip=False):
    """与GT相同的原生crop/D4/右下reflect，不改变纹理尺度。"""
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB source must be an HWC array with three channels')
    top, left, height, width = map(int, box)
    if (min(top, left) < 0 or min(height, width) < 1 or max(height, width) > crop_size
            or top + height > rgb.shape[0] or left + width > rgb.shape[1]):
        raise ValueError('Native RGB box lies outside source or exceeds crop_size')
    view = np.rot90(rgb[top:top + height, left:left + width], int(rotation) % 4)
    if flip:
        view = view[:, ::-1]
    result = np.array(view, copy=True, order='C')
    bottom, right = crop_size - result.shape[0], crop_size - result.shape[1]
    if bottom or right:
        mode = 'reflect' if min(result.shape[:2]) > 1 else 'edge'
        result = np.pad(result, ((0, bottom), (0, right), (0, 0)), mode=mode)
    return np.ascontiguousarray(result)


class BoundaryRGBDataset(BoundaryNAFRealDataset):
    """默认保留真实T；显式enabled时只在线擦已有支持短段，监督不变。"""

    def __init__(self, rows_or_manifest, config=None, epoch=1, *, real_input_manifest=None):
        outer = {} if config is None else dict(config)
        cfg = outer.get('boundary_naf', outer)
        self.input_mode = cfg.get('input_mode', 'rgb_boundary')
        if self.input_mode not in INPUT_MODES:
            raise ValueError('input_mode must be boundary or rgb_boundary')
        self.input_normalization = input_normalization_settings(cfg.get('input_normalization'))
        self.normalization_enabled = self.input_normalization['mode'] != 'none'
        self.erasure_settings = erasure_settings(cfg.get('erasure'))
        self.erasure_enabled = self.erasure_settings['enabled']
        ridge = cfg.get('ridge_supervision')
        if ridge is not None and (not isinstance(ridge, dict)
                or type(ridge.get('enabled', False)) is not bool):
            raise ValueError('ridge_supervision must be a mapping with boolean enabled')
        self.ridge_supervision_enabled = ridge is not None and ridge.get('enabled', False)
        if self.ridge_supervision_enabled and (not self.erasure_enabled
                or (self.input_mode == 'rgb_boundary'
                    and cfg.get('restored_rgb_manifest') is None)
                or not isinstance(cfg.get('model', {}), dict)
                or cfg.get('model', {}).get('upsampling') != 'project_bilinear_v1'):
            raise ValueError('Enabled ridge supervision requires erasure, project_bilinear_v1 and explicit restored RGB for RGB mode')
        requested_restored = cfg.get('restored_rgb_manifest')
        if requested_restored is not None and (not isinstance(requested_restored, (str, Path))
                                              or not str(requested_restored)):
            raise ValueError('restored_rgb_manifest must explicitly name a manifest')
        if requested_restored is not None and self.input_mode != 'rgb_boundary':
            raise ValueError('Restored RGB conditioning requires rgb_boundary input mode')
        if self.erasure_enabled and (self.input_normalization != {
                'mode': 'skeleton_dilate', 'kernel_size': 8} or
                (self.input_mode == 'rgb_boundary' and requested_restored is None)):
            raise ValueError('Enabled erasure requires whole-source skeleton_dilate kernel8 and explicit restored RGB for RGB mode')
        super().__init__(rows_or_manifest, config=outer, epoch=epoch,
                         real_input_manifest=real_input_manifest)
        teacher_target = self.target_identity.get('target_source_kind') == 'model_teacher'
        if teacher_target and self.erasure_enabled:
            raise ValueError('Teacher supervision has no original_covered provenance for the legacy erasure qualification')
        self.input_boundaries, self.rgb_images = {}, {}
        self.model_rgb_images, self.restored_rgb_records = {}, []
        self.restored_rgb_cache = None
        if requested_restored is not None:
            restored_path = Path(requested_restored)
            if not restored_path.is_absolute():
                if 'paths' in outer:
                    from utils.config import project_path
                    restored_path = Path(project_path(outer, str(restored_path)))
                else:
                    base = self.manifest_path.parent if self.manifest_path else Path(cfg.get('data', {}).get('manifest_base', '.'))
                    restored_path = base / restored_path
            self.restored_rgb_cache = load_restored_rgb_manifest(restored_path)
            if (self.restored_rgb_cache['domain'] != 'train'
                    or set(self.restored_rgb_cache['records']) != set(self.sources)):
                raise ValueError('Training restored RGB manifest must match training sources exactly')
            if self.restored_rgb_cache['manifest']['restoration']['model_file_sha256'] != self.identity['real_input_models']['restorer']:
                raise ValueError('Restored RGB and real N1 input must use the same restorer file')
        self.raw_input_boundaries = {}
        self.rgb_input_records = []
        self.erasure_sources = {}
        for row in self.real_input_records:
            source = row['source']
            with np.load(row['heat_path'], allow_pickle=False) as arrays:
                boundary = arrays['boundary']
            binary, normalization = normalize_boundary_input(boundary, settings=self.input_normalization)
            raw_binary = (np.ascontiguousarray(boundary >= .5, dtype=bool)
                          if self.normalization_enabled else binary)
            bgr = read_image(self.by_source[source]['image'], cv2.IMREAD_COLOR)
            if bgr.dtype != np.uint8 or bgr.shape != (*binary.shape, 3):
                raise ValueError('Original RGB and native N1/GT shapes differ: ' + source)
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            self.input_boundaries[source] = binary
            self.raw_input_boundaries[source] = raw_binary
            self.rgb_images[source] = rgb
            if self.restored_rgb_cache is not None:
                conditioned, restored_receipt = restored_rgb_for_source(self.restored_rgb_cache, source, rgb,
                    image_sha256=self.by_source[source]['image_sha256'])
                self.model_rgb_images[source] = conditioned
                self.restored_rgb_records.append(restored_receipt)
            self.rgb_input_records.append(dict(
                source=source, binary_sha256=_array_sha(binary), rgb_array_sha256=_array_sha(rgb),
                binary_pixels=int(binary.sum()), shape=list(binary.shape),
                heat_sha256=row['heat_sha256'], image_sha256=row['image_sha256'],
                gt_sha256=row['gt_sha256'], classes_sha256=row['classes_sha256']))
            if self.normalization_enabled:
                self.rgb_input_records[-1].update(raw_binary_sha256=_array_sha(raw_binary),
                                                 normalization=normalization)
            if self.erasure_enabled:
                with np.load(self.by_source[source]['gt'], allow_pickle=False) as payload:
                    gt = payload['instance_map'].copy()
                    original = payload['original_covered'].copy()
                target, valid = self.cache[source]
                self.erasure_sources[source] = prepare_erasure_source(gt, original, target, valid,
                                                                      self.input_skeletons[source])
        self.identity.update(
            format='boundary_rgb_data_v1', input_mode=self.input_mode,
            input_representation='original native RGB plus N1 threshold>=0.5 without thinning; identical native crop and D4',
            input_channels=['R', 'G', 'B', 'binary_boundary'] if self.input_mode == 'rgb_boundary' else ['binary_boundary'],
            real_input_sources=self.rgb_input_records,
            rgb_decoded=True, rgb_normalization='uint8 RGB / 255; no image-domain correction',
            input_threshold=.5, input_skeletonized=False,
            diagnostic_gap_definition='valid GT positive with no unthinned T in the radius-2 L-infinity square',
            policy='processed GT known/filled supervised; unknown/frame/crop rim/padding ignored; GT never masks RGB or T',
            limitation='Same seen training sources; not independent accuracy validation. Parent skeleton cache is provenance-only, never a model input.')
        if self.normalization_enabled:
            self.identity.update(format='boundary_rgb_data_v2', input_normalization=dict(self.input_normalization),
                input_representation='native N1 threshold>=0.5; whole-image skeleton then controlled ellipse dilation; crop and D4 after normalization',
                input_skeletonized=True,
                diagnostic_gap_definition='valid GT positive with no normalized T in the radius-2 L-infinity square',
                limitation='Seen training sources; normalization can change line position/connectivity; no accuracy or topology preservation claim.')
        if teacher_target:
            self.identity.update(target_source_kind='model_teacher', model_predictions_as_target=True,
                policy='teacher known-positive targets; ID0/unknown/frame/crop/padding ignored; GT never masks student N1 input',
                limitation='Frozen C teacher self-distillation; no independent label accuracy claim.')
            if self.target_identity.get('teacher_boundary_policy') == 'explicit_separators_v1':
                self.identity.update(policy='teacher boundary uses certified A-0-B center separators; ID0 instance/semantic/affinity endpoints ignored; other zero/frame/crop/padding ignored; GT never masks student N1')
        if self.erasure_enabled:
            heat_hashes = {row['source']: row['heat_sha256'] for row in self.real_input_records}
            records = [dict(source=source, gt_file_sha256=self.by_source[source]['gt_sha256'],
                heat_file_sha256=heat_hashes[source],
                **self.erasure_sources[source]['summary']) for source in self.sources]
            self.identity.update(format='boundary_rgb_data_v3', artificial_corruption=True,
                erasure=dict(self.erasure_settings), erasure_sources=records,
                erasure_sources_sha256=canonical_sha(records),
                erasure_mask_definition='requested rectangle in native coordinates; only input T8 erased; zero mask padding',
                erasure_rng='independent SHA256(version, seed, epoch, draw, source) numpy Generator; source/D4 plan unchanged',
                policy='unchanged processed GT/valid/positive-negative; only existing supported nonjunction input T8 segments erased; unknown/frame/crop rim/padding excluded')
        if self.restored_rgb_cache is not None:
            self.identity.update(format='boundary_rgb_data_v4',
                rgb_source='frozen_restored_rgb_manifest',
                rgb_normalization='lossless native float32 restored RGB in [0,1]; no quantization',
                restored_rgb_manifest_sha256=self.restored_rgb_cache['sha256'],
                restored_rgb_coordinate_contract=self.restored_rgb_cache['manifest']['coordinate_contract'],
                restored_rgb_restoration=dict(self.restored_rgb_cache['manifest']['restoration']),
                restored_rgb_sources=self.restored_rgb_records,
                restored_rgb_sources_sha256=canonical_sha(self.restored_rgb_records),
                raw_RGB_retained_for_display=True,
                input_representation='frozen native restored RGB plus controlled whole-source N1 T8; same native crop/D4; erasure changes T only')

    def __getitem__(self, index):
        draw = int(index)
        spec = self.plan[draw]
        source = spec['source']
        original_target, original_valid = self.cache[source]
        target, valid, geometry = _crop_pair(original_target, original_valid, spec['box'],
            self.crop_size, spec['rotation'], spec['flip'], self.crop_margin)
        boundary, input_geometry = _full_input_crop(self.input_boundaries[source], original_valid,
            spec, self.crop_size, self.crop_margin)
        raw_boundary, raw_geometry = _full_input_crop(self.raw_input_boundaries[source], original_valid,
            spec, self.crop_size, self.crop_margin)
        if geometry != input_geometry:
            raise RuntimeError('RGB experiment input and GT geometry differ')
        if geometry != raw_geometry:
            raise RuntimeError('Raw and normalized boundary crop geometry differ')
        if not valid.any():
            raise RuntimeError('Prechecked window lost all valid support')
        raw_rgb = crop_native_rgb(self.rgb_images[source], spec['box'], self.crop_size,
                                  spec['rotation'], spec['flip'])
        rgb = (crop_native_rgb(self.model_rgb_images[source], spec['box'], self.crop_size,
                              spec['rotation'], spec['flip']) if self.restored_rgb_cache is not None else raw_rgb)
        # 自然缺口沿用干净T8的旧共同分母，不能把人工删线混入原gap统计。
        clean_boundary = boundary
        near_input = cv2.dilate(boundary.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        gap = target & valid & ~near_input
        near_raw = cv2.dilate(raw_boundary.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        raw_gap = target & valid & ~near_raw
        erasure_metadata, extra = None, {}
        ridge_instances = None
        if self.ridge_supervision_enabled:
            # ID只供额外形状项筛选可信界面，不进入模型，也不改变原GT/valid/采样。
            prepared_source = self.erasure_sources.get(source)
            if not isinstance(prepared_source, dict) or 'gt' not in prepared_source:
                raise ValueError('Ridge supervision requires retained processed instance GT: ' + source)
            ids, _ = native_crop(prepared_source['gt'], spec['box'], self.crop_size,
                                 spec['rotation'], spec['flip'])
            # native_crop按同一box/D4裁块，但ID补零而不是把reflect内容当作标注。
            ridge_instances = torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int64))[None]
        if self.erasure_enabled:
            native_mask, erasure_metadata = make_erasure_mask(self.erasure_sources[source],
                self.input_boundaries[source], box=spec['box'], crop_margin=self.crop_margin,
                settings=self.erasure_settings,
                seed=erasure_draw_seed(self.seed, self.epoch, draw, source))
            mask = transform_erasure_mask(native_mask, self.crop_size, spec['rotation'], spec['flip'])
            boundary = clean_boundary.copy()
            boundary[mask] = False
            near_damaged = cv2.dilate(boundary.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            artificial_gap = target & valid & near_input & ~near_damaged
            erasure_metadata['artificial_gap_gt_points'] = int(artificial_gap.sum())
            extra = dict(clean_input_boundary=clean_boundary, erasure_mask=mask,
                         artificial_gap_mask=artificial_gap)
        tensors = {key: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))[None]
                   for key, value in dict(target=target, valid=valid, input_boundary=boundary,
                                         gap_mask=gap, fully_known_mask=valid, **extra).items()}
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32))
        if self.restored_rgb_cache is None:
            rgb_tensor = rgb_tensor / 255.
        image_tensor = (torch.from_numpy(np.ascontiguousarray(raw_rgb.transpose(2, 0, 1), dtype=np.float32)) / 255.
                        if self.restored_rgb_cache is not None else rgb_tensor)
        model_input = (torch.cat([rgb_tensor, tensors['input_boundary']], dim=0)
                       if self.input_mode == 'rgb_boundary' else tensors['input_boundary'])
        metadata = dict(source=source, epoch=self.epoch, draw=draw, draw_seed=spec['seed'],
            source_draw=spec['source_draw'], augmentation=spec['augmentation'], geometry=geometry,
            mode='real_rgb_boundary' if self.input_mode == 'rgb_boundary' else 'real_boundary',
            requested_mode=self.input_mode, gaps=[], artificial_corruption=False,
            no_artificial_corruption=True, valid_pixels=int(valid.sum()),
            boundary_pixels=int((target & valid).sum()), gap_pixels=int(gap.sum()),
            gap_kind='natural_GT_positive_without_T_in_radius2_L_infinity_square',
            input_boundary_pixels=int(boundary.sum()), input_skeletonized=False,
            padding_supervised=False, resize_scale=1.0, rgb_order='RGB',
            input_order=self.identity['input_channels'])
        if self.normalization_enabled:
            metadata.update(input_normalization=dict(self.input_normalization), input_skeletonized=True,
                raw_input_boundary_pixels=int(raw_boundary.sum()),
                input_normalization_scope='whole_native_source_before_crop_and_D4',
                input_normalization_anchor_xy=[self.input_normalization['kernel_size'] // 2] * 2,
                input_normalization_polarity='high_is_boundary',
                gap_kind='natural_GT_positive_without_normalized_T_in_radius2_L_infinity_square')
        if self.erasure_enabled:
            metadata.update(erasure=erasure_metadata,
                artificial_corruption=erasure_metadata['applied'],
                no_artificial_corruption=not erasure_metadata['applied'])
        if self.restored_rgb_cache is not None:
            metadata.update(rgb_source='frozen_restored_rgb_manifest',
                restored_rgb_manifest_sha256=self.restored_rgb_cache['sha256'],
                restored_rgb_array_sha256=self.restored_rgb_cache['records'][source]['restored_array_sha256'],
                rgb_quantization='none', image_role='original_RGB_display_only')
        result = dict(**tensors, input=model_input, rgb=rgb_tensor, image=image_tensor,
                    raw_input_boundary=torch.from_numpy(np.ascontiguousarray(raw_boundary, dtype=np.float32))[None],
                    raw_gap_mask=torch.from_numpy(np.ascontiguousarray(raw_gap, dtype=np.float32))[None],
                    identity=torch.tensor(bool(np.array_equal(boundary, target))),
                    source=source, box=list(spec['box']), draw=draw, meta=metadata)
        if ridge_instances is not None:
            result['ridge_instances'] = ridge_instances
        return result

    def verify_sources(self):
        super().verify_sources()
        if (self.restored_rgb_cache is not None and
                file_sha256(self.restored_rgb_cache['path']) != self.restored_rgb_cache['sha256']):
            raise RuntimeError('Restored RGB manifest changed')
        for row in self.rgb_input_records:
            source = row['source']
            if _array_sha(self.input_boundaries[source]) != row['binary_sha256']:
                raise RuntimeError('In-memory unthinned T changed: ' + source)
            if self.normalization_enabled and _array_sha(self.raw_input_boundaries[source]) != row['raw_binary_sha256']:
                raise RuntimeError('In-memory raw T changed: ' + source)
            if _array_sha(self.rgb_images[source]) != row['rgb_array_sha256']:
                raise RuntimeError('In-memory RGB changed: ' + source)
            if file_sha256(self.by_source[source]['image']) != row['image_sha256']:
                raise RuntimeError('RGB source changed: ' + source)
            if self.restored_rgb_cache is not None:
                restored = self.restored_rgb_cache['records'][source]
                if _array_sha(self.model_rgb_images[source]) != restored['restored_array_sha256']:
                    raise RuntimeError('In-memory restored RGB changed: ' + source)
                if file_sha256(restored['restored_path']) != restored['restored_sha256']:
                    raise RuntimeError('Restored RGB file changed: ' + source)
            if self.erasure_enabled:
                prepared = self.erasure_sources[source]
                for field, identity in (('gt', 'gt_sha256'), ('original', 'original_covered_sha256'),
                        ('skeleton', 'gt_skeleton_sha256'), ('input_skeleton', 'input_skeleton_sha256'),
                        ('candidates', 'candidates_sha256')):
                    if erasure_array_sha(prepared[field]) != prepared['summary'][identity]:
                        raise RuntimeError('Erasure in-memory qualification changed: ' + source + '/' + field)
        return canonical_sha(self.identity)


def collate_boundary_rgb(samples):
    keys = ('input', 'input_boundary', 'rgb', 'image', 'target', 'valid',
            'gap_mask', 'fully_known_mask', 'identity')
    if all('raw_input_boundary' in sample for sample in samples):
        keys = (*keys, 'raw_input_boundary')
    if all('raw_gap_mask' in sample for sample in samples):
        keys = (*keys, 'raw_gap_mask')
    for key in ('clean_input_boundary', 'erasure_mask', 'artificial_gap_mask', 'ridge_instances'):
        if any(key in sample for sample in samples):
            if not all(key in sample for sample in samples):
                raise ValueError('Cannot collate mixed enabled/disabled erasure samples')
            keys = (*keys, key)
    return {**{key: torch.stack([sample[key] for sample in samples]) for key in keys},
            'source': [sample['source'] for sample in samples],
            'box': [sample['box'] for sample in samples],
            'draw': [sample['draw'] for sample in samples],
            'meta': [sample['meta'] for sample in samples]}
