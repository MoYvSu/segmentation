# -*- coding: utf-8 -*-
"""真实N1骨架→处理后GT的NAF配对；目标/采样复用原数据集。

输入仅由封存N1全图热图生成，先整图细化，再作相同原生裁块和D4。
GT只提供目标、有效域及诊断掩码，不用于修剪或合成输入。
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from skimage.morphology import skeletonize
import torch

from data.boundary_naf import (
    BoundaryNAFDataset, _crop_pair, canonical_sha, file_sha256,
)


FORMAT = 'boundary_naf_real_heatmaps_v1'
LOCAL_RESTORATION = {'mode': 'zero', 'model': 'restorer'}


def _sha_text(value, description):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(description + ' needs a lowercase SHA256')
    return value


def _expected_models(config):
    models = config.get('real_input_models')
    if not isinstance(models, dict):
        raise ValueError('real_input_models must explicitly bind backend/restorer SHA256')
    result = {}
    for name in ('backend', 'restorer'):
        value = models.get(name)
        if isinstance(value, dict):
            value = value.get('sha256')
        result[name] = _sha_text(value, 'Expected ' + name)
    return result


def _full_input_crop(skeleton, valid, spec, crop_size, crop_margin):
    # _crop_pair只修改返回的valid，不将GT/valid乘到输入上；padding仍reflect。
    image, _, geometry = _crop_pair(skeleton, valid, spec['box'], crop_size,
        spec['rotation'], spec['flip'], crop_margin)
    return image, geometry


class BoundaryNAFRealDataset(BoundaryNAFDataset):
    """保留原GT、原采样计划与D4，只将输入改为同图真实N1骨架。"""

    def __init__(self, rows_or_manifest, config=None, epoch=1, *, real_input_manifest=None):
        outer_config = {} if config is None else dict(config)
        cfg = outer_config.get('boundary_naf', outer_config)
        cfg = dict(cfg)
        data_cfg = dict(cfg.get('data', cfg))
        expected_models = _expected_models(cfg)
        threshold = cfg.get('real_input_threshold', .5)
        if type(threshold) not in (int, float) or float(threshold) != .5:
            raise ValueError('Real N1 input contract requires threshold 0.5')
        requested_manifest = real_input_manifest
        if requested_manifest is None:
            requested_manifest = cfg.get('real_input_manifest', data_cfg.get('real_input_manifest'))
        if not isinstance(requested_manifest, (str, Path)):
            raise ValueError('real_input_manifest must name a sealed N1 heatmap manifest')

        # 复用同一GT/valid、候选窗与局部随机流，保证旧控制无需重训。
        super().__init__(rows_or_manifest, config=cfg, epoch=epoch)
        self.target_identity = copy.deepcopy(self.identity)
        heat_manifest = Path(requested_manifest)
        if not heat_manifest.is_absolute():
            if 'paths' in outer_config:
                from utils.config import project_path
                heat_manifest = Path(project_path(outer_config, str(heat_manifest)))
            else:
                base = self.manifest_path.parent if self.manifest_path else Path(data_cfg.get('manifest_base', '.'))
                heat_manifest = base / heat_manifest
        self.real_input_manifest_path = heat_manifest.resolve()
        self.real_input_manifest_sha256 = file_sha256(self.real_input_manifest_path)
        document = json.loads(self.real_input_manifest_path.read_text(encoding='utf-8'))
        if (not isinstance(document, dict) or document.get('format') != FORMAT
                or document.get('complete') is not True
                or document.get('no_test_training') is not True):
            raise ValueError('A complete no-test N1 heatmap manifest is required')
        if document.get('local_restoration') != LOCAL_RESTORATION:
            raise ValueError('Real input must use N1 zero/restorer local restoration')
        declared_gt_manifest = document.get('processed_gt_manifest_sha256')
        if declared_gt_manifest is not None:
            _sha_text(declared_gt_manifest, 'Processed GT manifest binding')
            if self.manifest_sha256 is not None and declared_gt_manifest != self.manifest_sha256:
                raise ValueError('N1 heatmap processed-GT manifest SHA256 differs')
        model_receipt = document.get('models')
        if not isinstance(model_receipt, dict):
            raise ValueError('N1 heatmap manifest model identities are required')
        for name, expected in expected_models.items():
            value = model_receipt.get(name)
            actual = value.get('sha256') if isinstance(value, dict) else value
            if actual != expected:
                raise ValueError('N1 heatmap ' + name + ' model SHA256 differs')
        rows = document.get('rows', document.get('files'))
        if not isinstance(rows, list) or len(rows) != len(self.sources):
            raise ValueError('N1 heatmap source count differs from processed GT')
        names = [row.get('source') if isinstance(row, dict) else None for row in rows]
        if (any(not isinstance(name, str) or re.fullmatch(r'train_[0-9]{3,4}', name) is None for name in names)
                or len(set(names)) != len(names) or set(names) != set(self.sources)):
            raise ValueError('N1 heatmaps must match unique named training sources exactly')

        self.input_skeletons = {}
        self.real_input_records = []
        for row in sorted(rows, key=lambda row: row['source']):
            source = row['source']
            target_record = self.by_source[source]
            for key in ('image_sha256', 'gt_sha256', 'classes_sha256'):
                if row.get(key) != target_record[key]:
                    raise ValueError('N1 heatmap source binding differs: ' + source + '/' + key)
            path = Path(row.get('heat_path', ''))
            if not row.get('heat_path'):
                raise ValueError('N1 heat_path is required')
            if not path.is_absolute():
                path = self.real_input_manifest_path.parent / path
            path = path.resolve()
            expected_sha = _sha_text(row.get('heat_sha256'), source + ' heatmap')
            if file_sha256(path) != expected_sha:
                raise ValueError('N1 heatmap file SHA256 differs: ' + source)
            with np.load(path, allow_pickle=False) as arrays:
                if 'boundary' not in arrays.files:
                    raise ValueError('N1 heatmap needs a boundary array')
                boundary = arrays['boundary']
            shape = self.cache[source][0].shape
            if (boundary.dtype != np.float32 or boundary.shape != shape
                    or not np.isfinite(boundary).all() or np.any((boundary < 0) | (boundary > 1))):
                raise ValueError('N1 boundary must be finite native float32 HW probability: ' + source)
            if 'shape' in row and list(row['shape']) != list(shape):
                raise ValueError('N1 heatmap declared shape differs: ' + source)
            # 输入骨架在整张源图上求一次，不能让逐裁窗产生的端点进入输入。
            skeleton = np.ascontiguousarray(skeletonize(boundary >= .5), dtype=bool)
            skeleton_sha = hashlib.sha256(skeleton.tobytes()).hexdigest()
            self.input_skeletons[source] = skeleton
            self.real_input_records.append(dict(source=source, heat_path=str(path),
                heat_sha256=expected_sha, skeleton_sha256=skeleton_sha,
                image_sha256=row['image_sha256'], gt_sha256=row['gt_sha256'],
                classes_sha256=row['classes_sha256'], shape=list(shape),
                skeleton_pixels=int(skeleton.sum())))

        ignored = {key: self.target_identity[key] for key in (
            'identity_probability', 'gap_count_range', 'gap_length_native_range', 'gap_half_width_native_range')}
        self.identity.update(format='boundary_naf_real_data_v1',
            target_dataset_format=self.target_identity['format'],
            input_representation='N1 full-image threshold>=0.5 then skeletonize; native crop then identical D4',
            real_input_manifest_sha256=self.real_input_manifest_sha256,
            real_input_models=expected_models,
            real_input_sources=[{key: value for key, value in row.items() if key != 'heat_path'}
                                for row in self.real_input_records],
            local_restoration=copy.deepcopy(LOCAL_RESTORATION),
            model_predictions_used=True, model_predictions_as_target=False,
            artificial_corruption=False, identity_probability=None,
            gap_count_range=None, gap_length_native_range=None, gap_half_width_native_range=None,
            legacy_synthetic_settings_ignored_for_real_input=ignored,
            diagnostic_gap_definition='valid GT positive farther than 2 native pixels from input skeleton; not artificial erasure',
            policy='unchanged processed GT/valid and source-balanced native/D4 plan; no GT mask applied to input',
            limitation='Training sources seen; N1 prediction errors paired with recorded GT, not independent correctness.')
        if self.target_identity.get('target_source_kind') == 'model_teacher':
            self.identity.update(model_predictions_as_target=True, target_source_kind='model_teacher',
                policy='student N1 input paired with frozen teacher instances; ID0 ignored; no manual target provenance',
                limitation='Self-distillation on seen training sources; teacher errors are not independent GT.')
            if self.target_identity.get('teacher_boundary_policy') == 'explicit_separators_v1':
                self.identity.update(policy='student N1 input; teacher boundary includes certified A-0-B center separators; ID0 semantic/affinity endpoints ignored; other zero/frame/crop/padding ignored')

    def __getitem__(self, index):
        draw = int(index)
        spec = self.plan[draw]
        target, valid = self.cache[spec['source']]
        target, valid, geometry = _crop_pair(target, valid, spec['box'], self.crop_size,
            spec['rotation'], spec['flip'], self.crop_margin)
        if not valid.any():
            raise RuntimeError('Prechecked window lost all valid support')
        inputs, input_geometry = _full_input_crop(self.input_skeletons[spec['source']],
            self.cache[spec['source']][1], spec, self.crop_size, self.crop_margin)
        if geometry != input_geometry:
            raise RuntimeError('Real N1 input and GT crop geometry differs')
        if inputs.any():
            distance = cv2.distanceTransform((~inputs).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            gap = target & valid & (distance > 2.)
        else:
            gap = target & valid
        identity = bool(np.array_equal(inputs, target))
        tensors = {key: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))[None]
                   for key, value in dict(input=inputs, target=target, valid=valid,
                       gap_mask=gap, fully_known_mask=valid).items()}
        metadata = dict(source=spec['source'], epoch=self.epoch, draw=draw,
            draw_seed=spec['seed'], source_draw=spec['source_draw'], augmentation=spec['augmentation'],
            geometry=geometry, mode='real_input', requested_mode='real_input',
            gaps=[], artificial_corruption=False, no_artificial_corruption=True,
            legacy_identity_and_gap_settings_ignored_for_real_input=True,
            valid_pixels=int(valid.sum()), boundary_pixels=int((target & valid).sum()),
            gap_pixels=int(gap.sum()), gap_kind='natural_GT_positive_farther_than_2px_from_input',
            input_skeleton_pixels=int(inputs.sum()), padding_supervised=False, resize_scale=1.0)
        return dict(**tensors, identity=torch.tensor(identity), source=spec['source'],
                    box=list(spec['box']), draw=draw, meta=metadata)

    def verify_sources(self):
        super().verify_sources()
        if file_sha256(self.real_input_manifest_path) != self.real_input_manifest_sha256:
            raise RuntimeError('Real N1 heatmap manifest changed')
        for row in self.real_input_records:
            if file_sha256(row['heat_path']) != row['heat_sha256']:
                raise RuntimeError('Real N1 heatmap changed: ' + row['source'])
            actual = hashlib.sha256(self.input_skeletons[row['source']].tobytes()).hexdigest()
            if actual != row['skeleton_sha256']:
                raise RuntimeError('Real N1 in-memory input skeleton changed: ' + row['source'])
        return canonical_sha(self.identity)
