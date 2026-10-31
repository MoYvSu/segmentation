# -*- coding: utf-8 -*-
"""冻结关系头的原生分块评分及一次互斥合并，不读取GT或重跑后端。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re

import numpy as np
import torch

from utils.boundary_relation_merge import merge_relation_pairs
from utils.boundary_relation_supervision import (
    adjacent_region_pairs, native_relation_patch, representative_anchor,
)


PARAMETER_COUNT = 211865


def _sha_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _array_record(array):
    value = np.ascontiguousarray(array)
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())


def _inputs_record(rgb, N1, NAF, instances, classes):
    return dict(rgb=_array_record(rgb), N1=_array_record(N1), NAF=_array_record(NAF),
                instances=_array_record(instances), classes_sha256=hashlib.sha256(
                    json.dumps(classes, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest())


def _model_module():
    # 不触发 models.__init__ 中的SAM2/旧FPN导入。
    path = Path(__file__).resolve().parents[1]/'models/boundary_relation.py'
    spec = importlib.util.spec_from_file_location('_boundary_relation_postprocess_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _preserve_runtime():
    """新头不改旧路径的RNG/TF32/卷积开关，包含异常退出。"""
    python, numpy = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state().clone()
    initialized = torch.cuda.is_initialized()
    cuda = torch.cuda.get_rng_state_all() if initialized else None
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield dict(CUDA_previously_initialized=initialized)
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = flags


def _tensor_patch(raw, mode):
    """与 BoundaryRelationDataset.tensors(augment=False)逐运算一致。"""
    rgb = torch.from_numpy(raw['rgb']).float().div_(127.5).sub_(1.)
    if mode == 'maps':
        rgb.zero_()
    maps = torch.from_numpy(raw['maps']).float()  # helper已先量化为float16
    masks = torch.from_numpy(raw['masks']).float()
    return torch.cat((rgb, maps), dim=0).contiguous(), masks.contiguous()


def _seam_components(flat, shape):
    """稀疏8连通检查，不为每对重复分配全图canvas。仅作读出范围审计。"""
    h, w = shape
    remaining = set(map(int, flat))
    count = 0
    while remaining:
        count += 1
        pending = [remaining.pop()]
        while pending:
            current = pending.pop()
            y, x = divmod(current, w)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dx or dy) and 0 <= y+dy < h and 0 <= x+dx < w:
                        neighbor = current+dy*w+dx
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            pending.append(neighbor)
    return count


class BoundaryRelationPostprocessor:
    """checkpoint惰性加载；threshold=0/无相邻对可无需权重精确回放。

    parameter_count是部署参数声明，不因某张图走零操作而减小。
    同一pair只取训练约定的全共享缝中位锚点，不使用GT或多窗投票。
    """

    parameter_count = PARAMETER_COUNT

    def __init__(self, checkpoint=None, *, threshold=.1, device='cuda', batch_size=64,
                 expected_model_sha=None):
        if (isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1
                or type(batch_size) is not int or not 1 <= batch_size <= 64):
            raise ValueError('threshold须在[0,1]，batch_size须为1..64')
        if expected_model_sha is not None and (not isinstance(expected_model_sha, str) or
                re.fullmatch('[0-9a-f]{64}', expected_model_sha) is None):
            raise ValueError('expected_model_sha须为小写SHA256')
        self.checkpoint = None if checkpoint is None else Path(checkpoint).expanduser().resolve()
        self.threshold, self.batch_size = float(threshold), batch_size
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('本适配器仅支持cpu/cuda')
        self.expected_model_sha = expected_model_sha
        self._network = self._module = None
        self._loaded = dict(mode=None, model_sha256=None, checkpoint_sha256=None,
                            global_step=None, loaded_parameter_count=0)

    @property
    def mode(self):
        return self._loaded['mode']

    def provenance(self):
        return dict(checkpoint=None if self.checkpoint is None else str(self.checkpoint),
                    expected_model_sha256=self.expected_model_sha, parameter_count=self.parameter_count,
                    device=str(self.device), batch_size=self.batch_size, threshold=self.threshold,
                    **self._loaded)

    def _load(self):
        if self._network is not None:
            self.verify_unchanged()
            return
        if self.checkpoint is None or not self.checkpoint.is_file():
            raise ValueError('非零评分且存在相邻对时，必须提供1000步关系权重')
        file_before = _sha_file(self.checkpoint)
        saved = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        identity = saved.get('identity')
        if (type(saved.get('global_step')) is not int or saved['global_step'] != 1000
                or saved.get('smoke_only', False) is not False or not isinstance(identity, dict)
                or identity.get('mode') not in ('maps', 'rgb')
                or identity.get('no_GT_features') is not True
                or identity.get('no_C_weights_loaded') is not True):
            raise ValueError('必须是带合法来源/模式的1000步正式关系checkpoint，不能用e5/smoke')
        model_sha = saved.get('model_sha256')
        if self.expected_model_sha is not None and model_sha != self.expected_model_sha:
            raise ValueError('关系模型SHA与显式门槛不符')
        module = _model_module()
        network = module.load_boundary_relation(self.checkpoint, device=self.device)
        parameters = sum(p.numel() for p in network.parameters())
        if parameters != self.parameter_count or any(p.dtype != torch.float32 for p in network.parameters()):
            raise ValueError('关系头参数量/FP32合同不符')
        if _sha_file(self.checkpoint) != file_before:
            raise RuntimeError('关系权重在加载期间改变')
        self._module, self._network = module, network
        self._loaded = dict(mode=identity['mode'], model_sha256=model_sha,
                            checkpoint_sha256=file_before, global_step=1000,
                            loaded_parameter_count=parameters)
        self.verify_unchanged()

    def verify_unchanged(self):
        if self._network is not None and (any(m.training for m in self._network.modules()) or
                any(p.requires_grad for p in self._network.parameters()) or
                self._module.state_sha256(self._network) != self._loaded['model_sha256']):
            raise RuntimeError('冻结关系模型的权重/模式发生改变')
        return True

    @staticmethod
    def _validate(rgb, N1, NAF, instances):
        if (not isinstance(instances, np.ndarray) or instances.dtype != np.uint16 or
                instances.ndim != 2 or not instances.size or min(instances.shape) < 1):
            raise ValueError('instances须为非空原生单通道uint16')
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (*instances.shape, 3):
            raise ValueError('RGB须为与实例同原生尺寸的uint8 RGB，不是恢复RGB')
        for name, array in (('N1', N1), ('NAF', NAF)):
            if (not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.shape != instances.shape
                    or not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1)):
                raise ValueError(name + '须为同原生尺寸的float32 [0,1]概率')

    def process(self, rgb, N1, NAF, instances, classes):
        self._validate(rgb, N1, NAF, instances)
        # 复用合并器的完整类别验证，避免另一套ID/语义规范；该步不做邻接或合并。
        baseline, original_classes, _ = merge_relation_pairs(instances, classes, [], threshold=0)
        before = _inputs_record(rgb, N1, NAF, instances, classes)
        scored, batches, pair_audit = [], [], None
        no_op = None
        with _preserve_runtime() as runtime:
            try:
                self.verify_unchanged()
                if self.threshold == 0:
                    no_op = 'threshold_zero'
                    new_ids, new_classes, merge_audit = merge_relation_pairs(baseline, original_classes, [], threshold=0)
                else:
                    pairs, pair_audit = adjacent_region_pairs(baseline)
                    if not pairs:
                        no_op = 'no_adjacent_pairs'
                    else:
                        self._load()
                        areas = np.bincount(baseline.ravel(), minlength=65536)
                        for start in range(0, len(pairs), self.batch_size):
                            selected = pairs[start:start+self.batch_size]
                            patches, values, masks, anchors = [], [], [], []
                            for pair in selected:
                                anchor = representative_anchor(pair, baseline.shape[1])
                                raw = native_relation_patch(rgb, N1, NAF, baseline, pair, anchor, size=256)
                                value, mask = _tensor_patch(raw, self.mode)
                                patches.append(raw); values.append(value); masks.append(mask); anchors.append(anchor)
                            x = torch.stack(values).to(self.device, dtype=torch.float32)
                            mm = torch.stack(masks).to(self.device, dtype=torch.float32)
                            with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
                                self._network.validate_inputs(x, mm, check_values=True)
                                probability = self._network(x, mm).sigmoid().cpu().tolist()
                            batches.append(len(selected))
                            for pair, raw, anchor, score in zip(selected, patches, anchors, probability):
                                if not np.isfinite(score) or not 0 <= score <= 1:
                                    raise RuntimeError('关系模型产生非法keep评分')
                                xa, ya, xb, yb = raw['metadata']['native_valid_box_xyxy']
                                yy, xx = np.divmod(pair['seam_flat'], baseline.shape[1])
                                visible = int(((xx >= xa) & (xx < xb) & (yy >= ya) & (yy < yb)).sum())
                                a, b = map(int, pair['pair_ids'])
                                scored.append(dict(pair_ids=[a,b], keep_score=float(score),
                                    anchor_xy=list(anchor), region_areas=[int(areas[a]),int(areas[b])],
                                    predicted_classes=[original_classes[str(a)],original_classes[str(b)]],
                                    seam_pixels=int(pair['seam_pixels']), seam_visible_pixels=visible,
                                    seam_visible_fraction=visible/int(pair['seam_pixels']),
                                    full_seam_components_8=_seam_components(pair['seam_flat'], baseline.shape),
                                    whole_pair_not_fully_observed=visible < int(pair['seam_pixels']),
                                    **raw['metadata']))
                    new_ids, new_classes, merge_audit = merge_relation_pairs(
                        baseline, original_classes, scored, threshold=self.threshold)
                audit = dict(format='boundary_relation_inference_v1', **self.provenance(),
                    input_contract=before, candidate_definition=pair_audit,
                    scored_all_original_geometry_pairs=no_op is None,
                    scoring_anchor='training canonical median-nearest point of the FULL original shared seam',
                    native_patch_size=256, maps_quantization='float32→reflect→float16→float32',
                    RGB_source='original uint8 RGB /127.5-1; maps mode is exactly zero after normalization',
                    no_GT_features=True, no_multi_window_vote=True, no_revote=True,
                    pairs=scored, scoring_batch_sizes=batches, no_op_reason=no_op,
                    runtime=runtime, merge=merge_audit,
                    output_contract=dict(instances=_array_record(new_ids), classes_sha256=hashlib.sha256(
                        json.dumps(new_classes,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()))
            finally:
                if _inputs_record(rgb, N1, NAF, instances, classes) != before:
                    raise RuntimeError('关系处理改变了旧输入数组或类别表')
                self.verify_unchanged()
        audit.update(input_unchanged=True, existing_RNG_and_runtime_flags_restored=True)
        return new_ids, new_classes, audit
