# -*- coding: utf-8 -*-
"""教师零缝短跑的控制复用与实际监督域检查；不修改原教师候选。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from utils.affinity_loss import build_affinity_targets_torch
from utils.offset_letterbox import letterbox_instance_geometry


_FORMAT = 'affinity_gt_source_v1'
_COMMON_KEYS = ('epoch', 'step', 'seed', 'manual', 'pseudo', 'manual_draw', 'pseudo_draw',
                'manual_crop', 'pseudo_crop', 'training_view')
_SIGNATURE_KEYS = {'image', 'clean_image', 'restored', 'valid_pixels', 'valid_edges', 'shape'}
_CONTRACT_KEYS = {'format', 'recipe_sha256', 'manifest_sha256', 'checkpoint_sha256',
                  'restorer_checkpoint_sha256', 'frozen_state_sha256',
                  'restorer_state_sha256', 'initial_head_sha256', 'training_code_sha256'}
_REPAIR_CONTRACT_KEYS = {'variant', 'seam_manifest_sha256', 'baseline_steps_sha256',
                         'baseline_status_sha256'}


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def _tensor_sha(value):
    """与既有 batch_signature 的 tensor_digest([('value', tensor)]) 完全相同。"""
    digest = hashlib.sha256(b'value')
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _baseline_recipe(config):
    value = deepcopy(config)
    opts = value.get('gt_source')
    if not isinstance(opts, dict):
        raise ValueError('GT-source recipe required for teacher control reuse')
    opts.pop('seam_repair', None)
    opts.pop('output_dir', None)
    return value


def validate_repaired_peer(status, contract, config, end):
    """复用完整原teacher控制；仅修补字段、输出目录与训练源码版本可以不同。"""
    if isinstance(end, bool) or not isinstance(end, int) or end != 5:
        raise ValueError('Same five-epoch teacher control required')
    if (not isinstance(status, dict) or status.get('format') != _FORMAT
            or status.get('arm') != 'teacher' or status.get('status') != 'complete'
            or status.get('epoch') != end or status.get('epochs_requested') != end
            or isinstance(status.get('updates'), bool) or status.get('updates') != end * 64
            or not all(status.get(key) is True for key in
                       ('frozen_unchanged', 'restorer_unchanged', 'strict_reload_equal'))):
        raise ValueError('Complete same-budget strictly reloaded teacher control required')
    peer_config = status.get('config')
    peer_contract = status.get('contract')
    if (not isinstance(peer_config, dict) or not isinstance(peer_contract, dict)
            or set(peer_contract) != _CONTRACT_KEYS
            or set(contract) != _CONTRACT_KEYS | _REPAIR_CONTRACT_KEYS
            or peer_config.get('gt_source', {}).get('seam_repair') is not None
            or not isinstance(config.get('gt_source', {}).get('seam_repair'), dict)):
        raise ValueError('Original teacher and explicit repair contracts required')
    opts = config['gt_source']['seam_repair']
    if (opts.get('format') != 'teacher_seam_support_v1'
            or contract.get('variant') != 'teacher_zero_seam_v1'
            or contract.get('seam_manifest_sha256') != opts.get('manifest_sha256')
            or contract.get('baseline_steps_sha256') != opts.get('baseline_steps_sha256')
            or contract.get('baseline_status_sha256') != opts.get('baseline_status_sha256')
            or any(not isinstance(contract.get(key), str) or len(contract[key]) != 64
                   or any(char not in '0123456789abcdef' for char in contract[key])
                   for key in _REPAIR_CONTRACT_KEYS - {'variant'})):
        raise ValueError('Explicit repair variant and manifest/control SHA bindings required')
    if (_canonical_sha(peer_config) != peer_contract['recipe_sha256']
            or _canonical_sha(config) != contract['recipe_sha256']
            or _canonical_sha(_baseline_recipe(peer_config)) != _canonical_sha(_baseline_recipe(config))):
        raise ValueError('Teacher control differs outside explicit seam repair and output directory')
    for key in _CONTRACT_KEYS - {'recipe_sha256', 'training_code_sha256'}:
        if contract[key] != peer_contract[key]:
            raise ValueError('Teacher control initialization or frozen contract differs: ' + key)
    if contract['format'] != _FORMAT:
        raise ValueError('Compatible GT-source contract format required')
    return dict(passed=True, updates=end * 64, same_recipe_except_seam_repair_and_output=True,
                training_code_may_differ=True, teacher_recipe_sha256=peer_contract['recipe_sha256'])


def compare_repaired_rows(row, reference):
    """人工源只允许监督域变化；SAM2输入与监督域逐项保持完全相同。"""
    for key in _COMMON_KEYS:
        if key not in row or key not in reference or row[key] != reference[key]:
            raise RuntimeError('Repaired/control common draw or input differs: ' + key)
    actual, prior = row.get('inputs'), reference.get('inputs')
    if (not isinstance(actual, list) or not isinstance(prior, list)
            or len(actual) != 2 or len(prior) != 2
            or any(not isinstance(value, dict) or set(value) != _SIGNATURE_KEYS
                   for value in actual + prior)):
        raise RuntimeError('Exactly manual and SAM2 complete input signatures required')
    for key in _SIGNATURE_KEYS - {'valid_pixels', 'valid_edges'}:
        if actual[0][key] != prior[0][key]:
            raise RuntimeError('Repaired/control manual input differs: ' + key)
    if actual[1] != prior[1]:
        raise RuntimeError('Repaired/control SAM2 input or supervision domain differs')
    audit = row.get('seam_support')
    if (not isinstance(audit, dict) or audit.get('old_support_subset') is not True
            or audit.get('old_targets_unchanged') is not True
            or audit.get('old_target_changed_edges') != 0):
        raise RuntimeError('Actual repaired supervision subset and target audit required')
    for audit_key, input_key in [('old_valid_pixels_sha256', 'valid_pixels'),
                                  ('old_valid_edges_sha256', 'valid_edges')]:
        if audit.get(audit_key) != prior[0][input_key]:
            raise RuntimeError('Rebuilt old teacher support differs from control: ' + input_key)
    for audit_key, input_key in [('new_valid_pixels_sha256', 'valid_pixels'),
                                  ('new_valid_edges_sha256', 'valid_edges')]:
        if audit.get(audit_key) != actual[0][input_key]:
            raise RuntimeError('Actual repaired support differs from input signature: ' + input_key)


def _single(value, name):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError('Batch-one draw field required: ' + name)
        return value.detach().cpu().item()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    raise ValueError('Collated batch-one draw field required: ' + name)


def _relative_file(config, value):
    base = Path(config['paths']['project_root']).resolve()
    relative = Path(value)
    if (relative.is_absolute() or '..' in relative.parts or '\\' in str(value)
            or ':' in str(value)):
        raise ValueError('Project-relative original target path required')
    result = (base / relative).resolve()
    if not result.is_relative_to(base) or not result.is_file():
        raise ValueError('Original target missing or outside project')
    return result


def _transform(value, horizontal, vertical, turns):
    # 既有 PairedDegradationDataset：先横翻，再竖翻，最后逆时针rot90。
    if horizontal:
        value = value.flip(-1)
    if vertical:
        value = value.flip(-2)
    return torch.rot90(value, turns, (-2, -1)).contiguous()


def audit_repaired_batch(batch, config, rows_index):
    """在实际crop／nearest／增强后检验监督扩展，哈希绑定旧320步控制回执。"""
    name = _single(batch.get('name'), 'name')
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('Bare training filename required')
    source = Path(name).stem
    if not source.startswith('train_') or source not in rows_index:
        raise ValueError('Training source outside confirmed teacher cohort')
    row = rows_index[source]
    if row.get('source') != source:
        raise ValueError('Declared source identity differs')
    with np.load(_relative_file(config, row['pair_npz']), allow_pickle=False) as payload:
        old_ids = payload['teacher_ids'].copy()
        old_known = payload['common_known'].astype(bool)
    with np.load(_relative_file(config, row['manual_gt']), allow_pickle=False) as payload:
        processed = payload['instance_map'].copy()
    if (old_ids.ndim != 2 or processed.shape != old_ids.shape or old_known.shape != old_ids.shape
            or not np.array_equal(old_known, old_ids > 0)
            or np.any(old_known & (processed == 0))):
        raise ValueError('Original teacher support and processed provenance inconsistent')
    box = batch.get('crop_box')
    if not torch.is_tensor(box) or tuple(box.shape) != (1, 4) or box.dtype.is_floating_point:
        raise ValueError('Actual batch-one integer crop box required')
    y, x, y1, x1 = box.detach().cpu().tolist()[0]
    if not 0 <= y < y1 <= old_ids.shape[0] or not 0 <= x < x1 <= old_ids.shape[1]:
        raise ValueError('Crop box outside original source')
    horizontal = _single(batch.get('horizontal_flip'), 'horizontal_flip')
    vertical = _single(batch.get('vertical_flip'), 'vertical_flip')
    turns = _single(batch.get('rotation_k'), 'rotation_k')
    if (not isinstance(horizontal, bool) or not isinstance(vertical, bool)
            or isinstance(turns, bool) or not isinstance(turns, int) or not 0 <= turns <= 3):
        raise ValueError('Actual draw requires boolean flips and integer rotation_k in 0..3')
    ids = batch.get('affinity_instance_map', batch.get('instance_map'))
    pixels = batch.get('affinity_valid_content', batch.get('valid_content'))
    if (not torch.is_tensor(ids) or ids.ndim != 3 or ids.shape[0] != 1
            or ids.shape[1] != ids.shape[2] or ids.dtype.is_floating_point
            or not torch.is_tensor(pixels) or tuple(pixels.shape) != (1, 1, *ids.shape[1:])
            or pixels.dtype != torch.bool):
        raise ValueError('Batch-one square affinity IDs and boolean valid content required')
    new_ids, new_pixels = ids.detach().cpu(), pixels.detach().cpu()
    size = int(config.get('data', {}).get('image_size', 1024))
    grid = int(ids.shape[-1])
    old_grid, content, _ = letterbox_instance_geometry(old_ids[y:y1, x:x1], size, grid)
    processed_grid, processed_content, _ = letterbox_instance_geometry(processed[y:y1, x:x1], size, grid)
    if not np.array_equal(content, processed_content):
        raise RuntimeError('Original and processed letterbox content differ')
    old_tensor = _transform(torch.from_numpy(old_grid.astype(np.int64))[None], horizontal, vertical, turns)
    old_pixels = _transform(torch.from_numpy(content & (old_grid > 0))[None, None],
                            horizontal, vertical, turns)
    allowed_pixels = _transform(torch.from_numpy(content & (processed_grid > 0))[None, None],
                                horizontal, vertical, turns)
    if torch.any(new_pixels & ~allowed_pixels):
        raise RuntimeError('Repaired supervision enters processed unknown or padding')
    if not torch.equal(new_pixels, (new_ids > 0)[:, None] & allowed_pixels):
        raise RuntimeError('Actual repaired validity differs from label and known content')
    if torch.any(old_pixels & ~new_pixels):
        raise RuntimeError('Repaired pixels removed original teacher support')
    if not torch.equal(new_ids[old_pixels[:, 0]], old_tensor[old_pixels[:, 0]]):
        raise RuntimeError('Repaired labels changed original nonzero teacher ownership')
    old_targets, old_edges = build_affinity_targets_torch(old_tensor, old_pixels)
    new_targets, new_edges = build_affinity_targets_torch(new_ids, new_pixels)
    if torch.any(old_edges & ~new_edges):
        raise RuntimeError('Repaired legal edges removed original teacher support')
    changed = old_edges & (old_targets != new_targets)
    if changed.any():
        raise RuntimeError('Repaired targets changed original legal teacher answers')
    added = new_edges & ~old_edges
    per_channel = [dict(channel=index, added_edges=int(added[0, index].sum()),
                        added_same=int((added[0, index] & (new_targets[0, index] == 1)).sum()),
                        added_different=int((added[0, index] & (new_targets[0, index] == 0)).sum()))
                   for index in range(new_edges.shape[1])]
    return dict(source=source, old_support_subset=True, old_targets_unchanged=True,
                old_target_changed_edges=0, old_valid_pixels_sha256=_tensor_sha(old_pixels),
                old_valid_edges_sha256=_tensor_sha(old_edges),
                new_valid_pixels_sha256=_tensor_sha(new_pixels), new_valid_edges_sha256=_tensor_sha(new_edges),
                old_targets_sha256=_tensor_sha(old_targets),
                old_legal_edges=int(old_edges.sum()), new_legal_edges=int(new_edges.sum()),
                added_pixels=int((new_pixels & ~old_pixels).sum()), added_edges=int(added.sum()),
                added_same=sum(value['added_same'] for value in per_channel),
                added_different=sum(value['added_different'] for value in per_channel),
                per_channel=per_channel, unknown_and_padding_ignored=True)
