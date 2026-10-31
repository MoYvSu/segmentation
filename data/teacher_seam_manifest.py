# -*- coding: utf-8 -*-
"""派生窄零缝监督的来源/内容门禁；原教师人工确认不被改写。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

FORMAT = 'teacher_seam_support_v1'
POLICY = {'max_distance_native': 1.0, 'context_radius_native': 2.0}
CODE_PATHS = ('utils/teacher_seam_support.py', 'data/teacher_seam_manifest.py',
              'tools/prepare_teacher_seams.py')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative_file(config, value):
    root = Path(config['paths']['project_root']).resolve()
    path = Path(value)
    if (path.is_absolute() or '..' in path.parts or '\\' in str(value)
            or ':' in str(value)):
        raise ValueError('Seam manifest inputs must be project-relative paths')
    result = (root / path).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError('Seam manifest input missing or outside project')
    return result


def seam_options(config):
    opts = config['gt_source']['seam_repair']
    if opts.get('format') != FORMAT or opts.get('policy') != POLICY:
        raise ValueError('Explicit version and fixed native seam policy required')
    return opts


def _checked_file(config, value, expected, protected):
    path = relative_file(config, value)
    if not isinstance(expected, str) or len(expected) != 64 or file_sha(path) != expected:
        raise RuntimeError('Seam manifest input SHA differs: ' + str(value))
    protected[str(path)] = expected
    return path


def _ids(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.uint16:
        raise ValueError(name + ' must be matching native uint16 IDs')
    return array


def validate_seam_arrays(values, old_pair, full_manual, raw_teacher):
    """不改变旧共同域、未知和实例集合；assigned只能出自原教师零像素。"""
    if not {'teacher_ids', 'assigned_mask'}.issubset(values):
        raise ValueError('Derived seam fields incomplete')
    shape = full_manual.shape
    new = _ids(values['teacher_ids'], shape, 'teacher_ids')
    old = _ids(old_pair['teacher_ids'], shape, 'old teacher_ids')
    raw = _ids(raw_teacher, shape, 'raw teacher PNG')
    assigned = np.asarray(values['assigned_mask'])
    if (assigned.shape != shape or assigned.dtype.kind not in 'bu'
            or not np.isin(assigned, [0, 1]).all()):
        raise ValueError('assigned_mask must be matching binary HW')
    assigned = assigned.astype(bool)
    old_known = np.asarray(old_pair['common_known'], dtype=bool)
    if (not np.array_equal(old_known, (full_manual > 0) & (raw > 0))
            or not np.array_equal(old, np.where(old_known, raw, 0))):
        raise ValueError('Original paired targets differ from bound raw teacher')
    if (not np.array_equal(new > 0, old_known | assigned)
            or not np.array_equal(new[old_known], old[old_known])
            or np.any(new[full_manual == 0])
            or np.any(assigned & ((raw != 0) | (full_manual == 0)))
            or not np.array_equal(np.unique(new[new > 0]), np.unique(old[old > 0]))):
        raise ValueError('Seam repair changed old labels, unknown, or instance set')
    return new, assigned


def load_seam_manifest(config, parent_manifest, expected_parent_sha):
    """返回派生manifest与保护SHA；调用者仍须先执行原load_targets教师门禁。"""
    opts = seam_options(config)
    protected = {}
    parent_path = _checked_file(config, config['gt_source']['manifest'],
                                expected_parent_sha, protected)
    disk_parent = json.loads(parent_path.read_text(encoding='utf-8'))
    if disk_parent != parent_manifest:
        raise ValueError('Passed parent manifest differs from bound file')
    parent_rows = parent_manifest.get('rows', [])
    sources = {row['source'] for row in parent_rows}
    if (parent_manifest.get('format') != 'affinity_gt_source_targets_v1'
            or parent_manifest.get('status') != 'complete'
            or len(parent_rows) != 32 or len(sources) != 32
            or set(parent_manifest.get('allowed_training_sources', [])) != sources):
        raise ValueError('Exact completed original32 parent cohort required')
    approval_path = relative_file(config, config['gt_source']['approval_file'])
    approval_sha = file_sha(approval_path)
    approval = json.loads(approval_path.read_text(encoding='utf-8'))
    if (approval.get('status') != 'explicit_user_confirmed'
            or approval.get('manifest_sha256') != expected_parent_sha
            or approval.get('review_checked_target_count') != 32):
        raise RuntimeError('Original user confirmation must remain bound to parent manifest')
    protected[str(approval_path)] = approval_sha
    path = _checked_file(config, opts['manifest'], opts.get('manifest_sha256'), protected)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('format') != FORMAT or manifest.get('status') != 'complete'
            or manifest.get('parent_manifest_sha256') != expected_parent_sha
            or manifest.get('parent_approval_sha256') != approval_sha
            or manifest.get('policy') != POLICY
            or manifest.get('user_authorized_transform') is not True
            or manifest.get('derived_targets_individually_human_confirmed') is not False
            or manifest.get('semantic_labels_used_for_assignment') is not False
            or manifest.get('test_training') is not False):
        raise ValueError('Bound complete authorized seam derivative required')
    code = manifest.get('protected_code', {})
    if set(code) != set(CODE_PATHS):
        raise ValueError('Exact seam assignment, preparation and loader code identity required')
    for value, digest in code.items():
        _checked_file(config, value, digest, protected)
    rows = manifest.get('rows', [])
    if (len(rows) != 32 or {row['source'] for row in rows} != sources
            or len({row['source'] for row in rows}) != 32):
        raise ValueError('Derivative must retain exact unique original32 source cohort')
    parents = {row['source']: row for row in parent_rows}
    for row in rows:
        old = parents[row['source']]
        for field in ('image_sha256', 'manual_gt_sha256'):
            if row.get(field) != old[field]:
                raise ValueError('Derived source identity changed: ' + field)
        if row.get('old_pair_sha256') != old['pair_sha256']:
            raise ValueError('Derived old pair identity changed')
        image = _checked_file(config, old['image'], old['image_sha256'], protected)
        gt_path = _checked_file(config, old['manual_gt'], old['manual_gt_sha256'], protected)
        old_pair = _checked_file(config, old['pair_npz'], old['pair_sha256'], protected)
        teacher_expected = old['prediction_hashes']['png_sha256']
        if row.get('teacher_png_sha256') != teacher_expected:
            raise ValueError('Derived raw teacher identity changed')
        expected_teacher_path = parent_path.parent / 'teacher' / (row['source'] + '_inst.png')
        teacher_path = _checked_file(config, row['teacher_png'], teacher_expected, protected)
        if teacher_path != expected_teacher_path.resolve():
            raise ValueError('Teacher must be original parent prediction PNG')
        pair_path = _checked_file(config, row['pair_npz'], row['pair_sha256'], protected)
        if not pair_path.is_relative_to(path.parent / 'pairs'):
            raise ValueError('Derived pair must be inside sealed manifest pairs directory')
        if image.stem != row['source'] or not row['source'].startswith('train_'):
            raise ValueError('Derived source is not an allowed training image')
        raw_teacher = cv2.imread(str(teacher_path), cv2.IMREAD_UNCHANGED)
        with np.load(gt_path, allow_pickle=False) as gt, np.load(old_pair, allow_pickle=False) as old_blob, np.load(pair_path, allow_pickle=False) as new_blob:
            if raw_teacher is None:
                raise ValueError('Raw teacher PNG unreadable')
            _, assigned = validate_seam_arrays(new_blob, old_blob, gt['instance_map'], raw_teacher)
            if int(assigned.sum()) != row['audit']['pixels']['assigned']:
                raise ValueError('Assigned pixel audit differs from derivative')
        if 'semantic_class_json' in row:
            class_path = _checked_file(config, row['semantic_class_json'],
                                       row['semantic_class_sha256'], protected)
            if class_path != gt_path.with_name(row['source'] + '_class.json'):
                raise ValueError('Semantic audit must use existing processed manual classes')
    return manifest, protected
