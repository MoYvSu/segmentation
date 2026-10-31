# -*- coding: utf-8 -*-
"""验证派生候选门禁，不读取赛题数据。"""
from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from data.teacher_seam_manifest import (FORMAT, POLICY, CODE_PATHS, file_sha, load_seam_manifest,
                                        validate_seam_arrays)


def _json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def cohort(tmp_path):
    original = tmp_path / 'outputs' / 'gt_source' / 'targets'
    derived = tmp_path / 'outputs' / 'teacher_seams' / 'targets'
    original.mkdir(parents=True)
    (original / 'teacher').mkdir()
    (original / 'pairs').mkdir()
    derived.mkdir(parents=True)
    (derived / 'pairs').mkdir()
    code = {}
    for value in CODE_PATHS:
        path = tmp_path / value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic fixture code\n', encoding='utf-8')
        code[value] = file_sha(path)
    rows, new_rows = [], []
    manual = np.array([[1, 1, 2], [1, 1, 2]], dtype=np.uint16)
    teacher = np.array([[7, 0, 8], [7, 0, 8]], dtype=np.uint16)
    known = teacher > 0
    changed = np.array([[7, 7, 8], [7, 7, 8]], dtype=np.uint16)
    relative = lambda path: path.relative_to(tmp_path).as_posix()
    for index in range(32):
        source = 'train_' + str(index + 1).zfill(3)
        image = original / (source + '.jpg')
        cv2.imwrite(str(image), np.full((2, 3, 3), index, np.uint8))
        gt = original / (source + '_gt.npz')
        np.savez(gt, instance_map=manual)
        old = original / 'pairs' / (source + '.npz')
        np.savez(old, teacher_ids=teacher, common_known=known)
        png = original / 'teacher' / (source + '_inst.png')
        cv2.imwrite(str(png), teacher)
        new = derived / 'pairs' / (source + '.npz')
        np.savez(new, teacher_ids=changed, assigned_mask=(~known).astype(np.uint8))
        row = dict(source=source, image=relative(image), image_sha256=file_sha(image),
                   manual_gt=relative(gt), manual_gt_sha256=file_sha(gt),
                   pair_npz=relative(old), pair_sha256=file_sha(old),
                   prediction_hashes={'png_sha256': file_sha(png)})
        rows.append(row)
        new_rows.append(dict(source=source, pair_npz=relative(new), pair_sha256=file_sha(new),
                   old_pair_sha256=file_sha(old), image_sha256=file_sha(image),
                   manual_gt_sha256=file_sha(gt), teacher_png=relative(png),
                   teacher_png_sha256=file_sha(png), audit={'pixels': {'assigned': 2}}))
    parent = dict(format='affinity_gt_source_targets_v1', status='complete', rows=rows,
                  allowed_training_sources=[row['source'] for row in rows])
    parent_file = original / 'manifest.json'
    _json(parent_file, parent)
    parent_sha = file_sha(parent_file)
    approval_file = original.parent / 'quality_review.json'
    _json(approval_file, dict(status='explicit_user_confirmed', manifest_sha256=parent_sha,
                              review_checked_target_count=32))
    derived_manifest = dict(format=FORMAT, status='complete', policy=POLICY,
              parent_manifest_sha256=parent_sha, parent_approval_sha256=file_sha(approval_file),
              user_authorized_transform=True, derived_targets_individually_human_confirmed=False,
              semantic_labels_used_for_assignment=False, test_training=False, rows=new_rows)
    derived_manifest['protected_code'] = code
    derived_file = derived / 'manifest.json'
    _json(derived_file, derived_manifest)
    config = dict(paths={'project_root': str(tmp_path)}, gt_source=dict(manifest=relative(parent_file),
              approval_file=relative(approval_file), seam_repair=dict(format=FORMAT,
                manifest=relative(derived_file), manifest_sha256=file_sha(derived_file), policy=POLICY)))
    return config, parent, parent_sha, derived_manifest, derived_file


def test_exact_parent_derivative_and_all_sources_protected(cohort):
    config, parent, parent_sha, manifest, _ = cohort
    actual, protected = load_seam_manifest(config, parent, parent_sha)
    assert actual == manifest
    assert len(protected) == 6 + 32 * 5
    assert len(actual['rows']) == 32


@pytest.mark.parametrize('change', ['manifest_hash', 'parent_approval', 'duplicate_source',
                                    'invent_confirmation', 'source_hash', 'assignment_count'])
def test_reject_changed_binding_and_audits(cohort, change):
    config, parent, parent_sha, manifest, path = cohort
    config, manifest = deepcopy(config), deepcopy(manifest)
    if change == 'manifest_hash':
        config['gt_source']['seam_repair']['manifest_sha256'] = '0' * 64
    else:
        if change == 'parent_approval':
            manifest['parent_approval_sha256'] = '0' * 64
        elif change == 'duplicate_source':
            manifest['rows'][0]['source'] = manifest['rows'][1]['source']
        elif change == 'invent_confirmation':
            manifest['derived_targets_individually_human_confirmed'] = True
        elif change == 'source_hash':
            manifest['rows'][0]['image_sha256'] = '0' * 64
        elif change == 'assignment_count':
            manifest['rows'][0]['audit']['pixels']['assigned'] = 999
        _json(path, manifest)
        config['gt_source']['seam_repair']['manifest_sha256'] = file_sha(path)
    with pytest.raises((ValueError, RuntimeError)):
        load_seam_manifest(config, parent, parent_sha)


@pytest.mark.parametrize('change', ['old_positive', 'new_id', 'unknown', 'assigned_positive'])
def test_reject_content_corruption_even_with_rebound_file_sha(cohort, change):
    config, parent, parent_sha, manifest, manifest_path = cohort
    config, manifest = deepcopy(config), deepcopy(manifest)
    path = manifest_path.parent / 'pairs' / 'train_001.npz'
    with np.load(path, allow_pickle=False) as blob:
        teacher, assigned = blob['teacher_ids'].copy(), blob['assigned_mask'].copy()
    if change == 'old_positive':
        teacher[0, 0] = 8
    elif change == 'new_id':
        teacher[0, 1] = 9
    elif change == 'unknown':
        # Removing assigned-support consistency must be rejected as well.
        teacher[0, 1] = 0
    else:
        assigned[0, 0] = 1
    np.savez(path, teacher_ids=teacher, assigned_mask=assigned)
    manifest['rows'][0]['pair_sha256'] = file_sha(path)
    _json(manifest_path, manifest)
    config['gt_source']['seam_repair']['manifest_sha256'] = file_sha(manifest_path)
    with pytest.raises(ValueError):
        load_seam_manifest(config, parent, parent_sha)


def test_true_unknown_cannot_be_filled():
    manual = np.array([[1, 0, 2]], np.uint16)
    teacher = np.array([[7, 0, 8]], np.uint16)
    old = {'teacher_ids': teacher, 'common_known': teacher > 0}
    with pytest.raises(ValueError, match='changed old labels, unknown'):
        validate_seam_arrays({'teacher_ids': np.array([[7, 7, 8]], np.uint16),
                              'assigned_mask': np.array([[0, 1, 0]], np.uint8)}, old, manual, teacher)


def test_prepare_complete32_smoke_preserves_sources_and_refuses_overwrite(cohort):
    import torch
    from tools.prepare_teacher_seams import prepare
    from train_affinity_gt_source import R1_SHA, D5A_SHA
    config, parent, _, _, _ = cohort
    root = __import__('pathlib').Path(config['paths']['project_root'])
    parent = deepcopy(parent)
    parent.update(teacher_checkpoint_sha256=R1_SHA, restorer_checkpoint_sha256=D5A_SHA,
                  semantic_labels_used_for_teacher_training=False)
    manual = np.array([[1, 1, 2], [1, 1, 2]], np.uint16)
    teacher = np.array([[7, 0, 8], [7, 0, 8]], np.uint16)
    known = teacher > 0
    for index, row in enumerate(parent['rows']):
        image = root / row['image']
        cv2.imwrite(str(image), np.random.default_rng(index).integers(0, 255, (2, 3, 3), dtype=np.uint8))
        row['image_sha256'] = file_sha(image)
        gt = root / row['manual_gt']
        np.savez(gt, instance_map=manual, original_covered=np.ones(manual.shape, np.uint8),
                 filled=np.zeros(manual.shape, np.uint8), residual_unknown=np.zeros(manual.shape, np.uint8))
        row['manual_gt_sha256'] = file_sha(gt)
        _json(gt.with_name(row['source'] + '_class.json'), {'1': 1, '2': 1})
        pair = root / row['pair_npz']
        np.savez(pair, manual_ids=np.where(known, manual, 0).astype(np.uint16), teacher_ids=teacher,
                 common_known=known, original_covered=np.ones(manual.shape, np.uint8),
                 filled=np.zeros(manual.shape, np.uint8))
        row['pair_sha256'] = file_sha(pair)
    parent_path = root / config['gt_source']['manifest']
    _json(parent_path, parent)
    parent_sha = file_sha(parent_path)
    _json(root / config['gt_source']['approval_file'], dict(status='explicit_user_confirmed',
          manifest_sha256=parent_sha, review_checked_target_count=32))
    config['gt_source']['seam_repair']['manifest'] = 'outputs/smoke_seams/targets/manifest.json'
    config['gt_source']['seam_repair']['manifest_sha256'] = ''
    threads = torch.get_num_threads()
    torch.set_num_threads(min(4, threads))
    try:
        result, digest = prepare(config)
    finally:
        torch.set_num_threads(threads)
    assert result['status'] == 'complete'
    assert len(result['rows']) == 32
    assert result['summary']['native_pixels']['assigned'] == 64
    assert result['summary']['whole1024_grid512']['added_legal_edges'] > 0
    assert file_sha(parent_path) == parent_sha
    assert len(digest) == 64
    config['gt_source']['seam_repair']['manifest_sha256'] = digest
    actual, _ = load_seam_manifest(config, parent, parent_sha)
    assert actual['status'] == 'complete'
    assert (root / 'outputs/smoke_seams/targets/contact32.jpg').is_file()
    with pytest.raises(FileExistsError, match='refusing overwrite'):
        prepare(config)
