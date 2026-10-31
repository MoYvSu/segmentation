# -*- coding: utf-8 -*-
"""新GT回执必须包含补区，并拒绝目标/未知域不一致。"""
from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from tools.verify_affinity_newgt import audit_new_gt


@pytest.fixture
def cohort(tmp_path):
    raw = tmp_path / 'data/raw'
    labels = tmp_path / 'outputs/newgt'
    raw.mkdir(parents=True)
    labels.mkdir(parents=True)
    ids = np.ones((4, 6), np.uint16)
    ids[-1, -1] = 0
    original = np.zeros_like(ids, np.uint8)
    original[:, :2] = 1
    filled = ((ids > 0) & ~original.astype(bool)).astype(np.uint8)
    unknown = (ids == 0).astype(np.uint8)
    for i in range(32):
        name = f'fiction_{i:02d}'
        cv2.imwrite(str(raw / (name + '.png')), np.full((4, 6, 3), 127, np.uint8))
        # 故意不提供合法旧多边形；它只确认源图身份，不能变成监督目标。
        (raw / (name + '.json')).write_text('SOURCE IDENTITY ONLY', encoding='utf8')
        np.savez(labels / (name + '_gt.npz'), instance_map=ids, original_covered=original,
                 filled=filled, residual_unknown=unknown)
        (labels / (name + '_class.json')).write_text('{"1": 1}', encoding='utf8')
    manifest = dict(format='manual_target_v2', version=1, sample_count=32,
        source='LabelMe instance seeds + narrow Lab-gradient watershed',
        max_fill_distance_native=8., blur_sigma=1.2,
        samples=[f'fiction_{i:02d}' for i in range(32)], summary=dict(
            num_images=32, total_pixels=24*32, filled_pixels=int(filled.sum())*32,
            residual_unknown_pixels=32, original_uncovered_pixels=16*32,
            original_coverage=8/24, completed_coverage=23/24))
    (labels / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    config = dict(paths=dict(project_root=str(tmp_path), raw_data_dir='data/raw'),
                  backend_adaptation=dict(completed_gt_dir='outputs/newgt', expected_manual_sources=32))
    return config, labels, raw


def change_npz(path, key, change):
    with np.load(path, allow_pickle=False) as blob:
        values = {k: blob[k].copy() for k in blob.files}
    change(values[key])
    np.savez(path, **values)


def test_completed_pixels_are_supervised_and_old_polygons_are_identity_only(cohort):
    config, labels, raw = cohort
    report = audit_new_gt(config)
    assert report['passed'] and report['manual_sources'] == 32
    assert report['totals']['completed_known'] == 23*32
    assert report['totals']['original_covered'] == 8*32
    assert report['totals']['filled'] == 15*32
    assert report['totals']['completion_only_sources'] == 32
    assert report['original_covered_is_supervision_mask'] is False
    assert report['filled_and_original_equal_weight'] is True
    assert report['completed_gt_dir'] == 'outputs/newgt'
    assert len(report['samples']) == 32
    assert all(len(row['files_sha256']) == 4 for row in report['samples'])


@pytest.mark.parametrize('key,point,value', [
    ('filled', (0, 4), 0), ('original_covered', (3, 5), 1),
    ('filled', (0, 4), 2), ('residual_unknown', (3, 5), 0)])
def test_invalid_completion_relationship_is_rejected_not_repaired(cohort, key, point, value):
    config, labels, _ = cohort
    change_npz(labels / 'fiction_00_gt.npz', key, lambda a: a.__setitem__(point, value))
    with pytest.raises(ValueError):
        audit_new_gt(config)


@pytest.mark.parametrize('classes', ['{"1": true}', '{}', '{"1": 1, "2": 0}', '{"1": 1, "1": 0}'])
def test_classes_are_exact_integer_mapping_and_duplicates_rejected(cohort, classes):
    config, labels, _ = cohort
    (labels / 'fiction_00_class.json').write_text(classes, encoding='utf8')
    with pytest.raises(ValueError):
        audit_new_gt(config)


def test_guard_receipt_changes_on_class_change_even_when_geometry_unchanged(cohort):
    config, labels, _ = cohort
    before = audit_new_gt(config)
    (labels / 'fiction_00_class.json').write_text('{"1": 0}', encoding='utf8')
    after = audit_new_gt(config)
    assert before['npz_cohort_sha256'] == after['npz_cohort_sha256']
    assert before['class_cohort_sha256'] != after['class_cohort_sha256']
    assert before['samples'] != after['samples']


def test_source_identity_change_does_not_reinterpret_old_polygon_targets(cohort):
    config, _, raw = cohort
    before = audit_new_gt(config)
    (raw / 'fiction_00.json').write_text('DIFFERENT IDENTITY BYTES', encoding='utf8')
    after = audit_new_gt(config)
    assert before['npz_cohort_sha256'] == after['npz_cohort_sha256']
    assert before['class_cohort_sha256'] == after['class_cohort_sha256']
    assert before['samples'] != after['samples']


def test_extra_target_outside_approved_source_set_is_rejected(cohort):
    config, labels, _ = cohort
    (labels / 'unused_gt.npz').write_bytes((labels / 'fiction_00_gt.npz').read_bytes())
    with pytest.raises(ValueError, match='exact canonical'):
        audit_new_gt(config)


def test_target_shape_and_manifest_summary_cannot_drift(cohort):
    config, labels, _ = cohort
    path = labels / 'manifest.json'
    value = json.loads(path.read_text())
    value['summary']['filled_pixels'] -= 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='summary'):
        audit_new_gt(config)


def test_no_new_holdout_or_count_override_is_accepted(cohort):
    config, _, _ = cohort
    config = deepcopy(config)
    config['backend_adaptation']['expected_manual_sources'] = 31
    with pytest.raises(ValueError, match='32'):
        audit_new_gt(config)
