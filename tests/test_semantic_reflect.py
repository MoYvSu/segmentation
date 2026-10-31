# -*- coding: utf-8 -*-
"""固定reflect语义探针的像素契约、分区对应、可信域与条件均值诊断。"""
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from tools.probe_semantic_reflect import (
    ferrite_composition, grouped_changes, gt_contract, load_fixed,
    partition_mapping, probability_mixture, training_probability_diagnostic,
    validate_prediction, verify_scored_bytes,
)
from tools.probe_semantic_fixed import probability_details
from tools.semantic_crossover import revote_instances


def test_relabelled_identical_shapes_are_unchanged():
    old = np.array([[1, 1, 0, 2], [1, 1, 0, 2]], np.uint16)
    current = np.array([[9, 9, 0, 7], [9, 9, 0, 7]], np.uint16)
    result = partition_mapping(old, current)
    assert result['current']['9']['kind'] == 'unchanged'
    assert result['current']['7']['kind'] == 'unchanged'
    assert result['current']['9']['old_ids'] == [1]
    assert result['group_counts'] == {'unchanged': 2}


def test_split_and_merge_taxonomy_includes_all_positive_pixels_once():
    old = np.array([[1, 1, 2, 3], [1, 1, 2, 3]], np.uint16)
    current = np.array([[4, 5, 6, 6], [4, 5, 6, 6]], np.uint16)
    result = partition_mapping(old, current)
    assert result['current']['4']['kind'] == result['current']['5']['kind'] == 'split'
    assert result['current']['6']['kind'] == 'merged'
    assert sum(row['area'] for row in result['current'].values()) == old.size
    assert result['old_to_current'] == {'1': [4, 5], '2': [6], '3': [6]}


def test_many_many_new_lost_and_background_are_not_fake_parents():
    old = np.array([[1, 1, 2, 2, 3], [1, 1, 2, 2, 0]], np.uint16)
    current = np.array([[4, 5, 4, 5, 0], [4, 5, 4, 5, 8]], np.uint16)
    result = partition_mapping(old, current)
    assert result['current']['4']['kind'] == result['current']['5']['kind'] == 'complex'
    assert result['current']['8']['kind'] == 'new'
    assert result['current']['8']['old_ids'] == []
    assert result['lost_old_ids'] == [3]
    assert all(row['old'] > 0 and row['current'] > 0 for row in result['overlaps'])


def test_one_to_one_changed_support_is_reshaped_not_unchanged():
    old = np.array([[0, 1, 1], [0, 0, 0]], np.uint16)
    current = np.array([[2, 2, 0], [0, 0, 0]], np.uint16)
    assert partition_mapping(old, current)['current']['2']['kind'] == 'reshaped'


def test_significant_overlap_is_separate_from_raw_topology():
    old = np.ones((10, 20), np.uint16)
    old[:, 10:] = 2
    current = old.copy()
    current[0, 9] = 2
    result = partition_mapping(old, current)
    assert len(result['overlaps']) == 3
    assert len(result['significant_overlaps']) == 2
    assert result['current']['2']['kind'] == 'complex'


def test_native_map_contract_rejects_wrong_dtype_classes_and_grid():
    ids = np.ones((2, 3), np.uint16)
    validate_prediction(ids, {'1': 1}, (2, 3))
    for image, classes, shape in ((ids.astype(np.uint8), {'1': 1}, None),
                                  (ids, {'1': True}, None), (ids, {'2': 0}, None),
                                  (ids, {'1': 1}, (3, 2))):
        with pytest.raises(RuntimeError):
            validate_prediction(image, classes, shape)


def test_gt_censoring_mixed_phase_and_oracle_no_forced_class():
    gt = np.zeros((9, 12), np.uint16)
    gt[2:5, 2:5] = 1
    gt[2:5, 5:8] = 2
    gt[5:7, 2:5] = 3
    gt[0:2, 9:11] = 4
    ids = np.zeros_like(gt)
    ids[2:5, 2:8] = 9  # 混合相粘连，不可用多数相强赋oracle。
    ids[5:7, 2:5] = 10  # 填充域，不可强赋oracle。
    ids[0:2, 9:11] = 11  # 图框删失。
    trust = gt > 0
    trust[5:7, 2:5] = False
    classes = {1: 1, 2: 0, 3: 1, 4: 1}
    result = gt_contract(gt, trust, classes, ids)
    assert result['predictions']['9']['mixed_phase']
    assert result['predictions']['9']['oracle_class'] is None
    assert result['predictions']['10']['filled_pixels'] == 6
    assert result['predictions']['10']['oracle_class'] is None
    assert result['predictions']['11']['frame_censored']
    assert result['predictions']['11']['oracle_class'] is None
    assert result['oracle_eligible_instances'] == 0
    assert all(not row['uncensored'] for row in result['pairs'])


def test_full_pure_original_nonframe_object_can_have_oracle_diagnosis():
    gt = np.zeros((7, 7), np.uint16)
    gt[2:5, 2:5] = 3
    result = gt_contract(gt, gt > 0, {3: 1}, gt)
    assert result['predictions']['3']['oracle_eligible']
    assert result['predictions']['3']['oracle_class'] == 1
    assert result['pairs'][0]['uncensored']


def test_probability_conditioning_cannot_hide_wrong_class_or_unmatched_gt():
    gt = np.zeros((9, 13), np.uint16)
    gt[2:5, 2:5], gt[2:5, 6:9], gt[6:8, 2:5] = 1, 2, 3
    ids = gt.copy()
    ids[ids == 3] = 0
    trust = gt > 0
    gc = {1: 1, 2: 0, 3: 1}
    contract = gt_contract(gt, trust, gc, ids)
    p = np.full(gt.shape, .8, np.float32)
    classes = {'1': 1, '2': 1}
    row = training_probability_diagnostic(contract, gt, trust, gc, classes, p)
    assert row['conditional_iou_mean'] == 1.
    assert row['class_correct_valid_matches'] == 1
    assert row['fixed_list_class_sensitive_score'] == pytest.approx(1/3)
    assert row['fixed_pair_confusion']['true_P_pred_F'] == 1
    assert row['GT_region_probability']['3']['predicted_original'] == 1
    assert row['pure_known_nonframe_oracle']['changed_ids'] == ['2']


def test_filled_only_gt_reported_but_never_added_to_fixed_gt_list():
    gt = np.zeros((7, 7), np.uint16)
    gt[1:3, 1:3], gt[4:6, 4:6] = 1, 2
    trust = (gt == 1)
    gc = {1: 1, 2: 0}
    contract = gt_contract(gt, trust, gc, gt)
    p = np.full(gt.shape, .5, np.float32)
    row = training_probability_diagnostic(contract, gt, trust, gc, {'1': 0, '2': 0}, p)
    assert row['fixed_gt_list_size'] == 1
    assert row['GT_region_probability']['2']['probability_original'] is None
    assert row['GT_region_probability']['2']['probability_filled'] == .5
    assert row['GT_region_probability']['2']['completed_region_class'] == 0


def test_unknown_inside_prediction_is_censored_even_if_pair_iou_is_one():
    gt = np.zeros((7, 7), np.uint16)
    gt[2:5, 2:5] = 1
    ids = gt.copy()
    ids[1, 3] = 1
    result = gt_contract(gt, gt > 0, {1: 1}, ids)
    assert result['pairs'][0]['iou'] == 1.
    assert not result['pairs'][0]['uncensored']
    assert result['predictions']['1']['unknown_pixels'] == 1
    assert result['predictions']['1']['oracle_class'] is None


def test_strict_vote_core_disagreement_and_ferrite_tails_are_diagnostic_only():
    ids = np.ones((30, 30), np.uint16)
    p = np.full(ids.shape, .25, np.float32)
    p[7:23, 7:23] = .9
    classes, scores = revote_instances(ids, p)
    details = probability_details(ids, p, scores)
    mixture = probability_mixture(ids, p, details)
    assert classes == {'1': 0}
    assert mixture['core_vs_all_disagreements'] == 1
    assert ferrite_composition(classes, details)['instances'] == 0
    classes = {'1': 1}
    composition = ferrite_composition(classes, details)
    assert composition['pixels'] == 900
    assert composition['tails']['200to999']['instances'] == 1
    assert sum(row['pixels'] for row in composition['tails'].values()) == 900


def test_grouped_changes_cover_each_new_instance_once():
    old = np.array([[1, 1, 2, 2]], np.uint16)
    ids = np.array([[4, 5, 6, 6]], np.uint16)
    p = np.full(ids.shape, .5, np.float32)
    base, scores = revote_instances(ids, p)
    details = probability_details(ids, p, scores)
    current = {'4': 1, '5': 0, '6': 1}
    grouped = grouped_changes(base, current, details, details, partition_mapping(old, ids))
    assert grouped['split']['P_to_F_instances'] == 1
    assert grouped['unchanged']['P_to_F_instances'] == 1
    assert sum(row['instances'] for row in grouped.values()) == 3
    assert sum(row['pixels'] for row in grouped.values()) == ids.size


def test_fixed_png_load_and_scored_zip_bytes(tmp_path):
    png = tmp_path/'test_000_inst.png'
    labels = tmp_path/'test_000_class.json'
    cv2.imwrite(str(png), np.ones((2, 2), np.uint16))
    labels.write_text(json.dumps({'1': 1}), encoding='utf8')
    ids, classes, _, _ = load_fixed(tmp_path, 'test_000')
    assert ids.dtype == np.uint16 and classes == {'1': 1}
    package = tmp_path/'scored.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        for i in range(100):
            archive.writestr(f'results/test_{i:03}_inst.png', png.read_bytes())
            archive.writestr(f'results/test_{i:03}_class.json', labels.read_bytes())
    with zipfile.ZipFile(package) as archive:
        verify_scored_bytes(archive, {'name': 'test_000'}, png, labels)
        labels.write_text(json.dumps({'1': 0}), encoding='utf8')
        with pytest.raises(RuntimeError, match='bytes differ'):
            verify_scored_bytes(archive, {'name': 'test_000'}, png, labels)


def test_duplicate_scored_zip_basenames_rejected(tmp_path):
    package = tmp_path/'duplicate.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        archive.writestr('a/x_inst.png', b'a')
        archive.writestr('b/x_inst.png', b'a')
    with zipfile.ZipFile(package) as archive:
        with pytest.raises(RuntimeError, match='Duplicate'):
            verify_scored_bytes(archive, {'name': 'x'}, Path('x_inst.png'), Path('x_class.json'))
