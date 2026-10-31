# -*- coding: utf-8 -*-
"""铁素体机制诊断：人为小图区分误切、泄漏、粘连及完整面积口径。"""
from copy import deepcopy

import numpy as np
import pytest

from tools.analyze_ferrite_native import analyze_ferrite_native, select_ferrite_cases, selftest


def board():
    gt = np.ones((48, 48), np.int32)
    gt[12:32, 12:32] = 2
    return gt


def ferrite_row(report, identifier=2):
    return next(row for row in report['gt_rows'] if row['gt'] == identifier)


def full_area(report):
    return report['summary']['safe_complete_F_area']


def test_true_internal_split_and_boundary_leakage_are_separate_mechanisms():
    gt = board(); classes = {1: 0, 2: 1}
    split = gt.astype(np.uint16); split[12:32, 22:32] = 3
    internal = analyze_ferrite_native(gt, split, classes, {'1': 0, '2': 1, '3': 1})
    row = ferrite_row(internal)
    assert row['internal_split_candidate'] and not row['boundary_leakage_candidate']
    assert set(row['contained_significant_pred_ids']) == {2, 3}
    assert row['neighbor_leakage_pred_ids'] == []
    assert not row['safe_one_to_one_area_eligible'] and full_area(internal)['n'] == 0

    leak = gt.astype(np.uint16); leak[12:32, 28:32] = 1
    boundary = analyze_ferrite_native(gt, leak, classes, {1: 0, 2: 1})
    row = ferrite_row(boundary)
    assert row['boundary_leakage_candidate'] and not row['internal_split_candidate']
    assert row['contained_significant_pred_ids'] == [2]
    assert row['neighbor_leakage_pred_ids'] == [1]
    assert not row['safe_one_to_one_area_eligible'] and full_area(boundary)['n'] == 0
    assert boundary['ff_merge_rows'] == []  # 两个预测ID不等于两个真实F粘连。


def test_adjacent_ferrite_merge_is_distinct_from_ferrite_pearlite_mix():
    gt = board(); gt[12:32, 22:32] = 3
    pred = gt.astype(np.uint16); pred[pred == 3] = 2
    ff = analyze_ferrite_native(gt, pred, {1: 0, 2: 1, 3: 1}, {1: 0, 2: 1})
    assert len(ff['ff_merge_rows']) == 1 and ff['fp_mixed_rows'] == []
    assert set(ff['ff_merge_rows'][0]['gt_ids']) == {2, 3}
    assert ff['ff_merge_rows'][0]['adjacent_gt_pairs'] == [[2, 3]]
    assert not ff['ff_merge_rows'][0]['censored']
    assert full_area(ff)['n'] == 0

    fp = analyze_ferrite_native(gt, pred, {1: 0, 2: 1, 3: 0}, {1: 0, 2: 1})
    assert fp['ff_merge_rows'] == [] and len(fp['fp_mixed_rows']) == 1
    assert full_area(fp)['n'] == 0  # 同F类别投票不使含P的预测变成完整单F。


def test_two_nonadjacent_ferrite_contributions_are_not_reported_as_direct_FF_merge():
    gt = board(); gt[12:32, 23:32] = 3; gt[12:32, 21:23] = 1
    pred = gt.astype(np.uint16); pred[pred == 3] = 2; pred[12:32, 21:23] = 2
    report = analyze_ferrite_native(gt, pred, {1: 0, 2: 1, 3: 1}, {1: 0, 2: 1})
    assert report['ff_merge_rows'] == []  # F之间隔着真实P，不能假定共同界面。
    assert full_area(report)['n'] == 0


def test_small_leak_from_large_F_is_not_promoted_to_complete_FF_merge():
    gt = np.ones((96, 96), np.int32)
    gt[10:70, 10:70] = 2  # 大F3600像素。
    gt[10:30, 70:90] = 3  # 相邻小F400像素。
    pred = gt.astype(np.uint16); pred[10:30, 65:70] = 3
    report = analyze_ferrite_native(gt, pred, {1: 0, 2: 1, 3: 1}, {1: 0, 2: 1, 3: 1})
    # 100像素占预测3的20%，但仅占自身GT2的2.78%；是邻区漏出。
    row = ferrite_row(report)
    assert row['any_crossGT_overlap_candidate'] and row['foreign_neighbor_overlap_candidate']
    assert not row['significant_foreign_neighbor_invasion_candidate']
    assert not row['boundary_leakage_candidate'] and not row['internal_split_candidate']
    assert report['ff_merge_rows'] == []
    assert len(report['ff_cooccupancy_rows']) == 1


def test_geometry_matching_exactly_reuses_existing_known_domain_assignment():
    from tools.probe_semantic_fixed import class_independent_pairs
    gt = board(); split = gt.astype(np.uint16); split[12:32, 22:32] = 3
    unknown = gt.copy(); unknown[18:20, 18:20] = 0
    shrink = gt.astype(np.uint16); shrink[12:32, 30:32] = 0
    for truth, pred, classes in [(gt, split, {1: 0, 2: 1, 3: 1}),
                                  (unknown, split, {1: 0, 2: 1, 3: 1}),
                                  (gt, shrink, {1: 0, 2: 1})]:
        report = analyze_ferrite_native(truth, pred, {1: 0, 2: 1}, classes)
        expected, _ = class_independent_pairs(truth, truth > 0, pred)
        assert report['pairs'] == expected


def test_core_cpu_selftest():
    assert selftest()['passed']


def test_same_parent_partition_with_small_neighbor_spill_is_candidate_not_proven_split():
    gt = np.ones((64, 64), np.int32); gt[10:40, 10:40] = 2
    pred = gt.astype(np.uint16); pred[10:40, 25:40] = 3
    pred[10:12, 40:50] = 3  # 两片各450，第三ID向邻P仅溢20。
    report = analyze_ferrite_native(gt, pred, {1: 0, 2: 1}, {1: 0, 2: 1, 3: 1})
    row = ferrite_row(report)
    assert not row['internal_split_candidate'] and row['same_parent_internal_partition_candidate']
    assert row['neighbor_leakage_pred_ids'] == [3]
    chosen = next(c for c in select_ferrite_cases(report) if c['kind'] == 'internal_split')
    assert chosen['evidence_rule'] == 'same_parent_candidate_not_proven_split'
    contributions = {c['gt']: c for c in next(p for p in report['pred_rows'] if p['pred'] == 3)['contributions']}
    assert contributions[1]['pixels'] == 20 and contributions[2]['pixels'] == 450
    assert contributions[2]['pred_fraction'] == pytest.approx(450/470)


def test_boolean_phase_is_not_accepted_as_integer_class_annotation():
    gt = board(); pred = gt.astype(np.uint16)
    with pytest.raises(ValueError):
        analyze_ferrite_native(gt, pred, {1: 0, 2: True}, {1: 0, 2: 1})


@pytest.mark.parametrize('defect', ['unknown_hole', 'frame_GT', 'frame_pred', 'invalid_content'])
def test_whole_object_censor_excludes_incomplete_F_area(defect):
    gt = board(); pred = gt.astype(np.uint16); content = np.ones(gt.shape, bool)
    if defect == 'unknown_hole':
        gt[18:20, 18:20] = 0  # 预测仍填满洞；unknown面积不能当作可信F。
    elif defect == 'frame_GT':
        gt[:12, 12:32] = 2; pred = gt.astype(np.uint16)
    elif defect == 'frame_pred':
        pred[20, :12] = 2  # GT完整，但预测连到图框，不能用裁切交集冒充完整面积。
    else:
        content[18, 18] = False
    report = analyze_ferrite_native(gt, pred, {1: 0, 2: 1}, {1: 0, 2: 1}, content)
    row = ferrite_row(report)
    if defect != 'frame_pred':
        assert row['censored']
    else:
        assert next(p for p in report['pred_rows'] if p['pred'] == 2)['censored']
    assert not row['safe_one_to_one_area_eligible']
    assert full_area(report)['n'] == 0
    assert full_area(report)['mean_absolute_relative_error'] is None


def test_safe_complete_F_area_uses_full_GT_area_and_preserves_missing_prediction_error():
    gt = board(); pred = gt.astype(np.uint16); pred[12:32, 30:32] = 0
    gt_before, pred_before = gt.copy(), pred.copy(); classes = {1: 0, 2: 1}
    classes_before = deepcopy(classes)
    report = analyze_ferrite_native(gt, pred, classes, classes)
    row = ferrite_row(report); area = full_area(report)
    assert row['safe_one_to_one_area_eligible']
    assert row['area_relative_error'] == pytest.approx(-.1)
    assert area['n'] == 1 and area['paired_gt_pixels'] == 400 and area['paired_pred_pixels'] == 360
    assert area['mean_absolute_relative_error'] == pytest.approx(.1)
    assert area['median_signed_relative_error'] == pytest.approx(-.1)
    assert np.array_equal(gt, gt_before) and np.array_equal(pred, pred_before) and classes == classes_before


def test_semantics_zero_P_one_F_geometry_match_does_not_hide_wrong_vote():
    gt = board(); pred = gt.astype(np.uint16)
    wrong = analyze_ferrite_native(gt, pred, {1: 0, 2: 1}, {1: 0, 2: 0})
    row = ferrite_row(wrong)
    assert row['matched_iou'] == 1 and row['predicted_phase'] == 0 and row['vote_wrong']
    assert not row['internal_split_candidate'] and not row['boundary_leakage_candidate']
    assert not row['safe_one_to_one_area_eligible'] and full_area(wrong)['n'] == 0
    correct = analyze_ferrite_native(gt, pred, {1: 0, 2: 1}, {1: 0, 2: 1})
    assert not ferrite_row(correct)['vote_wrong'] and full_area(correct)['n'] == 1
    assert full_area(correct)['mean_absolute_relative_error'] == 0


def test_empty_F_and_uint16_limit_have_no_invented_area_or_overflow():
    empty = np.zeros((12, 12), np.uint16)
    report = analyze_ferrite_native(empty, empty, {}, {})
    assert report['gt_rows'] == [] and report['pred_rows'] == []
    assert full_area(report)['n'] == 0 and full_area(report)['median_signed_relative_error'] is None
    p_only = np.ones((12, 12), np.uint16)
    assert analyze_ferrite_native(p_only, p_only, {1: 0}, {1: 0})['gt_rows'] == []
    gt = board(); pred = gt.astype(np.uint16); pred[pred == 2] = 65535
    high = analyze_ferrite_native(gt, pred, {1: 0, 2: 1}, {'1': 0, '65535': 1})
    assert ferrite_row(high)['matched_pred'] == 65535 and full_area(high)['n'] == 1
    with pytest.raises(ValueError):
        analyze_ferrite_native(gt, pred.astype(np.int64) + 65535, {1: 0, 2: 1}, {})
    with pytest.raises(ValueError):
        analyze_ferrite_native(gt, pred, {1: 0, 2: 1}, {1: 0})
    with pytest.raises(ValueError):
        analyze_ferrite_native(gt, pred, {1: 0, 2: 1}, {1: 0, 65535: 2})
