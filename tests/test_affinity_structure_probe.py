# -*- coding: utf-8 -*-
"""完整结构诊断的真实CPU读出及收益门禁；仅使用虚构小图。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tools.probe_affinity_structure as probe
from tools.probe_affinity_bottleneck import measure


@pytest.fixture
def config():
    return {'inference': dict(boundary_threshold=.65,
        marker_boundary_low_threshold=.45, marker_boundary_reconstruction_steps=8,
        marker_border_seal_width=2, bridge_width=1, watershed_dilate_width=1,
        min_instance_area=50, max_instance_id=65535, threshold=.5,
        semantic_vote_mode='probability_mean', semantic_vote_erode_width=0,
        semantic_vote_threshold=.5)}


def test_pearlite_regression_target_accepts_expected_zero_and_rejects_wrong_class(config):
    gt = np.ones((96, 128), np.uint16)
    boundary = np.full(gt.shape, .03, np.float32)
    semantic = torch.full((1, 1, *gt.shape), -2.)
    raw = semantic.sigmoid()[0, 0].numpy().copy()
    decoded, _, _ = probe.decode_reflect(semantic, boundary, config)
    metrics, votes = measure(decoded, raw, gt, {1: 0})
    case = dict(case_id='synthetic_P', direction='positive', gt_ids=[1],
                points=[[20, 20], [70, 100]], censored=True)
    result = probe.target_result(decoded, gt, np.ones_like(gt, bool), case,
                                 metrics, {1: 0}, votes)
    # P对象不在full_ferrite列表中，仍必须按其真实类别0检查回归。
    assert metrics['full_ferrite']['gt_rows'] == []
    assert result['full_target_pass']
    assert result['matched_gt'][0]['predicted_phase'] == 0
    wrong = probe.target_result(decoded, gt, np.ones_like(gt, bool), case,
        metrics, {1: 0}, {key: 1 for key in votes})
    assert not wrong['full_target_pass']


def test_selection_excludes_receipt_ambiguity_and_already_resolved_reflect():
    cases = [dict(case_id='ambiguous', direction='positive', gt_ids=[48]),
             dict(case_id='resolved', direction='negative', phase_kind='FF', gt_ids=[1, 2]),
             dict(case_id='remaining', direction='negative', phase_kind='FF', gt_ids=[3, 4])]
    rows = [dict(case_id=case['case_id'], full_known_quality=dict(whole_goal=case['case_id']=='resolved'),
                 stage_trace=dict(stages=[dict(known_target_domain=dict(connected=True))]))
            for case in cases]
    record = dict(source=dict(cases=cases, reference_receipt='synthetic_identity.json'),
                  variants=dict(reflect=dict(cases=rows)))
    before = deepcopy(record)
    selected, skipped = probe.select_cases(record, 'interior')
    assert selected == []
    assert skipped == [dict(case_id='ambiguous',
        reason='known_GT_structure_ambiguity_positive_excluded')]
    selected, skipped = probe.select_cases(record, 'interface')
    assert [case['case_id'] for case in selected] == ['remaining']
    assert skipped == [dict(case_id='resolved', reason='already_resolved_by_reflect')]
    assert record == before


def test_marker_only_cross_keeps_real_elevation_but_full_edit_changes_it(config):
    boundary = np.full((96, 128), .03, np.float32)
    edited = boundary.copy()
    edited[:, 60:68] = .9
    semantic = torch.zeros((1, 1, *boundary.shape))
    original, _, _ = probe.decode_reflect(semantic, boundary, config)
    full, _, _ = probe.decode_reflect(semantic, edited, config)
    marker, _, _ = probe.decode_reflect(semantic, edited, config, real_boundary=boundary)
    assert marker['elevation_sha256'] == original['elevation_sha256']
    assert full['elevation_sha256'] != original['elevation_sha256']
    assert np.array_equal(marker['actual_markers'], full['actual_markers'])
    assert not np.array_equal(marker['actual_markers'], original['actual_markers'])
    assert marker['capture_audit']['actual_watershed_calls'] == 1
    assert marker['instances'].dtype == np.uint16
    assert int(marker['instances'].max()) <= 65535


def test_no_new_failures_does_not_hide_measurable_iou_and_area_regression(config):
    def rectangle_boundary(right):
        result = np.full((96, 128), .03, np.float32)
        result[20:24, 30:right] = .9
        result[71:75, 30:right] = .9
        result[20:75, 30:34] = .9
        result[20:75, right-4:right] = .9
        return result

    gt = np.full((96, 128), 2, np.uint16)
    gt[22:73, 32:93] = 1
    raw = np.full(gt.shape, .1, np.float32)
    raw[gt == 1] = .9
    semantic = torch.from_numpy(np.log(raw / (1-raw)))[None, None]
    reference, _, _ = probe.decode_reflect(semantic, rectangle_boundary(95), config)
    before, _ = measure(reference, raw, gt, {1: 1, 2: 0})
    # 只把闭合边界向外移4px，依然一一匹配，却扩大了铁素体面积。
    candidate, _, _ = probe.decode_reflect(semantic, rectangle_boundary(99), config)
    after, _ = measure(candidate, raw, gt, {1: 1, 2: 0}, before)
    safety = probe.safety_metrics(before, after)
    assert safety['no_new_failures']
    assert not safety['no_average_regression']
    assert safety['fixed_matching']['missing_gt_ids'] == []
    assert safety['fixed_iou_decreased_gt'][0]['gt'] == 1
    assert safety['same_fixed_area_MAE']['denominator'] == 1
    assert safety['same_fixed_area_MAE']['delta'] > 0
    assert safety['fixed_area_status']['incomparable_count'] == 0


def test_existing_output_is_rejected_before_receipt_read_or_directory_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'ROOT', tmp_path)
    output = tmp_path / 'outputs' / 'geometry_structure'
    output.mkdir(parents=True)
    status = output / 'status.json'
    status.write_bytes(b'{"status":"complete","protected":true}')
    before = status.read_bytes()
    # input不存在；应先拒绝既有输出，而非加载输入或覆盖其状态。
    with pytest.raises(FileExistsError, match='Fresh ignored outputs'):
        probe.run(SimpleNamespace(output='outputs/geometry_structure', input='does_not_exist.json'))
    assert status.read_bytes() == before
    assert sorted(path.name for path in output.iterdir()) == ['status.json']


def test_unmatched_gt_is_geometry_loss_not_observed_class_flip():
    reference = dict(full_ferrite=dict(gt_rows=[dict(gt=1,matched_pred=2,matched_iou=.8,predicted_phase=1)]))
    candidate = dict(full_ferrite=dict(gt_rows=[dict(gt=1,matched_pred=None,matched_iou=None,predicted_phase=None)]),
        fixed_area=dict(original_fixed_count=0,comparable_count=0,failure_counts={},rows=[]), error_changes={})
    safety = probe.safety_metrics(reference, candidate)
    assert safety['fixed_matching']['missing_gt_ids'] == [1]
    assert safety['newly_wrong_F_class_gt_ids'] == []
    assert safety['fixed_F_class_changes'] == []
    assert safety['class_unassessed_unmatched_gt_ids'] == [1]
    assert not safety['no_new_failures']
