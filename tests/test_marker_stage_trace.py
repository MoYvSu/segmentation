# -*- coding: utf-8 -*-
"""虚构全图：阶段重开、未知域绕行、整GT贡献和只读契约。"""
from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from utils.marker_stage_trace import classify_stage_failure, trace_case_stages


def maps(free):
    count, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    return dict(labels=labels, component_count=int(count - 1))


def negative():
    gt = np.zeros((24, 28), np.int32)
    gt[2:22, 2:14] = 7
    gt[2:22, 14:26] = 9
    case = dict(direction='negative', gt_ids=[7, 9], points=[[12, 7], [12, 20]], censored=True)
    return gt, case


def split_map(shape, column=14):
    free = np.ones(shape, bool)
    free[:, column] = False
    return maps(free)


def trace(stage_maps, gt, case, instances=None, markers=None):
    final_labels = list(stage_maps.values())[-1]['labels']
    markers = final_labels.copy() if markers is None else markers
    instances = final_labels.astype(np.uint16) if instances is None else instances
    return trace_case_stages(stage_maps, markers, instances, gt, case)


def test_negative_contiguous_positive_separation_then_merge_is_reopened():
    gt, case = negative()
    stages = {'threshold': split_map(gt.shape), 'skeleton': maps(np.ones(gt.shape, bool))}
    result = trace(stages, gt, case)
    failure = classify_stage_failure(result)
    assert result['stage_order'] == ['threshold', 'skeleton', 'marker_labels', 'instances']
    assert failure['reopened'] and failure['reopened_full'] and failure['reopened_known_target_domain']
    assert failure['full_transitions'][0]['before'] == 'threshold'
    assert failure['full_transitions'][0]['after'] == 'skeleton'
    assert failure['full_transitions'][0]['transition'] == 'merge'
    assert failure['endpoint_evidence_only'] and not failure['physical_error_conclusion']
    assert result['stages'][-1]['labels_are_final_instance_ids']
    assert result['stages'][-2]['labels_are_actual_seed_ids']
    json.dumps(result, allow_nan=False)
    json.dumps(failure, allow_nan=False)


def test_unknown_bypass_has_full_reopen_but_no_known_domain_reopen():
    gt, case = negative()
    cropped_barrier = np.ones(gt.shape, bool)
    cropped_barrier[2:22, 14] = False
    stages = {'closed': split_map(gt.shape), 'retreated': maps(cropped_barrier)}
    # 终态只是独立提供的虚构划分，不让 final free>0 抹掉不同实例身份。
    instances = np.ones(gt.shape, np.uint16)
    instances[:, 14:] = 2
    result = trace(stages, gt, case, instances=instances)
    failure = classify_stage_failure(result)
    assert result['stages'][1]['connected'] is True
    assert result['stages'][1]['known_target_domain']['connected'] is False
    assert failure['reopened_full'] and not failure['reopened_known_target_domain']
    assert not failure['reopened']
    assert result['stages'][-1]['known_target_domain']['connected'] is False


@pytest.mark.parametrize('excluded', [0, 23])
def test_known_domain_excludes_unknown_and_third_gt_detours(excluded):
    gt, case = negative()
    gt[:2, :] = excluded
    free = np.zeros(gt.shape, bool)
    free[0:2, 2:26] = True
    free[2:22, 2:14] = True
    free[2:22, 15:26] = True
    result = trace({'detour': maps(free)}, gt, case)
    row = result['stages'][0]
    assert row['connected'] is True
    assert row['known_all']['connected'] is (excluded == 23)
    assert row['known_target_domain']['connected'] is False


def test_third_gt_reopen_is_distinct_from_target_domain_reopen():
    gt, case = negative()
    gt[:2, :] = 23
    free = np.zeros(gt.shape, bool)
    free[0:2, 2:26] = True
    free[2:22, 2:14] = True
    free[2:22, 15:26] = True
    result = trace({'before': split_map(gt.shape), 'after': maps(free)}, gt, case)
    failure = classify_stage_failure(result)
    assert failure['reopened_full'] and failure['reopened_known_all']
    assert not failure['reopened_known_target_domain'] and not failure['reopened']


def test_zero_endpoint_is_unknown_not_split_and_cannot_create_reopen():
    gt, case = negative()
    barrier = maps(np.ones(gt.shape, bool))
    barrier['labels'][tuple(case['points'][0])] = 0
    result = trace({'on_barrier': barrier, 'free': maps(np.ones(gt.shape, bool))}, gt, case)
    assert result['stages'][0]['connected'] is None
    assert result['stages'][0]['endpoint_relation_goal'] is None
    assert result['stages'][0]['known_target_domain']['connected'] is None
    failure = classify_stage_failure(result)
    assert not failure['reopened_full'] and not failure['reopened']
    assert failure['full_transitions'] == []


def test_only_adjacent_stages_can_be_reopen():
    gt, case = negative()
    on_barrier = maps(np.ones(gt.shape, bool))
    on_barrier['labels'][tuple(case['points'][1])] = 0
    result = trace({'a': split_map(gt.shape), 'b': on_barrier, 'c': maps(np.ones(gt.shape, bool))}, gt, case)
    assert not classify_stage_failure(result)['reopened']


def test_positive_merge_and_split_are_diagnostic_without_reopen_claim():
    gt = np.ones((24, 28), np.int32) * 7
    case = dict(direction='positive', gt_ids=[7], points=[[12, 7], [12, 20]], censored=False)
    result = trace({'a': split_map(gt.shape), 'b': maps(np.ones(gt.shape, bool)), 'c': split_map(gt.shape)}, gt, case)
    failure = classify_stage_failure(result)
    assert [row['transition'] for row in failure['full_transitions']] == ['merge', 'split']
    assert not failure['reopened'] and not failure['reopened_full']
    assert not failure['physical_error_conclusion']


def test_whole_known_contributions_keep_zero_tiny_parts_and_exact_deep_definition():
    gt = np.ones((24, 28), np.int32) * 7
    case = dict(direction='positive', gt_ids=[7], points=[[12, 7], [12, 20]], censored=True)
    labels = np.ones(gt.shape, np.int32)
    labels[:2, :2] = 3
    labels[3, 3] = 0
    labels[:, 16:] = 2
    result = trace({'labels': {'labels': labels}}, gt, case)
    row = result['stages'][0]['gt_rows'][0]
    assert {part['label'] for part in row['contributions']} == {0, 1, 2, 3}
    assert sum(part['pixels'] for part in row['contributions']) == gt.size
    assert row['significant_label_ids'] == [1, 2]
    assert row['zero_pixels'] == 1 and row['dominant_label'] == 1
    assert row['dominant_fraction'] < 1
    expected_deep = cv2.erode((gt == 7).astype(np.uint8), np.ones((9, 9), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    assert row['deep_pixels'] == int(expected_deep.sum())
    assert row['dominant_deep_fraction'] == float((expected_deep & (labels == 1)).sum() / expected_deep.sum())
    assert result['censored'] and result['known_domain_only']
    # 少量 GT 没有深部时不报告 95% 通过，也不丢掉贡献。
    tiny = np.ones((3, 4), np.int32) * 7
    small_case = dict(direction='positive', gt_ids=[7], points=[[1, 0], [1, 3]])
    small = trace({'a': maps(np.ones(tiny.shape, bool))}, tiny, small_case)
    assert small['stages'][0]['gt_rows'][0]['deep_pixels'] == 0
    assert not small['stages'][0]['all_dominant_deep_fraction_at_least_95']


def test_read_only_arrays_and_case():
    gt, case = negative()
    stages = {'a': split_map(gt.shape)}
    markers = stages['a']['labels'].copy()
    instances = markers.astype(np.uint16)
    arrays = [gt, markers, instances, stages['a']['labels']]
    snapshots = [array.copy() for array in arrays]
    old_case = deepcopy(case)
    for array in arrays:
        array.flags.writeable = False
    trace_case_stages(stages, markers, instances, gt, case)
    assert all(np.array_equal(a, old) for a, old in zip(arrays, snapshots))
    assert case == old_case


def test_actual_connectivity_maps_interface_with_synthetic_boundary():
    from tools.semantic_marker_diagnostic import connectivity_maps, marker_stages

    gt, case = negative()
    boundary = np.zeros(gt.shape, np.float32)
    boundary[:, 14] = .9
    config = {'inference': {'boundary_threshold': .5, 'bridge_width': 1,
                           'watershed_dilate_width': 1, 'min_instance_area': 10}}
    stages = marker_stages(boundary, config)
    stage_maps = connectivity_maps(stages)
    result = trace_case_stages(stage_maps, stages['marker_labels'],
                               stages['marker_labels'].astype(np.uint16), gt, case)
    assert result['stage_order'] == list(stage_maps) + ['marker_labels', 'instances']
    assert all(row['connected'] is False for row in result['stages'])
    assert not classify_stage_failure(result)['reopened']


def test_optional_watershed_order_and_full_shape_contract():
    gt, case = negative()
    stages = {'threshold': split_map(gt.shape)}
    markers = stages['threshold']['labels'].copy()
    watershed = np.ones(gt.shape, np.int32)
    instances = watershed.astype(np.uint16)
    result = trace_case_stages(stages, markers, instances, gt, case, watershed=watershed)
    assert result['stage_order'] == ['threshold', 'marker_labels', 'watershed', 'instances']
    assert result['stages'][2]['kind'] == 'actual_watershed_partition'
    assert result['stages'][2]['labels_are_actual_watershed_ids']
    with pytest.raises(ValueError):
        trace_case_stages(stages, markers, instances, gt, case, watershed=watershed[:-1])
    with pytest.raises(ValueError):
        trace_case_stages({'watershed': stages['threshold']}, markers, instances, gt, case)


@pytest.mark.parametrize('change', [
    'float_gt', 'negative_gt', 'float_stage', 'bool_stage', 'cropped_stage',
    'float_markers', 'int32_instances', 'same_gt_negative', 'absent_gt', 'float_gt_id',
    'point_gt_mismatch', 'point_oob', 'point_float', 'point_bool', 'same_points',
    'positive_two_gt', 'empty_stages', 'reserved_name', 'negative_count', 'bool_count',
])
def test_strict_input_contract(change):
    gt, case = negative()
    stages = {'a': split_map(gt.shape)}
    markers = stages['a']['labels'].copy()
    instances = markers.astype(np.uint16)
    if change == 'float_gt': gt = gt.astype(np.float32)
    elif change == 'negative_gt': gt[0, 0] = -1
    elif change == 'float_stage': stages['a']['labels'] = markers.astype(np.float32)
    elif change == 'bool_stage': stages['a']['labels'] = markers > 0
    elif change == 'cropped_stage': stages['a']['labels'] = markers[:-1]
    elif change == 'float_markers': markers = markers.astype(np.float32)
    elif change == 'int32_instances': instances = markers.copy()
    elif change == 'same_gt_negative': case['gt_ids'] = [7, 7]
    elif change == 'absent_gt': case['gt_ids'] = [7, 99]
    elif change == 'float_gt_id': case['gt_ids'] = [7., 9]
    elif change == 'point_gt_mismatch': case['points'][1] = [12, 8]
    elif change == 'point_oob': case['points'][1] = [50, 20]
    elif change == 'point_float': case['points'][1] = [12., 20]
    elif change == 'point_bool': case['points'][1] = [True, 20]
    elif change == 'same_points': case = dict(direction='positive', gt_ids=[7], points=[[12, 7], [12, 7]])
    elif change == 'positive_two_gt': case['direction'] = 'positive'
    elif change == 'empty_stages': stages = {}
    elif change == 'reserved_name': stages = {'instances': stages['a']}
    elif change == 'negative_count': stages['a']['component_count'] = -1
    elif change == 'bool_count': stages['a']['component_count'] = True
    with pytest.raises(ValueError):
        trace_case_stages(stages, markers, instances, gt, case)
