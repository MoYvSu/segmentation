# -*- coding: utf-8 -*-
"""真实分割log-IQR面积过滤；门槛只估计一次，覆盖完整保存路径。"""
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from utils.post_process import _restored_marker_watershed, post_process_prediction_boundary


def _image(shape):
    return np.full((*shape, 3), 128, np.uint8)


def _fixed_quantiles(threshold):
    # 只为隔离回填/保存路径；真实面积分布估计由另外的测试验证。
    return np.array([np.log(threshold), np.log(threshold)])


def test_real_distribution_filters_existing_small_region_without_new_partition():
    shape = (60, 100)
    markers = np.zeros(shape, np.int32)
    markers[1:-1, 1:-1] = 1
    markers[3:7, 3:7] = 2
    markers[10:30, 10:60] = 3
    markers[32:52, 10:60] = 4
    previous = markers.copy()
    audit = {}
    with patch('utils.post_process.np.quantile', wraps=np.quantile) as estimate:
        result = _restored_marker_watershed(_image(shape), markers, np.zeros(shape, np.uint8), 0, 50, 1.5, audit)
    assert estimate.call_count == 1
    assert sorted(np.rint(np.exp(estimate.call_args.args[0])).astype(int).tolist()) == [16, 1000, 1000, 3668]
    assert audit['area_threshold_native'] == 46
    assert audit['area_filter_method'] == 'per_image_log_iqr' and audit['area_filter_scope'] == 'all_instances'
    assert audit['split_parent_count'] == 0
    assert audit['filtered_regions'] == [{'marker_id': 2, 'area': 16, 'pass_index': 1}]
    assert audit['watershed_passes'] == 2 and audit['final_marker_count'] == 3
    assert (result[3:7, 3:7] == 1).all() and not (result == 0).any()
    assert np.array_equal(markers, previous)


def test_estimator_receives_full_watershed_areas_not_tiny_seed_areas():
    shape = (40, 80)
    markers = np.zeros(shape, np.int32)
    markers[20, 20], markers[20, 60] = 1, 2
    audit = {}
    with patch('utils.post_process.np.quantile', return_value=_fixed_quantiles(200)) as estimate:
        result = _restored_marker_watershed(_image(shape), markers, np.zeros(shape, np.uint8), 0, 1, 1.5, audit)
    assert sorted(np.rint(np.exp(estimate.call_args.args[0])).astype(int).tolist()) == [1444, 1482]
    assert int((markers == 1).sum()) == int((markers == 2).sum()) == 1
    assert int((result == 1).sum()) > 200 and int((result == 2).sum()) > 200
    assert not audit['filtered_regions'] and audit['final_marker_count'] == 2


def _region_map(shape, areas):
    result = np.full(shape, -1, np.int32)
    offset = 0
    for iid, area in enumerate(areas, 1):
        result.flat[offset:offset + area] = iid
        offset += area
    assert offset <= result.size
    return result


def _five_seeds(shape):
    markers = np.zeros(shape, np.int32)
    for iid in range(1, 6):
        markers[5, 5 * iid] = iid
    return markers


def test_area_rescaling_preserves_outliers_and_scales_unrounded_iqr_threshold():
    records = []
    for scale in [1, 2]:
        shape = (30 * scale, 40 * scale)
        markers = _five_seeds(shape)
        s = scale ** 2
        outputs = [_region_map(shape, [10 * s] + [100 * s] * 4),
                   _region_map(shape, [0, 110 * s] + [100 * s] * 3)]
        audit = {}
        with patch('utils.post_process.cv2.watershed', side_effect=outputs):
            _restored_marker_watershed(_image(shape), markers, np.zeros(shape, np.uint8), 0, 1, 1.5, audit)
        assert [r['marker_id'] for r in audit['filtered_regions']] == [1]
        initial = audit['initial_watershed']
        raw = np.exp(initial['log_area_q25'] - 1.5 * (initial['log_area_q75'] - initial['log_area_q25']))
        assert 0 <= raw - audit['area_threshold_native'] < 1
        records.append(raw)
    assert records[1] == pytest.approx(4 * records[0])


def test_multiple_pruning_passes_reuse_first_threshold_instead_of_reestimating():
    shape = (30, 40)
    markers = _five_seeds(shape)
    outputs = [_region_map(shape, [10, 100, 100, 100, 100]),
               _region_map(shape, [0, 90, 110, 110, 100]),
               _region_map(shape, [0, 0, 120, 120, 170])]
    audit = {}
    with patch('utils.post_process.cv2.watershed', side_effect=outputs) as watershed:
        with patch('utils.post_process.np.quantile', wraps=np.quantile) as estimate:
            result = _restored_marker_watershed(_image(shape), markers, np.zeros(shape, np.uint8), 0, 1, 1.5, audit)
    assert watershed.call_count == 3 and estimate.call_count == 1
    assert audit['threshold_estimated_once'] and audit['area_threshold_native'] == 100
    assert [(r['marker_id'], r['pass_index']) for r in audit['filtered_regions']] == [(1, 1), (2, 2)]
    assert audit['final_marker_count'] == 3 and np.array_equal(result, outputs[-1])


@pytest.mark.parametrize('shape', [(12, 12), (40, 100), (100, 100)])
def test_single_region_is_not_deleted_due_to_exp_log_roundoff(shape):
    audit = {}
    result = _restored_marker_watershed(_image(shape), np.ones(shape, np.int32), np.zeros(shape, np.uint8), 0, 1, 1.5, audit)
    assert audit['final_marker_count'] == 1 and not audit['filtered_regions']
    assert np.any(result == 1)


def test_equal_sized_regions_have_no_lower_area_outliers():
    shape = (22, 22)
    markers = np.zeros(shape, np.int32)
    markers[1:11, 1:11], markers[1:11, 11:21] = 1, 2
    markers[11:21, 1:11], markers[11:21, 11:21] = 3, 4
    audit = {}
    result = _restored_marker_watershed(_image(shape), markers, np.zeros(shape, np.uint8), 0, 1, 1.5, audit)
    assert audit['final_marker_count'] == 4 and not audit['filtered_regions']
    assert [int((result == iid).sum()) for iid in range(1, 5)] == [100] * 4


def test_filtered_new_child_is_refilled_without_a_hole():
    shape = (40, 100)
    markers = np.zeros(shape, np.int32)
    markers[1:-1, 1:-1] = 1
    source = np.zeros(shape, np.uint8)
    source[:, 94:97] = 1
    audit = {}
    with patch('utils.post_process.np.quantile', return_value=_fixed_quantiles(200)):
        result = _restored_marker_watershed(_image(shape), markers, source, 0, 50, 1.5, audit)
    assert audit['split_parent_count'] == 1 and audit['added_marker_count'] == 1
    assert audit['splits'][0]['children_ids'] == [1, 2] and audit['splits'][0]['core_areas'][1] == 76
    removed = audit['filtered_regions'][0]
    assert removed['marker_id'] == 2 and 76 < removed['area'] < 200
    assert audit['watershed_passes'] == 2 and audit['final_marker_count'] == 1
    assert (result[1:-1, 1:-1] == 1).all() and not (result == 0).any()


def _example_output():
    shape = (100, 100)
    boundary = np.zeros(shape, np.float32)
    cv2.rectangle(boundary, (8, 8), (24, 24), 1, 3)
    output = torch.stack([torch.full(shape, 1.4),
        torch.from_numpy(np.where(boundary > 0, 8., -8.).astype(np.float32))])[None]
    return shape, output


def _full_output(directory, restore=None):
    shape, output = _example_output()
    return post_process_prediction_boundary(output, shape, str(directory), 'synthetic',
        min_instance_area=50, boundary_threshold=.65, watershed_dilate_width=1,
        bridge_width=1, marker_border_seal_width=2, semantic_vote_mode='probability_mean',
        save_visualization=False, marker_partition_restore=restore)


def _assert_saved_contract(directory, instances, classes):
    directory = Path(directory)
    loaded = cv2.imread(str(directory / 'synthetic_inst.png'), cv2.IMREAD_UNCHANGED)
    assert loaded.dtype == np.uint16 and loaded.ndim == 2
    assert np.array_equal(loaded, instances) and int(loaded.max()) <= 65535
    disk_classes = json.loads((directory / 'synthetic_class.json').read_text(encoding='utf-8'))
    assert disk_classes == {str(k): v for k, v in classes.items()}
    assert set(int(k) for k in disk_classes) == set(np.unique(instances)) - {0}
    assert set(disk_classes.values()).issubset({0, 1})
    confidence = json.loads((directory / 'synthetic_class_confidence.json').read_text(encoding='utf-8'))
    assert set(confidence['instances']) == set(disk_classes)
    for iid, details in confidence['instances'].items():
        assert details['area'] == int((instances == int(iid)).sum())
        assert details['class'] == disk_classes[iid]


def test_enabled_config_reaches_save_prediction_and_filtered_artifacts_are_legal(tmp_path):
    from train_backend_adaptation import save_prediction

    shape, output = _example_output()
    config = {'inference': {'boundary_threshold': .65, 'watershed_dilate_width': 1,
        'bridge_width': 1, 'marker_border_seal_width': 2, 'semantic_vote_mode': 'probability_mean',
        'marker_partition_restore': {'enabled': True, 'min_core_area': 50, 'area_iqr_multiplier': 1.5}}}
    with patch('utils.post_process.np.quantile', return_value=_fixed_quantiles(300)):
        instances, classes = save_prediction(np.zeros((*shape, 3), np.uint8), output[:, :1],
            torch.sigmoid(output[:, 1:2]), config, tmp_path, 'synthetic')
    audit_path = tmp_path / 'synthetic_marker_restore.json'
    assert audit_path.is_file()
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['area_filter_method'] == 'per_image_log_iqr' and audit['split_parent_count'] == 0
    assert audit['filtered_region_count'] >= 1 and len(classes) == 1
    assert (instances[1:-1, 1:-1] == 1).all()
    _assert_saved_contract(tmp_path, instances, classes)


def test_default_and_explicit_disabled_keep_complete_outputs_identical(tmp_path):
    with patch('utils.post_process._restored_marker_watershed',
               side_effect=AssertionError('Disabled path must not invoke restoration or filtering')):
        paths_a, instances_a, classes_a = _full_output(tmp_path / 'default')
        paths_b, instances_b, classes_b = _full_output(tmp_path / 'disabled', {'enabled': False, 'area_iqr_multiplier': 0})
    assert np.array_equal(instances_a, instances_b) and classes_a == classes_b
    for key in ['inst_path', 'class_json_path', 'class_confidence_path']:
        assert Path(paths_a[key]).read_bytes() == Path(paths_b[key]).read_bytes()
    assert 'marker_restoration_audit' not in paths_a and 'marker_restoration_audit' not in paths_b
    _assert_saved_contract(tmp_path / 'default', instances_a, classes_a)


@pytest.mark.parametrize('multiplier', [-1, float('nan'), float('inf')])
def test_invalid_iqr_multiplier_is_rejected(multiplier):
    shape = (20, 20)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        _restored_marker_watershed(_image(shape), np.ones(shape, np.int32), np.zeros(shape, np.uint8), 0, 50, multiplier, {})


def test_restoration_only_keeps_small_old_regions_and_runs_watershed_once():
    shape = (60, 100)
    markers = np.zeros(shape, np.int32)
    markers[1:-1, 1:-1] = 1
    markers[3:7, 3:7] = 2
    markers[10:30, 10:60] = 3
    markers[32:52, 10:60] = 4
    expected = cv2.watershed(_image(shape), markers.copy())
    audit = {}
    with patch('utils.post_process.np.quantile', side_effect=AssertionError('Disabled estimator called')):
        result = _restored_marker_watershed(_image(shape), markers, np.zeros(shape, np.uint8),
            0, 50, 1.5, audit, filter_area=False)
    assert np.array_equal(result, expected)
    assert (result[3:7, 3:7] == 2).all()
    assert audit['watershed_passes'] == 1 and not audit['filtered_regions']
    assert audit['area_threshold_native'] is None and audit['area_filter_method'] == 'disabled'
    assert not audit['area_filter_enabled'] and not audit['threshold_estimated_once']


def test_restoration_only_still_recovers_partition_without_repeated_pruning():
    from utils.marker_restoration import restore_marker_partitions
    shape = (40, 100)
    markers = np.zeros(shape, np.int32)
    markers[1:-1, 1:-1] = 1
    source = np.zeros(shape, np.uint8)
    source[:, 94:97] = 1
    seeds, _ = restore_marker_partitions(markers, source, 0, 50)
    expected = cv2.watershed(_image(shape), seeds.copy())
    audit = {}
    result = _restored_marker_watershed(_image(shape), markers, source, 0, 50, 1.5, audit,
        filter_area=False)
    assert np.array_equal(result, expected)
    assert audit['split_parent_count'] == 1 and audit['final_marker_count'] == 2
    assert audit['watershed_passes'] == 1 and audit['filtered_region_count'] == 0


def test_restoration_only_config_reaches_native_decoder_and_audit():
    from tools.affinity_connectivity_views import decode_native
    _, output = _example_output()
    config = {'inference': {'boundary_threshold': .65, 'watershed_dilate_width': 1,
        'bridge_width': 1, 'marker_border_seal_width': 2, 'semantic_vote_mode': 'probability_mean',
        'marker_partition_restore': {'enabled': True, 'min_core_area': 50, 'area_filter_enabled': False}}}
    audit = {}
    with patch('utils.post_process.np.quantile', side_effect=AssertionError('Disabled estimator called')):
        ids, classes = decode_native(output[:, :1], torch.sigmoid(output[:, 1:2]), config, marker_audit=audit)
    assert ids.dtype == np.uint16 and ids.max() <= 65535
    assert set(classes) == set(np.unique(ids)) - {0}
    assert audit['area_filter_method'] == 'disabled' and audit['watershed_passes'] == 1
