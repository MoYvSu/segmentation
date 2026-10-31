# -*- coding: utf-8 -*-
"""面积拆分：旧输出兼容、核心与终态独立、reflect/replay生产路径传递。"""
from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from tools.affinity_connectivity_views import decode_native
from tools.probe_affinity_marker_replay import build_marker_variant, decode_marker_replay
from tools.semantic_marker_diagnostic import marker_stages
from utils.affinity_deployment import postprocess
from utils.instance_area_policy import (
    has_explicit_area_policy, instance_area_kwargs, resolve_instance_area_policy,
)
from utils.post_process import boundary_watershed_separation, post_process_prediction_boundary


def sample():
    config = {'inference': dict(
        boundary_threshold=.65, marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8, marker_border_seal_width=2,
        bridge_width=1, watershed_dilate_width=1, min_instance_area=50,
        max_instance_id=65535, threshold=.5, semantic_vote_mode='probability_mean')}
    boundary = np.full((96, 112), .03, np.float32)
    boundary[:, 69:73] = .95
    cv2.rectangle(boundary, (12, 12), (34, 34), .95, 3)
    semantic = torch.linspace(-2, 2, 112)[None, None, None].expand(1, 1, 96, 112).clone()
    return semantic, torch.from_numpy(boundary)[None, None], config


@pytest.mark.parametrize('legacy', [50, 73, '73', np.int64(73), 0])
def test_alias_independently_supplies_each_missing_threshold(legacy):
    base = {'min_instance_area': legacy}
    resolved = resolve_instance_area_policy(base)
    assert resolved == dict(seed_min_area=int(legacy), final_min_instance_area=int(legacy),
                            final_small_instance_action='legacy_drop',
                            pre_vote_final_min_area=int(legacy))
    assert instance_area_kwargs(base) == {} and not has_explicit_area_policy(base)
    assert resolve_instance_area_policy({**base, 'seed_min_area': 12})['final_min_instance_area'] == int(legacy)
    assert resolve_instance_area_policy({**base, 'final_min_instance_area': 200})['seed_min_area'] == int(legacy)


def test_default_and_explicit_values_do_not_modify_input():
    assert resolve_instance_area_policy()['seed_min_area'] == 50
    source = dict(min_instance_area=71, seed_min_area=8, final_min_instance_area=0,
                  final_small_instance_action='neighbor_merge_v1')
    before = source.copy()
    assert resolve_instance_area_policy(source) == dict(
        seed_min_area=8, final_min_instance_area=0,
        final_small_instance_action='neighbor_merge_v1', pre_vote_final_min_area=0)
    assert instance_area_kwargs(source) == {key: value for key, value in source.items()
                                           if key != 'min_instance_area'}
    assert source == before


@pytest.mark.parametrize('key,bad', [
    ('seed_min_area', 0), ('seed_min_area', -1), ('seed_min_area', True),
    ('seed_min_area', 50.), ('seed_min_area', '50'),
    ('final_min_instance_area', -1), ('final_min_instance_area', False),
    ('final_min_instance_area', 200.), ('final_min_instance_area', '200'),
    ('final_small_instance_action', 'merge'), ('final_small_instance_action', False),
    ('min_instance_area', -1), ('min_instance_area', float('inf')),
])
def test_explicit_illegal_areas_or_actions_fail(key, bad):
    with pytest.raises(ValueError):
        resolve_instance_area_policy({key: bad})


def test_new_50_50_matches_legacy_whole_decode_and_marker_replay():
    semantic, boundary, config = sample()
    original = decode_marker_replay(semantic, boundary, config)
    explicit = deepcopy(config)
    explicit['inference'].update(seed_min_area=50, final_min_instance_area=50)
    result = decode_marker_replay(semantic, boundary, explicit)
    for field in ('instances', 'actual_markers', 'watershed'):
        assert np.array_equal(result[field], original[field])
    assert result['classes'] == original['classes']
    assert result['hashes'] == original['hashes']
    assert 'seed_min_area' not in original['stages']['resolved']
    assert result['stages']['resolved']['seed_min_area'] == 50


@pytest.mark.parametrize('mode', ['preseal', 'nothin', 'reflect'])
def test_final_threshold_never_changes_seed_variants_or_ws_elevation(mode):
    semantic, boundary, config = sample()
    base = marker_stages(boundary[0, 0].numpy(), config)
    original_marker = build_marker_variant(base, mode)['marker_labels']
    explicit = deepcopy(config)
    explicit['inference'].update(seed_min_area=50, final_min_instance_area=100000,
                                 final_small_instance_action='legacy_drop')
    changed = marker_stages(boundary[0, 0].numpy(), explicit)
    selected = build_marker_variant(changed, mode)
    assert np.array_equal(selected['marker_labels'], original_marker)
    before = decode_marker_replay(semantic, boundary, config, original_marker)
    after = decode_marker_replay(semantic, boundary, explicit, selected['marker_labels'])
    assert np.array_equal(before['actual_markers'], after['actual_markers'])
    assert np.array_equal(before['watershed'], after['watershed'])
    assert before['elevation_sha256'] == after['elevation_sha256']
    assert np.count_nonzero(before['instances']) > 0
    assert np.count_nonzero(after['instances']) == 0 and after['classes'] == {}


@pytest.mark.parametrize('mode', ['preseal', 'nothin', 'reflect'])
def test_explicit_seed_threshold_reaches_every_variant(mode):
    _, boundary, config = sample()
    config['inference'].update(seed_min_area=600, final_min_instance_area=1)
    stage = build_marker_variant(marker_stages(boundary[0, 0].numpy(), config), mode)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(stage['cores'], connectivity=8)
    expected = np.zeros(labels.shape, np.int32)
    count = 0
    assert any(50 <= int(area) < 600 for area in stats[1:, cv2.CC_STAT_AREA])
    for label in range(1, num):
        if stats[label, cv2.CC_STAT_AREA] >= 600:
            count += 1
            expected[labels == label] = count
    assert np.array_equal(stage['marker_labels'], expected)
    assert stage['marker_count'] == count


def test_neighbor_merge_keeps_all_regions_until_outer_raw_vote_cleanup():
    semantic, boundary, config = sample()
    original = decode_marker_replay(semantic, boundary, config)
    explicit = deepcopy(config)
    explicit['inference'].update(seed_min_area=50, final_min_instance_area=100000,
                                 final_small_instance_action='neighbor_merge_v1')
    result = decode_marker_replay(semantic, boundary, explicit)
    assert np.array_equal(original['instances'], result['instances'])
    assert original['classes'] == result['classes']
    assert result['stages']['resolved']['final_min_instance_area'] == 100000
    assert result['stages']['resolved']['pre_vote_final_min_area'] == 0


def test_small_ws_region_is_not_zeroed_before_deferred_cleanup():
    # 人造WS输出隔离“最终区域过滤”，不把种子面积和WS面积混成同一断言。
    ws = np.ones((32, 32), np.int32)
    ws[4:7, 4:7] = 2
    semantic = np.ones(ws.shape, np.uint8)
    boundary = np.zeros(ws.shape, np.uint8)
    with patch('utils.post_process.cv2.watershed', side_effect=lambda image, markers: ws.copy()):
        legacy, _ = boundary_watershed_separation(semantic, boundary, min_area=50)
        low, _ = boundary_watershed_separation(semantic, boundary, min_area=50,
                                               seed_min_area=50, final_min_instance_area=0)
        deferred, _ = boundary_watershed_separation(semantic, boundary, min_area=50,
            seed_min_area=50, final_min_instance_area=200,
            final_small_instance_action='neighbor_merge_v1')
    assert np.all(legacy[4:7, 4:7] == 0)
    assert np.all(low[4:7, 4:7] > 0)
    assert np.array_equal(low, deferred)
    assert deferred.dtype == np.uint16


def test_restoration_fallback_uses_seed_area_and_rejects_unsupported_merge():
    mask = np.ones((32, 32), np.uint8)
    with patch('utils.post_process._restored_marker_watershed',
               return_value=mask.astype(np.int32)) as restored:
        boundary_watershed_separation(mask, np.zeros_like(mask), seed_min_area=77,
            final_min_instance_area=1, marker_partition_restore={'enabled': True})
    assert restored.call_args.args[4] == 77
    with pytest.raises(ValueError, match='requires disabled marker partition'):
        boundary_watershed_separation(mask, np.zeros_like(mask),
            marker_partition_restore={'enabled': True},
            final_small_instance_action='neighbor_merge_v1')


def test_postprocess_file_entry_forwards_explicit_fields_and_preserves_old_call(tmp_path):
    captured = []

    def fake(semantic, boundary, **kwargs):
        captured.append(kwargs)
        return np.zeros(semantic.shape, np.uint16), {}

    with patch('utils.post_process.boundary_watershed_separation', side_effect=fake):
        for fields in ({}, dict(seed_min_area=50, final_min_instance_area=200,
                                 final_small_instance_action='neighbor_merge_v1')):
            post_process_prediction_boundary(torch.zeros(1, 2, 32, 32), (32, 32),
                str(tmp_path), 'synthetic', save_visualization=False, **fields)
    assert 'seed_min_area' not in captured[0]
    for key, value in fields.items():
        assert captured[1][key] == value


def test_shared_deployment_forwards_both_thresholds(tmp_path):
    inference = dict(seed_min_area=50, final_min_instance_area=200,
                     final_small_instance_action='neighbor_merge_v1')
    with patch('utils.affinity_deployment.post_process_prediction_boundary') as target:
        postprocess(None, (32, 32), tmp_path, 'synthetic', inference, .65, False)
    for key, value in inference.items():
        assert target.call_args.kwargs[key] == value


def test_decode_native_propagates_seed_area_to_actual_core_capture():
    semantic, boundary, config = sample()
    config['inference'].update(seed_min_area=600, final_min_instance_area=1)
    stage = marker_stages(boundary[0, 0].numpy(), config)
    selected = decode_marker_replay(semantic, boundary, config)
    assert np.array_equal(selected['actual_markers'], stage['marker_labels'])
    assert selected['capture_audit']['original_marker_exact']
    direct, classes = decode_native(semantic, boundary, config)
    assert np.array_equal(selected['instances'], direct) and selected['classes'] == classes
