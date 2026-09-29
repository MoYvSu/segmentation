# -*- coding: utf-8 -*-
"""核对只读种子诊断与真实后处理传入 watershed 的数组一致。"""
from unittest.mock import patch
from contextlib import nullcontext
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from tools.semantic_marker_diagnostic import anchored_marker_stages, component_seed_relation, connectivity_maps, connectivity_stages, marker_stages
from utils.affinity_deployment import postprocess, probability_to_logit


def _config(reconstruct=True, seal=2):
    return {'inference': {
        'boundary_threshold': .65,
        'marker_boundary_low_threshold': .45 if reconstruct else None,
        'marker_boundary_reconstruction_steps': 8,
        'marker_border_seal_width': seal,
        'min_instance_area': 50,
        'semantic_vote_mode': 'probability_mean',
    }}


@pytest.mark.parametrize('reconstruct,seal,content', [
    (True, 2, 'lines'), (False, 2, 'lines'),
    (True, 0, 'lines'), (False, 0, 'lines'),
    (True, 2, 'empty'), (True, 2, 'full'),
])
def test_markers_and_barrier_match_actual_postprocess(tmp_path, reconstruct, seal, content):
    boundary = np.full((128, 144), .03, np.float32)
    if content == 'lines':
        boundary[:, 70:73] = .9
        boundary[53:63, 70:73] = .52
        boundary[70:73, :] = .9
        boundary[70:73, 45:49] = .52
    elif content == 'empty':
        boundary.fill(0)
    else:
        boundary.fill(1)
    config = _config(reconstruct, seal)
    stages = marker_stages(boundary, config)
    captured = []
    original = cv2.watershed

    def capture(image, markers):
        captured.append((image.copy(), markers.copy()))
        return original(image, markers)

    output = torch.cat([
        torch.zeros(1, 1, *boundary.shape),
        probability_to_logit(torch.from_numpy(boundary)[None, None]),
    ], dim=1)
    with patch.object(cv2, 'watershed', side_effect=capture):
        _, instances, _ = postprocess(output, boundary.shape, tmp_path, 'synthetic',
            config['inference'], .65, False)
    if stages['marker_count']:
        assert len(captured) == 1
        assert np.array_equal(captured[0][1], stages['marker_labels'])
        image = np.full((*boundary.shape, 3), 128, np.uint8)
        overlay = image.copy()
        overlay[stages['barrier_belt'] > 0] = 255
        expected = cv2.addWeighted(image, .7, overlay, .3, 0)
        assert np.array_equal(captured[0][0], expected)
    else:
        assert not captured and not instances.any()


def test_component_relation_distinguishes_same_different_and_unseeded_regions():
    mask = np.ones((100, 100), bool)
    probability = np.full(mask.shape, .1, np.float32)
    probability[:, :50] = .9
    markers = np.ones(mask.shape, np.int32)
    same = component_seed_relation(mask, probability, markers)
    assert same['same_seed'] is True
    assert same['ferrite_component']['dominant_fraction'] == 1
    markers[:, 50:] = 2
    assert component_seed_relation(mask, probability, markers)['same_seed'] is False
    markers[:50] = 0
    partly_unseeded = component_seed_relation(mask, probability, markers)
    assert partly_unseeded['same_seed'] is None
    for name in ['ferrite_component', 'pearlite_component']:
        assert partly_unseeded[name]['unseeded_pixels'] > 0
        assert partly_unseeded[name]['dominant_fraction'] == .5


def test_no_opposite_strong_component_does_not_claim_two_regions_share_seed():
    mask = np.ones((100, 100), bool)
    relation = component_seed_relation(mask, np.full(mask.shape, .9, np.float32),
                                      np.ones(mask.shape, np.int32))
    assert relation['same_seed'] is None
    assert relation['pearlite_component']['area'] == 0


@pytest.mark.parametrize('setting,value', [('center_seeds', True),
    ('sem_edge_merge_weight', .1), ('sem_edge_boost_alpha', .1)])
def test_diagnostic_rejects_out_of_scope_deployment(setting, value):
    config = _config()
    config['inference'][setting] = value
    with pytest.raises(ValueError):
        marker_stages(np.zeros((100, 100), np.float32), config)


@pytest.mark.parametrize('gap_value,reconstructed_connected', [(.52, False), (.03, True)])
def test_fixed_points_follow_reconstructed_weak_gap_or_unrepaired_missing_gap(gap_value, reconstructed_connected):
    mask = np.ones((128, 144), bool)
    probability = np.full(mask.shape, .1, np.float32)
    probability[:, :72] = .9
    boundary = np.full(mask.shape, .03, np.float32)
    boundary[:, 70:73] = .9
    boundary[53:63, 70:73] = gap_value
    stages = marker_stages(boundary, _config())
    report = connectivity_stages(stages, mask, probability)
    assert report['qualifying_components']
    assert report['stages']['high_mask']['same_component'] is True
    for stage in ['marker_boundary_mask', 'bridged_marker_mask', 'marker_belt']:
        assert report['stages'][stage]['same_component'] is reconstructed_connected
    for phase, point in report['points'].items():
        assert point['selected_from_positive_marker']
        assert stages['marker_labels'][point['y'], point['x']] == point['final_marker_id'] > 0
        assert probability[point['y'], point['x']] == (np.float32(.9) if phase == 'ferrite' else np.float32(.1))


def test_cached_connectivity_matches_uncached_and_preserves_global_point_coordinates():
    mask = np.zeros((200, 220), bool)
    mask[30:130, 40:140] = True
    probability = np.full(mask.shape, .1, np.float32)
    probability[:, :90] = .9
    boundary = np.zeros(mask.shape, np.float32)
    boundary[:, 89:92] = .9
    stages = marker_stages(boundary, _config())
    expected = connectivity_stages(stages, mask, probability)
    maps = connectivity_maps(stages)
    assert list(maps) == ['high_mask', 'marker_boundary_mask', 'bridged_marker_mask',
                          'skeleton_only', 'dilated_skeleton_before_seal', 'marker_belt']
    # 命中缓存后不再调用整图连通域计算；实例连通块使用独立的 WithStats 接口。
    with patch.object(cv2, 'connectedComponents', side_effect=AssertionError('cache not reused')):
        actual = connectivity_stages(stages, mask, probability, maps=maps)
    assert actual == expected
    for point in actual['points'].values():
        assert 40 <= point['x'] < 140 and 30 <= point['y'] < 130
        assert mask[point['y'], point['x']]
        assert stages['marker_labels'][point['y'], point['x']] == point['final_marker_id']


def test_preseal_stage_matches_final_when_border_seal_disabled():
    boundary = np.zeros((128, 144), np.float32)
    boundary[:, 70:73] = .9
    maps = connectivity_maps(marker_stages(boundary, _config(seal=0)))
    assert np.array_equal(maps['dilated_skeleton_before_seal']['labels'], maps['marker_belt']['labels'])
    assert maps['dilated_skeleton_before_seal']['component_count'] == maps['marker_belt']['component_count']


@pytest.mark.parametrize('backend', ['native', 'fallback'])
def test_preseal_prevents_fallback_border_tip_recession_without_changing_real_barrier(backend):
    if backend == 'native' and not hasattr(getattr(cv2, 'ximgproc', None), 'thinning'):
        pytest.skip('Native ximgproc thinning is unavailable')
    scope = (patch.object(cv2, 'ximgproc', SimpleNamespace(), create=True)
             if backend == 'fallback' else nullcontext())
    with scope:
        boundary = np.full((200, 240), .03, np.float32)
        boundary[:, 113:128] = .9
        original = marker_stages(boundary, _config())
        snapshot = {key: value.copy() for key, value in original.items() if isinstance(value, np.ndarray)}
        anchored = anchored_marker_stages(original)
    if backend == 'fallback':
        # 同一条贯穿上下图边的粗边界，在原skimage末端收缩后可从两端绕行。
        assert original['marker_count'] == 1
        assert original['marker_labels'][100, 60] == original['marker_labels'][100, 180] > 0
    assert anchored['marker_count'] == 2
    left, right = anchored['marker_labels'][100, 60], anchored['marker_labels'][100, 180]
    assert left > 0 and right > 0 and left != right
    assert anchored['diagnostic_only'] and not anchored['real_barrier_changed']
    assert anchored['barrier_belt'] is original['barrier_belt']
    assert 'diagnostic_variant' not in original['resolved']
    for key, previous in snapshot.items():
        assert np.array_equal(original[key], previous)
