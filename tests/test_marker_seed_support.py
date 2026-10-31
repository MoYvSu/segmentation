# -*- coding: utf-8 -*-
"""种子支撑过滤的纯CPU语义测试；无赛题数据、权重或最终实例面积过滤。"""
import json

import numpy as np
import pytest

from utils.marker_seed_support import filter_unsupported_seeds


def test_high_response_hole_is_removed_while_supported_core_keeps_original_id():
    markers = np.zeros((14, 16), np.int32)
    markers[2:5, 2:5] = 7  # 强边界带内的空腔种子。
    markers[7:12, 8:14] = 41
    reference = np.full(markers.shape, .8, np.float32)
    reference[8:11, 9:13] = .2
    before_markers, before_reference = markers.copy(), reference.copy()

    filtered, audit = filter_unsupported_seeds(markers, reference)

    assert np.all(filtered[markers == 7] == 0)
    assert np.array_equal(filtered[markers == 41], markers[markers == 41])
    assert np.array_equal(filtered[markers == 0], markers[markers == 0])
    assert filtered.dtype == np.int32 and filtered.shape == markers.shape
    assert audit['removed_ids'] == [7] and audit['retained_ids'] == [41]
    assert audit['input_seed_count'] == 2 and audit['removed_seed_count'] == 1
    assert audit['removed_marker_pixels'] == 9 and audit['retained_marker_pixels'] == 30
    assert audit['seeds'] == [dict(seed_id=7, area=9, low_support_pixels=0, retained=False),
                              dict(seed_id=41, area=30, low_support_pixels=12, retained=True)]
    assert audit['input_marker_sha256'] != audit['output_marker_sha256']
    assert np.array_equal(markers, before_markers) and np.array_equal(reference, before_reference)
    assert not np.shares_memory(filtered, markers)
    json.dumps(audit, allow_nan=False)


def test_one_inclusive_low_pixel_keeps_entire_seed_without_fraction_or_area_gate():
    markers = np.zeros((9, 11), np.int32)
    markers[1:8, 1:10] = 8
    reference = np.ones(markers.shape, np.float64)
    reference[4, 5] = .45
    filtered, audit = filter_unsupported_seeds(markers, reference, .45)
    assert np.array_equal(filtered, markers)
    assert audit['seeds'][0] == dict(seed_id=8, area=63, low_support_pixels=1, retained=True)
    assert audit['markers_unchanged']
    assert audit['input_marker_sha256'] == audit['output_marker_sha256']


def test_tiny_supported_path_is_not_deleted_by_size():
    markers = np.zeros((8, 10), np.int32)
    markers[4, 1:9] = 300
    reference = np.ones(markers.shape, np.float32)
    reference[4, 1:9] = .1
    filtered, audit = filter_unsupported_seeds(markers, reference)
    assert np.array_equal(filtered, markers)
    assert audit['retained_marker_pixels'] == 8 and audit['removed_ids'] == []


def test_sparse_max_int32_id_does_not_allocate_by_largest_id_or_relabel():
    huge = np.iinfo(np.int32).max
    markers = np.array([[0, huge, 0], [51, huge, 0]], dtype=np.int32)
    reference = np.array([[0, .1, 0], [.9, .9, 0]], dtype=np.float32)
    filtered, audit = filter_unsupported_seeds(markers, reference)
    assert filtered.tolist() == [[0, huge, 0], [0, huge, 0]]
    assert audit['removed_ids'] == [51] and audit['retained_ids'] == [huge]
    # int32种子不是最终uint16 PNG；最终读出另行重编号及限幅验收。
    assert filtered.max() > 65535


def test_low_probability_background_does_not_support_a_positive_seed():
    markers = np.array([[0, 0, 4], [0, 0, 4]], dtype=np.int32)
    reference = np.array([[0, 0, .9], [0, 0, .9]], dtype=np.float32)
    before = markers.copy()
    with pytest.raises(ValueError, match='All positive seeds'):
        filter_unsupported_seeds(markers, reference)
    assert np.array_equal(markers, before)


def test_originally_empty_seed_map_is_independent_identity_copy():
    markers = np.zeros((5, 7), np.int32)
    filtered, audit = filter_unsupported_seeds(markers, np.zeros(markers.shape, np.float32))
    assert np.array_equal(filtered, markers) and not np.shares_memory(filtered, markers)
    assert audit['input_seed_count'] == audit['retained_seed_count'] == audit['removed_seed_count'] == 0
    assert audit['seeds'] == [] and audit['markers_unchanged']


def test_noncontiguous_views_preserve_logical_positions():
    markers = np.array([[0, 9, 9, 0], [0, 12, 12, 0]], np.int32)[:, ::-1]
    reference = np.array([[1, .1, 1, 1], [1, 1, 1, 1]], np.float32)[:, ::-1]
    filtered, audit = filter_unsupported_seeds(markers, reference)
    expected = markers.copy(); expected[expected == 12] = 0
    assert not markers.flags.c_contiguous and np.array_equal(filtered, expected)
    assert audit['removed_ids'] == [12]


def test_disconnected_positions_of_same_id_are_not_split_or_renamed():
    markers = np.array([[5, 0, 5], [0, 0, 0]], np.int32)
    reference = np.array([[.2, 1, .9], [1, 1, 1]], np.float32)
    filtered, audit = filter_unsupported_seeds(markers, reference)
    assert np.array_equal(filtered, markers)
    assert audit['seeds'][0] == dict(seed_id=5, area=2, low_support_pixels=1, retained=True)


@pytest.mark.parametrize('markers', [
    np.zeros((2, 2), np.int64), np.zeros((2, 2), np.uint16),
    np.zeros((2, 2), np.float32), np.zeros((2, 2), bool),
    np.array([[0, -1]], np.int32), np.zeros((0, 2), np.int32),
    np.zeros((2,), np.int32), np.zeros((1, 2, 2), np.int32), [[0, 1]],
])
def test_invalid_seed_contract_is_rejected(markers):
    with pytest.raises(ValueError, match='markers'):
        filter_unsupported_seeds(markers, np.zeros((2, 2), np.float32))


@pytest.mark.parametrize('reference', [
    np.zeros((2, 3), np.float32), np.zeros((2, 2), np.uint8),
    np.zeros((2, 2), bool), np.full((2, 2), np.nan),
    np.full((2, 2), np.inf), np.full((2, 2), -.01),
    np.full((2, 2), 1.01), [[0., .2], [.3, .1]],
])
def test_invalid_reference_contract_is_rejected(reference):
    with pytest.raises(ValueError, match='reference_boundary'):
        filter_unsupported_seeds(np.ones((2, 2), np.int32), reference)


@pytest.mark.parametrize('threshold', [True, np.bool_(False), None, '0.45', -.01, 1.01, np.nan, np.inf])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match='low_threshold'):
        filter_unsupported_seeds(np.ones((2, 2), np.int32), np.zeros((2, 2), np.float32), threshold)


@pytest.mark.parametrize('threshold,reference_value', [(0., 0.), (1., 1.)])
def test_probability_threshold_endpoints_are_valid(threshold, reference_value):
    markers = np.ones((2, 2), np.int32)
    filtered, audit = filter_unsupported_seeds(markers, np.full((2, 2), reference_value), threshold)
    assert np.array_equal(filtered, markers)
    assert audit['seeds'][0]['low_support_pixels'] == 4
