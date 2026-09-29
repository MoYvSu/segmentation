# -*- coding: utf-8 -*-
"""只恢复既有参考分隔的种子；覆盖恒等、弱支持和原种子保留契约。"""
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from utils.marker_restoration import restore_marker_partitions
from utils.post_process import _boundary_skeleton_belt


def test_no_reference_separation_is_exact_identity_and_inputs_are_unchanged():
    markers = np.zeros((80, 120), np.int32)
    markers[3:35, 3:55] = 4
    markers[45:75, 65:115] = 9
    boundary = np.zeros(markers.shape, np.uint8)
    before, source = markers.copy(), boundary.copy()
    restored, audit = restore_marker_partitions(markers, boundary, 1)
    assert np.array_equal(restored, before)
    assert np.array_equal(markers, before) and np.array_equal(boundary, source)
    assert restored is not markers
    assert audit['split_parent_count'] == audit['added_marker_count'] == 0
    assert audit['old_marker_count'] == audit['new_marker_count'] == 2


def test_split_preserves_nontriggering_marker_old_ids_shape_and_zero_background():
    markers = np.zeros((100, 120), np.int32)
    markers[10:40, 10:110] = 3
    markers[60:90, 10:50] = 7
    boundary = np.zeros_like(markers, np.uint8)
    boundary[:, 59:62] = 1
    before, source = markers.copy(), boundary.copy()
    restored, audit = restore_marker_partitions(markers, boundary, 1)
    assert restored.dtype == np.int32 and restored.shape == markers.shape
    assert np.array_equal(restored == 7, markers == 7)
    assert {3, 7}.issubset(set(np.unique(restored)))
    assert set(np.unique(restored)) == {0, 3, 7, 8}
    assert not np.any((markers == 0) & (restored > 0))
    assert np.array_equal(markers, before) and np.array_equal(boundary, source)
    assert audit['split_parent_count'] == 1 and audit['preserved_parent_count'] == 1
    split = audit['splits'][0]
    assert split['parent_id'] == 3 and split['children_ids'] == [3, 8]
    assert len(set(split['reference_ids'])) == 2
    assert split['core_areas'][0] >= split['core_areas'][1]
    for child, area in zip(split['children_ids'], split['core_areas']):
        assert int((restored == child).sum()) == area
    assert split['core_total_area'] + split['released_pixels'] == split['parent_area']
    second, repeated_audit = restore_marker_partitions(markers, boundary, 1)
    assert np.array_equal(restored, second) and audit == repeated_audit


def test_multiple_old_markers_sharing_reference_regions_never_merge():
    markers = np.zeros((100, 120), np.int32)
    markers[10:40, 10:110] = 5
    markers[60:90, 10:110] = 12
    boundary = np.zeros_like(markers, np.uint8)
    boundary[:, 59:62] = 1
    restored, audit = restore_marker_partitions(markers, boundary, 1)
    assert audit['old_marker_count'] == 2 and audit['new_marker_count'] == 4
    assert [item['parent_id'] for item in audit['splits']] == [5, 12]
    assert [item['children_ids'] for item in audit['splits']] == [[5, 13], [12, 14]]
    assert audit['splits'][0]['reference_ids'] == audit['splits'][1]['reference_ids']
    for item in audit['splits']:
        for child in item['children_ids']:
            assert np.all(markers[restored == child] == item['parent_id'])


@pytest.mark.parametrize('all_boundary', [False, True])
def test_insufficient_reference_support_preserves_entire_old_marker(all_boundary):
    markers = np.ones((20, 100), np.int32)
    boundary = np.zeros(markers.shape, np.uint8)
    if all_boundary:
        boundary.fill(1)
    else:
        boundary[:, 96:99] = 1  # 右侧仅20像素，未达到50像素支持。
    restored, audit = restore_marker_partitions(markers, boundary, 0)
    assert np.array_equal(restored, markers)
    assert audit['splits'] == [] and audit['new_marker_count'] == 1


def test_reference_connectivity_is_full_image_not_cropped_to_old_marker():
    markers = np.zeros((100, 100), np.int32)
    markers[20:80, 20:80] = 1
    boundary = np.zeros(markers.shape, np.uint8)
    boundary[:80, 49:52] = 1  # 两侧在旧种子框外仍连通，不应被诊断成两个参考区域。
    restored, audit = restore_marker_partitions(markers, boundary, 0)
    assert np.array_equal(restored, markers)
    assert audit['reference_components'] == 1 and audit['split_parent_count'] == 0


def test_only_largest_connected_overlap_per_reference_is_retained():
    markers = np.zeros((100, 100), np.int32)
    markers[10:80, 52:90] = 1
    markers[10:20, 39:52] = 1
    markers[30:36, 39:52] = 1
    boundary = np.zeros(markers.shape, np.uint8)
    boundary[:, 49:52] = 1
    restored, audit = restore_marker_partitions(markers, boundary, 0)
    assert audit['new_marker_count'] == 2
    assert audit['splits'][0]['core_areas'] == [70 * 38, 100]
    assert not restored[30:36, 39:49].any()
    assert (restored[10:20, 39:49] == 2).all()


def test_disconnected_small_overlaps_are_not_summed_to_pass_support_threshold():
    markers = np.zeros((100, 100), np.int32)
    markers[10:40, 52:90] = 1
    markers[10:14, 39:52] = 1
    markers[20:23, 39:52] = 1
    boundary = np.zeros(markers.shape, np.uint8)
    boundary[:, 49:52] = 1
    restored, audit = restore_marker_partitions(markers, boundary, 0)
    assert np.array_equal(restored, markers)
    assert not audit['splits']


def test_fallback_thick_border_line_recovers_existing_separation_without_new_frame():
    source = np.zeros((200, 240), np.uint8)
    source[:, 113:128] = 1
    with patch.object(cv2, 'ximgproc', SimpleNamespace(), create=True):
        _, belt = _boundary_skeleton_belt(source, 1, 1)
    belt[:2] = belt[-2:] = 255
    belt[:, :2] = belt[:, -2:] = 255
    _, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(belt), connectivity=8)
    markers = np.zeros(source.shape, np.int32)
    next_id = 1
    for iid in range(1, len(stats)):
        if stats[iid, cv2.CC_STAT_AREA] >= 50:
            markers[labels == iid] = next_id
            next_id += 1
    assert next_id == 2
    restored, audit = restore_marker_partitions(markers, source, 1)
    assert audit['old_marker_count'] == 1 and audit['new_marker_count'] == 2
    assert restored[100, 60] > 0 and restored[100, 180] > 0
    assert restored[100, 60] != restored[100, 180]
    assert not restored[:2].any() and not restored[-2:].any()
    assert not restored[:, :2].any() and not restored[:, -2:].any()
    assert np.all(source[:, 113:128] == 1) and not source[:, :113].any()


def test_restoration_does_not_add_a_frame_or_close_a_missing_border_connection():
    markers = np.ones((100, 100), np.int32)
    boundary = np.zeros(markers.shape, np.uint8)
    boundary[8:, 49:52] = 1  # 图顶有实际开口，桥接1像素之后仍然开放。
    restored, audit = restore_marker_partitions(markers, boundary, 1)
    assert audit['reference_components'] == 1
    assert np.array_equal(restored, markers)
    assert (restored[0] == 1).all() and (restored[:, 0] == 1).all()


def test_all_zero_markers_remain_empty():
    markers = np.zeros((30, 30), np.int32)
    boundary = np.zeros(markers.shape, np.uint8)
    boundary[:, 14:16] = 1
    restored, audit = restore_marker_partitions(markers, boundary, 1)
    assert not restored.any()
    assert audit['old_marker_count'] == audit['new_marker_count'] == 0


@pytest.mark.parametrize('markers,boundary,bridge,minimum', [
    (np.ones((4, 4), np.float32), np.zeros((4, 4)), 1, 50),
    (-np.ones((4, 4), np.int32), np.zeros((4, 4)), 1, 50),
    (np.ones((4, 4), np.int32), np.zeros((3, 4)), 1, 50),
    (np.ones((4, 4), np.int32), np.zeros((4, 4)), -1, 50),
    (np.ones((4, 4), np.int32), np.zeros((4, 4)), 1, 0),
])
def test_invalid_inputs_are_rejected(markers, boundary, bridge, minimum):
    with pytest.raises(ValueError):
        restore_marker_partitions(markers, boundary, bridge, minimum)
