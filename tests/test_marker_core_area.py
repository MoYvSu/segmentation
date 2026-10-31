# -*- coding: utf-8 -*-
"""独立种子核心门槛的CPU契约；不使用赛题数据或最终实例预测。"""
import json

import numpy as np
import pytest

from utils.marker_core_area import rebuild_core_markers


def _boundary_areas():
    labels = np.zeros((11, 20), dtype=np.int32)
    flat = labels.ravel()
    start = 3
    for label, area in [(1, 24), (2, 25), (3, 49), (4, 50), (5, 60)]:
        flat[start:start + area] = label
        start += area + 1
    return labels


def test_default_fifty_exactly_matches_legacy_sorted_source_label_loop():
    labels = _boundary_areas()
    legacy = np.zeros_like(labels, dtype=np.int32)
    next_id = 0
    for label in range(1, int(labels.max()) + 1):
        if int(np.count_nonzero(labels == label)) < 50:
            continue
        next_id += 1
        legacy[labels == label] = next_id

    markers, audit = rebuild_core_markers(labels)

    assert np.array_equal(markers, legacy)
    assert markers.dtype == np.int32 and markers.shape == labels.shape
    assert audit['retained_core_count'] == 2 and audit['reference_core_count'] == 2
    assert audit['added_cores'] == [] and audit['removed_cores'] == []
    assert audit['reference_selection_unchanged']


def test_twenty_five_only_adds_twenty_five_through_forty_nine_cores():
    labels = _boundary_areas()
    control, _ = rebuild_core_markers(labels, 50)
    candidate, audit = rebuild_core_markers(labels, 25, reference_min_core_area=50)

    assert np.all(candidate[labels == 1] == 0)  # 面积24仍然被排除。
    assert [(row['source_label'], row['area']) for row in audit['added_cores']] == [(2, 25), (3, 49)]
    assert audit['added_core_count'] == 2 and audit['added_core_pixels'] == 74
    assert audit['retained_core_count'] == 4 and audit['removed_core_count'] == 0
    assert np.array_equal(candidate > 0, (control > 0) | (labels == 2) | (labels == 3))
    assert np.unique(candidate).tolist() == [0, 1, 2, 3, 4]
    rows = {row['source_label']: row for row in audit['cores']}
    assert rows[4]['marker_id'] == 3 and rows[4]['reference_marker_id'] == 1
    assert rows[5]['marker_id'] == 4 and rows[5]['reference_marker_id'] == 2
    assert not audit['final_instance_filter_applied']
    assert not audit['final_instance_filter_modified']
    json.dumps(audit, allow_nan=False)


def test_zero_background_never_becomes_seed_even_if_larger_than_threshold():
    labels = np.zeros((30, 30), dtype=np.int64)
    labels[3:8, 4:9] = 8
    markers, audit = rebuild_core_markers(labels, 25)
    assert np.all(markers[labels == 0] == 0)
    assert np.all(markers[labels == 8] == 1)
    assert audit['source_core_count'] == 1 and audit['cores'][0]['source_label'] == 8
    assert audit['zero_background_preserved']


def test_sparse_source_ids_keep_ascending_legacy_order_without_huge_lookup():
    labels = np.array([[2**40, 0, 7], [2**40, 0, 7]], dtype=np.int64)
    markers, audit = rebuild_core_markers(labels, np.int64(2), reference_min_core_area=2)
    assert markers.tolist() == [[2, 0, 1], [2, 0, 1]]
    assert [row['source_label'] for row in audit['cores']] == [7, 2**40]


def test_core_positions_are_preserved_without_recomputing_components():
    # 相同源label即使局部显示为不相邻，也不在此处重新拆分；调用方必须传整图。
    labels = np.array([[3, 0, 3], [0, 0, 0]], dtype=np.int16)
    markers, audit = rebuild_core_markers(labels, 2)
    assert markers.tolist() == [[1, 0, 1], [0, 0, 0]]
    assert audit['requires_complete_native_core_labels']


@pytest.mark.parametrize('shape', [(0, 0), (0, 7), (8, 0), (4, 6)])
def test_empty_or_all_zero_map_returns_independent_int32_zero_map(shape):
    labels = np.zeros(shape, dtype=np.uint32)
    markers, audit = rebuild_core_markers(labels, 25)
    assert markers.shape == shape and markers.dtype == np.int32
    assert markers.size == 0 or not markers.any()
    assert not np.shares_memory(markers, labels)
    assert audit['source_core_count'] == 0 and audit['retained_core_count'] == 0
    assert audit['reference_selection_unchanged'] and audit['cores'] == []


def test_read_only_noncontiguous_input_is_not_modified_or_aliased():
    labels = _boundary_areas()[:, ::-1]
    before = labels.copy()
    labels.flags.writeable = False
    markers, _ = rebuild_core_markers(labels, 25)
    assert np.array_equal(labels, before)
    assert not np.shares_memory(markers, labels)
    assert markers.flags.c_contiguous and markers.flags.writeable


def test_larger_candidate_threshold_reports_removed_reference_cores():
    labels = _boundary_areas()
    markers, audit = rebuild_core_markers(labels, 60)
    assert np.all(markers[labels == 4] == 0) and np.all(markers[labels == 5] == 1)
    assert audit['removed_core_count'] == 1
    assert audit['removed_cores'][0]['source_label'] == 4
    assert audit['removed_core_pixels'] == 50 and audit['added_core_count'] == 0


@pytest.mark.parametrize('bad', [True, np.bool_(False), 0, -1, 25.0, np.nan, '25', None])
@pytest.mark.parametrize('name', ['min_core_area', 'reference_min_core_area'])
def test_non_positive_or_non_integer_thresholds_are_rejected(bad, name):
    with pytest.raises(ValueError, match=name):
        rebuild_core_markers(np.zeros((2, 3), np.int32), **{name: bad})


@pytest.mark.parametrize('bad', [
    [[0, 1]], np.zeros(3, np.int32), np.zeros((2, 3, 1), np.int32),
    np.array([[0., 1.]], np.float32), np.array([[True, False]]),
    np.array([[0, -1]], np.int32), np.array([[1]], dtype=object),
])
def test_invalid_core_map_contract_is_rejected(bad):
    with pytest.raises(ValueError, match='nonnegative integer HW'):
        rebuild_core_markers(bad, 25)
