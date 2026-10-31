# -*- coding: utf-8 -*-
"""最终面积合并的纯 CPU 契约测试，全部使用合成小图。"""
import json

import cv2
import numpy as np
import pytest

from utils.final_instance_cleanup import merge_final_instances


def example():
    ids = np.zeros((24, 38), np.uint16)
    ids[1:23, 1:24] = 100
    ids[8:12, 25:28] = 5
    ids[1:23, 30:37] = 65535
    return ids, {'100': 1, '5': 0, '65535': 0}


def test_zero_is_independent_identity_and_small_real_instances_are_not_silently_deleted():
    ids, classes = example()
    out, result, audit = merge_final_instances(ids, classes, 0)
    assert np.array_equal(out, ids) and result == classes
    assert not np.shares_memory(out, ids) and out.dtype == np.uint16
    assert audit['merged_count'] == 0 and audit['candidate_small_count'] == 0
    # 同一真实小区在 T50 下应被处理，不能为了“恒等”测试预先删掉。
    changed, _, record = merge_final_instances(ids, classes, 50)
    assert record['merged_ids'] == [5] and not np.array_equal(changed, ids)


def test_one_pixel_seam_connects_target_with_class_inheritance_and_preserves_third_party():
    ids, classes = example()
    original = ids.copy()
    out, result, audit = merge_final_instances(ids, classes, 50)
    assert result == {'100': 1, '65535': 0}
    assert np.all(out[ids == 5] == 100) and np.all(out[8:12, 24] == 100)
    assert np.all(out[:, 29] == 0) and np.array_equal(out[ids == 65535], ids[ids == 65535])
    assert np.array_equal(ids, original) and audit['source_input_unchanged']
    assert np.array_equal(out[ids == 100], ids[ids == 100])
    assert np.array_equal(out[[0, -1]], ids[[0, -1]]) and np.array_equal(out[:, [0, -1]], ids[:, [0, -1]])
    assert out.dtype == np.uint16 and int(out.max()) == 65535
    _, labels = cv2.connectedComponents((out == 100).astype(np.uint8), connectivity=8)
    assert len(set(labels[out == 100])) == 1
    json.dumps(audit, allow_nan=False)


def test_longest_contact_ties_use_larger_area_then_smaller_id_and_not_class():
    ids = np.zeros((15, 16), np.uint16)
    ids[2:12, 1:6] = 2
    ids[2:12, 9:14] = 3
    ids[6, 7] = 7
    for classes in ({'2': 0, '3': 1, '7': 1}, {'2': 1, '3': 0, '7': 0}):
        out, result, audit = merge_final_instances(ids, classes, 10)
        assert out[6, 7] == 2 and audit['decisions'][0]['target_id'] == 2
        assert result['2'] == classes['2'] and '7' not in result


def test_isolated_small_and_small_with_disconnected_piece_are_retained():
    ids = np.zeros((28, 30), np.uint16)
    ids[2:15, 2:15] = 20
    ids[8:10, 16:18] = 7
    ids[24, 24] = 7
    out, classes, audit = merge_final_instances(ids, {'20': 0, '7': 1}, 10)
    assert np.array_equal(out, ids) and classes == {'20': 0, '7': 1}
    assert audit['decisions'][0]['reason'] == 'would_leave_disconnected_ID_or_unfilled_seam'
    ids[8:10, 16:18] = 0
    out, _, audit = merge_final_instances(ids, {'20': 0, '7': 1}, 10)
    assert np.array_equal(out, ids) and audit['merged_count'] == 0
    assert audit['decisions'][0]['reason'] == 'no_direct_retained_large_neighbour'


def test_large_thin_region_and_exact_threshold_region_are_not_filtered_by_width():
    ids = np.zeros((30, 80), np.uint16)
    ids[2:27, 2:4] = 7  # 正好50像素。
    ids[2:27, 5:75] = 8
    out, classes, audit = merge_final_instances(ids, {'7': 1, '8': 0}, 50)
    assert np.array_equal(out, ids) and classes == {'7': 1, '8': 0}
    assert audit['candidate_small_count'] == 0


def test_original_frame_small_region_kept_and_empty_map_supported():
    ids = np.zeros((14, 14), np.uint16)
    ids[0, 2] = 1
    ids[2:12, 1:12] = 2
    out, _, audit = merge_final_instances(ids, {'1': 1, '2': 0}, 10)
    assert np.array_equal(out, ids) and audit['decisions'][0]['reason'] == 'touches_original_image_frame'
    empty = np.zeros((1, 1), np.uint16)
    out, classes, audit = merge_final_instances(empty, {}, 50)
    assert np.array_equal(out, empty) and classes == {} and audit['final_contract_valid']


def test_integer_class_keys_normalize_without_changing_input_mapping():
    ids, _ = example()
    classes = {100: 1, 5: 0, 65535: 0}
    _, result, _ = merge_final_instances(ids, classes, 0)
    assert result == {'100': 1, '5': 0, '65535': 0} and set(classes) == {100, 5, 65535}


@pytest.mark.parametrize('bad', [-1, True, 1.0, '50', None])
def test_invalid_threshold_is_rejected(bad):
    ids, classes = example()
    with pytest.raises(ValueError, match='nonnegative integer'):
        merge_final_instances(ids, classes, bad)


@pytest.mark.parametrize('bad', [{'100': 1, '5': 0}, {'100': True, '5': 0, '65535': 0},
    {'0100': 1, '5': 0, '65535': 0}, {100: 1, '100': 1, '5': 0, '65535': 0}])
def test_invalid_or_ambiguous_classes_are_rejected(bad):
    ids, _ = example()
    with pytest.raises(ValueError):
        merge_final_instances(ids, bad, 50)


def test_invalid_map_and_unknown_action_are_rejected():
    ids, classes = example()
    with pytest.raises(ValueError, match='uint16'):
        merge_final_instances(ids.astype(np.int32), classes, 50)
    with pytest.raises(ValueError, match='Unknown'):
        merge_final_instances(ids, classes, 50, action='drop')
