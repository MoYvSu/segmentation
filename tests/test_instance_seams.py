# -*- coding: utf-8 -*-
"""一次原图观察的细缝归属、类别保护和ID重编号反例。"""
from copy import deepcopy

import numpy as np
import pytest

from tools.assign_instance_seams import assign_instance_seams


def test_same_phase_seam_keeps_original_pixels_IDs_classes_and_input_immutable():
    ids = np.full((7, 7), 7, np.uint16)
    ids[:, 4:] = 65535
    ids[:, 3] = 0
    original = ids.copy()
    ids.flags.writeable = False
    classes = {'7': 1, '65535': 1}
    saved_classes = deepcopy(classes)
    candidate, receipt = assign_instance_seams(ids, classes)
    np.testing.assert_array_equal(ids, original)
    np.testing.assert_array_equal(candidate[original > 0], original[original > 0])
    assert classes == saved_classes
    assert candidate.dtype == np.uint16
    assert set(np.unique(candidate[candidate > 0])) == {7, 65535}
    assert receipt['assigned_pixels'] == 5
    assert receipt['assigned_by_phase'] == {'P': 0, 'F': 5}
    assert receipt['zero_partition'] == dict(frame=2, interior_without_positive_neighbor=0,
                                             interior_mixed_phase=0, eligible=5)
    assert candidate[0, 3] == candidate[-1, 3] == 0
    assert receipt['positive_pixels_unchanged'] and receipt['instance_ids_unchanged']
    assert receipt['classes_unchanged'] and receipt['input_sha256'] != receipt['output_sha256']


def test_mixed_phase_seam_is_never_filled_even_when_nearest_is_unambiguous():
    ids = np.full((7, 7), 7, np.uint16)
    ids[:, 4:] = 91
    ids[:, 3] = 0
    candidate, receipt = assign_instance_seams(ids, {7: 0, 91: 1})
    np.testing.assert_array_equal(candidate, ids)
    assert receipt['eligible_pixels'] == receipt['assigned_pixels'] == 0
    assert receipt['zero_partition']['interior_mixed_phase'] == 5
    # 铁相斜邻同样阻止只看四邻珠光体的填补。
    ids = np.zeros((7, 7), np.uint16)
    ids[3, 2] = 7
    ids[2, 4] = 91
    candidate, _ = assign_instance_seams(ids, {7: 0, 91: 1})
    assert candidate[3, 3] == 0


def test_wide_hole_fills_only_original_adjacent_layer_without_propagation():
    ids = np.full((9, 9), 412, np.uint16)
    ids[2:7, 2:7] = 0
    candidate, receipt = assign_instance_seams(ids, {412: 0})
    assert receipt['assigned_pixels'] == 16
    assert receipt['assigned_by_phase'] == {'P': 16, 'F': 0}
    assert receipt['zero_pixels_before'] == 25 and receipt['zero_pixels_after'] == 9
    np.testing.assert_array_equal(candidate[3:6, 3:6], np.zeros((3, 3), np.uint16))
    assert receipt['zero_partition']['interior_without_positive_neighbor'] == 9
    assert receipt['single_pass'] and receipt['original_positive_sources_only']


def test_euclidean_nearest_prefers_cardinal_to_diagonal_without_ID_priority():
    ids = np.zeros((7, 7), np.uint16)
    ids[3, 2] = 65535  # 四邻距离1。
    ids[2, 4] = 7      # 斜邻距离sqrt(2)，ID更小无优先权。
    candidate, _ = assign_instance_seams(ids, {'65535': 1, '7': 1})
    assert candidate[3, 3] == 65535
    assert candidate[0].sum() == candidate[-1].sum() == 0
    assert candidate[:, 0].sum() == candidate[:, -1].sum() == 0


def test_equal_distance_tie_is_invariant_under_instance_ID_renumbering():
    ids = np.zeros((7, 7), np.uint16)
    ids[3, 2] = 9
    ids[3, 4] = 65000
    candidate, receipt = assign_instance_seams(ids, {9: 0, 65000: 0})
    renamed = ids.copy()
    renamed[ids == 9] = 65000
    renamed[ids == 65000] = 9
    changed, changed_receipt = assign_instance_seams(renamed, {65000: 0, 9: 0})
    expected = candidate.copy()
    expected[candidate == 9] = 65000
    expected[candidate == 65000] = 9
    np.testing.assert_array_equal(changed, expected)
    assert candidate[3, 3] in (9, 65000)
    assert receipt['eligible_tied_nearest_pixel_pixels'] > 0
    assert receipt['eligible_pixels'] == changed_receipt['eligible_pixels']
    assert receipt['tie_rule'] == changed_receipt['tie_rule']


def test_disabled_is_exact_noop_and_does_not_call_distance_transform(monkeypatch):
    ids = np.ones((5, 5), np.uint16)
    ids[2, 2] = 0
    def fail(*args, **kwargs):
        raise AssertionError('Disabled assignment must not compute nearest ownership')
    monkeypatch.setattr('tools.assign_instance_seams.distance_transform_edt', fail)
    candidate, receipt = assign_instance_seams(ids, {1: 1}, enabled=False)
    np.testing.assert_array_equal(candidate, ids)
    assert receipt['eligible_pixels'] == 1 and receipt['assigned_pixels'] == 0
    assert receipt['input_sha256'] == receipt['output_sha256']
    candidate[1, 1] = 0
    assert ids[1, 1] == 1


def test_no_positive_or_no_zero_is_safe_noop_without_fake_distance_sources():
    for ids, classes in [(np.zeros((5, 5), np.uint16), {}),
                         (np.full((5, 5), 65535, np.uint16), {'65535': 0}),
                         (np.zeros((1, 7), np.uint16), {})]:
        candidate, receipt = assign_instance_seams(ids, classes)
        np.testing.assert_array_equal(candidate, ids)
        assert receipt['eligible_pixels'] == receipt['assigned_pixels'] == 0
        assert receipt['input_sha256'] == receipt['output_sha256']


@pytest.mark.parametrize('classes', [
    {}, {1: 0, 2: 1}, {0: 0, 1: 0}, {1: 2}, {1: True}, {1: 1.0},
    {1: '1'}, {'-1': 1}, {'1.0': 1}, {True: 1}, {1: 0, '1': 0}, {65536: 1},
])
def test_incomplete_or_malformed_class_tables_are_rejected(classes):
    with pytest.raises(ValueError):
        assign_instance_seams(np.ones((5, 5), np.uint16), classes)


@pytest.mark.parametrize('ids', [np.ones((5, 5), np.int32), np.ones((5, 5), np.float32),
                                np.ones((1, 5, 5), np.uint16), np.empty((0, 5), np.uint16)])
def test_output_contract_rejects_wrong_input_dtype_shape_or_empty_grid(ids):
    with pytest.raises(ValueError, match='uint16'):
        assign_instance_seams(ids, {1: 1})


def test_disabled_still_requires_valid_class_contract_and_boolean_switch():
    ids = np.ones((5, 5), np.uint16)
    with pytest.raises(ValueError, match='Class table'):
        assign_instance_seams(ids, {}, enabled=False)
    with pytest.raises(ValueError, match='boolean'):
        assign_instance_seams(ids, {1: 1}, enabled=1)
