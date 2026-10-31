# -*- coding: utf-8 -*-
"""缓存预检的来源资格和完整seed/内核判定，不运行真实数据或网络。"""
import numpy as np

from tools.probe_affinity_gap_cut import (
    choose_interface_views, compare_core_assignments, core_assignments,
    crop_geometry, seed_assignment, fixed_views_for_source,
)


def test_views_require_original_both_sides_and_valid_contact():
    ids = np.ones((32, 32), np.uint16)
    ids[:, 16:] = 2
    original = np.ones_like(ids, bool)
    boundary = np.full(ids.shape, .4, np.float32)
    views = choose_interface_views(ids, original, original, boundary, size=24, limit=2)
    assert len(views) == 1 and views[0]['gt_pair'] == [1, 2]
    original[:, 16:] = False
    assert not choose_interface_views(ids, original, ids > 0, boundary, size=24)
    original[:] = True
    valid = np.ones_like(original); valid[:, 15:17] = False
    assert not choose_interface_views(ids, original, valid, boundary, size=24)


def test_view_selection_is_fixed_and_does_not_modify_inputs():
    ids = np.ones((40, 80), np.uint16)
    ids[:, 25:55] = 2; ids[:, 55:] = 3
    boundary = np.full(ids.shape, .8, np.float32); boundary[:, 54:56] = .1
    before = boundary.copy(); original = np.ones_like(ids, bool)
    a = choose_interface_views(ids, original, original, boundary, size=16, limit=1)
    b = choose_interface_views(ids, original, original, boundary, size=16, limit=1)
    assert a == b and a[0]['gt_pair'] == [2, 3]
    assert np.array_equal(boundary, before)


def test_seed_assignment_requires_both_nonzero_disjoint_full_sets():
    labels = np.array([[1, 2], [2, 0]], np.int32)
    first = np.array([[1, 1], [0, 0]], bool)
    second = np.array([[0, 0], [1, 1]], bool)
    a = seed_assignment(labels, first, second)
    assert not a['separated'] and a['shared_ids'] == [2]
    labels[1, 0] = 3
    assert seed_assignment(labels, first, second)['separated']
    labels[1] = 0
    assert not seed_assignment(labels, first, second)['separated']


def test_known_core_safety_tracks_new_fragments_and_lost_adjacencies():
    core = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], np.int64)
    old = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], np.uint16)
    new = np.array([[1, 3, 1, 1], [1, 3, 1, 1]], np.uint16)
    before = core_assignments(old, core, minimum=1)
    after = core_assignments(new, core, minimum=1)
    result = compare_core_assignments(before, after, [(1, 2)])
    assert result['additional_known_core_fragment_GT'] == [1]
    assert result['lost_known_core_adjacent_separations'] == [[1, 2]]
    assert not compare_core_assignments(before, before, [(1, 2)])['lost_known_core_adjacent_separations']


def test_geometry_crop_keeps_native_scalar_metadata_and_values():
    geometry = {'ids': np.arange(48).reshape(6, 8), 'scale': 1., 'identity': 'static'}
    result = crop_geometry(geometry, [1, 2, 4, 6])
    assert result['scale'] == 1. and result['identity'] == 'static'
    assert result['ids'].shape == (3, 4)
    assert np.array_equal(result['ids'], geometry['ids'][1:4, 2:6])


def test_fixed_views_reuse_exact_coordinates_pair_and_order():
    case = dict(source='toy', view=0, gt_pair=[8, 9], center_yx=[10, 15], box_yxyx=[2, 3, 20, 25],
                OO_contact_pixels=24, OO_contact_mean_B=.3)
    result = fixed_views_for_source([case], 'toy', (32, 32))
    assert len(result) == 1 and result[0]['box_yxyx'] == case['box_yxyx']
    assert result[0]['gt_pair'] == [8, 9]
    assert case['view'] == 0 and case['source'] == 'toy'


def test_fixed_views_reject_shape_or_order_drift():
    import pytest
    case = dict(source='toy', view=0, gt_pair=[8, 9], center_yx=[10, 15], box_yxyx=[2, 3, 20, 25],
                OO_contact_pixels=24, OO_contact_mean_B=.3)
    with pytest.raises(ValueError):
        fixed_views_for_source([case], 'toy', (16, 16))
    with pytest.raises(ValueError):
        fixed_views_for_source([{**case, 'view': 1}], 'toy', (32, 32))
