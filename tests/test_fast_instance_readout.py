# -*- coding: utf-8 -*-
"""整数快路径与冻结旧函数对照；不运行模型或修改旧入口。"""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from utils import compact_instance_readout as original
from utils import fast_instance_readout as fast


def policy(*, minimum=0, maximum=65535, action='legacy_drop'):
    return dict(seed_min_area=1, final_min_instance_area=minimum,
                final_small_instance_action=action, max_instance_id=maximum)


def compare(watershed, inference):
    before = watershed.copy()
    config_before = deepcopy(inference)
    expected = original.finish_watershed(watershed, inference)
    actual = fast.finish_watershed(watershed, inference)
    assert np.array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    assert actual[0].dtype == np.uint16 and int(actual[0].max()) <= 65535
    assert np.array_equal(watershed, before) and inference == config_before
    assert not np.shares_memory(watershed, actual[0])
    return actual


@pytest.mark.parametrize('action', ['legacy_drop', 'neighbor_merge_v1'])
@pytest.mark.parametrize('minimum', [0, 3, 4, 5, 7])
def test_sparse_ids_equal_area_and_exact_filter_threshold(action, minimum):
    source = np.array([[99, 99, 0, 7, 7, 0, 4000, 4000, -1],
                       [99, 99, 0, 7, 7, 0, 4000, 4000, -2],
                       [0, 0, 0, 0, 0, 0, 4000, 4000, 0]], np.int32)
    ids, classes, proof = compare(source, policy(minimum=minimum, action=action))
    if minimum <= 4 or action == 'neighbor_merge_v1':
        # 面积6的4000优先，面积4的7在99之前，门槛相等保留。
        assert (ids[0, 6], ids[0, 3], ids[0, 0]) == (1, 2, 3)
        assert classes == {1: 0, 2: 0, 3: 0}
    assert proof['pre_vote_final_min_area'] == (minimum if action == 'legacy_drop' else 0)


@pytest.mark.parametrize('maximum', [1, 2, 3, 65535])
@pytest.mark.parametrize('isolated', [False, True])
def test_original_cap_merge_chain_ties_seams_and_isolated_deletion(maximum, isolated):
    source = np.zeros((15, 23), np.int32)
    spacing = 7 if isolated else 4
    for i, label in enumerate([13, 3, 101, 29]):
        y, x = (i // 2) * spacing + 1, (i % 2) * spacing + 1
        source[y:y + 3, x:x + 3] = label
    compare(source, policy(maximum=maximum))


def test_cap_receives_exact_original_ids_and_filtered_array():
    source = np.array([[1, 1, 0, 60000, 60000, 0, 100000, 100000],
                       [1, 1, 0, 60000, 60000, 0, 100000, 100000],
                       [0, 0, 0, 0, 0, 0, -1, -1]], np.int32)
    real_cap = fast._merge_region_map_to_cap
    seen = []

    def cap(candidate, maximum):
        assert candidate.dtype == np.int32
        seen.append(candidate.copy())
        return real_cap(candidate, maximum)

    with patch.object(fast, '_merge_region_map_to_cap', new=cap):
        for maximum in (1, 2, 65535):
            compare(source, policy(maximum=maximum))
        compare(source, policy(minimum=5))
    assert all(np.array_equal(value, np.maximum(source, 0)) for value in seen[:3])
    assert not seen[3].any()


def test_very_sparse_int32_ids_without_cap_keeps_original_behavior():
    small = np.array([[1, 1, 0, 2, 2, 0, 3, 3],
                      [1, 1, 0, 2, 2, 0, 3, 3],
                      [0, 0, 0, 0, 0, 0, 0, 0]], np.int32)
    label_map = np.array([0, 17, 2 ** 30, np.iinfo(np.int32).max], np.int32)
    huge = label_map[small]
    for minimum in (0, 4, 5):
        compare(huge, policy(minimum=minimum))


@pytest.mark.parametrize('value', [0, -1, np.iinfo(np.int32).min])
def test_all_background_and_no_input_mutation(value):
    ids, classes, _ = compare(np.full((5, 7), value, np.int32), policy())
    assert not ids.any() and classes == {}


def test_noncontiguous_readonly_input_and_legacy_area_alias():
    source = np.array([[0, 3, 3, 0], [7, 7, 0, -1], [99, 0, 99, 99]], np.int32).T
    source.flags.writeable = False
    compare(source, dict(min_instance_area=2, max_instance_id=2))


def test_all_65535_output_ids_remain_uint16_without_wraparound():
    # 每区面积1，原ID已升序，因此预期结果无需复刻待测的计数/排序实现。
    source = np.arange(1, 65536, dtype=np.int32).reshape(255, 257)
    ids, classes, proof = fast.finish_watershed(source, policy())
    assert ids.dtype == np.uint16 and np.array_equal(ids, source)
    assert len(classes) == 65535 and classes[1] == classes[65535] == 0
    assert set(classes.values()) == {0} and proof['cap_merge_edges'] == []


def test_deterministic_randomized_small_maps_all_metadata_exact():
    rng = np.random.default_rng(20261101)
    for _ in range(24):
        source = rng.choice(np.array([-1, 0, 3, 17, 99, 1009], np.int32), (9, 13))
        compare(source, policy(minimum=int(rng.integers(0, 25)),
                               maximum=int(rng.choice([1, 2, 4, 65535])),
                               action=str(rng.choice(['legacy_drop', 'neighbor_merge_v1']))))


@pytest.mark.parametrize('bad', [np.zeros((0, 2), np.int32), np.zeros((2, 2), np.int64),
                               np.zeros((2, 2, 1), np.int32), [[1]]])
def test_invalid_label_input_same_error(bad):
    for function in (original.finish_watershed, fast.finish_watershed):
        with pytest.raises(ValueError, match='nonempty HW watershed labels'):
            function(bad, policy())


@pytest.mark.parametrize('maximum', [0, -1, 65536])
def test_invalid_cap_same_error(maximum):
    for function in (original.finish_watershed, fast.finish_watershed):
        with pytest.raises(ValueError, match='max_instance_id'):
            function(np.ones((3, 4), np.int32), policy(maximum=maximum))


@pytest.mark.parametrize('mode', ['C', 'F0'])
def test_real_watershed_only_finish_swapped_same_ids_classes_and_audit(mode):
    # 真实OpenCV WS，避免只测人工标签图而漏掉实际shape/dtype/调用边界。
    after = np.full((64, 80), .1, np.float32)
    after[:, 39:42] = .95
    after[30:33, :] = .9
    semantic = torch.linspace(-3, 3, after.size).reshape(1, 1, *after.shape)
    if mode == 'F0':
        semantic.zero_()
    config = dict(inference=dict(policy(minimum=200, action='neighbor_merge_v1'),
        boundary_threshold=.65, bridge_width=1, watershed_dilate_width=1,
        marker_border_seal_width=2, marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8, semantic_vote_mode='probability_mean',
        semantic_vote_erode_width=0, semantic_vote_threshold=.5))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        expected, old_audit = original.compact_supported_readout(
            semantic, after, after.copy(), config, geometry_mode=mode)
        with patch.object(original, 'finish_watershed', new=fast.finish_watershed):
            actual, new_audit = original.compact_supported_readout(
                semantic, after, after.copy(), config, geometry_mode=mode)
        assert np.array_equal(actual[0], expected[0]) and actual[1] == expected[1]
        assert new_audit == old_audit
    finally:
        torch.set_num_threads(old_threads)
