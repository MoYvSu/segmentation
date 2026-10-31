# -*- coding: utf-8 -*-
"""P邻接连通组合并：最终F不变、原生单零接触与输出合同。"""
from copy import deepcopy
import json

import numpy as np
import pytest

from utils.pearlite_merge import merge_adjacent_pearlite


def run(value, classes):
    ids = np.asarray(value, dtype=np.uint16)
    before, table = ids.copy(), deepcopy(classes)
    output, result, audit = merge_adjacent_pearlite(ids, classes)
    assert np.array_equal(ids, before) and table == classes
    assert output.dtype == np.uint16 and not np.shares_memory(output, ids)
    assert {str(int(v)) for v in np.unique(output) if v} == set(result)
    assert audit['F_final_ID_map_and_classes_exact'] and audit['final_contract_valid']
    json.dumps(audit, allow_nan=False)
    return output, result, audit


@pytest.mark.parametrize('value,classes',[
    ([[0, 0], [0, 0]], {}),
    ([[1, 0, 65535]], {'1': 1, '65535': 1}),
    ([[1, 0, 1], [0, 0, 0]], {'1': 0}),
    ([[1]], {'1': 0}),
    ([[65535]], {'65535': 1}),
])
def test_noop_is_exact_independent_copy_without_compaction(value, classes):
    output, result, audit = run(value, classes)
    assert np.array_equal(output, value) and result == classes
    assert audit['removed_P_instance_count'] == audit['filled_seam_pixels'] == 0


def test_direct4_contact_keeps_minimum_ID_and_does_not_touch_F():
    output, table, audit = run([[65535, 65535, 9, 3], [65535, 9, 9, 3]],
                              {'65535': 0, '9': 0, '3': 1})
    assert np.array_equal(output, [[9, 9, 9, 3], [9, 9, 9, 3]])
    assert table == {'9': 0, '3': 1}
    assert audit['direct4_pairs'] == [[9, 65535]]
    assert audit['P_groups'] == [dict(retained_id=9, original_ids=[9, 65535],
        original_areas=[3, 3], filled_seam_pixels=0, final_area=6)]


def test_chain_joins_full_P_group_not_just_nonoverlapping_pairs():
    ids = np.tile(np.array([7, 7, 0, 3, 3, 0, 65535, 65535], np.uint16), (5, 1))
    output, table, audit = run(ids, {'7': 0, '3': 0, '65535': 0})
    assert table == {'3': 0} and np.all(output[ids > 0] == 3)
    assert np.all(output[1:-1, 2] == 3) and np.all(output[1:-1, 5] == 3)
    assert np.all(output[[0, -1], 2] == 0) and np.all(output[[0, -1], 5] == 0)
    assert audit['removed_P_instance_count'] == 2 and audit['chain_merging']
    assert audit['filled_seam_pixels'] == 6


def test_one_zero_gap_only_fills_original_internal_contact():
    ids = np.zeros((7, 9), np.uint16)
    ids[1:-1, 1:4] = 2
    ids[1:-1, 5:8] = 8
    output, table, audit = run(ids, {'2': 0, '8': 0})
    assert table == {'2': 0} and np.all(output[1:-1, 4] == 2)
    assert audit['zero_bridge_pairs'] == [[2, 8]]
    assert audit['filled_seam_pixels'] == 5
    assert np.all(output[0] == 0) and np.all(output[-1] == 0)


def test_F_thin_path_never_becomes_P_or_connects_two_P_regions():
    ids = np.tile(np.array([3, 3, 9, 7, 7], np.uint16), (5, 1))
    output, table, audit = run(ids, {'3': 0, '9': 1, '7': 0})
    assert np.array_equal(output, ids) and table == {'3': 0, '9': 1, '7': 0}
    assert audit['union_pair_count'] == 0


def test_three_P_junction_can_join_and_fill_without_pairwise_quota():
    ids = np.zeros((5, 5), np.uint16)
    ids[1, 2], ids[2, 1], ids[2, 3] = 8, 5, 2
    output, table, audit = run(ids, {'8': 0, '5': 0, '2': 0})
    assert output[2, 2] == 2 and table == {'2': 0}
    assert audit['zero_bridge_pair_count'] == 3
    assert audit['P_groups'][0]['original_ids'] == [2, 5, 8]


def test_PPF_junction_is_not_a_bridge_or_filled_even_if_P_merge_elsewhere():
    ids = np.zeros((7, 7), np.uint16)
    ids[3, 2], ids[3, 4], ids[2, 3] = 1, 2, 3
    output, table, audit = run(ids, {'1': 0, '2': 0, '3': 1})
    assert output[3, 3] == 0 and table == {'1': 0, '2': 0, '3': 1}
    assert audit['union_pair_count'] == 0
    ids[5, 2:5] = [1, 0, 2]  # A/B在另一处合法连接，仍不填有F邻居的交点。
    output, table, _ = run(ids, {'1': 0, '2': 0, '3': 1})
    assert table == {'1': 0, '3': 1} and output[5, 3] == 1
    assert output[3, 3] == 0 and output[2, 3] == 3


def test_two_zero_wide_gap_does_not_close_or_grow():
    ids = np.tile(np.array([1, 1, 0, 0, 2, 2], np.uint16), (5, 1))
    output, table, audit = run(ids, {'1': 0, '2': 0})
    assert np.array_equal(output, ids) and len(table) == 2
    assert audit['filled_seam_pixels'] == audit['union_pair_count'] == 0


def test_diagonal_without_internal_single_zero_bridge_does_not_join():
    output, table, audit = run([[1, 0], [0, 2]], {'1': 0, '2': 0})
    assert np.array_equal(output, [[1, 0], [0, 2]]) and len(table) == 2
    assert audit['union_pair_count'] == 0


def test_single_zero_can_turn_corner_when_all_positive_neighbors_are_P():
    ids = np.zeros((5, 5), np.uint16)
    ids[1, 2], ids[2, 1] = 1, 2
    output, table, audit = run(ids, {'1': 0, '2': 0})
    assert table == {'1': 0} and output[2, 2] == 1
    assert audit['filled_seam_pixels'] == 2  # 两条对称的一零转角通路。


@pytest.mark.parametrize('rotation', range(4))
def test_outer_frame_zero_stays_zero_even_if_it_only_separates_P(rotation):
    ids = np.tile(np.array([1, 1, 0, 2, 2], np.uint16), (5, 1))
    output, table, audit = run(np.rot90(ids, rotation), {'1': 0, '2': 0})
    output = np.rot90(output, -rotation)
    assert output[0, 2] == output[-1, 2] == 0 and np.all(output[1:-1, 2] == 1)
    assert table == {'1': 0} and audit['original_outer_frame_zero_pixels_unchanged']


def test_existing_multicomponent_same_ID_is_not_split_or_force_connected():
    ids = np.zeros((7, 9), np.uint16)
    ids[1, 1], ids[4:6, 2:4], ids[4:6, 4:6] = 9, 9, 13
    output, table, audit = run(ids, {'9': 0, '13': 0})
    assert output[1, 1] == 9 and np.all(output[4:6, 2:6] == 9)
    assert np.all(output[2:4] == 0) and table == {'9': 0}
    assert audit['existing_multicomponent_IDs_not_repaired']


def test_single_pass_does_not_iterate_newly_filled_zero_into_another_bridge():
    ids = np.zeros((7, 9), np.uint16)
    ids[3, 2], ids[2, 3], ids[3, 5] = 1, 2, 3
    first, table, audit = run(ids, {'1': 0, '2': 0, '3': 0})
    assert table == {'1': 0, '3': 0} and first[3, 4] == 0
    assert audit['removed_P_instance_count'] == 1
    # 一次填缝能改变下一次调用的邻接；入口必须只执行一次，不能循环到稳定。
    second, merged, _ = run(first, table)
    assert merged == {'1': 0} and second[3, 4] == 1


def test_second_application_is_exact_and_class_order_does_not_change_result():
    ids = np.tile(np.array([9, 9, 0, 2, 2, 0, 5], np.uint16), (5, 1))
    output, table, audit = run(ids, {'9': 0, '2': 0, '5': 1})
    again, same_table, second = run(output, table)
    other, other_table, _ = run(ids, {'5': 1, '2': 0, '9': 0})
    assert np.array_equal(again, output) and same_table == table
    assert np.array_equal(other, output) and other_table == table
    assert second['filled_seam_pixels'] == second['removed_P_instance_count'] == 0
    assert audit['input_ID_sha256'] != audit['output_ID_sha256']


@pytest.mark.parametrize('bad', [
    {}, {'1': 0}, {'1': 0, '2': 1, '3': 0}, {'01': 0, '2': 0},
    {1: 0, '2': 0}, {'1': True, '2': 0}, {'1': np.int64(0), '2': 0},
    {'1': 2, '2': 0}, {'1': 0.0, '2': 0}, {'1': 0, '2': '0'},
])
def test_invalid_or_incomplete_class_table_rejected(bad):
    with pytest.raises(ValueError):
        merge_adjacent_pearlite(np.array([[1, 2]], np.uint16), bad)


@pytest.mark.parametrize('bad', [
    np.array([[1, 2]], np.int32), np.array([[1, 2]], np.uint8),
    np.array([1, 2], np.uint16), np.zeros((0, 2), np.uint16),
    np.zeros((2, 2, 1), np.uint16), [[1, 2]],
    np.ma.array([[1, 2]], dtype=np.uint16),
])
def test_invalid_instance_contract_rejected(bad):
    with pytest.raises(ValueError):
        merge_adjacent_pearlite(bad, {'1': 0, '2': 0})


def test_readonly_strided_source_supported_and_unchanged():
    ids = np.tile(np.array([7, 7, 0, 4, 4, 3], np.uint16), (7, 1))[:, ::-1]
    ids.flags.writeable = False
    output, table, audit = run(ids, {'7': 0, '4': 0, '3': 1})
    assert table == {'4': 0, '3': 1} and output.flags.writeable
    assert audit['source_input_unchanged']
