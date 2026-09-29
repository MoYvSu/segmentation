# -*- coding: utf-8 -*-
"""反事实报告按掩码重叠关联，不能把实例重新编号误记为几何变化。"""
import numpy as np
from tools.probe_marker_anchor import partition_changes


def test_partition_relabel_is_not_split_or_merge():
    old = np.zeros((30, 30), np.uint16)
    old[2:15, 2:28] = 1
    old[16:28, 2:28] = 2
    new = np.array([0, 2, 1], np.uint16)[old]
    result = partition_changes(old, new)
    assert result['split_old_ids'] == result['merged_new_ids'] == []
    assert result['contour_changed_pixels'] == 0
    assert result['foreground_lost'] == result['foreground_added'] == 0


def test_partition_split_and_inverse_merge():
    old = np.zeros((40, 40), np.uint16)
    old[1:35, 1:35] = 1
    new = old.copy()
    new[18:35, 1:35] = 2
    result = partition_changes(old, new)
    assert result['split_old_ids'] == result['split_border_old_ids'] == [1]
    assert result['merged_new_ids'] == []
    inverse = partition_changes(new, old)
    assert inverse['merged_new_ids'] == [1]
    assert inverse['split_old_ids'] == []
