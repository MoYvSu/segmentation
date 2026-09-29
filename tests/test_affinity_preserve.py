# -*- coding: utf-8 -*-
"""纯种子恢复对照的新增错误诊断及完整解码契约。"""
import numpy as np
import torch

from tools.run_affinity_preserve import trusted_changes, decode_pair


def test_added_significant_split_is_counted_even_when_not_in_previous_errors():
    gt = np.ones((40, 80), np.uint16)
    before = np.ones_like(gt)
    after = before.copy()
    after[:, 40:] = 2
    result = trusted_changes(gt, gt > 0, before, after)
    assert result['summary']['new_split_pairs'] == 1
    assert result['summary']['gt_regions_more_fragments'] == 1
    assert result['summary']['gt_regions_with_new_split'] == 1
    assert result['summary']['fixed_anchor_merges'].get('new_pairs', 0) == 0


def test_new_merge_of_previously_separated_gt_is_counted():
    gt = np.ones((40, 80), np.uint16)
    gt[:, 40:] = 2
    result = trusted_changes(gt, gt > 0, gt, np.ones_like(gt))
    assert result['summary']['fixed_anchor_merges']['new_pairs'] == 1
    assert len(result['new_merge_pairs']) == 1
    assert result['summary']['new_split_pairs'] == 0


def test_unknown_only_changes_do_not_create_gt_error_counts():
    gt = np.ones((40, 80), np.uint16)
    trusted = np.zeros_like(gt, bool)
    trusted[:, :35] = True
    before = np.ones_like(gt)
    after = before.copy()
    after[:, 40:] = 2
    result = trusted_changes(gt, trusted, before, after)
    assert result['summary']['new_split_pairs'] == 0
    assert result['summary']['gt_regions_more_fragments'] == 0


def test_zero_new_output_is_not_reported_as_successful_merge_correction():
    gt = np.ones((40, 80), np.uint16)
    gt[:, 40:] = 2
    result = trusted_changes(gt, gt > 0, np.ones_like(gt), np.zeros_like(gt))
    assert result['summary']['fixed_anchor_merges']['base_pairs'] == 1
    assert result['summary']['fixed_anchor_merges']['blocked_after'] == 1
    assert result['summary']['fixed_anchor_merges'].get('fixed_pairs', 0) == 0


def test_no_reference_split_is_identical_after_complete_raw_semantic_revote():
    config = {'inference': {'boundary_threshold': .65, 'min_instance_area': 50,
        'bridge_width': 1, 'watershed_dilate_width': 1, 'marker_border_seal_width': 2,
        'marker_boundary_low_threshold': .45, 'marker_boundary_reconstruction_steps': 8,
        'semantic_vote_mode': 'probability_mean',
        'marker_partition_restore': {'enabled': True, 'min_core_area': 50, 'area_filter_enabled': False}}}
    semantic = torch.ones(1, 1, 50, 70)
    # 原图概率反向，验证末端复投票与几何语义来源分开。
    before, after, bc, ac, audit, _, _ = decode_pair(semantic, torch.zeros_like(semantic),
        np.full((50, 70), .1, np.float32), config)
    assert np.array_equal(before, after) and bc == ac == {'1': 0}
    assert audit['split_parent_count'] == 0 and audit['watershed_passes'] == 1
