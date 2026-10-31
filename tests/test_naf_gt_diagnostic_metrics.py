# -*- coding: utf-8 -*-
"""新增线位置和实际已知域切分的独立构造检查，不镜像具体模型。"""
import numpy as np
import pytest

from tools.naf_gt_diagnostic_metrics import added_boundary_diagnostic, known_domain_splits


def masks(ids):
    original = ids > 0
    return original, np.zeros(ids.shape, bool), ids == 0


def test_near_gt_added_line_is_not_deep_interior():
    ids = np.ones((64, 64), np.int32); ids[:, 32:] = 2
    before = np.zeros(ids.shape, bool); after = before.copy(); after[:, 32] = True
    summary, image = added_boundary_diagnostic(before, after, ids, *masks(ids), safe_margin=2)
    group = summary['groups']['original_covered']
    assert group['added_skeleton_pixels'] > 0
    assert group['near_recorded_boundary_pixels'] == group['added_skeleton_pixels']
    assert group['deep_recorded_interior_pixels'] == 0
    assert group['supported_oo_only'] == group['near_recorded_boundary_pixels']


def test_normal_thickening_is_not_added_line():
    ids = np.ones((64, 64), np.int32)
    before = np.zeros(ids.shape, bool); before[8:56, 32] = True
    after = np.zeros_like(before); after[8:56, 31:34] = True
    summary, _ = added_boundary_diagnostic(before, after, ids, *masks(ids), safe_margin=2)
    assert summary['groups']['all_known_safe']['added_skeleton_pixels'] == 0


def test_unknown_and_its_margin_are_ignored():
    ids = np.ones((64, 64), np.int32); ids[:, :16] = 0
    before = np.zeros(ids.shape, bool); after = before.copy(); after[:, 18] = True
    summary, _ = added_boundary_diagnostic(before, after, ids, *masks(ids), safe_margin=4)
    assert summary['raw_added_pixels'] > 0
    assert summary['groups']['all_known_safe']['added_skeleton_pixels'] == 0
    assert summary['ignored_unknown_or_margin_pixels'] == summary['raw_added_pixels']


def test_filled_boundary_support_not_original_original():
    ids = np.ones((64, 64), np.int32); ids[:, 32:] = 2
    original, filled, unknown = masks(ids)
    original[:, 30:34] = False; filled = (ids > 0) & ~original
    before = np.zeros(ids.shape, bool); after = before.copy(); after[:, 32] = True
    summary, _ = added_boundary_diagnostic(before, after, ids, original, filled, unknown, safe_margin=2)
    group = summary['groups']['filled']
    assert group['supported_with_fill_only'] > 0
    assert group['supported_oo_only'] == 0


def test_partial_internal_stroke_does_not_split_gt():
    ids = np.ones((64, 64), np.int32)
    before = np.zeros(ids.shape, bool); after = before.copy(); after[20:44, 32] = True
    summary, _ = added_boundary_diagnostic(before, after, ids, *masks(ids), safe_margin=2)
    assert summary['groups']['original_covered']['deep_recorded_interior_pixels'] > 0
    cuts = known_domain_splits(before, after, ids, *masks(ids), safe_margin=2)
    assert cuts['actual']['substantial_deep_split_parents'] == 0


def test_full_internal_stroke_splits_same_known_domain():
    ids = np.ones((64, 64), np.int32)
    before = np.zeros(ids.shape, bool); after = before.copy(); after[:, 32] = True
    cuts = known_domain_splits(before, after, ids, *masks(ids), safe_margin=2)
    assert cuts['actual']['substantial_deep_split_parents'] == 1
    assert cuts['actual']['original_supported_split_parents'] == 1
    assert cuts['addition_only']['substantial_deep_split_parents'] == 1
    assert cuts['actual']['deep_addition_causally_required_parents'] == 1
    assert cuts['actual']['original_supported_deep_causal_split_parents'] == 1


def test_provenance_hole_does_not_define_connectivity():
    ids = np.ones((64, 64), np.int32)
    original, _, unknown = masks(ids); original[:, 30:34] = False
    filled = (ids > 0) & ~original
    before = np.zeros(ids.shape, bool)
    cuts = known_domain_splits(before, before, ids, original, filled, unknown, safe_margin=2)
    assert cuts['actual']['parent_components_with_multiple_children'] == 0
    after = before.copy(); after[:, 32] = True
    cuts = known_domain_splits(before, after, ids, original, filled, unknown, safe_margin=2)
    assert cuts['actual']['original_supported_split_parents'] == 1


def test_invalid_provenance_rejected():
    ids = np.ones((32, 32), np.int32)
    original, filled, unknown = masks(ids); filled[2, 2] = True
    with pytest.raises(ValueError, match='inconsistent'):
        added_boundary_diagnostic(original, original, ids, original, filled, unknown)


def test_preexisting_split_or_removed_barrier_is_not_a_new_deep_split():
    ids = np.ones((64, 64), np.int32)
    before = np.zeros(ids.shape, bool); before[:, 32] = True
    same = known_domain_splits(before, before, ids, *masks(ids), safe_margin=2)
    removed = known_domain_splits(before, np.zeros_like(before), ids, *masks(ids), safe_margin=2)
    assert same['actual']['substantial_deep_split_parents'] == 0
    assert removed['actual']['substantial_deep_split_parents'] == 0


def test_tiny_piece_is_reported_even_when_not_substantial():
    ids = np.ones((64, 64), np.int32)
    before = np.zeros(ids.shape, bool); after = before.copy()
    after[:, 4] = True
    cuts = known_domain_splits(before, after, ids, *masks(ids), safe_margin=2,
                               min_piece_pixels=500, min_piece_fraction=.05)
    row = cuts['actual']['cases'][0]
    assert row['after_children'] == 2
    assert row['substantial_children'] == 1
    assert row['all_children'][1]['pixels'] > 0
    assert cuts['actual']['substantial_deep_split_parents'] == 0


def test_true_gt_interface_is_not_an_internal_split():
    ids = np.ones((64, 64), np.int32); ids[:, 32:] = 2
    before = np.zeros(ids.shape, bool); after = before.copy(); after[:, 31:33] = True
    cuts = known_domain_splits(before, after, ids, *masks(ids), safe_margin=2)
    assert cuts['actual']['substantial_deep_split_parents'] == 0


def test_irrelevant_deep_stroke_does_not_explain_a_near_boundary_split():
    ids = np.full((80, 80), 2, np.int32)
    ids[12:68, 12:68] = 1
    before = np.zeros(ids.shape, bool); after = before.copy()
    # 上边缘内4px的横线真正切下薄条；远处内部短枝不造成切分。
    after[16, 12:68] = True
    after[36:44, 40] = True
    cuts = known_domain_splits(before, after, ids, *masks(ids), safe_margin=2,
                               min_piece_pixels=50, min_piece_fraction=.05)
    row = next(r for r in cuts['actual']['cases'] if r['gt'] == 1)
    assert row['substantial_split_with_deep_additions']
    assert row['newly_deep_barrier_pixels'] > 0
    assert not row['deep_addition_contributes_to_split']
    assert not row['deep_addition_causally_required_for_split']
