# -*- coding: utf-8 -*-
"""原生结构反事实的合成支持域、来源和输入保护测试。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from utils.affinity_structure import (apply_structure, interface_mask, interior_mask,
                                     mask_provenance)


def split_gt(shape=(17, 21)):
    gt = np.ones(shape, np.uint16)
    gt[:, shape[1] // 2:] = 2
    return gt, np.ones(shape, np.bool_)


def test_interface_counts_each_four_neighbor_edge_and_both_endpoints():
    gt, covered = split_gt((5, 8))
    out = interface_mask(gt, covered, [1, 2], radius=0)
    expected = np.zeros(gt.shape, bool)
    expected[:, 3:5] = True
    assert out['contact_count'] == 5
    assert out['strict_contact_count'] == 5
    for key in ('mask', 'strict_mask', 'contact_mask', 'strict_contact_mask'):
        assert np.array_equal(out[key], expected)
    assert out['provenance']['masked_pixels'] == 10
    assert out['provenance']['filled_pixels'] == 0


def test_diagonal_only_touch_is_not_a_four_neighbor_contact():
    gt = np.array([[1, 0], [0, 2]], np.uint16)
    out = interface_mask(gt, gt > 0, [1, 2], radius=2)
    assert out['contact_count'] == 0
    assert out['strict_contact_count'] == 0
    assert not out['mask'].any() and not out['strict_mask'].any()


def test_interface_dilation_uses_ellipse_and_excludes_unknown_and_third_gt():
    gt, covered = split_gt()
    gt[2:5, 8] = 0
    gt[6:9, 8] = 3
    covered = gt > 0
    before_gt = gt.copy()
    before_coverage = covered.copy()
    out = interface_mask(gt, covered, [1, 2], radius=2)
    expected = cv2.dilate(out['contact_mask'].astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) & np.isin(gt, [1, 2])
    assert np.array_equal(out['mask'], expected)
    assert not out['mask'][gt == 0].any()
    assert not out['mask'][gt == 3].any()
    assert np.array_equal(gt, before_gt)
    assert np.array_equal(covered, before_coverage)
    boundary = np.full(gt.shape, .25, np.float32)
    after, audit = apply_structure(boundary, out['mask'], 1, gt=gt, original_covered=covered)
    assert np.array_equal(after[gt == 0], boundary[gt == 0])
    assert np.array_equal(after[gt == 3], boundary[gt == 3])
    assert audit['outside_mask_exact']


def test_strict_contacts_require_both_original_endpoints_then_coverage_clip():
    gt, covered = split_gt((9, 12))
    covered[:] = False
    covered[4, 5] = True  # 仅一端支持，不构成strict接触。
    one_end = interface_mask(gt, covered, [1, 2], radius=2)
    assert one_end['strict_contact_count'] == 0
    assert not one_end['strict_mask'].any()
    covered[4, 6] = True
    covered[3, 5] = True  # 可被合法双端种子膨胀覆盖。
    both = interface_mask(gt, covered, [1, 2], radius=2)
    assert both['strict_contact_count'] == 1
    assert both['strict_mask'][4, 5] and both['strict_mask'][4, 6]
    assert both['strict_mask'][3, 5]
    assert not both['strict_mask'][~covered].any()
    assert both['strict_provenance']['filled_pixels'] == 0
    assert both['provenance']['filled_pixels'] > 0


def test_interior_default_margin_four_and_constant_zero_image_border():
    gt = np.ones((15, 17), np.uint16)
    covered = np.ones(gt.shape, bool)
    out = interior_mask(gt, covered, 1)
    expected = np.zeros(gt.shape, bool)
    expected[4:-4, 4:-4] = True
    assert np.array_equal(out['mask'], expected)
    assert np.array_equal(out['strict_mask'], expected)
    assert out['margin'] == 4


def test_interior_respects_external_gt_and_strict_filled_holes():
    gt, covered = split_gt((13, 17))
    covered[:, 3] = False
    out = interior_mask(gt, covered, 1, margin=1)
    expected = cv2.erode((gt == 1).astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    assert np.array_equal(out['mask'], expected)
    assert np.array_equal(out['strict_mask'], expected & covered)
    assert not out['mask'][gt == 2].any()
    assert out['strict_components'] == 2
    assert out['strict_excluded_filled_pixels'] > 0
    assert out['strict_provenance']['filled_pixels'] == 0


def test_provenance_and_changed_counts_keep_original_vs_filled_separate():
    gt = np.array([[1, 1, 0], [2, 2, 0]], np.uint16)
    covered = np.array([[True, False, False], [True, False, False]])
    mask = gt > 0
    boundary = np.array([[.2, 1., .8], [.4, .3, .7]], np.float32)
    before = boundary.copy()
    after, audit = apply_structure(boundary, mask, 1, gt=gt, original_covered=covered)
    assert after.dtype == np.float32
    assert audit['masked_pixels'] == 4 and audit['changed_pixels'] == 3
    assert audit['provenance']['original_covered_pixels'] == 2
    assert audit['provenance']['filled_pixels'] == 2
    assert audit['changed_provenance']['original_covered_pixels'] == 2
    assert audit['changed_provenance']['filled_pixels'] == 1
    assert np.array_equal(boundary, before)
    assert np.array_equal(after[~mask], boundary[~mask])
    assert not np.shares_memory(after, boundary)


def test_zero_mask_is_exact_and_interior_zero_value_preserves_all_inputs():
    gt, covered = split_gt()
    boundary = np.linspace(0, 1, gt.size, dtype=np.float32).reshape(gt.shape)
    zeros = np.zeros(gt.shape, bool)
    after, audit = apply_structure(boundary, zeros, 0, gt=gt, original_covered=covered)
    assert np.array_equal(after, boundary)
    assert audit['masked_pixels'] == audit['changed_pixels'] == 0
    mask = interior_mask(gt, covered, 1, margin=1)['mask']
    saved = [x.copy() for x in (boundary, gt, covered, mask)]
    reduced, audit = apply_structure(boundary, mask, 0, gt=gt, original_covered=covered)
    assert not reduced[mask].any()
    assert np.array_equal(reduced[~mask], boundary[~mask])
    for actual, old in zip((boundary, gt, covered, mask), saved):
        assert np.array_equal(actual, old)


@pytest.mark.parametrize('gt', [np.ones((2, 2), bool), np.ones((2, 2), np.float32),
    np.array([[-1, 1], [1, 1]], np.int32), np.ones(3, np.uint16), np.empty((0, 2), np.uint16)])
def test_invalid_gt_dtype_shape_or_range_rejected(gt):
    with pytest.raises(ValueError):
        mask_provenance(gt, np.zeros(gt.shape, bool), np.zeros(gt.shape, bool))


@pytest.mark.parametrize('coverage', [np.ones((3, 3), bool), np.ones((2, 2), np.uint8)])
def test_invalid_coverage_dtype_or_shape_rejected(coverage):
    with pytest.raises(ValueError):
        mask_provenance(np.ones((2, 2), np.uint16), coverage, np.ones((2, 2), bool))


def test_original_coverage_unknown_or_unknown_selected_is_rejected():
    gt = np.array([[1, 0], [1, 1]], np.uint16)
    covered = gt > 0
    mask = np.ones(gt.shape, bool)
    with pytest.raises(ValueError, match='unknown'):
        mask_provenance(gt, covered, mask)
    with pytest.raises(ValueError, match='unknown'):
        apply_structure(np.zeros(gt.shape, np.float32), mask, 1,
                        gt=gt, original_covered=covered)
    with pytest.raises(ValueError, match='unknown'):
        mask_provenance(gt, np.ones(gt.shape, bool), covered)


@pytest.mark.parametrize('boundary', [np.ones((2, 2), np.float64), np.ones(3, np.float32),
    np.empty((0, 2), np.float32), np.array([[np.nan, .2], [.4, .5]], np.float32),
    np.array([[np.inf, .2], [.4, .5]], np.float32), np.full((2, 2), -1, np.float32),
    np.full((2, 2), 1.1, np.float32)])
def test_boundary_contract_rejects_invalid_dtype_shape_range_or_finite(boundary):
    with pytest.raises(ValueError):
        apply_structure(boundary, np.ones((2, 2), bool), 1,
                        gt=np.ones((2, 2), np.uint16), original_covered=np.ones((2, 2), bool))


@pytest.mark.parametrize('mask', [np.ones((2, 2), np.uint8), np.ones((3, 2), bool)])
def test_mask_dtype_shape_rejected(mask):
    gt = np.ones((2, 2), np.uint16)
    with pytest.raises(ValueError):
        apply_structure(np.zeros(gt.shape, np.float32), mask, 1,
                        gt=gt, original_covered=np.ones(gt.shape, bool))


@pytest.mark.parametrize('value', [True, False, np.nan, np.inf, -.01, 1.01, '1'])
def test_invalid_replacement_value_rejected(value):
    gt = np.ones((2, 2), np.uint16)
    with pytest.raises(ValueError):
        apply_structure(np.zeros(gt.shape, np.float32), np.ones(gt.shape, bool), value,
                        gt=gt, original_covered=np.ones(gt.shape, bool))


@pytest.mark.parametrize('pair', [[1, 1], [1, 3], [0, 2], [True, 2], [1.0, 2], [1], '1,2'])
def test_pair_validation(pair):
    gt, covered = split_gt()
    with pytest.raises(ValueError):
        interface_mask(gt, covered, pair)


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_radius_and_margin_must_be_nonnegative_integer(value):
    gt, covered = split_gt()
    with pytest.raises(ValueError):
        interface_mask(gt, covered, [1, 2], radius=value)
    with pytest.raises(ValueError):
        interior_mask(gt, covered, 1, margin=value)


def test_missing_interior_gt_and_shape_mismatch_rejected():
    gt, covered = split_gt()
    with pytest.raises(ValueError):
        interior_mask(gt, covered, 3)
    with pytest.raises(ValueError):
        apply_structure(np.zeros((2, 2), np.float32), np.zeros((2, 2), bool), 0,
                        gt=gt, original_covered=covered)
