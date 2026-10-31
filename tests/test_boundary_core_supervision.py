# -*- coding: utf-8 -*-
"""原生可信内核与分隔诊断的CPU契约，不依赖赛题文件。"""
import json

import cv2
import numpy as np
import pytest

from utils.boundary_core_supervision import (
    adjacent_instance_pairs, build_trusted_cores, candidate_interface,
    compare_core_separation, compare_marker_atoms, core_marker_statistics,
    pair_core_relation,
)


def two_grains(shape=(40, 60)):
    gt = np.ones(shape, np.int32)
    gt[:, shape[1]//2:] = 2
    return gt, np.ones(shape, bool)


def compact_cores():
    gt, original = two_grains((12, 20))
    return build_trusted_cores(gt, original, radius=0, min_component_pixels=1)


def test_per_instance_erosion_keeps_both_sides_away_from_known_contact():
    gt, original = two_grains()
    cores = build_trusted_cores(gt, original)
    assert not cores['core_map'][:, 26:34].any()
    assert not cores['core_map'][:4].any()
    assert cores['summary']['gt_with_cores'] == 2
    assert np.all(gt[cores['core_map'] > 0] == cores['core_map'][cores['core_map'] > 0])


def test_filled_cannot_create_core_or_bridge_two_components():
    gt = np.ones((50, 80), np.int32)
    original = np.ones(gt.shape, bool)
    original[:, 36:44] = False
    cores = build_trusted_cores(gt, original)
    assert cores['summary']['retained_components'] == 2
    assert cores['gt_stats'][0]['retained_component_count'] == 2
    assert not cores['core_map'][:, 32:48].any()
    assert not np.any((cores['core_map'] > 0) & ~original)


def test_all_large_components_retained_small_component_reported():
    gt = np.zeros((30, 60), np.int32)
    gt[1:9, 1:9] = 7
    gt[12:22, 12:22] = 7
    gt[1:8, 30:37] = 7
    cores = build_trusted_cores(gt, gt > 0, radius=0, min_component_pixels=64)
    row = cores['gt_stats'][0]
    assert row['core_pixels'] == 164
    assert row['retained_component_count'] == 2
    assert row['removed_small_pixels'] == 49
    assert row['removed_component_count'] == 1
    assert [v['pixels'] for v in cores['components']] == [64, 100]


def test_missing_original_instance_stays_in_denominator_cohort():
    gt, original = two_grains()
    original[gt == 2] = False
    cores = build_trusted_cores(gt, original)
    stats = core_marker_statistics(cores, np.ones_like(gt))
    assert stats['gt']['2']['core_pixels'] == 0
    assert stats['gt']['2']['coverage'] is None
    relation = pair_core_relation(stats, (1, 2))
    assert not relation['eligible_both_have_cores']
    result = compare_core_separation(stats, stats, (1, 2))
    assert not result['target_separated_and_covered']


@pytest.mark.parametrize('which', ['shape', 'unknown', 'float_gt', 'negative_gt', 'oversized_gt'])
def test_invalid_gt_and_original_fail(which):
    gt, original = two_grains()
    if which == 'shape': original = original[:-1]
    elif which == 'unknown': gt[0, 0] = 0
    elif which == 'float_gt': gt = gt.astype(np.float32)
    elif which == 'oversized_gt': gt[0, 0] = 65536
    else: gt[0, 0] = -1
    with pytest.raises(ValueError): build_trusted_cores(gt, original)


def test_adjacency_does_not_jump_unknown_or_diagonal():
    gt = np.array([[1, 0, 2], [0, 3, 2], [4, 4, 0]], np.int32)
    assert adjacent_instance_pairs(gt) == [(2, 3), (3, 4)]


def test_all_and_significant_shared_marker_have_explicit_full_denominator():
    cores = compact_cores()
    labels = np.zeros_like(cores['core_map'])
    for gid in (1, 2):
        ys, xs = np.where(cores['core_map'] == gid)
        labels[ys[:32], xs[:32]] = 5
        labels[ys[32:33], xs[32:33]] = 7
        labels[ys[33:], xs[33:]] = 10+gid
    stats = core_marker_statistics(cores, labels)
    pair = pair_core_relation(stats, (2, 1))
    assert pair['pair'] == [1, 2]
    assert pair['all']['marker_count'] == 2
    assert pair['all']['shared_core_pixels'] == 66
    assert pair['significant']['marker_ids'] == [5]
    assert pair['significant']['shared_core_pixels'] == 64
    assert pair['a_core_pixels'] == pair['b_core_pixels'] == 120
    assert stats['gt']['1']['marker_counts']['0'] == 0
    assert not pair['dominant_same_at_least_threshold']
    json.dumps(stats, allow_nan=False)
    json.dumps(pair, allow_nan=False)


def test_dominance_uses_all_core_including_zero():
    cores = compact_cores()
    labels = np.ones_like(cores['core_map'])*9
    for gid in (1, 2):
        ys, xs = np.where(cores['core_map'] == gid)
        labels[ys[:13], xs[:13]] = 0
    relation = pair_core_relation(core_marker_statistics(cores, labels), (1, 2))
    assert not relation['dominant_same_at_least_threshold']
    assert relation['a_coverage'] == 107/120


def test_seed_swallowed_by_zero_is_not_success():
    cores = compact_cores()
    before = core_marker_statistics(cores, np.ones_like(cores['core_map']))
    after_labels = np.ones_like(cores['core_map'])*2
    after_labels[cores['core_map'] == 2] = 0
    after = core_marker_statistics(cores, after_labels)
    result = compare_core_separation(before, after, (1, 2))
    assert result['target_relation']['after']['all_positive_labels_disjoint']
    assert not result['target_separated_and_covered']
    assert result['summary']['newly_unseeded_components'] == 1
    assert result['summary']['zero_pixel_delta'] == 120


def test_two_sample_points_separate_but_rest_of_cores_share_marker():
    cores = compact_cores()
    before_labels = np.ones_like(cores['core_map'])*9
    after_labels = before_labels.copy()
    after_labels[5, 5], after_labels[5, 15] = 10, 11
    a = core_marker_statistics(cores, before_labels)
    b = core_marker_statistics(cores, after_labels)
    result = compare_core_separation(a, b, (1, 2))
    assert not result['target_separated_and_covered']
    assert result['target_relation']['after']['all']['marker_ids'] == [9]
    assert result['summary']['newly_split_components'] == 2


def test_unknown_bypass_is_not_cut_by_cropping_to_known_union():
    gt, original = two_grains((24, 30))
    gt[:4] = 0;original[:4] = False
    # 正中屏障不延伸到顶部unknown，整图左右仍绕行。不要先裁掉unknown再算连通。
    background = np.ones_like(gt, np.uint8)
    background[4:, 14:16] = 0
    _, markers = cv2.connectedComponents(background, connectivity=8)
    cores = build_trusted_cores(gt, original, radius=2, min_component_pixels=8)
    stats = core_marker_statistics(cores, markers)
    assert pair_core_relation(stats, (1, 2))['all']['marker_count'] == 1
    assert not compare_core_separation(stats, stats, (1, 2))['target_separated_and_covered']


def test_clean_whole_core_separation_passes_but_reports_third_gt_harm():
    gt = np.repeat(np.array([[1]*12+[2]*12+[3]*12], np.int32), 16, axis=0)
    cores = build_trusted_cores(gt, gt > 0, radius=0, min_component_pixels=1)
    old = np.ones_like(gt);old[gt == 3] = 3
    new = gt.copy();new[:8, 24:] = 4
    result = compare_core_separation(core_marker_statistics(cores, old), core_marker_statistics(cores, new), (1, 2))
    assert result['target_separated_and_covered']
    assert result['summary']['newly_split_components'] == 1
    assert len(result['all_gt_changes']) == 3


def test_interface_support_needs_both_original_endpoints():
    gt, original = two_grains()
    original[gt == 2] = False
    result = candidate_interface(gt, original, (1, 2))
    assert result['summary']['full_contact_edges'] == 40
    assert result['summary']['supported_contact_edges'] == 0
    assert not result['mask'].any()


def test_thin_fill_can_use_only_own_id_original_support():
    gt, original = two_grains((30, 40))
    original[:, 18:22] = False
    thin = candidate_interface(gt, original, (1, 2), support_policy='thin4')
    strict = candidate_interface(gt, original, (1, 2))
    assert thin['summary']['full_interface_supported']
    assert thin['provenance']['filled_pixels'] > 0
    assert not strict['mask'].any()
    original[:, 20:] = False
    other_nearby = candidate_interface(gt, original, (1, 2), support_policy='thin4')
    assert not other_nearby['support_mask'][gt == 2].any()
    assert not other_nearby['mask'].any()


def test_thick_own_fill_rejected_even_if_other_id_is_nearby():
    gt, original = two_grains((30, 40))
    original[:, 20:27] = False
    result = candidate_interface(gt, original, (1, 2), support_policy='thin4')
    assert not result['support_mask'][:, 20:23].any()
    assert not result['contact_mask'].any()
    assert not result['mask'].any()


def test_optimistic_interface_is_explicit_and_unknown_unchanged():
    gt, original = two_grains()
    original[:, 26:34] = False
    gt[10:15, 29:31] = 0;original[gt == 0] = False
    gt_before, orig_before = gt.copy(), original.copy()
    result = candidate_interface(gt, original, (1, 2), support_policy='processed_known_optimistic')
    assert result['summary']['optimistic_only']
    assert not result['summary']['reliable_for_training']
    assert result['provenance']['filled_pixels'] > 0
    assert not result['mask'][gt == 0].any()
    np.testing.assert_array_equal(gt, gt_before)
    np.testing.assert_array_equal(original, orig_before)


def test_statistics_reject_changed_core_and_wrong_marker_shape():
    cores = compact_cores()
    with pytest.raises(ValueError): core_marker_statistics(cores, np.ones((2, 2), np.int32))
    cores['core_map'][0, 0] = 2
    with pytest.raises(ValueError): core_marker_statistics(cores, np.ones((12, 20), np.int32))


def test_atoms_report_one_pixel_split_and_zero_without_threshold():
    cores = compact_cores();old = np.ones((12, 20), np.int32)
    new = old.copy();new[0, 0] = 2;new[1, 0] = 0
    result = compare_marker_atoms(cores, old, new)
    assert result['summary']['atom_splits'] == 1
    assert result['summary']['nonmajor_pixels'] == 1
    assert result['summary']['zero_pixels_from_old_positive'] == 1
    assert not result['summary']['atom_partition_preserved']
    assert result['summary']['new_merged_atom_pairs'] == 0
    json.dumps(result, allow_nan=False)


def test_atoms_track_new_cross_gt_merger_not_existing_shared_marker():
    cores = compact_cores();gt = cores['core_map']
    old = gt.copy();new = np.ones_like(gt)*8
    result = compare_marker_atoms(cores, old, new)
    assert result['summary']['new_merged_atom_pairs'] == 1
    assert result['new_merged_atom_pairs'][0]['a_shared_pixels'] == 120
    assert result['summary']['atom_splits'] == 0
    already_shared = compare_marker_atoms(cores, new, new+1)
    assert already_shared['summary']['new_merged_atom_pairs'] == 0
    assert already_shared['summary']['atom_partition_preserved']


def test_atoms_do_not_force_different_old_components_of_same_gt_to_stay_apart():
    gt = np.ones((12, 20), np.int32)
    cores = build_trusted_cores(gt, gt > 0, radius=0, min_component_pixels=1)
    old = gt.copy();old[:, 10:] = 2
    result = compare_marker_atoms(cores, old, gt)
    assert result['summary']['atom_count'] == 2
    assert result['summary']['new_merged_atom_pairs'] == 0
    assert result['summary']['atom_partition_preserved']


def test_probe_stability_rejects_old_zero_core_joining_another_gt():
    from tools.probe_boundary_core_supervision import core_intervention_is_stable
    # 目标1/2被正确分开，但原无种子的4进入3的种子。
    # 仅检查旧正种子atom会漏掉后一种副作用。
    gt = np.repeat(np.array([[1, 2, 3, 4]], np.int32), 4, axis=0)
    cores = build_trusted_cores(gt, gt > 0, radius=0, min_component_pixels=1)
    before = np.repeat(np.array([[1, 1, 3, 0]], np.int32), 4, axis=0)
    after = np.repeat(np.array([[1, 2, 3, 3]], np.int32), 4, axis=0)
    comparison = compare_core_separation(core_marker_statistics(cores, before),
                                         core_marker_statistics(cores, after), (1, 2))
    atoms = compare_marker_atoms(cores, before, after)
    assert comparison['target_separated_and_covered']
    assert atoms['summary']['atom_partition_preserved']
    assert comparison['new_cross_gt_shared_pairs'] == [[3, 4]]
    assert not core_intervention_is_stable(comparison, atoms)
    # 原0保持0时，相同目标分隔没有这种额外合并。
    after[:, 3] = 0
    comparison = compare_core_separation(core_marker_statistics(cores, before),
                                         core_marker_statistics(cores, after), (1, 2))
    assert core_intervention_is_stable(comparison, compare_marker_atoms(cores, before, after))
