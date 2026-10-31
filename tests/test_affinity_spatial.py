# -*- coding: utf-8 -*-
"""真实空间口径的合成回归：未知保护、组件像素量、数字连通与纯函数契约。"""
import json

import numpy as np
import pytest

from tools.probe_affinity_equal_recall import fused_targets
from tools.probe_affinity_spatial import compare_spatial_boundaries, spatial_boundary_diagnostic, self_test
from utils.affinity_graph import build_affinity_targets, DEFAULT_AFFINITY_OFFSETS


def scene(size=31):
    ids = np.ones((size, size), np.int64)
    return np.zeros(ids.shape, np.float32), ids, np.ones(ids.shape, bool)


def report(scores, ids, trust, **kwargs):
    return spatial_boundary_diagnostic(scores, ids, trust, **kwargs)['report']


def test_interface_uses_identical_short_target_and_all_four_valid_contract():
    scores, ids, trust = scene()
    ids[:, 15:] = 2
    ids[14:, 8:23] = 3
    trust[12:15, 10:13] = False
    target, valid = build_affinity_targets(ids, trust, offsets=DEFAULT_AFFINITY_OFFSETS[:4])
    truth, eligible = fused_targets(target.astype(bool), valid)
    result = spatial_boundary_diagnostic(scores, ids, trust, return_maps=True)
    assert np.array_equal(result['maps']['true_interface'], truth & eligible)
    assert np.array_equal(result['maps']['short_target_eligible'], eligible)
    assert result['report']['domain']['unique_interface_anchor_pixels'] == int((truth & eligible).sum())


def test_gap_size_is_component_pixels_not_arc_length_or_doubled_interface_width():
    scores, ids, trust = scene()
    ids[:, 16:] = 2
    scores[:] = 1
    scores[12:15, 15] = .2
    r = report(scores, ids, trust)
    assert r['interfaces']['accepted_gap_components'] == 1
    assert r['interfaces']['longest_accepted_gap_component_pixels'] == 3
    assert r['gap_rows'][0]['component_pixels'] == 3
    assert r['gap_rows'][0]['bbox_yxyx'] == [12, 15, 15, 16]
    assert r['settings']['gap_size'].endswith('not_arc_length')


def test_gap_components_touching_unknown_or_image_frame_are_censored():
    scores, ids, trust = scene()
    ids[:, 16:] = 2
    scores[:] = 1
    scores[0:3, 15] = .1
    r = report(scores, ids, trust)
    assert r['interfaces']['accepted_gap_components'] == 0
    assert r['interfaces']['censored_gap_components'] == 1
    scores[:] = 1
    trust[15, 14] = False
    scores[12:15, 15] = .1
    r = report(scores, ids, trust)
    assert r['interfaces']['accepted_gap_components'] == 0
    assert r['interfaces']['censored_gap_components'] == 1


def test_known_weak_interface_is_not_counted_as_false_internal_response():
    scores, ids, trust = scene()
    ids[:, 16:] = 2
    scores[:, 15:17] = 1
    r = report(scores, ids, trust)
    assert r['internal']['considered_components'] == 0


def test_single_isolated_interior_pixel_and_threshold_equality():
    scores, ids, trust = scene()
    scores[15, 15] = .65
    r = report(scores, ids, trust)
    assert r['internal']['accepted_components'] == 1
    assert r['internal']['accepted_response_pixels'] == 1
    assert r['internal']['categories']['isolated'] == 1
    scores[15, 15] = .649
    assert report(scores, ids, trust)['internal']['accepted_components'] == 0


def test_open_line_and_branched_response_are_described_without_claiming_cut():
    scores, ids, trust = scene()
    scores[10:21, 15] = .9
    r = report(scores, ids, trust)
    row = r['internal_rows'][0]
    assert row['category'] == 'branch_or_line'
    assert row['hole_count'] == 0 and row['skeleton_endpoint_pixels'] == 2
    assert row['contact_patches'] == 0 and not row['multi_contact_candidate']
    scores[15, 15:21] = .9
    branched = report(scores, ids, trust)['internal_rows'][0]
    assert branched['category'] == 'branch_or_line'
    assert branched['skeleton_junction_pixels'] > 0


def test_closed_ring_has_trusted_hole_and_is_only_a_candidate():
    scores, ids, trust = scene()
    scores[12:19, 12] = scores[12:19, 18] = .9
    scores[12, 12:19] = scores[18, 12:19] = .9
    r = report(scores, ids, trust)
    row = r['internal_rows'][0]
    assert row['category'] == 'closed_loop_candidate'
    assert row['hole_count'] == 1 and row['hole_pixels'] == 25
    assert r['internal']['accepted_response_pixels'] == 24
    assert not row['censored']


def test_unknown_enclosed_in_loop_censors_whole_component_even_without_direct_touch():
    scores, ids, trust = scene(41)
    scores[10:31, 10] = scores[10:31, 30] = .9
    scores[10, 10:31] = scores[30, 10:31] = .9
    trust[20, 20] = False
    r = report(scores, ids, trust)
    assert r['internal']['accepted_components'] == 0
    assert r['internal']['censored_components'] == 1
    assert 'encloses_unknown' in r['internal_rows'][0]['censor_reasons']
    assert 'touches_unknown' not in r['internal_rows'][0]['censor_reasons']


def test_other_gt_inside_ring_is_not_mislabeled_as_trusted_false_internal_loop():
    scores, ids, trust = scene(41)
    scores[10:31, 10] = scores[10:31, 30] = .9
    scores[10, 10:31] = scores[30, 10:31] = .9
    ids[19:22, 19:22] = 2
    r = report(scores, ids, trust)
    assert r['internal']['accepted_components'] == 0
    assert 'encloses_other_gt' in r['internal_rows'][0]['censor_reasons']


def test_complete_component_is_tracked_before_deep_filter_unknown_extension_not_hidden():
    scores, ids, trust = scene()
    scores[15, :20] = .9
    r = report(scores, ids, trust)
    assert r['internal']['considered_components'] == 1
    assert r['internal']['accepted_components'] == 0
    assert r['internal_rows'][0]['component_pixels'] == 20
    assert r['internal_rows'][0]['deep_pixels'] > 0
    assert 'touches_frame' in r['internal_rows'][0]['censor_reasons']
    scores[:]=0
    scores[15, 5:20] = .9
    trust[15, 4] = False
    r = report(scores, ids, trust)
    assert r['internal']['accepted_components'] == 0
    assert 'touches_unknown' in r['internal_rows'][0]['censor_reasons']


def test_two_boundary_contacts_are_disjoint_patches_not_adjacent_pixels():
    scores, ids, trust = scene()
    ids[:, :5] = 2
    ids[:, 26:] = 3
    scores[15, 6:25] = .9
    r = report(scores, ids, trust)
    row = r['internal_rows'][0]
    assert row['category'] == 'multi_contact_candidate'
    assert row['contact_patches'] == 2
    assert row['multi_contact_candidate']
    scores[:]=0
    scores[14:17, 6:16] = .9
    row = report(scores, ids, trust)['internal_rows'][0]
    assert row['contact_patches'] == 1
    assert not row['multi_contact_candidate']


def test_four_and_eight_connectivity_are_used_consistently_for_response_components():
    scores, ids, trust = scene()
    scores[12, 12] = scores[13, 13] = .9
    four = report(scores, ids, trust, connectivity=4)
    eight = report(scores, ids, trust, connectivity=8)
    assert four['internal']['accepted_components'] == 2
    assert eight['internal']['accepted_components'] == 1
    assert four['internal']['accepted_response_pixels'] == eight['internal']['accepted_response_pixels'] == 2
    assert four['settings']['hole_background_connectivity'] == 8
    assert eight['settings']['hole_background_connectivity'] == 4


def test_foreground_background_dual_connectivity_does_not_invent_diagonal_hole():
    scores, ids, trust = scene()
    scores[14, 15] = scores[15, 14] = scores[15, 16] = scores[16, 15] = .9
    four = report(scores, ids, trust, connectivity=4)
    eight = report(scores, ids, trust, connectivity=8)
    assert four['internal']['accepted_components'] == 4
    assert four['internal']['categories']['closed_loop_candidate'] == 0
    assert eight['internal']['accepted_components'] == 1
    assert eight['internal']['categories']['closed_loop_candidate'] == 1


def test_four_and_eight_connectivity_apply_to_gaps_as_well_as_high_responses():
    scores, ids, trust = scene()
    ids[:, 16:] = 2
    scores[:]=1
    scores[12, 15] = scores[13, 16] = .1
    four = report(scores, ids, trust, connectivity=4)
    eight = report(scores, ids, trust, connectivity=8)
    assert four['interfaces']['accepted_gap_components'] == 2
    assert eight['interfaces']['accepted_gap_components'] == 1
    assert four['interfaces']['accepted_gap_pixels_unique'] == eight['interfaces']['accepted_gap_pixels_unique'] == 2


def test_empty_trust_zero_id_padding_and_sparse_ids_do_not_create_supervision():
    scores, ids, trust = scene()
    scores[15, 15] = .9
    empty = report(scores, ids, np.zeros_like(trust))
    assert empty['domain']['gt_instances'] == 0
    assert empty['internal']['considered_components'] == empty['interfaces']['gap_components'] == 0
    ids[15, 15] = 0
    zero = report(scores, ids, trust)
    assert zero['internal']['considered_components'] == 0
    ids[:] = 9223372036854775000
    content = np.ones_like(trust)
    content[15, 15] = False
    padding = report(scores, ids, trust, valid_content=content)
    assert padding['internal']['considered_components'] == 0
    assert padding['domain']['gt_instances'] == 1
    json.dumps(empty, allow_nan=False)


def test_paired_comparison_uses_same_gt_and_returns_only_structural_differences():
    scores, ids, trust = scene()
    changed = scores.copy()
    changed[15, 15] = .9
    result = compare_spatial_boundaries({'gated': scores, 'short': changed}, ids, trust, return_maps=True)
    assert result['report']['arms']['gated']['domain'] == result['report']['arms']['short']['domain']
    assert result['report']['changes_relative_to_reference']['short']['internal.accepted_components'] == 1
    assert result['report']['interpretation'].endswith('not_accuracy')
    assert result['maps']['gated']['accepted_internal_high_response'].sum() == 0
    assert result['maps']['short']['accepted_internal_high_response'].sum() == 1
    json.dumps(result['report'], allow_nan=False)


def test_inputs_and_readonly_arrays_remain_byte_identical():
    scores, ids, trust = scene()
    scores[15, 15] = .9
    snapshots = [a.tobytes() for a in (scores, ids, trust)]
    for a in (scores, ids, trust):
        a.flags.writeable = False
    spatial_boundary_diagnostic(scores, ids, trust)
    assert snapshots == [a.tobytes() for a in (scores, ids, trust)]


@pytest.mark.parametrize('bad', [np.nan, np.inf, -.01, 1.01])
def test_probability_bounds_and_nonfinite_values_are_rejected_even_in_unknown(bad):
    scores, ids, trust = scene()
    scores[0, 0] = bad
    trust[0, 0] = False
    with pytest.raises(ValueError):
        report(scores, ids, trust)


@pytest.mark.parametrize('kwargs', [dict(connectivity=6), dict(connectivity=True), dict(deep_radius=0),
    dict(deep_radius=1.5), dict(threshold=0), dict(threshold=float('nan')), dict(return_maps=1)])
def test_invalid_configuration_is_not_silently_reinterpreted(kwargs):
    with pytest.raises(ValueError):
        report(*scene(), **kwargs)


def test_shapes_noninteger_ids_and_nonbinary_trust_are_rejected():
    scores, ids, trust = scene()
    for wrong_ids in (ids[:-1], ids.astype(float), -ids):
        with pytest.raises(ValueError):
            report(scores, wrong_ids, trust)
    weighted = trust.astype(float)
    weighted[15, 15] = .5
    with pytest.raises(ValueError):
        report(scores, ids, weighted)
    with pytest.raises(ValueError):
        report(scores, ids, trust, valid_content=trust[:-1])
    with pytest.raises(ValueError):
        compare_spatial_boundaries({'short': scores}, ids, trust)
    with pytest.raises(ValueError):
        compare_spatial_boundaries({'short': scores, 'other': scores}, ids, trust)


def test_synthetic_self_test_has_no_dataset_dependency():
    self_test()
