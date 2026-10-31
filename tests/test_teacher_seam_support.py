# -*- coding: utf-8 -*-
"""虚构教师窄缝修补与既有端点监督的CPU合同，不读取赛题数据。"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from utils.teacher_seam_support import repair_teacher_zero_seams, resize_seam_pair


ALLOW = [f'train_{i:03d}' for i in range(32)]


def sample():
    manual = np.ones((11, 17), np.uint16)
    manual[:, 9:] = 2
    teacher = np.full(manual.shape, 41, np.uint16)
    teacher[:, 9:] = 99
    teacher[:, 8] = 0
    original = manual > 0
    original[3:6, 6:11] = False
    return manual, teacher, original


def repair(manual, teacher, original, **kwargs):
    return repair_teacher_zero_seams(manual, teacher, original, source_name='train_001',
                                    allowed_training_sources=ALLOW, **kwargs)


def test_one_pixel_seam_repaired_by_nearest_coordinate_tie_not_id_and_inputs_unchanged():
    manual, teacher, original = sample()
    teacher[:, :8] = 500
    teacher[:, 9:] = 3
    before = [x.copy() for x in (manual, teacher, original)]
    pair, audit = repair(manual, teacher, original)
    assert pair['assigned_mask'][:, 8].all()
    assert np.all(pair['teacher_masked'][:, 8] == 500)  # 左源坐标更小，但ID更大。
    assert np.array_equal(pair['teacher_masked'][teacher > 0], teacher[teacher > 0])
    assert np.array_equal(np.unique(pair['teacher_masked']), [3, 500])
    assert np.all(pair['assigned_distance_native'] == 1.)
    assert np.array_equal(pair['nearest_sources_native'][:, 1], np.full(11, 7))
    assert audit['pixels']['assigned'] == 11
    assert audit['parameters']['max_distance_native'] == 1.
    assert audit['parameters']['context_radius_native'] == 2.
    assert audit['manual_instance_answers_used_for_assignment'] is False
    assert all(np.array_equal(x, old) for x, old in zip((manual, teacher, original), before))
    json.dumps(audit, allow_nan=False)


def test_crossing_seam_nearest_provenance_and_row_before_column_tie():
    manual = np.ones((9, 9), np.uint16)
    teacher = np.full(manual.shape, 31, np.uint16)
    teacher[4, 4] = 0
    teacher[3, 4] = 60000
    teacher[4, 3] = 7
    pair, _ = repair(manual, teacher, manual > 0)
    assert pair['teacher_masked'][4, 4] == 60000
    assert np.array_equal(pair['nearest_sources_native'], [[3, 4]])
    positions, sources = pair['assignment_coordinates_native'], pair['nearest_sources_native']
    y, x = positions.T
    sy, sx = sources.T
    assert np.all(manual[sy, sx] > 0) and np.all(teacher[sy, sx] > 0)
    assert np.array_equal(pair['teacher_masked'][y, x], teacher[sy, sx])
    assert np.allclose(pair['assigned_distance_native'], np.hypot(y - sy, x - sx))


def test_unknown_remains_ignore_and_unknown_teacher_id_cannot_be_a_donor():
    manual, teacher, original = sample()
    manual[:, :8] = 0
    original[:, :8] = False
    teacher[:, :8] = 777
    pair, audit = repair(manual, teacher, original)
    assert not pair['teacher_masked'][:, :9].any()
    assert not pair['assigned_mask'].any()  # 只剩known内99一个ID，不是两实例间窄缝。
    assert 777 not in pair['teacher_masked']
    assert audit['pixels']['assigned'] == 0


def test_single_instance_internal_hole_not_relabelled_as_a_boundary():
    manual = np.ones((11, 13), np.uint16)
    teacher = np.full(manual.shape, 9, np.uint16)
    teacher[4:7, 5:8] = 0
    pair, _ = repair(manual, teacher, manual > 0)
    assert not pair['assigned_mask'].any()
    assert np.array_equal(pair['teacher_masked'], teacher)


def test_wide_gap_has_no_filled_rim_even_with_two_teacher_ids_in_context():
    manual = np.ones((11, 17), np.uint16)
    teacher = np.ones_like(manual) * 41
    teacher[:, 5:10] = 0
    teacher[:, 10:] = 99
    pair, audit = repair(manual, teacher, manual > 0, context_radius_native=6.)
    assert audit['pixels']['nearby_two_teacher_ids'] > 0
    assert audit['pixels']['rejected_wide_void_neighbor'] > 0
    assert not pair['assigned_mask'].any()


def test_tiny_positive_instance_and_all_existing_positives_are_preserved():
    manual, teacher, original = sample()
    teacher[5, 9] = 65535
    pair, _ = repair(manual, teacher, original)
    assert pair['teacher_masked'][5, 9] == 65535
    assert np.array_equal(pair['teacher_masked'][teacher > 0], teacher[teacher > 0])
    assert np.array_equal(np.unique(pair['teacher_masked'][pair['common_known']]),
                          np.unique(teacher[teacher > 0]))


def test_identical_result_repeated_and_manual_instance_answers_do_not_assign_ids():
    manual, teacher, original = sample()
    a, aa = repair(manual, teacher, original)
    manual[:] = 432
    b, ba = repair(manual, teacher, original)
    assert np.array_equal(a['teacher_masked'], b['teacher_masked'])
    assert np.array_equal(a['assigned_mask'], b['assigned_mask'])
    c, ca = repair(manual, teacher, original)
    assert ba == ca
    assert np.array_equal(b['nearest_sources_native'], c['nearest_sources_native'])
    assert aa['hashes']['teacher_repaired'] == ba['hashes']['teacher_repaired']


@pytest.mark.parametrize('maximum,context', [(0, 2), (-1, 2), (np.nan, 2), (1, .5), (1, np.inf)])
def test_invalid_distance_settings_rejected(maximum, context):
    with pytest.raises(ValueError):
        repair(*sample(), max_distance_native=maximum, context_radius_native=context)


def test_empty_teacher_and_no_zero_seam_keep_empty_or_unchanged_support():
    manual, teacher, original = sample()
    teacher[:] = 0
    pair, audit = repair(manual, teacher, original)
    assert not pair['assigned_mask'].any()
    assert pair['nearest_sources_native'].shape == (0, 2)
    geometry, stats = resize_seam_pair(pair, output_grid=17)
    assert not geometry['edge_valid'].any()
    assert stats['added_legal_edges'] == 0
    json.dumps(stats, allow_nan=False)
    teacher[:] = 5
    pair, _ = repair(manual, teacher, original)
    assert np.array_equal(pair['teacher_masked'], teacher)


def test_nearest_letterbox_added_edges_existing_answers_and_filled_partition():
    pair, _ = repair(*sample())
    geometry, stats = resize_seam_pair(pair, input_size=1024, output_grid=17)
    oldlegal, legal = geometry['old_edge_valid'], geometry['edge_valid']
    assert not torch.any(oldlegal & ~legal)
    assert torch.equal(geometry['added_edge_valid'], legal & ~oldlegal)
    assert stats['old_legal_subset_verified'] and stats['old_teacher_answers_unchanged']
    assert stats['added_legal_edges'] > 0
    assert stats['new_legal_edges'] == stats['old_legal_edges'] + stats['added_legal_edges']
    all_added = stats['strata']['all_added']
    assert all_added['manual_same_teacher_same'] > 0
    assert all_added['manual_different_teacher_different'] > 0
    assert (stats['strata']['original_original_added']['selected_edges']
            + stats['strata']['involving_filled_added']['selected_edges'] == stats['added_legal_edges'])
    assert all(len(row['per_channel']) == 8 for row in stats['strata'].values())
    assert not geometry['edge_valid'][:, :, 11:, :].any()  # 等比letterbox下方padding。
    json.dumps(stats, allow_nan=False)


def test_actual_512_grid_can_remove_a_thin_seam_without_creating_extra_edge_support():
    manual = np.ones((1024, 1024), np.uint16)
    manual[:, 501:] = 2
    teacher = np.where(manual == 1, 41, 99).astype(np.uint16)
    teacher[:, 501] = 0  # nearest512取偶数列，零缝没有被采到。
    pair, audit = repair(manual, teacher, manual > 0)
    _, stats = resize_seam_pair(pair)
    assert audit['pixels']['assigned'] == 1024
    assert stats['added_legal_edges'] == 0
    assert stats['old_legal_edges'] == stats['new_legal_edges']


def test_semantic_ff_fp_pp_audit_never_changes_targets_or_assignment():
    pair, _ = repair(*sample())
    sem = np.ones(pair['manual_original'].shape, np.int8)
    sem[5:, :] = 0
    sem[0, :] = -1
    geometry, stats = resize_seam_pair(pair, output_grid=17, semantic=sem)
    ordinary_geometry, ordinary_stats = resize_seam_pair(pair, output_grid=17)
    assert torch.equal(geometry['teacher_targets'], ordinary_geometry['teacher_targets'])
    assert stats['strata']['all_added'] == ordinary_stats['strata']['all_added']
    names = ('FF_added', 'FP_added', 'PP_added', 'class_unknown_added')
    assert sum(stats['strata'][name]['selected_edges'] for name in names) == stats['added_legal_edges']
    assert all(stats['strata'][name]['selected_edges'] > 0 for name in names)
    with pytest.raises(ValueError):
        resize_seam_pair(pair, semantic=np.ones_like(sem) * 2)


def test_tampered_old_support_answer_or_unknown_assignment_rejected():
    pair, _ = repair(*sample())
    bad = deepcopy(pair)
    bad['teacher_masked'][4, 4] = 777
    with pytest.raises(ValueError, match='Original labels'):
        resize_seam_pair(bad, output_grid=17)
    bad = deepcopy(pair)
    bad['old_common_known'][2, 8] = True
    with pytest.raises(ValueError):
        resize_seam_pair(bad, output_grid=17)


def test_existing_source_allowlist_and_uint16_contract_are_still_required():
    manual, teacher, original = sample()
    with pytest.raises(ValueError, match='test'):
        repair_teacher_zero_seams(manual, teacher, original, source_name='test_101',
                                 allowed_training_sources=ALLOW)
    with pytest.raises(ValueError, match='uint16'):
        repair(manual, teacher.astype(np.uint8), original)
