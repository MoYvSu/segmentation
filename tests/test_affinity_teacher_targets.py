# -*- coding: utf-8 -*-
"""同源几何标签A/B、共同ignore与canonical resize的虚构CPU合同。"""
from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from utils.affinity_loss import build_affinity_targets_torch
from utils.affinity_teacher_targets import (build_teacher_pair, load_teacher_pair,
                                           resize_teacher_pair)
from utils.offset_letterbox import letterbox_instance_geometry


ALLOW = [f'train_{i:03d}.jpg' for i in range(32)]


def sample():
    manual = np.ones((17, 25), np.uint16)
    manual[:, 13:] = 2
    manual[0:2] = 0
    original = manual > 0
    original[8:10, 7:11] = False
    teacher = np.where(manual == 1, 41, np.where(manual == 2, 99, 31)).astype(np.uint16)
    teacher[:, 8] = 0
    return manual, teacher, original


def build(manual, teacher, original, **kwargs):
    return build_teacher_pair(manual, teacher, original, source_name='train_001.jpg',
                              allowed_training_sources=ALLOW, **kwargs)


def test_native_common_known_original_filled_loss_and_inputs_unchanged():
    manual, teacher, original = sample()
    before = [x.copy() for x in (manual, teacher, original)]
    pair, audit = build(manual, teacher, original,
                        filled=(manual > 0) & ~original, residual_unknown=manual == 0)
    common = (manual > 0) & (teacher > 0)
    assert np.array_equal(pair['common_known'], common)
    assert np.array_equal(pair['manual_masked'] > 0, pair['teacher_masked'] > 0)
    assert np.array_equal(pair['manual_masked'][common], manual[common])
    assert np.array_equal(pair['teacher_masked'][common], teacher[common])
    assert not pair['teacher_masked'][manual == 0].any()
    assert audit['pixels']['teacher_zero_on_manual_known'] == 15
    assert audit['pixels']['teacher_zero_on_filled'] == 2
    assert audit['pixels']['ignored_teacher_positive_outside_manual'] > 0
    assert audit['teacher_human_confirmation_granted'] is False
    json.dumps(audit, allow_nan=False)
    assert all(np.array_equal(x, old) for x, old in zip((manual, teacher, original), before))


def test_actual_nearest_letterbox_all_eight_targets_and_common_legal_match():
    manual, teacher, original = sample()
    pair, _ = build(manual, teacher, original)
    geometry, stats = resize_teacher_pair(pair, input_size=1024, output_grid=25)
    expected, valid, meta = letterbox_instance_geometry(pair['manual_masked'], 1024, 25)
    targets, legal = build_affinity_targets_torch(torch.from_numpy(expected.astype(np.int64))[None],
                                                torch.from_numpy(valid.copy())[None, None])
    assert np.array_equal(geometry['manual_grid'], expected)
    assert geometry['metadata'] == meta.to_dict()
    assert torch.equal(geometry['manual_targets'], targets) and torch.equal(geometry['edge_valid'], legal)
    assert stats['common_edge_mask_equal'] and stats['lost_manual_edges_due_teacher_zero'] > 0
    assert stats['strata']['all']['conflict_edges'] == 0  # IDs 1/2与41/99只是重编号。
    assert stats['strata']['all']['selected_edges'] == int(legal.sum())
    assert all(len(s['per_channel']) == 8 for s in stats['strata'].values())
    assert (stats['strata']['original_original']['selected_edges']
            +stats['strata']['involving_filled']['selected_edges'] == int(legal.sum()))
    json.dumps(stats, allow_nan=False)


def test_thin_native_teacher_zero_can_disappear_on_actual_nearest_grid():
    manual, teacher, original = sample()
    pair, audit = build(manual, teacher, original)
    geometry, stats = resize_teacher_pair(pair, output_grid=13)
    assert audit['pixels']['teacher_zero_on_manual_known'] == 15
    assert stats['lost_manual_edges_due_teacher_zero'] == 0
    original_grid, valid, _ = letterbox_instance_geometry(manual, 1024, 13)
    _, original_legal = build_affinity_targets_torch(
        torch.from_numpy(original_grid.astype(np.int64))[None],
        torch.from_numpy(valid.copy())[None, None])
    assert torch.equal(geometry['edge_valid'], original_legal)


def test_same_different_conflicts_both_directions_and_strata_preserved():
    manual, teacher, original = sample()
    teacher = np.ones_like(manual)*11
    teacher[:, :5] = 12  # manual1被teacher分开。
    pair, _ = build(manual, teacher, original)
    _, stats = resize_teacher_pair(pair, output_grid=25)
    all_stats = stats['strata']['all']
    assert all_stats['manual_same_teacher_different'] > 0
    assert all_stats['manual_different_teacher_same'] > 0
    assert all_stats['conflict_edges'] == (all_stats['manual_same_teacher_different']
                                          +all_stats['manual_different_teacher_same'])
    assert stats['strata']['involving_filled']['selected_edges'] > 0
    assert stats['lost_manual_edges_due_teacher_zero'] == 0


def test_all_teacher_zero_returns_empty_shared_legal_without_invented_boundaries():
    manual, teacher, original = sample()
    teacher[:] = 0
    pair, audit = build(manual, teacher, original)
    geometry, stats = resize_teacher_pair(pair, output_grid=11)
    assert not geometry['edge_valid'].any()
    assert not geometry['manual_targets'].any() and not geometry['teacher_targets'].any()
    assert stats['strata']['all']['conflict_fraction'] is None
    assert stats['lost_manual_edge_fraction'] == 1.
    assert audit['pixels']['common_known'] == 0


@pytest.mark.parametrize('source', ['test_101.jpg','train_999.jpg','../train_001.jpg'])
def test_test_unknown_or_path_sources_rejected(source):
    manual, teacher, original = sample()
    with pytest.raises(ValueError):
        build_teacher_pair(manual, teacher, original, source_name=source, allowed_training_sources=ALLOW)


def test_fixed_32_allowlist_and_gt_provenance_must_be_exact():
    manual, teacher, original = sample()
    with pytest.raises(ValueError, match='32'):
        build_teacher_pair(manual, teacher, original, source_name='train_001', allowed_training_sources=ALLOW[:31])
    with pytest.raises(ValueError, match='filled'):
        build(manual, teacher, original, filled=np.zeros_like(original))
    bad = original.copy(); bad[0, 0] = True
    with pytest.raises(ValueError, match='unknown'):
        build(manual, teacher, bad)
    with pytest.raises(ValueError, match='residual_unknown'):
        build(manual, teacher, original, residual_unknown=np.zeros_like(original))


@pytest.mark.parametrize('bad_teacher', [np.ones((17,25),np.uint8), np.ones((17,25,3),np.uint16),
                                        np.ones((16,25),np.uint16)])
def test_teacher_single_channel_uint16_and_same_shape_required(bad_teacher):
    manual, _, original = sample()
    with pytest.raises(ValueError):
        build(manual, bad_teacher, original)


def test_ids_overflow_negative_and_tampered_common_mask_rejected():
    manual, teacher, original = sample()
    big = manual.astype(np.int64); big[4,4] = 65536
    with pytest.raises(ValueError, match='65535'):
        build(big, teacher, original)
    big[4,4] = -1
    with pytest.raises(ValueError, match='65535'):
        build(big, teacher, original)
    pair, _ = build(manual, teacher, original)
    bad = deepcopy(pair); bad['teacher_masked'][4,8] = 99
    with pytest.raises(ValueError, match='inconsistent'):
        resize_teacher_pair(bad, output_grid=11)


def test_loader_readonly_hashes_uint16_and_declared_manual_source(tmp_path):
    manual, teacher, original = sample()
    mpath, tpath = tmp_path/'train_001_gt.npz', tmp_path/'train_001_inst.png'
    np.savez_compressed(mpath,instance_map=manual,original_covered=original,
                        filled=(manual > 0)&~original,residual_unknown=manual == 0)
    assert cv2.imwrite(str(tpath),teacher)
    before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (mpath,tpath)}
    pair,audit = load_teacher_pair(mpath,tpath,source_name='train_001',allowed_training_sources=ALLOW)
    assert audit['input_files'] == before
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == before[str(p)] for p in (mpath,tpath))
    assert pair['teacher_masked'].dtype == np.uint16
    assert audit['teacher_file_source_binding_verified_here'] is False
    with pytest.raises(ValueError,match='declared'):
        load_teacher_pair(mpath,tpath,source_name='train_002',allowed_training_sources=ALLOW)
