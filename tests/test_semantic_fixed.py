# -*- coding: utf-8 -*-
"""固定几何的语义配对、严格阈值和PNG字节复制。"""
import numpy as np

from tools.probe_semantic_fixed import class_independent_pairs, copy_fixed_mask, fixed_confusion, probability_details
from tools.semantic_crossover import revote_instances


def test_class_independent_pairs_ignore_unknown_and_confusion_changes_only():
    gt = np.array([[1, 1, 0, 2, 2], [1, 1, 0, 2, 2]], np.uint16)
    pred = np.array([[4, 4, 4, 7, 7], [4, 4, 4, 7, 7]], np.uint16)
    pairs, stats = class_independent_pairs(gt, gt > 0, pred)
    assert [(p['gt'], p['pred'], p['iou']) for p in pairs] == [(1, 4, 1.), (2, 7, 1.)]
    assert stats['matched'] == 2
    original = fixed_confusion(pairs, {1:1, 2:0}, {'4':0, '7':0})
    changed = fixed_confusion(pairs, {1:1, 2:0}, {'4':1, '7':0})
    assert original['true_F_pred_P'] == 1 and changed['true_F_pred_P'] == 0
    assert original['evaluated_gt_known_pixels'] == changed['evaluated_gt_known_pixels'] == 8


def test_merged_geometry_has_one_pair_not_two():
    gt = np.array([[1, 1, 2, 2]], np.uint16)
    pred = np.full_like(gt, 6)
    pairs, stats = class_independent_pairs(gt, gt > 0, pred)
    assert len(pairs) == 1 and pairs[0]['iou'] == .5 and stats['unmatched_gt'] == 1


def test_zero_prediction_and_empty_trust_do_not_generate_semantic_pair():
    gt = np.ones((2, 2), np.uint16)
    pairs, stats = class_independent_pairs(gt, gt > 0, np.zeros_like(gt))
    assert not pairs and stats['unmatched_gt'] == 1
    pairs, stats = class_independent_pairs(gt, np.zeros_like(gt, bool), gt)
    assert not pairs and stats['matched'] == 0


def test_strict_half_vote_and_core_are_diagnostic_only():
    ids = np.ones((7, 7), np.uint16)
    p = np.full(ids.shape, .5, np.float32)
    classes, scores = revote_instances(ids, p)
    assert classes == {'1':0}
    p[2:5, 2:5] = .9
    classes, scores = revote_instances(ids, p)
    details = probability_details(ids, p, scores)
    assert details['1']['core_pixels'] + details['1']['outer_pixels'] == ids.size
    assert details['1']['probability_core'] > details['1']['probability_all']
    assert classes == {'1':1}


def test_png_copy_preserves_bytes(tmp_path):
    source, target = tmp_path/'old.png', tmp_path/'new'/'fixed.png'
    value = b'\x89PNG\r\n\x1a\nopaque byte preservation test'
    source.write_bytes(value)
    copy_fixed_mask(source, target)
    assert target.read_bytes() == value and source.read_bytes() == value
