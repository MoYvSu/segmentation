# -*- coding: utf-8 -*-
"""局部GT关系、unknown保护与真实CPU部署捕获的必要反例。"""
from copy import deepcopy

import numpy as np
import pytest
import torch

from tools import probe_affinity_partition as subject
from tools.analyze_affinity_fused_gt import diagnose_partition
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_loss import build_affinity_targets_torch


def config():
    return {'inference': dict(boundary_threshold=.65, threshold=.5, min_instance_area=20,
        max_instance_id=65535, bridge_width=1, watershed_dilate_width=1,
        marker_border_seal_width=2, marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8, semantic_vote_mode='probability_mean',
        marker_partition_restore={'enabled': False})}


def logits_for(ids):
    ids = torch.as_tensor(ids)[None]
    valid = torch.ones((1, 1, *ids.shape[-2:]), dtype=torch.bool)
    target, edges = build_affinity_targets_torch(ids, valid)
    return torch.where(edges & (target == 0), -16., 16.)


def decode(logits):
    boundary = affinity_boundary_probability(logits, mode='gated', short_reduction='top2')
    return subject.decode_capture(torch.zeros_like(boundary), boundary, config())


def board(merge=False):
    gt = np.ones((96, 96), np.int32); gt[12:84, 12:84] = 2
    if merge:
        gt[12:84, 48:84] = 3
    return gt


def case(kind='split'):
    return dict(case_id='toy', kind=kind, gt_ids=[2, 2] if kind == 'split' else [2, 3],
        points=[[48, 26], [48, 68]], reference_prediction_ids=[2, 3] if kind == 'split' else [2, 2])


def test_actual_watershed_marker_capture_matches_deployed_stages_and_no_mutation():
    boundary = torch.zeros((1, 1, 96, 96)); boundary[:, :, 12:84, 46:50] = .95
    semantic = torch.zeros_like(boundary)
    original = boundary.clone(); cfg = config(); cfg_original = deepcopy(cfg)
    result = subject.decode_capture(semantic, boundary, cfg)
    assert result['capture_audit']['actual_watershed_calls'] == 1
    assert np.array_equal(result['actual_markers'], result['stages']['marker_labels'])
    assert result['instances'].dtype == np.uint16
    assert torch.equal(boundary, original) and cfg == cfg_original
    cfg['inference']['marker_partition_restore']['enabled'] = True
    with pytest.raises(ValueError, match='disabled marker'):
        subject.decode_capture(semantic, boundary, cfg)


def test_empty_markers_skip_cv2_without_fabricated_capture():
    boundary = torch.zeros((1, 1, 20, 20)); cfg = config()
    cfg['inference']['min_instance_area'] = 10000
    result = subject.decode_capture(torch.zeros_like(boundary), boundary, cfg)
    assert result['capture_audit']['actual_watershed_calls'] == 0
    assert result['capture_audit']['marker_exact']
    assert not result['instances'].any() and result['classes'] == {}


def test_short_all_and_fixed_reference_fragment_isolation():
    gt = board(); pred = gt.copy(); pred[12:84, 48:84] = 3
    logits = torch.linspace(-3, 3, 8*96*96).reshape(1, 8, 96, 96)
    before = logits.clone(); gt_before = gt.copy(); pred_before = pred.copy()
    short, smask, scount = subject.correct_local_relations(logits, gt, np.ones_like(gt, bool), case(), partition=pred)
    full, fmask, fcount = subject.correct_local_relations(logits, gt, np.ones_like(gt, bool), case(), 'all', partition=pred)
    assert scount['selected_edges'] > 0 and fcount['selected_edges'] > scount['selected_edges']
    assert not smask[:, 4:].any() and torch.equal(short[:, 4:], logits[:, 4:])
    assert torch.equal(short[~smask], logits[~smask]) and torch.equal(full[~fmask], logits[~fmask])
    assert (short[smask] == 16).all() and (full[fmask] == 16).all()
    assert not smask[0, :, 30, 30].any()  # 内部同碎片不因同GT被清空。
    assert scount['selection_scope'] == 'split_reference_fragments'
    assert torch.equal(logits, before) and np.array_equal(gt, gt_before) and np.array_equal(pred, pred_before)
    other = torch.randn_like(logits)
    _, other_mask, _ = subject.correct_local_relations(other, gt, np.ones_like(gt, bool), case(), partition=pred)
    assert torch.equal(other_mask, smask)  # 固定同参考partition，对两个头完全同掩码。


def test_unknown_or_padding_inside_long_segment_ignored_but_third_known_GT_reported():
    gt = np.full((10, 12), 2, np.int32); gt[:, 6:] = 3
    gt[4, 4] = 0  # radius4边[4,2]→[4,6]两端已知但跨unknown。
    valid = np.ones_like(gt, bool); valid[5, 4] = False
    gt[6, 4] = 4  # 第三已知实例不是unknown；沿用原端点关系定义。
    logits = torch.full((1, 8, 10, 12), 2.)
    corrected, mask, counts = subject.correct_local_relations(logits, gt, valid, case('merge'), 'all')
    assert not mask[0, 6, 4, 2] and not mask[0, 6, 5, 2]
    assert mask[0, 6, 6, 2] and corrected[0, 6, 6, 2] == -16
    assert counts['path_filtered_edges'] >= 2 and counts['third_known_GT_path_edges'] > 0
    assert torch.equal(corrected[~mask], logits[~mask])


def test_merge_only_requested_unordered_GT_pair_and_zero_contact_is_explicit():
    gt = np.full((12, 18), 2, np.int32); gt[:, 6:12] = 3; gt[:, 12:] = 4
    logits = torch.full((1, 8, 12, 18), 2.)
    corrected, mask, counts = subject.correct_local_relations(logits, gt, np.ones_like(gt, bool), case('merge'), 'all')
    assert counts['selected_edges'] > 0 and (corrected[mask] == -16).all()
    assert not mask[0, 0, 5, 11]  # 3/4分界不是本case。
    pred = np.full(gt.shape, 2, np.int32); pred[:, 10:] = 3
    _, no_mask, empty = subject.correct_local_relations(logits, gt, np.ones_like(gt, bool), case(), partition=pred)
    assert not no_mask.any() and empty['no_contact']


def test_select_union_reference_priority_and_unknown_whole_object_censor():
    gt = board(); classes = {1: 0, 2: 1}
    split = gt.astype(np.uint16); split[12:84, 48:84] = 3
    perfect = gt.astype(np.uint16)
    def select(r1, tail):
        predictions = dict(r1=r1, tail=tail)
        labels = {'1': 0, '2': 1, '3': 1}
        diagnostics = {a: diagnose_partition(gt, p, classes, labels) for a, p in predictions.items()}
        return subject.select_cases(gt, classes, predictions, diagnostics)
    rows = select(split, split)
    assert len(rows) == 1 and rows[0]['reference_arm'] == 'r1'
    assert rows[0]['baseline_relations'] == {'r1': 'different', 'tail': 'different'}
    assert not any(rows[0]['censor'].values())
    assert select(perfect, split)[0]['reference_arm'] == 'tail'
    gt[40:42, 40:42] = 0
    assert select(split, split) == []  # 完整GT对象碰unknown后不进入保守并集。


def test_select_merge_uses_actual_common_prediction_and_GT_phases():
    gt = board(merge=True); merged = gt.astype(np.uint16); merged[merged == 3] = 2
    perfect = gt.astype(np.uint16); classes = {1: 0, 2: 1, 3: 0}
    predictions = dict(r1=merged, tail=perfect)
    diagnostics = {a: diagnose_partition(gt, p, classes, {'1': 0, '2': 1, '3': 0})
                   for a, p in predictions.items()}
    rows = subject.select_cases(gt, classes, predictions, diagnostics)
    assert len(rows) == 1 and rows[0]['kind'] == 'merge'
    assert rows[0]['gt_ids'] == [2, 3] and rows[0]['gt_phases'] == [1, 0]
    assert rows[0]['baseline_relations'] == {'r1': 'same', 'tail': 'different'}


@pytest.mark.parametrize('kind', ['split', 'merge'])
def test_synthetic_local_relation_repairs_final_partition_via_actual_fusion(kind):
    gt = board(merge=kind == 'merge')
    reference = gt.copy()
    if kind == 'split':
        reference[12:84, 48:84] = 3
    else:
        reference[reference == 3] = 2
    logits = logits_for(reference); before = decode(logits)
    c = case(kind)
    assert subject._point_relation(before['instances'], c['points']) == ('different' if kind == 'split' else 'same')
    corrected, mask, counts = subject.correct_local_relations(logits, gt, np.ones_like(gt, bool), c,
                                                              'all', partition=reference if kind == 'split' else None)
    after = decode(corrected); trace = subject.trace_relations(after, gt, [c])
    assert counts['selected_edges'] > 0 and mask.any()
    assert trace['cases'][0]['final_desired']
    assert trace['order'] == list(subject.STAGE_ORDER)
    assert after['instances'].dtype == np.uint16 and int(after['instances'].max()) <= 65535
    assert torch.equal(corrected[~mask], logits[~mask])


def test_trace_full_route_through_unknown_is_distinct_from_known_connection():
    gt = np.full((40, 40), 2, np.int32); gt[:, 20:] = 3; gt[:, 19:21] = 0
    result = subject.decode_capture(torch.zeros((1, 1, 40, 40)), torch.zeros((1, 1, 40, 40)), config())
    c = dict(kind='merge', gt_ids=[2, 3], points=[[20, 10], [20, 30]])
    trace = subject.trace_relations(result, gt, [c])
    assert trace['stages']['threshold']['full'] == ['same']
    assert trace['stages']['threshold']['known_only'] == ['different']
    assert trace['stages']['final']['unknown_route_dependent'] == [True]


def test_malformed_content_and_case_are_rejected():
    gt = board(); logits = torch.zeros((1, 8, 96, 96))
    with pytest.raises(ValueError, match='pixel valid_content'):
        subject.correct_local_relations(logits, gt, torch.ones_like(logits, dtype=torch.bool), case())
    invalid = case(); invalid['gt_ids'] = [2, 3]
    with pytest.raises(ValueError, match='identical GT IDs'):
        subject.correct_local_relations(logits, gt, np.ones_like(gt, bool), invalid)
    with pytest.raises(ValueError, match='distinct positive prediction IDs'):
        invalid = case(); invalid['reference_prediction_ids'] = [2, 2]
        subject.correct_local_relations(logits, gt, np.ones_like(gt, bool), invalid, partition=gt)


def test_uint16_reference_partition_and_rectangular_grid_keep_valid_horizontal_links():
    gt = np.full((2, 12), 2, np.uint16); gt[:, 6:] = 3
    logits = torch.zeros((1, 8, 2, 12))
    corrected, mask, counts = subject.correct_local_relations(logits, gt, np.ones_like(gt, bool),
                                                              case('merge'), 'all', partition=gt)
    assert mask[0, 4].any() and mask[0, 6].any()
    assert not mask[0, 5].any() and not mask[0, 7].any()
    assert counts['selected_edges'] > 0 and corrected[mask].eq(-16).all()
