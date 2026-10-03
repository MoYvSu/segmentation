# -*- coding: utf-8 -*-
"""CPU验证实际预测区域语义监督、原生读出和冻结输入的边界。"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from utils.semantic_object_fit import (build_semantic_batch, canonical_sha, feature_copy,
    object_outcomes, paired_schedule, region_probability_loss, select_regions)


def _regions_case():
    gt = np.ones((12, 12), np.int32)
    gt[:, 6:] = 2
    gt[6, 2:4] = 3
    original = gt > 0
    pred = np.zeros_like(gt, dtype=np.uint16)
    pred[2:4, 2:4] = 1  # 全原覆盖F。
    pred[2:4, 8:10] = 2  # 全原覆盖P。
    pred[4:6, 5:7] = 3  # 同一预测跨两个相。
    pred[6:8, 2:4] = 4  # 两个GT身份，但同为F，仍是纯相区域。
    pred[8:10, 8:10] = 5
    original[8, 8] = False  # 只一个filled像素也应排除整个预测区域。
    pred[8:10, 2:4] = 6
    gt[8, 2] = 0
    original[8, 2] = False  # 只一个unknown像素也不能裁掉再池化。
    pred[0, 3] = 7  # 原生触框。
    return pred, gt, original, {1: 1, 2: 0, 3: 1}


def test_all_strict_pure_regions_selected_including_correct_and_same_phase_multiple_gt():
    pred, gt, original, classes = _regions_case()
    before = (pred.copy(), gt.copy(), original.copy(), deepcopy(classes))
    regions, audit = select_regions(pred, gt, original, classes)
    assert regions == [dict(id=1, cls=1, native_area=4), dict(id=2, cls=0, native_area=4),
                       dict(id=4, cls=1, native_area=4)]
    assert audit['selection_independent_of_predicted_class']
    assert audit['ferrite'] == 2 and audit['pearlite'] == 1
    assert audit['unknown_filled_and_frame_excluded']
    for old, new in zip(before[:3], (pred, gt, original)):
        np.testing.assert_array_equal(old, new)
    assert classes == before[3]


@pytest.mark.parametrize('mutation', ['pred_dtype', 'pred_shape', 'original_on_unknown',
                                     'mixed_gt_dtype', 'missing_class', 'invalid_class'])
def test_selection_rejects_wrong_geometry_or_gt_class_contract(mutation):
    pred, gt, original, classes = _regions_case()
    if mutation == 'pred_dtype':
        pred = pred.astype(np.int32)
    elif mutation == 'pred_shape':
        pred = pred[:-1]
    elif mutation == 'original_on_unknown':
        original[8, 2] = True
    elif mutation == 'mixed_gt_dtype':
        gt = gt.astype(np.float32)
    elif mutation == 'missing_class':
        classes.pop(3)
    else:
        classes[3] = 2
    with pytest.raises(ValueError):
        select_regions(pred, gt, original, classes)


def test_extra_class_keys_never_override_present_gt_truth():
    pred, gt, original, classes = _regions_case()
    expected, _ = select_regions(pred, gt, original, classes)
    for key in (-1, 0):
        contaminated = dict(classes, **{})
        contaminated[key] = 0
        try:
            actual, _ = select_regions(pred, gt, original, contaminated)
        except ValueError:
            continue
        assert actual == expected


def test_native_probability_mean_differs_from_mean_logits_and_uses_strict_deployment_threshold():
    logits = torch.tensor([8., 8., -20.], dtype=torch.float32).reshape(1, 1, 1, 3).requires_grad_(True)
    ids = np.ones((1, 3), np.uint16)
    regions = [dict(id=1, cls=1, native_area=3)]
    loss, stats = region_probability_loss(logits, ids, regions)
    expected = -torch.log(logits.sigmoid().mean())
    wrong_order = F.binary_cross_entropy_with_logits(logits.mean(), logits.new_tensor(1.))
    assert torch.allclose(loss, expected, atol=1e-6)
    assert not torch.isclose(loss, wrong_order)
    assert stats['region_rows'][0]['probability'] > .5
    assert logits.mean().sigmoid().item() < .5
    tie = torch.zeros_like(logits)
    _, tie_stats = region_probability_loss(tie, ids, regions)
    assert tie_stats['region_rows'][0]['probability'] == .5
    assert int(tie_stats['region_rows'][0]['probability'] > .5) == 0


def test_region_object_class_balance_is_equal_per_object_then_equal_per_present_class():
    probability = torch.tensor([.9, .8, .5, .6, .6], dtype=torch.float32)
    logits = torch.logit(probability).reshape(1, 1, 1, 5).requires_grad_(True)
    ids = np.array([[1, 2, 3, 4, 4]], np.uint16)
    regions = [dict(id=i, cls=int(i != 4), native_area=2 if i == 4 else 1) for i in range(1, 5)]
    loss, stats = region_probability_loss(logits, ids, regions)
    expected = ((-torch.log(probability[:3])).mean() - torch.log(1 - probability[3])) / 2
    pixel_average = F.binary_cross_entropy(probability, torch.tensor([1., 1., 1., 0., 0.]))
    assert torch.allclose(loss, expected, atol=1e-6)
    assert not torch.isclose(loss, pixel_average)
    assert stats['ferrite'] == 3 and stats['pearlite'] == 1 and stats['active_classes'] == 2
    json.dumps(stats, allow_nan=False)


def test_additional_native_logit_gradient_is_zero_outside_complete_selected_regions():
    logits = torch.zeros((1, 1, 5, 7), dtype=torch.float32, requires_grad=True)
    ids = np.zeros((5, 7), np.uint16)
    ids[1:3, 1:3] = 1
    ids[3, 4:6] = 2
    regions = [dict(id=1, cls=1, native_area=4)]
    loss, _ = region_probability_loss(logits, ids, regions)
    loss.backward()
    selected = torch.from_numpy(ids == 1)
    assert not logits.grad[0, 0][~selected].count_nonzero()
    assert logits.grad[0, 0][selected].count_nonzero() == 4
    assert (logits.grad[0, 0][selected] < 0).all()


def test_region_loss_reaches_decoder_through_native_restore_without_unfreezing_cached_features():
    # 原生附加监督仅一像素，插值祖先的四个邻点有梯度属于正常回传。
    source = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4) / 12
    cached = feature_copy(source, 'cpu')
    decoder = torch.nn.Conv2d(1, 1, 1).eval().requires_grad_(True)
    with torch.no_grad():
        decoder.weight.fill_(.3)
        decoder.bias.fill_(-.1)
    ancestor = decoder(cached)
    ancestor.retain_grad()
    native = F.interpolate(ancestor, size=(6, 8), mode='bilinear', align_corners=False).cpu()
    native.retain_grad()
    ids = np.zeros((6, 8), np.uint16)
    ids[2, 2] = 1
    loss, _ = region_probability_loss(native, ids, [dict(id=1, cls=1, native_area=1)])
    head_grads = torch.autograd.grad(loss, tuple(decoder.parameters()), retain_graph=True)
    assert all(torch.isfinite(g).all() and g.count_nonzero() for g in head_grads)
    loss.backward()
    selected = torch.from_numpy(ids == 1)
    assert native.grad[0, 0][selected].count_nonzero() == 1
    assert not native.grad[0, 0][~selected].count_nonzero()
    assert ancestor.grad.count_nonzero() == 4
    assert not cached.requires_grad and cached.grad is None and source.grad is None
    assert torch.equal(cached, source)


def test_empty_region_source_is_differentiable_zero_and_missing_class_uses_present_mean():
    logits = torch.zeros((1, 1, 2, 2), dtype=torch.float32, requires_grad=True)
    ids = np.ones((2, 2), np.uint16)
    zero, stats = region_probability_loss(logits, ids, [])
    assert zero.item() == 0 and stats['active_classes'] == 0
    zero.backward()
    assert not logits.grad.count_nonzero()
    logits.grad = None
    loss, stats = region_probability_loss(logits, ids, [dict(id=1, cls=0, native_area=4)])
    assert torch.allclose(loss, logits.new_tensor(np.log(2)), atol=1e-6)
    assert stats['active_classes'] == 1 and stats['ferrite'] == 0
    loss.backward()
    assert (logits.grad > 0).all()


@pytest.mark.parametrize('mutation', ['duplicate', 'missing_id', 'changed_area', 'invalid_class',
                                     'wrong_shape', 'nonfinite', 'wrong_precision'])
def test_region_loss_rejects_tampered_fixed_regions_and_probability_input(mutation):
    logits = torch.zeros((1, 1, 3, 4), dtype=torch.float32, requires_grad=True)
    ids = np.ones((3, 4), np.uint16)
    regions = [dict(id=1, cls=1, native_area=12)]
    if mutation == 'duplicate':
        regions *= 2
    elif mutation == 'missing_id':
        regions[0]['id'] = 2
    elif mutation == 'changed_area':
        regions[0]['native_area'] = 11
    elif mutation == 'invalid_class':
        regions[0]['cls'] = 2
    elif mutation == 'wrong_shape':
        ids = ids[:-1]
    elif mutation == 'nonfinite':
        logits = logits.detach()
        logits[0, 0, 0, 0] = float('nan')
    else:
        logits = logits.double()
    with pytest.raises(ValueError):
        region_probability_loss(logits, ids, regions)


def test_extreme_wrong_probabilities_remain_finite_without_using_unselected_pixels():
    logits = torch.full((1, 1, 2, 2), -1000., dtype=torch.float32, requires_grad=True)
    ids = np.ones((2, 2), np.uint16)
    loss, _ = region_probability_loss(logits, ids, [dict(id=1, cls=1, native_area=4)])
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_common_targets_keep_processed_fill_ignore_unknown_and_use_letterbox_without_squeeze():
    gt = np.ones((6, 12), np.int32)
    gt[:, 6:] = 2
    gt[2, 2] = 0
    original = gt > 0
    original[2, 8] = False
    batch = build_semantic_batch(gt, original, {1: 1, 2: 0}, input_size=12)
    target = batch['semantic_target'][0, 0]
    valid = batch['semantic_valid_content'][0, 0]
    assert target.shape == (12, 12)
    assert torch.all(target[6:] == -1) and not valid[6:].any()
    assert target[2, 2] == -1 and not valid[2, 2]
    assert target[2, 8] == 0 and valid[2, 8]  # 新GT填补仍参与共同pixel/core。
    assert torch.equal(batch['semantic_instance_map'][0, :6], torch.from_numpy(gt).long())
    assert batch['filled_pixels'] == 1


def test_actual_common_criterion_has_zero_gradient_on_unknown_padding_and_nonzero_on_fill():
    from train_direct_semantic_affinity import build_semantic_criterion, compute_semantic_loss
    gt = np.ones((6, 12), np.int32)
    gt[:, 6:] = 2
    gt[2, 2] = 0
    original = gt > 0
    original[2, 8] = False
    batch = build_semantic_batch(gt, original, {1: 1, 2: 0}, input_size=12)
    criterion = build_semantic_criterion({'direct_semantic_affinity': {'semantic_loss': {
        'dice_weight': .30, 'instance_weight': .75, 'pool_weight': .50,
        'core_radius': 3, 'core_min_pixels': 12, 'core_boundary_threshold': .20,
        'class_balance': True, 'ferrite_weight': 1., 'hard_gamma': .5,
        'hard_floor': .5, 'thin_weight': 1.5}}}, 'cpu')
    assert criterion.seg_dice_weight == .30 and criterion.semantic_instance_weight == .75
    assert criterion.semantic_instance_pool_weight == .50
    logits = torch.zeros((1, 1, 12, 12), requires_grad=True)
    loss = compute_semantic_loss(criterion, logits, batch)
    loss.backward()
    assert not logits.grad[batch['semantic_target'] < 0].count_nonzero()
    assert logits.grad[0, 0, 2, 8] != 0
    json.dumps(criterion.last_semantic_instance_stats, allow_nan=False)


def test_paired_schedule_repeats_same_full32_order_for64_independent_of_listing_order():
    names = ['fiction_%02d' % i for i in range(32)]
    schedule = paired_schedule(names)
    assert len(schedule) == 64 and schedule[:32] == schedule[32:]
    assert set(schedule[:32]) == set(names)
    assert schedule == paired_schedule(list(reversed(names)))


@pytest.mark.parametrize('names,seed', [(['only'], 8), (['repeated'] * 32, 8),
                                     (['fiction_%02d' % i for i in range(32)], True)])
def test_incomplete_repeated_or_invalid_seed_schedule_rejected(names, seed):
    with pytest.raises(ValueError):
        paired_schedule(names, seed)


@pytest.mark.parametrize('transform', ['identity', 'transpose', 'strided_slice'])
def test_feature_copy_preserves_actual_fp32_layout_and_does_not_alias_frozen_source(transform):
    source = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
    if transform == 'transpose':
        source = source.transpose(2, 3)
    elif transform == 'strided_slice':
        source = source[..., ::2]
    source.requires_grad_(True)
    before = source.detach().clone()
    result = feature_copy(source, 'cpu')
    assert result.shape == source.shape and result.dtype == source.dtype and result.stride() == source.stride()
    assert torch.equal(result, before) and not result.requires_grad
    result.add_(1)
    assert torch.equal(source.detach(), before)


def _outcome_case():
    ids = np.array([[1, 1, 2], [2, 3, 3]], np.uint16)
    regions = [dict(id=1, cls=1, native_area=2), dict(id=2, cls=0, native_area=2),
               dict(id=3, cls=0, native_area=2)]
    baseline = {'1': 0, '2': 0, '3': 1}
    current = {'1': 1, '2': 1, '3': 0}
    return regions, baseline, current, ids


def test_fixed_correct_regions_remain_in_denominator_and_bidirectional_regressions_are_separate():
    result = object_outcomes(*_outcome_case())
    assert result['fixed_regions'] == 3 and result['initial_wrong'] == 2
    assert result['corrected'] == 2 and result['original_correct_regressed'] == 1
    assert result['wrong'] == 1 and result['F_to_P_wrong'] == 0 and result['P_to_F_wrong'] == 1
    assert result['ferrite_count'] == 2 and result['ferrite_pixels'] == 4
    assert result['changed_instances'] == 3 and result['changed_pixels'] == 6


@pytest.mark.parametrize('all_pearlite', [False, True])
def test_actual_object_outcomes_serializes_all_report_values_as_native_json(all_pearlite):
    regions, baseline, current, ids = _outcome_case()
    if all_pearlite:
        current = {key: 0 for key in current}
    result = object_outcomes(regions, baseline, current, ids)
    restored = json.loads(json.dumps(result, allow_nan=False))
    assert restored == result
    for bucket in result['ferrite_size_buckets'].values():
        assert type(bucket['count']) is int and type(bucket['pixels']) is int


def test_valid_capture_config_json_roundtrip_preserves_identity_but_real_changes_do_not():
    from utils.config import load_config
    config = load_config('config/train/semantic_object_fit.yaml')
    assert config['data']['classes'] == {0: 'pearlite', 1: 'ferrite'}
    receipt = json.loads(json.dumps(config, allow_nan=False))
    assert receipt['data']['classes'] == {'0': 'pearlite', '1': 'ferrite'}
    assert receipt != config  # 直接字典比较会错误拒绝合法的持久化配置。
    expected = canonical_sha(config)
    assert canonical_sha(receipt) == expected
    for field, value in (('learning_rate', 2.5e-5), ('region_weight', .6)):
        changed = deepcopy(receipt)
        changed['semantic_object_fit'][field] = value
        assert canonical_sha(changed) != expected
    changed = deepcopy(receipt)
    changed['data']['classes'] = {'0': 'ferrite', '1': 'pearlite'}
    assert canonical_sha(changed) != expected
    changed['data']['classes'].pop('1')
    assert canonical_sha(changed) != expected


@pytest.mark.parametrize('mutation', ['missing_class', 'invalid_class', 'boolean_class',
                                     'duplicate_region', 'changed_region_area'])
def test_fixed_outcome_contract_rejects_missing_class_or_changed_supervised_cohort(mutation):
    regions, baseline, current, ids = _outcome_case()
    if mutation == 'missing_class':
        current.pop('3')
    elif mutation == 'invalid_class':
        current['3'] = 2
    elif mutation == 'boolean_class':
        current['3'] = False
    elif mutation == 'duplicate_region':
        regions.append(regions[0].copy())
    else:
        regions[0]['native_area'] = 1
    with pytest.raises(ValueError):
        object_outcomes(regions, baseline, current, ids)
