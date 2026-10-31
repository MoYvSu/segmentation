# -*- coding: utf-8 -*-
"""距离融合机制的可信域、阈值方向及因果剥离合成短测。"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from tools import probe_affinity_distance_fusion as probe
from utils.affinity_fusion import affinity_boundary_probability


def inputs(height=8, width=8):
    logits = torch.linspace(-3, 3, 8 * height * width).reshape(1, 8, height, width)
    target = torch.ones_like(logits, dtype=torch.bool)
    target[:, 0, :, width // 2] = False
    valid = torch.ones_like(target)
    return logits, target, valid


def test_same_logits_exact_deployed_fusion_and_predefined_ablations_only():
    logits, target, valid = inputs()
    options = dict(short_reduction='top2', distance2_weight=.5, distance4_weight=.25,
                   support_threshold=.2, support_temperature=.05)
    saved = [value.clone() for value in (logits, target, valid)]
    original_options = deepcopy(options)
    result = probe.fusion_diagnostic(logits, target, valid, options, return_maps=True)
    scores = result['maps']['scores']
    for name, mode, chosen in [('short', 'short', options), ('gated', 'gated', options),
            ('d2_only', 'gated', dict(options, distance4_weight=0.)),
            ('d4_only', 'gated', dict(options, distance2_weight=0.))]:
        expected = affinity_boundary_probability(logits, mode=mode, **chosen)[:, 0].numpy()
        assert np.array_equal(scores[name], expected)
    assert options == original_options
    assert all(torch.equal(actual, expected) for actual, expected in zip((logits, target, valid), saved))
    assert tuple(scores) == probe.VARIANTS
    assert result['report']['thresholds'] == [.45, .65]
    assert result['report']['recalls'] == [.9, .95, .97, .98]
    assert result['report']['instance_inference'] is False
    json.dumps(result['report'], allow_nan=False)


def test_unknown_short_endpoint_and_padding_never_become_negative_labels():
    logits, target, valid = inputs()
    valid[:, 1, 2, 2] = False
    target[:, 0, 2, 2] = False  # 即使未知关系有边界值，该anchor仍须ignore。
    content = torch.ones((1, 1, 8, 8), dtype=torch.bool)
    content[:, :, 0] = False
    result = probe.fusion_diagnostic(logits, target, valid, dict(short_reduction='top2'),
                                    valid_content=content, return_maps=True)
    eligible = result['maps']['valid']['shared_short']
    assert not eligible[0, 2, 2]
    assert not eligible[0, 0].any()
    assert result['report']['domains']['shared_short']['ignored_pixels'] == 9
    expected = result['report']['domains']['shared_short']
    for name in probe.VARIANTS:
        hist = result['histograms'][name]['shared_short']
        assert hist[0].sum() == expected['true_boundary_pixels']
        assert hist[1].sum() == expected['interior_pixels']


def test_deep_interior_uses_eight_known_connections_and_same_short_boundaries():
    logits, target, valid = inputs()
    target[:, 6, 2, 2] = False  # 短程内部、长程跨实例：不是真正深内部。
    valid[:, 7, 3, 3] = False  # 短程可信，长程未知：深内部忽略。
    valid[:, 7, 3, 4] = False  # 可信短程真边界不因长程未知丢失。
    result = probe.fusion_diagnostic(logits, target, valid, dict(short_reduction='top2'), return_maps=True)
    mask = result['maps']['valid']
    assert mask['shared_short'][0, 2, 2] and mask['shared_short'][0, 3, 3]
    assert not mask['deep_interior'][0, 2, 2] and not mask['deep_interior'][0, 3, 3]
    assert mask['deep_interior'][0, 3, 4]
    assert result['report']['domains']['shared_short']['true_boundary_pixels'] == result['report']['domains']['deep_interior']['true_boundary_pixels']


def test_gated_can_weaken_real_strong_boundary_not_only_add_boundaries():
    scores = torch.full((1, 8, 3, 3), .05)
    scores[:, :4] = .8
    logits = torch.logit(1 - scores)
    target = torch.ones_like(logits, dtype=torch.bool)
    target[:, 0, 1, 1] = False
    result = probe.fusion_diagnostic(logits, target, torch.ones_like(target), dict(short_reduction='top2'))
    changes = result['report']['comparisons']['short_to_gated']['domains']['shared_short']
    high = changes['threshold_crossings']['0.65']
    assert high['removed_true_boundary'] == 1
    assert high['removed_false_boundary'] == 8
    assert high['added_true_boundary'] == high['added_false_boundary'] == 0
    assert changes['continuous']['true_boundary']['weakened'] == 1
    assert changes['continuous']['interior']['weakened'] == 8
    low = changes['threshold_crossings']['0.45']
    assert low['removed_true_boundary'] == low['removed_false_boundary'] == 0


def test_fixed_threshold_added_true_and_false_boundary_are_distinguished():
    base = np.array([[[.2, .2, .8, .8, .8]]], np.float32)
    after = np.array([[[.7, .7, .2, .2, .2]]], np.float32)
    boundary = np.array([[[True, False, True, False, True]]])
    valid = np.array([[[True, True, True, True, False]]])
    changes = probe._changes(base, after, boundary, valid)
    for threshold in ['0.45', '0.65']:
        count = changes['threshold_crossings'][threshold]
        assert count['added_true_boundary'] == count['added_false_boundary'] == 1
        assert count['removed_true_boundary'] == count['removed_false_boundary'] == 1
    assert changes['continuous']['true_boundary']['pixels'] == 2
    assert changes['continuous']['interior']['pixels'] == 2


def test_weak_local_boundary_is_reinforced_only_where_support_is_present():
    scores = torch.full((1, 8, 12, 12), .01)
    scores[:, 4:] = .9
    scores[:, :4, 5, 5] = .8
    scores[:, :4, 5, 6] = .35
    scores[:, :4, 6, 5] = .35
    logits = torch.logit(1 - scores)
    target = torch.ones_like(logits, dtype=torch.bool)
    target[:, 0, 5, 6] = False
    result = probe.fusion_diagnostic(logits, target, torch.ones_like(target),
                                    dict(short_reduction='top2'), return_maps=True)
    crossings = result['report']['comparisons']['short_to_gated']['domains']['shared_short']['threshold_crossings']['0.45']
    assert crossings['added_true_boundary'] == 1
    assert crossings['added_false_boundary'] >= 1
    assert result['maps']['scores']['gated'][0, 0, 0] < .45


def test_zero_distance_weights_restore_short_map_exactly():
    logits, target, valid = inputs()
    result = probe.fusion_diagnostic(logits, target, valid,
        dict(short_reduction='top2', distance2_weight=0., distance4_weight=0.), return_maps=True)
    for name in probe.VARIANTS:
        assert np.array_equal(result['maps']['scores'][name], result['maps']['scores']['short'])
    for domain in probe.DOMAINS:
        crossings = result['report']['comparisons']['short_to_gated']['domains'][domain]['threshold_crossings']
        for row in crossings.values():
            assert all(row[key] == 0 for key in ['added_true_boundary', 'added_false_boundary',
                                                'removed_true_boundary', 'removed_false_boundary'])


def test_empty_trusted_domain_produces_no_fake_negatives_or_nan():
    logits, target, valid = inputs()
    result = probe.fusion_diagnostic(logits, target, torch.zeros_like(valid), dict(short_reduction='top2'))
    for name in probe.VARIANTS:
        for domain in probe.DOMAINS:
            assert result['histograms'][name][domain].sum() == 0
            assert result['report']['metrics'][name][domain]['equal_recall'] == [None] * 4
    json.dumps(result['report'], allow_nan=False)


@pytest.mark.parametrize('options', [dict(short_reduction='mean'), {},
    dict(short_reduction='top2', mode='short'), dict(short_reduction='top2', accidental_parameter=1),
    dict(short_reduction='top2', distance2_weight=-1),
    dict(short_reduction='top2', support_temperature=0),
    dict(short_reduction='top2', distance4_weight=float('nan'))])
def test_wrong_fusion_contract_is_not_silently_reinterpreted(options):
    logits, target, valid = inputs()
    with pytest.raises(ValueError):
        probe.fusion_diagnostic(logits, target, valid, options)


def test_bad_shapes_nonbinary_masks_and_nonfinite_logits_are_rejected():
    logits, target, valid = inputs()
    with pytest.raises(ValueError, match='same'):
        probe.fusion_diagnostic(logits, target[:, :4], valid, dict(short_reduction='top2'))
    wrong = target.float()
    wrong[0, 0, 0, 0] = .5
    with pytest.raises(ValueError, match='binary'):
        probe.fusion_diagnostic(logits, wrong, valid, dict(short_reduction='top2'))
    broken = logits.clone()
    broken[0, 0, 0, 0] = float('inf')
    with pytest.raises(ValueError, match='finite'):
        probe.fusion_diagnostic(broken, target, valid, dict(short_reduction='top2'))
    with pytest.raises(ValueError, match='valid_content'):
        probe.fusion_diagnostic(logits, target, valid, dict(short_reduction='top2'), valid_content=torch.ones((9, 9)))


def test_synthetic_cli_self_test():
    probe.self_test()
