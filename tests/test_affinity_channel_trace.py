# -*- coding: utf-8 -*-
"""原设备实际融合的可追溯分量与严格node/edge统计合同。"""
import json

import pytest
import torch

from utils.affinity_channel_trace import trace_fusion, selected_channel_stats
from utils.affinity_fusion import affinity_boundary_probability


def test_actual_gated_top2_exact_fp32_noncontiguous_without_grad_or_mutation():
    logits = torch.randn((2, 8, 13, 17), dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    logits.requires_grad_(True)
    before, stride = logits.detach().clone(), logits.stride()
    options = dict(short_reduction='top2', distance2_weight=.5, distance4_weight=.25)
    parts, audit = trace_fusion(logits, 'gated', options)
    actual = affinity_boundary_probability(logits, mode='gated', **options)
    assert torch.equal(parts['reconstruction'], actual)
    assert torch.equal(parts['short'], parts['top2_values'].mean(dim=1, keepdim=True))
    expected = (parts['short']+parts['numerator2']+parts['numerator4'])/parts['denominator']
    assert torch.equal(parts['reconstruction'], expected.clamp(0., 1.))
    assert torch.equal(logits, before) and logits.stride() == stride
    assert all(x.device == logits.device and not x.requires_grad for x in parts.values())
    assert audit['reconstruction_equal_actual'] and audit['normalized_component_sum_is_exact_gate'] is False
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize('mode, options', [('mean', {}), ('short', {'short_reduction':'mean'}),
                                        ('short', {'short_reduction':'softmax'}),
                                        ('gated', {'short_reduction':'mean'})])
def test_other_original_modes_and_reductions_are_exact(mode, options):
    logits = torch.randn((1, 8, 3, 7))
    parts, audit = trace_fusion(logits, mode, options)
    assert torch.equal(parts['reconstruction'], affinity_boundary_probability(logits, mode=mode, **options))
    assert audit['logits_shape'] == [1, 8, 3, 7]
    # 只接受最终logits，不假设encoder必须有某个特征级数。
    assert 'features' not in audit and parts['probability'].shape[1] == 8


def test_top2_cutoff_ties_preserve_all_candidates_without_claiming_unique_channel():
    logits = torch.zeros((1, 8, 2, 3))
    parts, audit = trace_fusion(logits, 'gated', {'short_reduction':'top2'})
    assert parts['top2_cutoff_candidates'].all()
    assert torch.all(parts['top2_cutoff_tie_count'] == 4)
    assert torch.all(parts['top2_selected_cutoff_count'] == 2)
    assert parts['top2_selection_ambiguous'].all() and parts['top2_order_tied'].all()
    assert audit['top2_selection_ambiguous_pixels'] == 6
    assert torch.equal(parts['top2_values'], torch.gather(parts['short_channels'], 1, parts['top2_indices']))
    # 两个同高通道、其余更低：顺序可同分，但选择集合没有cutoff歧义。
    logits[:, :2] = -2.
    logits[:, 2:4] = 2.
    parts, _ = trace_fusion(logits, 'gated', {'short_reduction':'top2'})
    assert parts['top2_order_tied'].all() and not parts['top2_selection_ambiguous'].any()


def test_long_probabilities_can_reduce_short_after_normalization():
    logits = torch.full((1, 8, 5, 5), 9.)
    logits[:, :4] = -1.
    parts, audit = trace_fusion(logits, 'gated', {'short_reduction':'top2'})
    assert torch.all(parts['fused'] < parts['short'])
    assert parts['contribution2'].min() > 0 and parts['contribution4'].min() > 0
    assert audit['long_terms_reduce_short_pixels'] == 25
    assert audit['long_terms_increase_short_pixels'] == 0


@pytest.mark.parametrize('bad', [torch.zeros((1, 7, 3, 3)), torch.zeros((8, 3, 3)),
                                torch.zeros((1, 8, 0, 3)), torch.zeros((1, 8, 3, 3),dtype=torch.float64),
                                torch.full((1, 8, 3, 3), float('nan'))])
def test_invalid_shape_dtype_or_nonfinite_rejected(bad):
    with pytest.raises(ValueError, match='FP32'):
        trace_fusion(bad, 'gated', {})
    with pytest.raises(ValueError, match='FP32'):
        selected_channel_stats(bad, {})


def test_edge_and_node_counts_are_explicit_and_empty_statistics_none():
    logits = torch.zeros((1, 8, 4, 5))
    edges = torch.zeros_like(logits, dtype=torch.bool)
    edges[0, 0, 1, 1] = True
    nodes = torch.zeros((4, 5), dtype=torch.bool)
    nodes[1, 1] = True
    result = selected_channel_stats(logits, {
        'edge':dict(kind='edge',mask=edges), 'node':dict(kind='node',mask=nodes),
        'empty':dict(kind='edge',mask=torch.zeros_like(edges))})
    assert result['edge']['per_channel'][0]['same_probability']['count'] == 1
    assert result['edge']['per_channel'][1]['same_probability']['count'] == 0
    assert all(x['same_probability']['count'] == 1 for x in result['node']['per_channel'])
    assert 'raw_node_top2_boundary_tail' not in result['edge']
    assert result['node']['relation_legality_inferred'] is False
    assert result['empty']['pooled_short_same_probability']['quantiles'] is None
    assert result['empty']['pooled_short_boundary_tail']['maximum'] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match='explicit'):
        selected_channel_stats(logits, {'unclear':edges})
    with pytest.raises(ValueError, match='edge mask'):
        selected_channel_stats(logits, {'wrong':dict(kind='edge',mask=nodes)})


def test_high_average_does_not_hide_rare_low_same_probability_tail():
    logits = torch.full((1, 8, 10, 10), 9.)
    logits[0, 0, 0, 0] = -9.
    edges = torch.ones_like(logits, dtype=torch.bool)
    row = selected_channel_stats(logits, {'local':dict(kind='edge',mask=edges)})['local']
    stats = row['per_channel'][0]['same_probability']
    assert stats['mean'] > .98 and stats['below_0_5_count'] == 1
    assert stats['below_0_5'] == .01 and stats['quantiles']['0.0'] < .001
    tail = row['per_channel'][0]['short_boundary_tail']
    assert tail['maximum'] > .999 and tail['at_least_0_65'] > 0
    assert row['per_channel'][1]['same_probability']['below_0_5_count'] == 0
