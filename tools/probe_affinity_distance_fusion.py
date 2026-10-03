# -*- coding: utf-8 -*-
"""同一组logits的固定距离融合诊断：不训练、不改阈值、不读赛题文件。

四个预设为短距top2、现有gated、仅保留距离2辅助、仅保留距离4辅助。
所有预设使用同一短距可信GT；未知端点、窗外与padding保持ignore。
这是连续边界图的机制诊断，不是完整实例划分或比赛精度。
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import inspect
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.probe_affinity_equal_recall import (
    curve_metrics, equal_recall, fused_targets, score_histogram)
from utils.affinity_fusion import affinity_boundary_probability

VARIANTS = ('short', 'gated', 'd2_only', 'd4_only')
DOMAINS = ('shared_short', 'deep_interior')
RECALLS = (.90, .95, .97, .98)
THRESHOLDS = (.45, .65)
DELTA_EPSILON = 1.e-6


def _binary_tensor(value, shape, name):
    tensor = torch.as_tensor(value)
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(name + ' must have the same [B,8,H,W] shape as logits')
    if tensor.dtype != torch.bool:
        if not torch.isfinite(tensor).all() or not ((tensor == 0) | (tensor == 1)).all():
            raise ValueError(name + ' must contain only binary values')
    return tensor.detach().to(device='cpu', dtype=torch.bool).numpy()


def _content_mask(value, batch, height, width):
    if value is None:
        return np.ones((batch, height, width), bool)
    tensor = torch.as_tensor(value)
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim == 2 and batch == 1:
        tensor = tensor[None]
    if tuple(tensor.shape) != (batch, height, width):
        raise ValueError('valid_content must have [B,H,W] or [B,1,H,W] shape')
    if tensor.dtype != torch.bool:
        if not torch.isfinite(tensor).all() or not ((tensor == 0) | (tensor == 1)).all():
            raise ValueError('valid_content must contain only binary values')
    return tensor.detach().to(device='cpu', dtype=torch.bool).numpy()


def _fusion_options(kwargs):
    if not isinstance(kwargs, Mapping):
        raise ValueError('fusion_kwargs must explicitly describe the deployed gated/top2 configuration')
    defaults = {name: parameter.default for name, parameter in
                inspect.signature(affinity_boundary_probability).parameters.items()
                if parameter.kind == inspect.Parameter.KEYWORD_ONLY}
    if set(kwargs) - set(defaults):
        raise ValueError('Unknown fusion option: ' + ', '.join(sorted(set(kwargs) - set(defaults))))
    if kwargs.get('mode', 'gated') != 'gated' or kwargs.get('short_reduction') != 'top2':
        raise ValueError('This comparison requires the current gated fusion with explicit top2 short reduction')
    options = dict(defaults, **kwargs)
    options.pop('mode')
    for name in ('distance2_weight', 'distance4_weight', 'support_threshold',
                 'support_temperature', 'short_softmax_temperature'):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('Fusion scalar must be finite numeric: ' + name)
        options[name] = float(value)
    if (options['distance2_weight'] < 0 or options['distance4_weight'] < 0
            or options['support_temperature'] <= 0 or options['short_softmax_temperature'] <= 0):
        raise ValueError('Distance weights must be nonnegative and fusion temperatures positive')
    return options


def _quantiles(value):
    if not value.size:
        return None
    return {name: float(np.quantile(value, quantile)) for name, quantile in
            (('minimum', 0.), ('p10', .1), ('median', .5), ('p90', .9), ('maximum', 1.))}


def _fixed_operating(scores, boundary, eligible, threshold):
    positive = eligible & boundary
    negative = eligible & ~boundary
    prediction = scores >= threshold
    tp, fp = int((prediction & positive).sum()), int((prediction & negative).sum())
    boundaries, interiors = int(positive.sum()), int(negative.sum())
    return dict(threshold=threshold, true_positive=tp, false_positive=fp,
        false_negative=boundaries - tp, true_negative=interiors - fp,
        true_boundary_recall=tp / boundaries if boundaries else None,
        interior_false_boundary_rate=fp / interiors if interiors else None,
        precision=tp / (tp + fp) if tp + fp else None)


def _changes(reference, candidate, boundary, eligible):
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    continuous = {}
    for kind, mask in (('true_boundary', eligible & boundary),
                       ('interior', eligible & ~boundary)):
        values = delta[mask]
        continuous[kind] = dict(pixels=int(values.size),
            reinforced=int((values > DELTA_EPSILON).sum()),
            weakened=int((values < -DELTA_EPSILON).sum()),
            unchanged_within_epsilon=int((np.abs(values) <= DELTA_EPSILON).sum()),
            mean_delta=float(values.mean()) if values.size else None,
            delta_quantiles=_quantiles(values))
    crossings = {}
    for threshold in THRESHOLDS:
        before, after = reference >= threshold, candidate >= threshold
        added, removed = ~before & after & eligible, before & ~after & eligible
        crossings[str(threshold)] = dict(
            added_true_boundary=int((added & boundary).sum()),
            added_false_boundary=int((added & ~boundary).sum()),
            removed_true_boundary=int((removed & boundary).sum()),
            removed_false_boundary=int((removed & ~boundary).sum()),
            both_above_true=int((before & after & eligible & boundary).sum()),
            both_above_false=int((before & after & eligible & ~boundary).sum()))
    return dict(continuous=continuous, threshold_crossings=crossings)


def fusion_diagnostic(logits, target, edge_valid, fusion_kwargs, *, valid_content=None,
                      bins=8192, return_histograms=True, return_maps=False):
    """返回report、可选histograms/maps；外层负责源图、视野及退化回执。

    logits、target、edge_valid均为[B,8,H,W]，target=1表示同实例连接。
    shared_short需四个短距关系均可信；任一短距跨实例即为真边界。
    deep_interior沿用旧等召回定义：真边界相同，内部须八个关系均为
    可信同实例连接。距离辅助的未知GT不被扩充为假边界负例。
    .45/.65只统计连续图阈值穿越，不执行部署实例划分或调阈值。
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != 8:
        raise ValueError('logits must be a floating tensor [B,8,H,W]')
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError('logits must be finite floating values')
    if min(logits.shape[0], logits.shape[-2], logits.shape[-1]) < 1:
        raise ValueError('logits must have a nonempty batch and spatial dimensions')
    if type(bins) is not int or bins < 2:
        raise ValueError('Histogram bins must be an integer >=2')
    options = _fusion_options(fusion_kwargs)
    truth = _binary_tensor(target, logits.shape, 'target')
    trusted = _binary_tensor(edge_valid, logits.shape, 'edge_valid')
    content = _content_mask(valid_content, logits.shape[0], logits.shape[-2], logits.shape[-1])
    boundary, domains = [], {name: [] for name in DOMAINS}
    for index in range(logits.shape[0]):
        labels, eligible = fused_targets(truth[index], trusted[index])
        deep_labels, deep_eligible = fused_targets(truth[index], trusted[index], strict_interior=True)
        if not np.array_equal(labels, deep_labels):
            raise AssertionError('Deep domain changed the shared short-range target')
        boundary.append(labels)
        domains['shared_short'].append(eligible & content[index])
        domains['deep_interior'].append(deep_eligible & content[index])
    boundary = np.stack(boundary)
    domains = {name: np.stack(value) for name, value in domains.items()}
    choices = {'short': ('short', options), 'gated': ('gated', options),
        'd2_only': ('gated', dict(options, distance4_weight=0.)),
        'd4_only': ('gated', dict(options, distance2_weight=0.))}
    maps = {}
    with torch.no_grad(), torch.autocast(device_type=logits.device.type, enabled=False):
        for name, (mode, selected) in choices.items():
            value = affinity_boundary_probability(logits.detach().float(), mode=mode, **selected)
            if tuple(value.shape) != (logits.shape[0], 1, logits.shape[-2], logits.shape[-1]):
                raise RuntimeError('Fusion returned a wrong scalar boundary shape')
            if not torch.isfinite(value).all() or torch.any(value < 0) or torch.any(value > 1):
                raise RuntimeError('Fusion produced nonfinite or invalid probabilities')
            maps[name] = value[:, 0].cpu().numpy()
    histograms, metrics = {}, {}
    for name, scores in maps.items():
        histograms[name], metrics[name] = {}, {}
        for domain, eligible in domains.items():
            hist = score_histogram(scores, boundary, eligible, bins=bins)
            histograms[name][domain] = hist
            curve = curve_metrics(hist)
            curve.pop('threshold_0_5', None)
            metrics[name][domain] = dict(**curve,
                equal_recall=[equal_recall(hist, recall) for recall in RECALLS],
                fixed_thresholds={str(value): _fixed_operating(scores, boundary, eligible, value)
                                  for value in THRESHOLDS})
    comparisons = {}
    for name, before, after in (('short_to_gated', 'short', 'gated'),
            ('short_to_d2_only', 'short', 'd2_only'), ('short_to_d4_only', 'short', 'd4_only'),
            ('remove_distance2', 'gated', 'd4_only'), ('remove_distance4', 'gated', 'd2_only')):
        comparisons[name] = dict(reference=before, candidate=after,
            domains={domain: _changes(maps[before], maps[after], boundary, eligible)
                     for domain, eligible in domains.items()})
    report = dict(training=False, epoch_selection=False, postprocessing_tuned=False,
        official_accuracy=False, instance_inference=False, same_logits=True,
        same_target_and_valid_for_all_variants=True,
        scope='Trusted continuous-boundary mechanism diagnostic, not final-instance accuracy',
        input_shape=list(logits.shape), input_dtype=str(logits.dtype), fusion_dtype='torch.float32',
        fusion_options=options, variants={name: dict(mode=mode, **selected) for name, (mode, selected) in choices.items()},
        thresholds=list(THRESHOLDS), recalls=list(RECALLS), histogram_bins=bins,
        continuous_delta_epsilon=DELTA_EPSILON,
        target_definition='Any one of four unit-offset targets=0 means boundary; all four endpoints/pairs must be trusted.',
        deep_definition='Same true boundaries; negatives require all eight targets=1 and all eight edge-valid masks.',
        domains={name: dict(eligible_pixels=int(eligible.sum()),
            true_boundary_pixels=int((eligible & boundary).sum()),
            interior_pixels=int((eligible & ~boundary).sum()),
            ignored_pixels=int((~eligible).sum())) for name, eligible in domains.items()},
        metrics=metrics, comparisons=comparisons,
        caveats=['Unknown GT and padding are ignored, never labelled false boundaries.',
                 'Fixed .45/.65 crossings describe continuous maps, not watershed seed counts.',
                 'Removing one distance also changes the gated normalization denominator; effects are not additive.',
                 'Each input cohort needs external immutable source/view/degradation receipts.'])
    result = {'report': report}
    if return_histograms:
        result['histograms'] = histograms
    if return_maps:
        result['maps'] = dict(scores=maps, boundary=boundary, valid=domains)
    return result


def self_test():
    target = torch.ones((1, 8, 12, 12), dtype=torch.bool)
    valid = torch.ones_like(target)
    target[:, 0, :, 5] = False
    valid[:, :, 0] = False
    logits = torch.full(target.shape, 3.)
    logits[:, :4, :, 5] = -2.
    result = fusion_diagnostic(logits, target, valid, dict(short_reduction='top2'),
                               bins=128, return_maps=True)
    reference = affinity_boundary_probability(logits, mode='gated', short_reduction='top2')
    if not np.array_equal(result['maps']['scores']['gated'], reference[:, 0].numpy()):
        raise AssertionError('Current deployed fusion was not reused exactly')
    if result['report']['domains']['shared_short']['ignored_pixels'] != 12:
        raise AssertionError('Unknown original pair mask became a negative label')
    expected = result['report']['domains']['shared_short']
    for arm in VARIANTS:
        hist = result['histograms'][arm]['shared_short']
        if hist[0].sum() != expected['true_boundary_pixels'] or hist[1].sum() != expected['interior_pixels']:
            raise AssertionError('Variants evaluated different trusted targets')
    print('DISTANCE_FUSION_CPU_PASS: deployed fusion exact, shared GT and unknown ignore preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    self_test()
