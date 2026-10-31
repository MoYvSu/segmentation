# -*- coding: utf-8 -*-
"""拆解原设备FP32 affinity融合，并区分关系掩码与像素位置统计。

本模块只诊断原有计算，不修正logits或监督。逐项归一再相加有舍入误差，
exact门禁始终使用原代码的先求分子、后除分母顺序。
"""
from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping

import torch

from utils.affinity_fusion import (affinity_boundary_probability,
                                   _reduce_boundary_channels,
                                   _soft_dilated_support)


def _logits(value):
    if (not torch.is_tensor(value) or value.ndim != 4 or value.shape[1] != 8
            or min(value.shape[0], *value.shape[-2:]) < 1
            or value.dtype != torch.float32 or not torch.isfinite(value).all()):
        raise ValueError('logits must be finite original-device FP32 [B,8,H,W]')


def _sha(value):
    array = value.detach().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(str(array.shape).encode('ascii'))
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _options(options):
    if not isinstance(options, Mapping) or 'mode' in options:
        raise ValueError('options must be fusion keyword mapping without mode')
    signature = inspect.signature(affinity_boundary_probability)
    defaults = {name:p.default for name,p in signature.parameters.items()
                if name not in ('affinity_logits', 'mode')}
    if set(options)-set(defaults):
        raise ValueError('Unknown fusion options: '+','.join(sorted(set(options)-set(defaults))))
    defaults.update(options)
    return defaults


@torch.no_grad()
def trace_fusion(logits, mode, options):
    """返回 ``parts, audit``；parts全为原设备无梯度张量，audit可写JSON。

    top2_indices遵循该实际设备的torch.topk。cutoff相同的候选不强行唯一
    排序：top2_cutoff_candidates和top2_selection_ambiguous显式保留歧义。
    contribution_*只作分量归因；reconstruction严格复用原融合计算顺序。
    """
    _logits(logits)
    opts = _options(options)
    if mode not in ('mean', 'short', 'gated'):
        raise ValueError('mode must be mean, short or gated')
    with torch.autocast(device_type=logits.device.type, enabled=False):
        actual = affinity_boundary_probability(logits, mode=mode, **opts)
        probability = logits.sigmoid()
        short_channels = 1.-probability[:, :4]
        values, indices = short_channels.topk(k=2, dim=1)
        cutoff = values[:, 1:2]
        candidates = short_channels == cutoff
        candidate_count = candidates.sum(dim=1, keepdim=True)
        selected_count = (values == cutoff).sum(dim=1, keepdim=True)
        parts = dict(probability=probability,
                     short_channels=short_channels,
                     top2_values=values, top2_indices=indices,
                     top2_cutoff=cutoff, top2_cutoff_candidates=candidates,
                     top2_cutoff_tie_count=candidate_count,
                     top2_selected_cutoff_count=selected_count,
                     top2_selection_ambiguous=candidate_count > selected_count,
                     top2_order_tied=values[:, :1] == values[:, 1:2])
        if mode == 'mean':
            reconstructed = 1.-probability.mean(dim=1, keepdim=True)
            parts['mean_affinity'] = probability.mean(dim=1, keepdim=True)
        else:
            short = _reduce_boundary_channels(short_channels, opts['short_reduction'],
                                               opts['short_softmax_temperature'])
            parts['short'] = short
            if opts['short_reduction'] == 'top2' and not torch.equal(short, values.mean(dim=1, keepdim=True)):
                raise RuntimeError('Actual top2 reduction differs from captured top2 values')
            if mode == 'short':
                reconstructed = short
            else:
                distance2 = 1.-probability[:, 4:6].mean(dim=1, keepdim=True)
                distance4 = 1.-probability[:, 6:8].mean(dim=1, keepdim=True)
                support2 = _soft_dilated_support(short, 2, opts['support_threshold'], opts['support_temperature'])
                support4 = _soft_dilated_support(short, 4, opts['support_threshold'], opts['support_temperature'])
                weighted2 = float(opts['distance2_weight'])*support2
                weighted4 = float(opts['distance4_weight'])*support4
                numerator2, numerator4 = weighted2*distance2, weighted4*distance4
                # 与原函数严格相同的运算先后，不以单项归一和替代它。
                numerator = short+numerator2+numerator4
                denominator = 1.+weighted2+weighted4
                reconstructed = (numerator/denominator).clamp(0., 1.)
                parts.update(distance2=distance2, distance4=distance4,
                             support2=support2, support4=support4,
                             weighted2=weighted2, weighted4=weighted4,
                             numerator2=numerator2, numerator4=numerator4,
                             numerator=numerator, denominator=denominator,
                             contribution_short=short/denominator,
                             contribution2=numerator2/denominator,
                             contribution4=numerator4/denominator,
                             fusion_minus_short=reconstructed-short)
                parts['normalized_sum'] = (parts['contribution_short']
                                           +parts['contribution2']+parts['contribution4'])
        if not torch.equal(reconstructed, actual):
            raise RuntimeError('FP32 reconstruction differs from actual original fusion')
        parts.update(reconstruction=reconstructed, fused=actual)
    if any(value.device != logits.device or value.requires_grad for value in parts.values()):
        raise RuntimeError('Trace parts must preserve actual device without gradients')
    audit = dict(format='affinity_channel_trace_v1', mode=mode, options=opts,
                 logits_shape=list(logits.shape), logits_dtype=str(logits.dtype), device=str(logits.device),
                 coordinate_space='logit tensor grid, not native image pixels',
                 logits_sha256=_sha(logits), input_mutated=False,
                 reconstruction_equal_actual=True,
                 reconstruction_order='original numerator additions, denominator additions, division, original clamp',
                 private_original_reduce_and_support_reused=True,
                 top2_selection_ambiguous_pixels=int(parts['top2_selection_ambiguous'].sum()),
                 top2_order_tied_pixels=int(parts['top2_order_tied'].sum()),
                 tie_policy='actual device topk indices; all cutoff candidates retained, no unique channel attribution on ties',
                 parts={name:dict(shape=list(value.shape), dtype=str(value.dtype), sha256=_sha(value))
                        for name,value in parts.items()})
    if mode == 'gated':
        difference = (parts['normalized_sum']-actual).abs()
        audit.update(normalized_component_sum_equal_actual=bool(torch.equal(parts['normalized_sum'], actual)),
                     normalized_component_sum_max_roundoff=float(difference.max()),
                     long_terms_reduce_short_pixels=int((actual < parts['short']).sum()),
                     long_terms_increase_short_pixels=int((actual > parts['short']).sum()),
                     normalized_component_sum_is_exact_gate=False)
    return parts, audit


def _mask(spec, logits, name):
    if not isinstance(spec, Mapping) or set(spec) != {'kind', 'mask'}:
        raise ValueError(name+' requires explicit {kind: node|edge, mask: bool tensor}')
    value, kind = spec['mask'], spec['kind']
    if not torch.is_tensor(value) or value.dtype != torch.bool:
        raise ValueError(name+' mask must be a boolean tensor')
    batch, _, height, width = logits.shape
    if kind == 'edge':
        if tuple(value.shape) != tuple(logits.shape):
            raise ValueError(name+' edge mask must be [B,8,H,W]')
        return kind, value.to(device=logits.device)
    if kind != 'node':
        raise ValueError(name+' mask kind must be node or edge')
    if value.ndim == 2 and batch == 1:
        value = value[None]
    elif value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if tuple(value.shape) != (batch, height, width):
        raise ValueError(name+' node mask must match HW(single B), BHW or B1HW')
    return kind, value.to(device=logits.device)


def _summary(values):
    count = values.numel()
    if not count:
        return dict(count=0, mean=None, quantiles=None, below_0_5=None, below_0_8=None,
                    below_0_9=None, below_0_5_count=0, below_0_8_count=0, below_0_9_count=0)
    levels = (0., .01, .05, .1, .5, .9, .95, .99, 1.)
    quantiles = torch.quantile(values, values.new_tensor(levels)).cpu().tolist()
    result = dict(count=count, mean=float(values.mean()),
                  quantiles={str(level):float(value) for level,value in zip(levels, quantiles)})
    for label, threshold in (('0_5', .5), ('0_8', .8), ('0_9', .9)):
        n = int((values < threshold).sum())
        result['below_'+label+'_count'] = n
        result['below_'+label] = n/count
    return result


def _boundary_tail(values):
    if not values.numel():
        return dict(count=0, mean=None, maximum=None, quantiles=None,
                    at_least_0_45=None, at_least_0_65=None)
    levels = (.5, .9, .95, .99, 1.)
    quantiles = torch.quantile(values, values.new_tensor(levels)).cpu().tolist()
    return dict(count=values.numel(), mean=float(values.mean()), maximum=float(values.max()),
                quantiles={str(level):float(value) for level,value in zip(levels, quantiles)},
                at_least_0_45=float((values >= .45).float().mean()),
                at_least_0_65=float((values >= .65).float().mean()))


@torch.no_grad()
def selected_channel_stats(logits, masks):
    """返回纯JSON：每个显式typed mask的八通道same概率及短程边界尾部。

    masks例：{'local':{'kind':'edge','mask':B8HW_bool},
             'ROI':{'kind':'node','mask':HW_bool}}。
    edge的合法性由调用者提供；这里不把node自动当合法关系，也不补齐边。
    node只统计对应原始位置，会包含未获GT关系监督的方向；其top2另标raw。
    """
    _logits(logits)
    if not isinstance(masks, Mapping):
        raise ValueError('masks must be a named mapping of explicit typed masks')
    with torch.autocast(device_type=logits.device.type, enabled=False):
        probability = logits.sigmoid()
        result = {}
        for name, spec in masks.items():
            if not isinstance(name, str):
                raise ValueError('Mask names must be strings')
            kind, mask = _mask(spec, logits, name)
            selected = mask if kind == 'edge' else mask[:, None].expand_as(logits)
            channels = []
            for channel in range(8):
                values = probability[:, channel][selected[:, channel]]
                row = dict(channel=channel, same_probability=_summary(values))
                if channel < 4:
                    row['short_boundary_tail'] = _boundary_tail(1.-values)
                channels.append(row)
            short = probability[:, :4][selected[:, :4]]
            row = dict(kind=kind, mask_shape=list(mask.shape), mask_sha256=_sha(mask),
                       coordinate_space='logit tensor grid, not native image pixels',
                       selected_mask_entries=int(mask.sum()), per_channel=channels,
                       pooled_short_same_probability=_summary(short),
                       pooled_short_boundary_tail=_boundary_tail(1.-short),
                       relation_legality_inferred=False,
                       scope=('caller-provided individual edge selection; no other channel substituted'
                              if kind == 'edge' else 'raw positions only; node presence does not establish legal relations'))
            if kind == 'node':
                top2 = _reduce_boundary_channels(1.-probability[:, :4], 'top2', .15)
                row['raw_node_top2_boundary_tail'] = _boundary_tail(top2[:, 0][mask])
            result[name] = row
    return result
