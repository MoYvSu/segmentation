# -*- coding: utf-8 -*-
"""短/中/长距离监督在共享权重上的梯度诊断；不训练、不修改参数或累积grad。

调用者传入真实forward产生的8通道logits及需要观察的共享参数，例如
affinity_decoder.affinity_head[0].weight。不要用最终8通道输出层判断组间冲突：
那些参数行各负责不同通道，本来就不重叠。可同时传入FPN参数，结果分别报告。
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch

from utils.affinity_loss import balanced_affinity_loss


GROUPS = {'short': (0, 4), 'distance2': (4, 6), 'distance4': (6, 8)}
CSHORT_CHANNEL_WEIGHTS = {'short': 1., 'distance2': .5, 'distance4': .25}
LOSS_OPTIONS = dict(negative_weight=1., hard_negative_weight=1.,
                    hard_negative_gamma=2., normalize_edge_weights=True)


def _norm(tensor):
    return float(torch.linalg.vector_norm(tensor.detach().double()))


def _reported_norm(tensor):
    value = _norm(tensor)
    return value if value > 0. else None


def _cosine(left, right):
    left_norm, right_norm = _norm(left), _norm(right)
    if not left_norm or not right_norm:
        return None
    value = float((left.detach().double() * right.detach().double()).sum()) / (left_norm * right_norm)
    return max(-1., min(1., value))


def _short_projection(mixture, short):
    """在单位短距离梯度方向上的有符号长度；正值表示方向一致。"""
    norm = _norm(short)
    if not norm or not _norm(mixture):
        return None
    return float((mixture.detach().double() * short.detach().double()).sum()) / norm


def _gradient(loss, parameters, *, retain_graph):
    values = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=retain_graph,
                                 allow_unused=True, create_graph=False)
    return {name: (value.detach() if value is not None else torch.zeros_like(parameter))
            for (name, parameter), value in zip(parameters.items(), values)}


def _validate(logits, target, edge_valid, parameters, edge_weight, options, rtol, atol):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != 8
            or not logits.is_floating_point() or not logits.requires_grad
            or not torch.isfinite(logits).all()):
        raise ValueError('Expected finite gradient-enabled [B,8,H,W] affinity logits')
    if (not isinstance(target, torch.Tensor) or target.shape != logits.shape
            or target.device != logits.device or not target.is_floating_point()
            or not torch.isfinite(target).all()):
        raise ValueError('Finite floating target must share logits shape/device')
    if (not isinstance(edge_valid, torch.Tensor) or edge_valid.shape != logits.shape
            or edge_valid.device != logits.device or edge_valid.dtype != torch.bool):
        raise ValueError('Boolean edge_valid must share logits shape/device')
    if (not isinstance(parameters, Mapping) or not parameters
            or any(not isinstance(name, str) or not name for name in parameters)):
        raise ValueError('shared_parameters must be a nonempty mapping of names to leaf parameters')
    seen = set()
    for name, parameter in parameters.items():
        if (not isinstance(parameter, torch.Tensor) or not parameter.is_leaf
                or not parameter.requires_grad or not parameter.is_floating_point()
                or parameter.device != logits.device or not torch.isfinite(parameter).all()
                or id(parameter) in seen):
            raise ValueError('Shared parameters must be unique finite gradient-enabled leaves: ' + name)
        seen.add(id(parameter))
    if edge_weight is not None and (not isinstance(edge_weight, torch.Tensor)
            or edge_weight.shape != logits.shape or edge_weight.device != logits.device
            or not torch.isfinite(edge_weight).all() or torch.any(edge_weight < 0)):
        raise ValueError('edge_weight must be finite, nonnegative and share logits shape/device')
    if set(options) != set(LOSS_OPTIONS) or type(options['normalize_edge_weights']) is not bool:
        raise ValueError('Only current balanced-affinity loss options are accepted')
    for key, expected in (('negative_weight', 1.), ('hard_negative_weight', 1.), ('hard_negative_gamma', 2.)):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
            raise ValueError('Distance mechanism check retains r1/h1/gamma2: ' + key)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
               for v in (rtol, atol)):
        raise ValueError('Gradient reconstruction tolerances must be finite and nonnegative')


def probe_distance_gradients(affinity_logits, target, edge_valid, shared_parameters, *,
                             edge_weight=None, loss_options=None, retain_graph=True,
                             rtol=2.e-5, atol=1.e-7):
    """返回可JSON序列化回执；只对已完成forward求autograd.grad。

    loss_options复用当前r1/h1/gamma2，并保留空间edge_weight与原class归一化。
    每组loss先按原公式完成各通道的类别归一化；原目标按有效通道数加权三组，
    Cshort则再乘1/.5/.25并除以有效通道的加权总数。不能将这些常数作为
    edge_weight传入原class归一化，后者会在分子/分母中抵消。

    有效通道指edge_valid里至少有一个关系，严格复现原loss的channel_losses；
    即使该通道空间edge_weight全零，也仍计入原平均的分母。
    缺少有效关系或零梯度时，范数/余弦/投影为None，不伪造冲突或正交。
    调用者负责整个模型的冻结状态和保存权重核对；本函数核验传入参数及其.grad不变。
    """
    options = dict(LOSS_OPTIONS)
    if loss_options is not None:
        if not isinstance(loss_options, Mapping) or not set(loss_options).issubset(options):
            raise ValueError('Unrecognized balanced-affinity loss option')
        options.update(loss_options)
    parameters = dict(shared_parameters) if isinstance(shared_parameters, Mapping) else shared_parameters
    _validate(affinity_logits, target, edge_valid, parameters, edge_weight, options, rtol, atol)
    if type(retain_graph) is not bool:
        raise ValueError('retain_graph must be boolean')
    snapshots = {name: (parameter.detach().clone(),
                       None if parameter.grad is None else parameter.grad.detach().clone())
                 for name, parameter in parameters.items()}
    counts, losses, gradients, edge_counts = {}, {}, {}, {}
    for name, (start, stop) in GROUPS.items():
        valid = edge_valid[:, start:stop]
        counts[name] = int(valid.flatten(2).any(2).any(0).sum())
        weights = None if edge_weight is None else edge_weight[:, start:stop]
        loss, details = balanced_affinity_loss(affinity_logits[:, start:stop], target[:, start:stop],
                                              valid, edge_weight=weights, **options)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite group loss: ' + name)
        losses[name] = loss
        edge_counts[name] = {key: details[key] for key in ('positive_edges', 'negative_edges')}
        gradients[name] = (_gradient(loss, parameters, retain_graph=True) if counts[name]
                           else {key: torch.zeros_like(value) for key, value in parameters.items()})
    total = sum(counts.values())
    weighted_total = sum(counts[name] * CSHORT_CHANNEL_WEIGHTS[name] for name in GROUPS)
    native_weights = {name: counts[name] / total if total else None for name in GROUPS}
    cshort_weights = {name: counts[name] * CSHORT_CHANNEL_WEIGHTS[name] / weighted_total
                     if weighted_total else None for name in GROUPS}
    full_loss, _ = balanced_affinity_loss(affinity_logits, target, edge_valid,
                                         edge_weight=edge_weight, **options)
    reconstructed_loss = (sum(losses[name] * native_weights[name] for name in GROUPS)
                          if total else affinity_logits.sum() * 0.)
    cshort_loss = (sum(losses[name] * cshort_weights[name] for name in GROUPS)
                  if total else affinity_logits.sum() * 0.)
    if not torch.isclose(full_loss.detach(), reconstructed_loss.detach(), rtol=rtol, atol=atol):
        raise RuntimeError('Equal-channel group combination does not reproduce the actual balanced loss')
    direct = _gradient(full_loss, parameters, retain_graph=retain_graph)
    reports = {}
    for parameter_name, parameter in parameters.items():
        current = {name: gradients[name][parameter_name] for name in GROUPS}
        native_mix = sum(current[name] * (native_weights[name] or 0.) for name in GROUPS)
        cshort_mix = sum(current[name] * (cshort_weights[name] or 0.) for name in GROUPS)
        if not all(torch.isfinite(value).all() for value in (*current.values(), native_mix, cshort_mix, direct[parameter_name])):
            raise FloatingPointError('Nonfinite shared-parameter gradient: ' + parameter_name)
        error = native_mix - direct[parameter_name]
        direct_norm, error_norm = _norm(direct[parameter_name]), _norm(error)
        if (not torch.allclose(native_mix, direct[parameter_name], rtol=rtol, atol=atol)
                or error_norm > atol + rtol * direct_norm):
            raise RuntimeError('Equal-channel gradients do not reproduce the actual loss: ' + parameter_name)
        delta = cshort_mix - native_mix
        reports[parameter_name] = dict(numel=parameter.numel(),
            group_norms={name: _reported_norm(value) if counts[name] else None for name, value in current.items()},
            pairwise_cosines={left + '__' + right: _cosine(current[left], current[right])
                for left, right in (('short', 'distance2'), ('short', 'distance4'), ('distance2', 'distance4'))},
            native_mixture_norm=_reported_norm(native_mix), cshort_mixture_norm=_reported_norm(cshort_mix),
            native_vs_cshort_cosine=_cosine(native_mix, cshort_mix),
            native_short_projection=_short_projection(native_mix, current['short']),
            cshort_short_projection=_short_projection(cshort_mix, current['short']),
            native_short_cosine=_cosine(native_mix, current['short']),
            cshort_short_cosine=_cosine(cshort_mix, current['short']),
            cshort_minus_native_norm=_reported_norm(delta),
            cshort_minus_native_short_projection=_short_projection(delta, current['short']),
            reconstruction=dict(passed=True, direct_norm=direct_norm, error_norm=error_norm,
                relative_l2_error=error_norm / direct_norm if direct_norm else None,
                maximum_absolute_error=float(error.abs().max()) if error.numel() else 0.))
    for name, parameter in parameters.items():
        value, grad = snapshots[name]
        if (not torch.equal(parameter.detach(), value)
                or (grad is None) != (parameter.grad is None)
                or (grad is not None and not torch.equal(parameter.grad.detach(), grad))):
            raise RuntimeError('A diagnostic changed parameter values or accumulated gradients: ' + name)
    return dict(format='affinity_distance_shared_gradients_v1', training=False,
        optimizer_step=False, parameter_values_exact=True, accumulated_gradients_untouched=True,
        group_channels={name: list(range(start, stop)) for name, (start, stop) in GROUPS.items()},
        active_channels=counts, total_active_channels=total, supervised_edges=edge_counts,
        loss_options=options, spatial_edge_weight_present=edge_weight is not None,
        group_losses={name: float(loss.detach()) if counts[name] else None for name, loss in losses.items()},
        native_group_weights=native_weights, cshort_channel_weights=CSHORT_CHANNEL_WEIGHTS.copy(),
        cshort_group_weights=cshort_weights, original_loss=float(full_loss.detach()),
        reconstructed_native_loss=float(reconstructed_loss.detach()), cshort_loss=float(cshort_loss.detach()),
        loss_reconstruction_passed=True, parameters=reports,
        scope='Read-only mechanism on shared parameters; gradient conflict does not establish final instance accuracy')
