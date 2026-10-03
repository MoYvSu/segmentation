# -*- coding: utf-8 -*-
"""新GT可信域内的部署融合排序附加项；不改变原八通道BCE或融合公式。

目标只要求真实分界比深内部高一个固定间隔，不把任一短程跨界
强制标成融合概率1。新GT的身份、来源和同步几何变换由调用者审计。
"""
from __future__ import annotations

from collections.abc import Mapping
import inspect
import math
from numbers import Integral, Real

import torch
from torch.nn import functional as F

from utils.affinity_fusion import affinity_boundary_probability

MARGIN = .10
TAIL_FRACTION = .10
MAX_PER_GROUP = 256
SUPPORT_RADIUS = 4


def _fusion_options(options):
    if not isinstance(options, Mapping):
        raise ValueError('fusion_kwargs must explicitly describe gated/top2 deployment')
    defaults = {name: value.default for name, value in
                inspect.signature(affinity_boundary_probability).parameters.items()
                if value.kind == inspect.Parameter.KEYWORD_ONLY}
    if set(options) - set(defaults):
        raise ValueError('Unknown fusion option')
    if options.get('mode', 'gated') != 'gated' or options.get('short_reduction') != 'top2':
        raise ValueError('The ranking auxiliary requires gated fusion with explicit top2')
    result = dict(defaults, **options)
    result['mode'] = 'gated'
    for name in ('distance2_weight', 'distance4_weight', 'support_threshold',
                 'support_temperature', 'short_softmax_temperature'):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError('Fusion scalar must be finite numeric: ' + name)
        result[name] = float(value)
    if (result['distance2_weight'] < 0 or result['distance4_weight'] < 0
            or not 0 <= result['support_threshold'] <= 1
            or result['support_temperature'] <= 0 or result['short_softmax_temperature'] <= 0):
        raise ValueError('Invalid deployed fusion scalar')
    return result


def _validate_targets(target, edge_valid):
    if (not torch.is_tensor(target) or target.ndim != 4 or target.shape[1] != 8
            or min(target.shape) < 1 or target.is_complex()
            or not bool(torch.isfinite(target).all())
            or not bool(((target == 0) | (target == 1)).all())):
        raise ValueError('target must contain finite binary GT relations in nonempty [B,8,H,W]')
    if (not torch.is_tensor(edge_valid) or edge_valid.dtype != torch.bool
            or edge_valid.shape != target.shape or edge_valid.device != target.device):
        raise ValueError('edge_valid must be matching bool GT relation validity on the same device')


def fused_ranking_domains(target, edge_valid):
    """返回BHW真分界/深内部；支持池窗口不能读到未知关系。

    四短程关系全可信且任一跨实例是真分界；八关系全可信且
    全同实例为深内部。gated在每个anchor还读取其长程关系和
    半径4支持窗口，因此中心八关系以及窗口全部四短关系均须可信。
    图框外明确当作未知；pool自身的零padding不能冒充可信内容。
    """
    _validate_targets(target, edge_valid)
    truth = target.detach().bool()
    known = edge_valid.detach()
    short_known = known[:, :4].all(dim=1, keepdim=True)
    unknown = (~short_known).float()
    padded = F.pad(unknown, (SUPPORT_RADIUS,) * 4, value=1.)
    support_known = F.max_pool2d(padded, 2 * SUPPORT_RADIUS + 1, stride=1)[:, 0] == 0
    eligible = known.all(dim=1) & support_known
    boundary = eligible & (~truth[:, :4]).any(dim=1)
    interior = eligible & truth.all(dim=1)
    return boundary, interior


def _mean(value):
    return float(value.detach().mean()) if value.numel() else None


def fused_boundary_ranking_loss(logits, target, edge_valid, *, fusion_kwargs,
                                margin=MARGIN, tail_fraction=TAIL_FRACTION,
                                max_per_group=MAX_PER_GROUP):
    """返回可反传loss与纯数值details；仅调用者决定是否用于人工源。

    每个样本选择最低ceil(10%)真实分界及最高ceil(10%)深内部，
    两侧各最多256个。选位detach，值与原gated支持池保持完整梯度。
    hinge=relu(margin + 内部响应 - 分界响应)，对全部组合平均，
    再对存在双侧监督的样本等权平均，不能只按活动组合归一化。
    空监督返回连接logits的零值，不丢失反向传播入口。
    """
    _validate_targets(target, edge_valid)
    if (not torch.is_tensor(logits) or logits.shape != target.shape
            or not logits.is_floating_point() or logits.device != target.device
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('logits must be matching finite floating [B,8,H,W] on the same device')
    for name, value in (('margin', margin), ('tail_fraction', tail_fraction)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(name + ' must be finite numeric')
    if not 0 < margin < 1 or not 0 < tail_fraction <= 1:
        raise ValueError('margin must be in (0,1), tail_fraction in (0,1]')
    if (isinstance(max_per_group, bool) or not isinstance(max_per_group, Integral)
            or not 1 <= max_per_group <= MAX_PER_GROUP):
        raise ValueError('max_per_group must be an integer within [1,256]')
    options = _fusion_options(fusion_kwargs)
    # FP16/BF16融合和组合均值统一升至FP32；FP64诊断保留原精度。
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = logits if logits.dtype == torch.float64 else logits.float()
        scores = affinity_boundary_probability(values, **options)[:, 0]
        boundary, interior = fused_ranking_domains(target, edge_valid)
        group_losses, groups, selected_true, selected_interior = [], [], [], []
        for index in range(logits.shape[0]):
            positive, negative = scores[index][boundary[index]], scores[index][interior[index]]
            row = dict(batch_index=index, eligible_true_boundary=positive.numel(),
                       eligible_deep_interior=negative.numel(), selected_true_boundary=0,
                       selected_deep_interior=0, total_pairs=0, active_pairs=0,
                       true_boundary_mean=_mean(positive), deep_interior_mean=_mean(negative),
                       selected_true_mean=None, selected_interior_mean=None, loss=None)
            if positive.numel() and negative.numel():
                npick = min(int(max_per_group), max(1, math.ceil(float(tail_fraction) * positive.numel())))
                nnick = min(int(max_per_group), max(1, math.ceil(float(tail_fraction) * negative.numel())))
                pi = positive.detach().topk(npick, largest=False).indices
                ni = negative.detach().topk(nnick, largest=True).indices
                selected_p, selected_n = positive[pi], negative[ni]
                violations = float(margin) + selected_n[None, :] - selected_p[:, None]
                loss = F.relu(violations).mean()
                group_losses.append(loss)
                selected_true.append(selected_p.detach())
                selected_interior.append(selected_n.detach())
                row.update(selected_true_boundary=npick, selected_deep_interior=nnick,
                           total_pairs=npick * nnick, active_pairs=int((violations.detach() > 0).sum()),
                           selected_true_mean=_mean(selected_p), selected_interior_mean=_mean(selected_n),
                           loss=float(loss.detach()))
            groups.append(row)
        # 使用一个有限标量接回计算图，避免大量有限极值求和溢出后inf*0。
        loss = torch.stack(group_losses).mean() if group_losses else values.reshape(-1)[0] * 0.
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite fused boundary ranking auxiliary')
    details = dict(margin=float(margin), tail_fraction=float(tail_fraction), max_per_group=int(max_per_group),
                   support_radius=SUPPORT_RADIUS, fusion_options=options,
                   valid_groups=len(group_losses), groups=groups, loss=float(loss.detach()),
                   eligible_true_boundary=int(boundary.sum()), eligible_deep_interior=int(interior.sum()),
                   selected_true_boundary=sum(r['selected_true_boundary'] for r in groups),
                   selected_deep_interior=sum(r['selected_deep_interior'] for r in groups),
                   total_pairs=sum(r['total_pairs'] for r in groups), active_pairs=sum(r['active_pairs'] for r in groups),
                   true_boundary_mean=_mean(scores[boundary]), deep_interior_mean=_mean(scores[interior]),
                   selected_true_mean=_mean(torch.cat(selected_true)) if selected_true else None,
                   selected_interior_mean=_mean(torch.cat(selected_interior)) if selected_interior else None)
    return loss, details
