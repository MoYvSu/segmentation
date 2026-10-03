# -*- coding: utf-8 -*-
"""在完整困难尾部内按响应秩取样；保留已封存融合排序的目标与梯度。

唯一变化是选位：先确定最难ceil(10%)，再覆盖其排序范围取最多256点。
这是响应秩的覆盖，不保证空间或实例均衡，也不引入新的随机流。
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch.nn import functional as F

from utils.affinity_fused_loss import (
    MARGIN, MAX_PER_GROUP, SUPPORT_RADIUS, TAIL_FRACTION,
    _fusion_options, _mean, _validate_targets, fused_ranking_domains,
)
from utils.affinity_fusion import affinity_boundary_probability

SAMPLING_FORMAT = 'tail_rank_stratified_v1'


def tail_rank_positions(tail_count, max_per_group=MAX_PER_GROUP, *, device=None):
    """返回0-based等间隔秩；多点时含两端，单点时取中位秩。

    Python整数算术避免大尾部计数在乘法中溢出。未触及上限时返回全部秩，
    与原topk的顺序完全相同。响应相等时沿用原torch.topk的同设备选位行为。
    """
    if (isinstance(tail_count, bool) or not isinstance(tail_count, Integral)
            or not 0 <= tail_count <= torch.iinfo(torch.int64).max):
        raise ValueError('tail_count must be a nonnegative int64-compatible integer')
    if (isinstance(max_per_group, bool) or not isinstance(max_per_group, Integral)
            or not 1 <= max_per_group <= MAX_PER_GROUP):
        raise ValueError('max_per_group must be an integer within [1,256]')
    count = min(int(tail_count), int(max_per_group))
    if not count:
        ranks = []
    elif count == 1:
        ranks = [int(tail_count) // 2]
    else:
        ranks = [(i * (int(tail_count)-1)) // (count-1) for i in range(count)]
    return torch.tensor(ranks, dtype=torch.long, device=device)


def tail_boundary_ranking_loss(logits, target, edge_valid, *, fusion_kwargs,
                               margin=MARGIN, tail_fraction=TAIL_FRACTION,
                               max_per_group=MAX_PER_GROUP):
    """返回loss及兼容原排序回执的details，新增完整尾部与选中秩记录。

    domains、GT/fusion校验及融合公式复用已封存实现。只有选位被detach，
    选中的响应和支持门仍参与反传。全部hinge组合均值及每样本等权不变。
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
    # 同时复用选秩函数的预算校验，不创建或消耗任何随机状态。
    tail_rank_positions(0, max_per_group)
    options = _fusion_options(fusion_kwargs)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = logits if logits.dtype == torch.float64 else logits.float()
        scores = affinity_boundary_probability(values, **options)[:, 0]
        boundary, interior = fused_ranking_domains(target, edge_valid)
        group_losses, groups, selected_true, selected_interior = [], [], [], []
        tail_true, tail_interior = [], []
        for index in range(logits.shape[0]):
            positive = scores[index][boundary[index]]
            negative = scores[index][interior[index]]
            nptail = math.ceil(float(tail_fraction) * positive.numel())
            nntail = math.ceil(float(tail_fraction) * negative.numel())
            row = dict(batch_index=index, eligible_true_boundary=positive.numel(),
                       eligible_deep_interior=negative.numel(), selected_true_boundary=0,
                       selected_deep_interior=0, total_pairs=0, active_pairs=0,
                       true_boundary_mean=_mean(positive), deep_interior_mean=_mean(negative),
                       selected_true_mean=None, selected_interior_mean=None, loss=None,
                       sampling_format=SAMPLING_FORMAT, tail_true_boundary=nptail,
                       tail_deep_interior=nntail, tail_true_mean=None, tail_interior_mean=None,
                       selected_true_ranks=[], selected_interior_ranks=[],
                       true_rank_range=None, interior_rank_range=None)
            if positive.numel() and negative.numel():
                # 完整尾部从最难至较易排序；不在完整尾部形成前应用256上限。
                ptail = positive.detach().topk(nptail, largest=False, sorted=True).indices
                ntail = negative.detach().topk(nntail, largest=True, sorted=True).indices
                pranks = tail_rank_positions(nptail, max_per_group, device=logits.device)
                nranks = tail_rank_positions(nntail, max_per_group, device=logits.device)
                selected_p, selected_n = positive[ptail[pranks]], negative[ntail[nranks]]
                violations = float(margin) + selected_n[None, :] - selected_p[:, None]
                loss = F.relu(violations).mean()
                group_losses.append(loss)
                selected_true.append(selected_p.detach())
                selected_interior.append(selected_n.detach())
                tail_true.append(positive[ptail].detach())
                tail_interior.append(negative[ntail].detach())
                plist, nlist = pranks.cpu().tolist(), nranks.cpu().tolist()
                row.update(selected_true_boundary=selected_p.numel(),
                           selected_deep_interior=selected_n.numel(),
                           total_pairs=selected_p.numel() * selected_n.numel(),
                           active_pairs=int((violations.detach() > 0).sum()),
                           selected_true_mean=_mean(selected_p), selected_interior_mean=_mean(selected_n),
                           loss=float(loss.detach()), tail_true_mean=_mean(positive[ptail]),
                           tail_interior_mean=_mean(negative[ntail]),
                           selected_true_ranks=plist, selected_interior_ranks=nlist,
                           true_rank_range=[plist[0], plist[-1]],
                           interior_rank_range=[nlist[0], nlist[-1]])
            groups.append(row)
        # 空监督也保留反传入口；避免有限极值求和溢出后inf*0。
        loss = torch.stack(group_losses).mean() if group_losses else values.reshape(-1)[0] * 0.
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite representative-tail boundary ranking auxiliary')
    details = dict(margin=float(margin), tail_fraction=float(tail_fraction),
                   max_per_group=int(max_per_group), support_radius=SUPPORT_RADIUS,
                   fusion_options=options, valid_groups=len(group_losses), groups=groups,
                   loss=float(loss.detach()), eligible_true_boundary=int(boundary.sum()),
                   eligible_deep_interior=int(interior.sum()),
                   selected_true_boundary=sum(r['selected_true_boundary'] for r in groups),
                   selected_deep_interior=sum(r['selected_deep_interior'] for r in groups),
                   total_pairs=sum(r['total_pairs'] for r in groups),
                   active_pairs=sum(r['active_pairs'] for r in groups),
                   true_boundary_mean=_mean(scores[boundary]), deep_interior_mean=_mean(scores[interior]),
                   selected_true_mean=_mean(torch.cat(selected_true)) if selected_true else None,
                   selected_interior_mean=_mean(torch.cat(selected_interior)) if selected_interior else None,
                   sampling_format=SAMPLING_FORMAT,
                   sampling_rule='sorted full ceil(fraction*N) tail; integer equal-rank endpoints; singleton midpoint',
                   tail_true_boundary=sum(r['tail_true_boundary'] for r in groups),
                   tail_deep_interior=sum(r['tail_deep_interior'] for r in groups),
                   tail_true_mean=_mean(torch.cat(tail_true)) if tail_true else None,
                   tail_interior_mean=_mean(torch.cat(tail_interior)) if tail_interior else None)
    return loss, details
