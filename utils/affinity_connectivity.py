# -*- coding: utf-8 -*-
"""用真实部署实例定位双向困难 affinity 边；不是 MALIS 或拓扑保证。

调用者先按原部署口径取得最终实例，再对齐到 logits 网格。本模块只改变
已有合法 GT 边的训练权重，不生成实例 GT，也不改变推理阈值或后处理。
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import torch
from torch.nn import functional as F

from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _line_points(dy, dx):
    """包含两端的整数 Bresenham 线段，禁止长边越过未知标注。"""
    y = x = 0
    ax, ay = abs(dx), abs(dy)
    sx, sy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    error = ax - ay
    points = []
    while True:
        points.append((y, x))
        if (y, x) == (dy, dx):
            return points
        doubled = 2 * error
        if doubled > -ay:
            error -= ay
            x += sx
        if doubled < ax:
            error += ax
            y += sy


def _empty_direction():
    return dict(candidate_edges=0, trusted_endpoint_edges=0,
                skipped_untrusted_endpoint_edges=0, skipped_unknown_path_edges=0,
                skipped_untrusted_path_edges=0, valid_edges=0, active_edges=0,
                candidate_groups=0, skipped_small_groups=0, active_groups=0,
                selected_groups=0, selected_edges=0,
                mean_selected_probability=None, mean_group_loss=None)


def deployment_connectivity_loss(
    logits, target, edge_valid, predicted_instances, gt_instances, *,
    trusted_pixels=None, max_groups=8, edges_per_group=16,
    minimum_group_edges=2, negative_margin=0.3, positive_margin=0.7,
    offsets=DEFAULT_AFFINITY_OFFSETS,
):
    """对部署误连/误断关系中的困难合法边做关系均衡 BCE。

    logits/target/edge_valid 为 BCHW；实例图为 BHW，trusted 为 BHW 或 B1HW。
    负方向：不同 GT 实例被预测成同一实例；按 (batch, GT无序对) 分组。
    正方向：同一 GT 实例被预测拆分；按 (batch, GT, 预测无序对) 分组。
    两个预测端点必须均为正 ID；线段内预测零缝不排除，但 GT 必须已知，
    trusted 若提供则整条线段都必须可信。unknown 永远不能变成负实例。

    minimum_group_edges 检查 margin 筛选前的可信候选，因而一个原本有充分
    支持的关系即使只剩一个未满足边也可学习。每组只选最多 edges_per_group
    个最难活动边。max_groups 是整个调用的关系总预算，双向均有候选时尽量
    均分，再将空余名额给另一方向；同方向按组内最难边平均难度排序。
    各组先平均 BCE，再在每个方向平均，最后对存在的方向等权平均。
    采样全部 detach；梯度仅流入所选原始 logits。空结果仍可反向传播。
    """
    max_groups = _positive_integer(max_groups, 'max_groups')
    edges_per_group = _positive_integer(edges_per_group, 'edges_per_group')
    minimum_group_edges = _positive_integer(minimum_group_edges, 'minimum_group_edges')
    if (isinstance(negative_margin, bool) or isinstance(positive_margin, bool)
            or not isinstance(negative_margin, Real) or not isinstance(positive_margin, Real)
            or not math.isfinite(negative_margin) or not math.isfinite(positive_margin)
            or not 0 < negative_margin < positive_margin < 1):
        raise ValueError('Margins must satisfy 0 < negative_margin < positive_margin < 1')
    if not torch.is_tensor(logits) or logits.ndim != 4 or min(logits.shape) < 1:
        raise ValueError('logits must be a nonempty BCHW tensor')
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError('logits must be finite floating point values')
    batch, channels, height, width = logits.shape
    offsets = tuple(offsets)
    if len(offsets) != channels:
        raise ValueError('offset count must equal logits channels')
    for offset in offsets:
        if (not isinstance(offset, (tuple, list)) or len(offset) != 2
                or any(isinstance(v, bool) or not isinstance(v, Integral) for v in offset)
                or tuple(offset) == (0, 0)):
            raise ValueError('Each offset must contain two integers and be nonzero')
    if len(set(tuple(offset) for offset in offsets)) != len(offsets):
        raise ValueError('Duplicate offsets are not allowed')
    if not torch.is_tensor(target) or target.shape != logits.shape:
        raise ValueError('target must match logits shape')
    if (not bool(torch.isfinite(target).all())
            or not bool(((target == 0) | (target == 1)).all())):
        raise ValueError('target must contain finite binary GT affinity values')
    if (not torch.is_tensor(edge_valid) or edge_valid.shape != logits.shape
            or edge_valid.dtype != torch.bool):
        raise ValueError('edge_valid must be a bool tensor matching logits')
    arrays = []
    for name, labels in [('predicted_instances', predicted_instances), ('gt_instances', gt_instances)]:
        if (not torch.is_tensor(labels) or labels.shape != (batch, height, width)
                or labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool
                or bool((labels < 0).any())):
            raise ValueError(f'{name} must contain nonnegative integer IDs in BHW shape')
        arrays.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))
    predicted, gt = arrays
    if trusted_pixels is None:
        trusted = np.ones((batch, height, width), dtype=bool)
    else:
        if not torch.is_tensor(trusted_pixels):
            raise ValueError('trusted_pixels must be a bool tensor')
        if trusted_pixels.shape == (batch, 1, height, width):
            trusted_pixels = trusted_pixels[:, 0]
        if trusted_pixels.shape != (batch, height, width) or trusted_pixels.dtype != torch.bool:
            raise ValueError('trusted_pixels must have bool BHW or B1HW shape')
        trusted = trusted_pixels.detach().cpu().numpy()
    valid = edge_valid.detach().cpu().numpy()
    truth = target.detach().float().cpu().numpy()
    probability = logits.detach().float().sigmoid().cpu().numpy()
    negative_cutoff = float(np.float32(negative_margin))
    positive_cutoff = float(np.float32(positive_margin))
    stats = {'negative': _empty_direction(), 'positive': _empty_direction()}
    groups = {'negative': {}, 'positive': {}}

    for channel, (dy, dx) in enumerate(offsets):
        dy, dx = int(dy), int(dx)
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        if y0 >= y1 or x0 >= x1:
            continue
        source = (slice(None), slice(y0, y1), slice(x0, x1))
        destination = (slice(None), slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
        gs, gd, ps, pd = gt[source], gt[destination], predicted[source], predicted[destination]
        eligible = valid[:, channel, y0:y1, x0:x1] & (gs > 0) & (gd > 0) & (ps > 0) & (pd > 0)
        endpoints_trusted = trusted[source] & trusted[destination]
        known_path = np.ones_like(eligible)
        trusted_path = np.ones_like(eligible)
        for sy, sx in _line_points(dy, dx)[1:-1]:
            segment = (slice(None), slice(y0 + sy, y1 + sy), slice(x0 + sx, x1 + sx))
            known_path &= gt[segment] > 0
            trusted_path &= trusted[segment]
        local_probability = probability[:, channel, y0:y1, x0:x1]
        for direction, relation in [('negative', (gs != gd) & (ps == pd)),
                                    ('positive', (gs == gd) & (ps != pd))]:
            counters = stats[direction]
            candidate = eligible & relation
            counters['candidate_edges'] += int(candidate.sum())
            endpoint_candidate = candidate & endpoints_trusted
            counters['trusted_endpoint_edges'] += int(endpoint_candidate.sum())
            counters['skipped_untrusted_endpoint_edges'] += int((candidate & ~endpoints_trusted).sum())
            counters['skipped_unknown_path_edges'] += int((endpoint_candidate & ~known_path).sum())
            counters['skipped_untrusted_path_edges'] += int((endpoint_candidate & known_path & ~trusted_path).sum())
            candidate = endpoint_candidate & known_path & trusted_path
            counters['valid_edges'] += int(candidate.sum())
            ib, iy, ix = np.nonzero(candidate)
            if ib.size == 0:
                continue
            expected = 0 if direction == 'negative' else 1
            if not np.all(truth[ib, channel, iy + y0, ix + x0] == expected):
                raise ValueError('GT affinity targets disagree with supplied instance IDs on selected relations')
            scores = local_probability[ib, iy, ix]
            for b, y, x, score in zip(ib.tolist(), iy.tolist(), ix.tolist(), scores.tolist()):
                if direction == 'negative':
                    key = (b, min(int(gs[b, y, x]), int(gd[b, y, x])),
                           max(int(gs[b, y, x]), int(gd[b, y, x])))
                    active = score > negative_cutoff
                    hardness = score
                else:
                    key = (b, int(gs[b, y, x]), min(int(ps[b, y, x]), int(pd[b, y, x])),
                           max(int(ps[b, y, x]), int(pd[b, y, x])))
                    active = score < positive_cutoff
                    hardness = 1.0 - score
                group = groups[direction].setdefault(key, {'candidate_edges': 0, 'active': []})
                group['candidate_edges'] += 1
                if active:
                    flat = ((b * channels + channel) * height + y + y0) * width + x + x0
                    group['active'].append((float(hardness), int(flat), float(score), channel))
                    counters['active_edges'] += 1

    ranked = {}
    for direction in ('negative', 'positive'):
        counters = stats[direction]
        counters['candidate_groups'] = len(groups[direction])
        proposals = []
        for key, group in groups[direction].items():
            if group['candidate_edges'] < minimum_group_edges:
                counters['skipped_small_groups'] += 1
                continue
            if not group['active']:
                continue
            selected = sorted(group['active'], key=lambda item: (-item[0], item[1]))[:edges_per_group]
            score = float(np.mean([item[0] for item in selected]))
            proposals.append((score, key, selected))
        proposals.sort(key=lambda item: (-item[0], item[1]))
        ranked[direction] = proposals
        counters['active_groups'] = len(proposals)

    # 总预算优先双向均分；无候选方向不浪费预算，奇数余量优先给更难关系。
    chosen = {direction: [] for direction in ranked}
    if ranked['negative'] and ranked['positive']:
        for direction in ranked:
            chosen[direction] = ranked[direction][:max_groups // 2]
    remaining = max_groups - sum(len(value) for value in chosen.values())
    leftovers = [(group[0], direction, group) for direction in ranked
                 for group in ranked[direction][len(chosen[direction]):]]
    leftovers.sort(key=lambda item: (-item[0], item[1], item[2][1]))
    for _, direction, group in leftovers[:remaining]:
        chosen[direction].append(group)

    flat_logits = logits.float().reshape(-1)
    direction_losses = []
    for direction in ('negative', 'positive'):
        group_losses, group_probabilities = [], []
        by_channel = [0] * channels
        for _, _, edges in chosen[direction]:
            index = torch.tensor([item[1] for item in edges], device=logits.device, dtype=torch.long)
            selected_logits = flat_logits[index]
            desired = torch.full_like(selected_logits, 0.0 if direction == 'negative' else 1.0)
            group_losses.append(F.binary_cross_entropy_with_logits(selected_logits, desired))
            group_probabilities.append(float(np.mean([item[2] for item in edges])))
            for edge in edges:
                by_channel[edge[3]] += 1
        counters = stats[direction]
        counters['selected_groups'] = len(group_losses)
        counters['selected_edges'] = sum(by_channel)
        counters['selected_edges_by_channel'] = by_channel
        if group_losses:
            direction_loss = torch.stack(group_losses).mean()
            direction_losses.append(direction_loss)
            counters['mean_group_loss'] = float(direction_loss.detach())
            counters['mean_selected_probability'] = float(np.mean(group_probabilities))
    loss = torch.stack(direction_losses).mean() if direction_losses else logits.float().sum() * 0.0
    metrics = dict(version='deployment_pair_hard_v1', loss=float(loss.detach()),
                   max_groups=max_groups, edges_per_group=edges_per_group,
                   minimum_group_edges=minimum_group_edges,
                   negative_margin=float(negative_margin), positive_margin=float(positive_margin),
                   trusted_path_check=True,
                   selected_groups=sum(value['selected_groups'] for value in stats.values()),
                   selected_edges=sum(value['selected_edges'] for value in stats.values()),
                   directions_present=len(direction_losses), **stats)
    return loss, metrics
