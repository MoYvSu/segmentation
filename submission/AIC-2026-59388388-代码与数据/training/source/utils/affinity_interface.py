# -*- coding: utf-8 -*-
"""连续短程界面监督；预测只决定选位，目标始终来自原标可信实例。"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.nn import functional as F

from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS, _edge_slices
from utils.affinity_connectivity import _line_points


def interface_loss(logits, target, edge_valid, predicted_instances, gt_instances, *,
                   trusted_pixels, max_groups=8, minimum_group_edges=2,
                   negative_margin=.3, positive_margin=.7, band_radius=4,
                   offsets=DEFAULT_AFFINITY_OFFSETS):
    """每组覆盖整段界面，不再按每组16条边截断。

    先为预测零缝指定最近预测区域，仅用于寻找界面，不写回预测或GT。
    正关系按GT实例分组：同GT内预测区域之间的短程界面扩展band_radius格，
    只监督仍属同GT且整段可信的distance1关系。负关系按无序GT对分组：
    只要该对的一处可信接触在预测中合并，便覆盖完整可信短程接触。
    margin仅决定该关系组是否仍困难；入选组的完整合法界面均参与BCE。
    两方向先组内平均，再方向平均，保留原总关系预算。
    """
    if (logits.ndim != 4 or logits.shape != target.shape or logits.shape != edge_valid.shape
            or edge_valid.dtype != torch.bool or not logits.is_floating_point()):
        raise ValueError('Expected matching floating BCHW logits/targets and bool edge_valid')
    if not bool(torch.isfinite(logits).all()) or not bool(((target == 0) | (target == 1)).all()):
        raise ValueError('Nonfinite logits or nonbinary targets')
    batch, channels, height, width = logits.shape
    offsets = tuple(offsets)
    if len(offsets) != channels:
        raise ValueError('Offset channels differ')
    for value in (max_groups, minimum_group_edges):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Group budgets must be positive integers')
    if isinstance(band_radius, bool) or not isinstance(band_radius, int) or not 0 <= band_radius <= 16:
        raise ValueError('band_radius must be an integer in 0..16')
    if not 0 < negative_margin < positive_margin < 1:
        raise ValueError('Invalid activity margins')
    if trusted_pixels.ndim == 4:
        trusted_pixels = trusted_pixels[:, 0]
    if trusted_pixels.shape != (batch, height, width) or trusted_pixels.dtype != torch.bool:
        raise ValueError('Expected bool BHW trust')
    for ids in (gt_instances, predicted_instances):
        if (ids.shape != (batch, height, width) or ids.is_floating_point()
                or ids.dtype == torch.bool or bool((ids < 0).any())):
            raise ValueError('Expected nonnegative integer BHW instances')
    pred = predicted_instances.detach().cpu().numpy().astype(np.int64)
    gt = gt_instances.detach().cpu().numpy().astype(np.int64)
    trust = trusted_pixels.detach().cpu().numpy()
    valid = edge_valid.detach().cpu().numpy()
    truth = target.detach().cpu().numpy()
    prob = logits.detach().float().sigmoid().cpu().numpy()
    short = [c for c, (dy, dx) in enumerate(offsets) if max(abs(dy), abs(dx)) == 1]
    if not short:
        raise ValueError('No distance1 channels')
    groups = {'negative': [], 'positive': []}
    eligible_counts = {key: 0 for key in groups}
    kernel = np.ones((2 * band_radius + 1, 2 * band_radius + 1), np.uint8)

    for b in range(batch):
        if not np.any(pred[b] > 0):
            continue
        # 无目标生成：最近区域仅给预测零缝提供临时ownership，用后即弃。
        _, nearest = distance_transform_edt(pred[b] == 0, return_indices=True)
        owners = pred[b][tuple(nearest)]
        interface_pixels = np.zeros((height, width), np.uint8)
        channel_data = []
        negative_edges = {}
        negative_trigger = set()
        for c in short:
            dy, dx = offsets[c]
            src, dst = _edge_slices(height, width, dy, dx)
            ys, xs = src[0].start, src[1].start
            gs, gd = gt[b][src], gt[b][dst]
            ps, pd = owners[src], owners[dst]
            legal = valid[b, c][src].copy() & (gs > 0) & (gd > 0)
            for oy, ox in _line_points(dy, dx):
                legal &= trust[b, ys+oy:src[0].stop+oy, xs+ox:src[1].stop+ox]
            same = gs == gd
            if np.any(legal & (truth[b, c][src] != same)):
                raise ValueError('GT IDs disagree with affinity targets')
            crossing = legal & same & (ps != pd) & (ps > 0) & (pd > 0)
            interface_pixels[src] |= crossing.astype(np.uint8)
            interface_pixels[dst] |= crossing.astype(np.uint8)
            channel_data.append((c, src, dst, legal, same))
            y, x = np.nonzero(legal & ~same)
            if len(y):
                keys = np.sort(np.stack([gs[y, x], gd[y, x]], 1), axis=1)
                for pair in np.unique(keys, axis=0):
                    take = np.all(keys == pair, axis=1)
                    yy, xx = y[take], x[take]
                    key = tuple(map(int, pair))
                    flat = ((b * channels + c) * height + yy + ys) * width + xx + xs
                    negative_edges.setdefault(key, []).extend(flat.tolist())
                    if np.any(ps[yy, xx] == pd[yy, xx]):
                        negative_trigger.add(key)
        positive_edges = {}
        if interface_pixels.any():
            # 各GT单独扩展，邻近晶粒的界面不能成为本GT的正监督来源。
            for iid in np.unique(gt[b][interface_pixels > 0]):
                support = (gt[b] == iid) & trust[b]
                ys, xs = np.nonzero(support)
                y0, y1 = max(0, int(ys.min())-band_radius), min(height, int(ys.max())+band_radius+1)
                x0, x1 = max(0, int(xs.min())-band_radius), min(width, int(xs.max())+band_radius+1)
                band = np.zeros((height, width), bool)
                seed = ((interface_pixels > 0) & support)[y0:y1, x0:x1].astype(np.uint8)
                band[y0:y1, x0:x1] = cv2.dilate(seed, kernel).astype(bool)
                for c, src, dst, legal, same in channel_data:
                    take = legal & same & (gt[b][src] == iid) & (band[src] | band[dst])
                    y, x = np.nonzero(take)
                    if len(y):
                        flat = ((b * channels + c) * height + y + src[0].start) * width + x + src[1].start
                        positive_edges.setdefault(int(iid), []).extend(flat.tolist())
        for direction, mapping in [('negative', negative_edges), ('positive', positive_edges)]:
            for key, edges in mapping.items():
                if direction == 'negative' and key not in negative_trigger:
                    continue
                if len(edges) < minimum_group_edges:
                    continue
                idx = np.asarray(edges, np.int64)
                scores = prob.reshape(-1)[idx]
                active = scores > negative_margin if direction == 'negative' else scores < positive_margin
                eligible_counts[direction] += 1
                if not active.any():
                    continue
                hardness = scores if direction == 'negative' else 1 - scores
                groups[direction].append((float(hardness[active].mean()), b, str(key), idx))

    for direction in groups:
        groups[direction].sort(key=lambda v: (-v[0], v[1], v[2]))
    chosen = {key: [] for key in groups}
    if all(groups.values()):
        chosen = {key: value[:max_groups // 2] for key, value in groups.items()}
    remaining = max_groups - sum(map(len, chosen.values()))
    spare = [(item[0], key, item) for key in groups for item in groups[key][len(chosen[key]):]]
    spare.sort(key=lambda v: (-v[0], v[1], v[2][1], v[2][2]))
    for _, key, item in spare[:remaining]:
        chosen[key].append(item)
    stats = {}
    losses = []
    flat_logits = logits.float().reshape(-1)
    for direction, selected in chosen.items():
        group_losses, indices = [], []
        for _, _, _, ids in selected:
            index = torch.as_tensor(ids, device=logits.device, dtype=torch.long)
            values = flat_logits[index]
            group_losses.append(F.binary_cross_entropy_with_logits(values, torch.full_like(values, float(direction == 'positive'))))
            indices.extend(ids.tolist())
        by_channel = np.bincount((np.asarray(indices, np.int64) // (height * width)) % channels,
                                  minlength=channels).tolist()
        zero_edges = 0
        for flat in indices:
            b, c, y, x = np.unravel_index(flat, logits.shape)
            dy, dx = offsets[c]
            zero_edges += int(pred[b, y, x] == 0 or pred[b, y+dy, x+dx] == 0)
        stats[direction] = dict(selected_edges=len(indices), selected_groups=len(selected),
            eligible_groups=eligible_counts[direction], active_groups=len(groups[direction]),
            selected_edges_by_channel=by_channel, selected_zero_endpoint_edges=zero_edges)
        if group_losses:
            losses.append(torch.stack(group_losses).mean())
    loss = torch.stack(losses).mean() if losses else logits.float().sum() * 0
    return loss, dict(version='continuous_interface_v1', loss=float(loss.detach()),
        band_radius=band_radius, max_groups=max_groups, trusted_path_check=True,
        selected_edges=sum(x['selected_edges'] for x in stats.values()),
        selected_groups=sum(x['selected_groups'] for x in stats.values()),
        directions_present=len(losses), **stats)
