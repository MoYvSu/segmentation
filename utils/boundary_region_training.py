# -*- coding: utf-8 -*-
"""可靠内核之间的完整分隔监督，不回归精确GT细线。

``region_loss(logits, batch, settings)``返回可反传标量及JSON审计；
``region_update(model, optimizer, batch, settings)``完成一次仅生成器更新。

settings是平面字典：high_weight/low_weight/separation_weight默认1；
positive_margin=.9；negative_margin=.1；min_atom_pixels=64；max_pairs_per_image=1（可设0关闭选对）；
max_pair_attempts=4；capacity_scale=1000；exterior=True。三项分别先在每个非空
样本内部按自身像素平均，再在非空样本间平均，空样本计数另报、不稀释有效项。
完整割仅在未达到positive_margin的位置回传正监督；几何证书仍检查整套割。
keep_low仅在当前p>=negative_margin的未满足位置归一；静态内核、seed身份及其
覆盖分母不因此改变。已满足低响应区域不稀释少数异常高点的纠错梯度。

种子只来自region_keep_low中不同GT身份的单个8连通原子，不强制同GT全域连通。
选割使用当前G的停止梯度概率，包括unknown的空间连接；unknown不产生新监督。
GT target张量不进入损失，既不使用GAN，也不使用PH或精确边界像素BCE。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from numbers import Integral, Real
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from utils.boundary_region_cut import select_region_cut


DEFAULT_SETTINGS = dict(high_weight=1.0, low_weight=1.0, separation_weight=1.0,
    positive_margin=.9, negative_margin=.1, min_atom_pixels=64, max_pairs_per_image=1,
    max_pair_attempts=4, capacity_scale=1000, exterior=True)


def _settings(values):
    if values is None:
        values = {}
    if not isinstance(values, dict) or set(values) - set(DEFAULT_SETTINGS):
        raise ValueError('Unknown or invalid region loss settings')
    result = {**DEFAULT_SETTINGS, **values}
    for key in ('high_weight', 'low_weight', 'separation_weight'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(key + ' must be finite and nonnegative')
        result[key] = float(value)
    margin = result['positive_margin']
    if isinstance(margin, bool) or not isinstance(margin, Real) or not math.isfinite(margin) or not 0 < margin < 1:
        raise ValueError('positive_margin must lie strictly within (0,1)')
    result['positive_margin'] = float(margin)
    negative = result['negative_margin']
    if (isinstance(negative, bool) or not isinstance(negative, Real) or not math.isfinite(negative)
            or not 0 < negative < result['positive_margin']):
        raise ValueError('negative_margin must lie strictly between zero and positive_margin')
    result['negative_margin'] = float(negative)
    for key in ('min_atom_pixels', 'max_pair_attempts', 'capacity_scale'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if type(result['max_pairs_per_image']) is not int or result['max_pairs_per_image'] not in (0, 1):
        raise ValueError('max_pairs_per_image must be 0 or 1')
    if result['exterior'] is not True:
        raise ValueError('Region training requires exterior=True; crop edges cannot be walls')
    return result


def _binary_tensor(value, shape, device, name):
    if (not torch.is_tensor(value) or value.shape != shape or value.device != device
            or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
            or bool(((value != 0) & (value != 1)).any())):
        raise ValueError(name + ' must be an aligned binary tensor on the logits device')
    return value.bool()


def _validate(logits, batch):
    if (not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[1] != 1
            or min(logits.shape) < 1 or not logits.is_floating_point()
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('Region logits must be finite nonempty floating B1HW')
    shape, device = logits.shape, logits.device
    masks = {key: _binary_tensor(batch[key], shape, device, key) for key in
        ('valid', 'region_original', 'region_allowed', 'region_keep_high', 'region_keep_low')}
    ids = {}
    for key in ('region_ids', 'region_core_ids'):
        value = batch[key]
        if (not torch.is_tensor(value) or value.shape != shape or value.device != device
                or value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or bool(((value < 0) | (value > 65535)).any())):
            raise ValueError(key + ' must be aligned integer IDs in [0,65535]')
        ids[key] = value
    valid = masks['valid']
    for key in ('region_allowed', 'region_keep_high', 'region_keep_low'):
        if bool((masks[key] & ~valid).any()):
            raise ValueError(key + ' may not supervise unknown, crop border, or padding')
    if bool((valid & (ids['region_ids'] <= 0)).any()):
        raise ValueError('Known region pixels require a GT instance ID')
    core = ids['region_core_ids'] > 0
    if (bool((core & ~valid).any()) or bool((core & ~masks['region_original']).any())
            or bool((core & (ids['region_core_ids'] != ids['region_ids'])).any())):
        raise ValueError('Reliable core identities must agree with known original GT')
    if bool((masks['region_keep_low'] & ~core).any()):
        raise ValueError('keep_low requires reliable instance cores')
    if bool((masks['region_keep_low'] & (masks['region_allowed'] | masks['region_keep_high'])).any()):
        raise ValueError('Low supervision cannot overlap the modifiable corridor or high supervision')
    metadata = batch.get('meta')
    if not isinstance(metadata, (list, tuple)) or len(metadata) != shape[0]:
        raise ValueError('A metadata record with draw_seed is required per sample')
    seeds = []
    for item in metadata:
        seed = item.get('draw_seed') if isinstance(item, dict) else None
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError('draw_seed must be a nonnegative integer')
        seeds.append(int(seed))
    return masks, ids, seeds


def _adjacent_ids(ids, valid):
    pairs = set()
    for a, b, va, vb in ((ids[1:], ids[:-1], valid[1:], valid[:-1]),
                          (ids[:, 1:], ids[:, :-1], valid[:, 1:], valid[:, :-1])):
        selected = va & vb & (a > 0) & (b > 0) & (a != b)
        if selected.any():
            values = np.stack([a[selected], b[selected]], 1)
            for first, second in np.unique(np.sort(values, 1), axis=0):
                pairs.add((int(first), int(second)))
    return pairs


def _atoms(core_ids, low, probability, margin, minimum):
    # 原子身份只由静态GT/N1可靠低域决定。不能由G涂墙后缩小原子或换seed。
    candidates = low & (core_ids > 0)
    atoms = []
    for gid in sorted(map(int, np.unique(core_ids[candidates]))):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (candidates & (core_ids == gid)).astype(np.uint8), connectivity=8)
        for component in range(1, count):
            size = int(stats[component, cv2.CC_STAT_AREA])
            if size < minimum:
                continue
            atoms.append(dict(gt_id=gid, component=component, pixels=size,
                mask=labels == component, bbox_xywh=[int(v) for v in stats[component, :4]]))
    return atoms, int((candidates & (probability >= margin)).sum())


def _certificate(probability, cut, source, sink, threshold):
    """独立8邻接证书；周边所有自由组件经外部节点相连。"""
    wall = (probability >= threshold) | cut
    if np.any(source & wall) or np.any(sink & wall):
        return dict(valid=False, reason='seed_blocked_by_wall_or_cut')
    _, labels = cv2.connectedComponents((~wall).astype(np.uint8), connectivity=8)
    left = set(map(int, np.unique(labels[source]))) - {0}
    right = set(map(int, np.unique(labels[sink]))) - {0}
    if not left or not right:
        return dict(valid=False, reason='empty_open_seed')
    exterior = set(map(int, np.unique(np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]])))) - {0}
    connected = bool(left & right) or bool(left & exterior) and bool(right & exterior)
    return dict(valid=not connected, reason='separated' if not connected else 'remaining_native_or_exterior_route',
        source_components=len(left), sink_components=len(right),
        source_reaches_exterior=bool(left & exterior), sink_reaches_exterior=bool(right & exterior))


def _mask_mean_per_sample(logits, masks, positive):
    rows = []
    for index in range(len(logits)):
        selected = masks[index]
        if bool(selected.any()):
            values = logits[index][selected]
            rows.append(F.softplus(-values if positive else values).mean())
    return (torch.stack(rows).mean() if rows else logits.sum() * 0.0), len(rows)


def region_loss(logits, batch, settings=None):
    """对完整可行分隔及已有可靠高/低响应分别监督；返回标量和小型审计。"""
    started = time.perf_counter()
    cfg = _settings(settings)
    masks, ids, seeds = _validate(logits, batch)
    detached_probability = logits.detach().sigmoid()
    active_low = masks['region_keep_low'] & (detached_probability >= cfg['negative_margin'])
    probability = detached_probability.float().cpu().numpy()[:, 0]
    arrays = {key: value.detach().cpu().numpy()[:, 0] for key, value in {**masks, **ids}.items()}
    low_active_counts = active_low.flatten(1).sum(1).detach().cpu().tolist()
    active_cuts = np.zeros(logits.shape, bool)
    images = []
    reason_totals = Counter()
    for index, p in enumerate(probability):
        high = arrays['region_keep_high'][index]
        low = arrays['region_keep_low'][index]
        atom_rows, excluded = _atoms(arrays['region_core_ids'][index], low, p,
                                    cfg['positive_margin'], cfg['min_atom_pixels'])
        adjacent = _adjacent_ids(arrays['region_ids'][index], arrays['valid'][index])
        preferred, other = [], []
        for first in range(len(atom_rows)):
            for second in range(first + 1, len(atom_rows)):
                a, b = atom_rows[first], atom_rows[second]
                if a['gt_id'] == b['gt_id']:
                    continue
                item = (first, second)
                (preferred if tuple(sorted((a['gt_id'], b['gt_id']))) in adjacent else other).append(item)
        rng = np.random.default_rng(seeds[index])
        candidates = ([preferred[int(i)] for i in rng.permutation(len(preferred))]
                      + [other[int(i)] for i in rng.permutation(len(other))])
        reasons = Counter()
        attempts, selected_pairs = [], []
        if not cfg['max_pairs_per_image'] or cfg['separation_weight'] == 0:
            reasons['separation_disabled'] += 1
        elif not candidates:
            reasons['no_different_GT_atom_pair'] += 1
        else:
            for first, second in candidates[:cfg['max_pair_attempts']]:
                a, b = atom_rows[first], atom_rows[second]
                cut, cut_audit = select_region_cut(p, arrays['region_allowed'][index],
                    a['mask'], b['mask'], threshold=cfg['positive_margin'],
                    capacity_scale=cfg['capacity_scale'], exterior=cfg['exterior'])
                cut = np.asarray(cut)
                if (cut.shape != p.shape or not np.isin(cut, [0, 1]).all()
                        or cut.dtype != np.bool_ or np.any(cut & ~arrays['region_allowed'][index])
                        or np.any(cut & (a['mask'] | b['mask']))):
                    raise RuntimeError('Selector returned illegal or seed-crossing cut pixels')
                cut = cut.astype(bool)
                certificate = _certificate(p, cut, a['mask'], b['mask'], cfg['positive_margin'])
                item = dict(gt_ids=[a['gt_id'], b['gt_id']], atom_indices=[first, second],
                    source_pixels=a['pixels'], sink_pixels=b['pixels'],
                    source_open_pixels=int((a['mask'] & (p < cfg['positive_margin'])).sum()),
                    sink_open_pixels=int((b['mask'] & (p < cfg['positive_margin'])).sum()),
                    source_open_fraction=float((a['mask'] & (p < cfg['positive_margin'])).sum()/a['pixels']),
                    sink_open_fraction=float((b['mask'] & (p < cfg['positive_margin'])).sum()/b['pixels']),
                    adjacent_GT=tuple(sorted((a['gt_id'], b['gt_id']))) in adjacent,
                    selector=cut_audit, independent_certificate=certificate,
                    cut_pixels=int(cut.sum()),
                    cut_sha256=hashlib.sha256(np.ascontiguousarray(cut).tobytes()).hexdigest())
                attempts.append(item)
                if not cut.any():
                    reason = ('already_separated' if certificate['valid'] else
                              'seed_coverage_blocked' if certificate['reason']=='seed_blocked_by_wall_or_cut' else
                              cut_audit.get('reason', 'no_complete_separator'))
                    reasons[str(reason)] += 1
                    continue
                if cut_audit.get('feasible') is not True or not certificate['valid']:
                    # 不能训练非法割的合法子集后声称完整分隔。
                    reasons['seed_coverage_blocked' if certificate['reason']=='seed_blocked_by_wall_or_cut'
                            else 'uncertified_complete_cut'] += 1
                    continue
                active = cut & (p < cfg['positive_margin'])
                if not active.any():
                    reasons['complete_cut_already_at_margin'] += 1
                    continue
                active_cuts[index, 0] |= active
                selected_pairs.append(item)
                reasons['selected_complete_separator'] += 1
                break
        images.append(dict(sample=index, draw_seed=seeds[index],
            high_pixels=int(high.sum()), low_pixels=int(low.sum()),
            active_low_pixels=int(low_active_counts[index]),
            atom_count=len(atom_rows), seed_wall_pixels=excluded,
            eligible_pairs=len(candidates), adjacent_pairs=len(preferred),
            atoms=[{k:v for k,v in row.items() if k != 'mask'} for row in atom_rows],
            attempts=attempts, selected_pairs=selected_pairs, reasons=dict(reasons),
            selected_cut_pixels=sum(item['cut_pixels'] for item in selected_pairs),
            active_cut_pixels=int(active_cuts[index].sum())))
        reason_totals.update(reasons)
    active = torch.from_numpy(active_cuts).to(logits.device)
    high_loss, high_samples = _mask_mean_per_sample(logits, masks['region_keep_high'], True)
    low_loss, low_active_samples = _mask_mean_per_sample(logits, active_low, False)
    low_samples = int(masks['region_keep_low'].flatten(1).any(1).sum())
    separator_loss, separator_samples = _mask_mean_per_sample(logits, active, True)
    loss = (cfg['high_weight'] * high_loss + cfg['low_weight'] * low_loss
            + cfg['separation_weight'] * separator_loss)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite region loss')
    supervised = ((cfg['high_weight'] > 0 and high_samples > 0)
                  or (cfg['low_weight'] > 0 and low_active_samples > 0)
                  or (cfg['separation_weight'] > 0 and separator_samples > 0))
    audit = dict(format='boundary_region_loss_v1', settings=cfg, samples=len(logits),
        loss_g=float(loss.detach()), loss_high=float(high_loss.detach()),
        loss_low=float(low_loss.detach()), loss_separation=float(separator_loss.detach()),
        high_samples=high_samples, low_samples=low_samples, separation_samples=separator_samples,
        low_active_samples=low_active_samples, active_low_pixels=int(active_low.sum()),
        valid_pixels=int(masks['valid'].sum()), high_pixels=int(masks['region_keep_high'].sum()),
        low_pixels=int(masks['region_keep_low'].sum()), active_cut_pixels=int(active.sum()),
        selected_pairs=sum(len(row['selected_pairs']) for row in images),
        pair_attempts=sum(len(row['attempts']) for row in images),
        reasons=dict(reason_totals), images=images, supervised=bool(supervised),
        no_exact_GT_pixel_BCE=True, no_GAN=True, no_PH=True,
        static_seed_atoms=True, same_GT_connectivity_forced=False,
        selection_stopped_gradient=True, elapsed_seconds=time.perf_counter()-started)
    return loss, audit


def region_update(model, optimizer, batch, settings=None, grad_clip=1.0):
    """仅G一次前向/反向/优化器步；没有有效监督时不走Adam动量。"""
    if (grad_clip is not None and (isinstance(grad_clip, bool) or not isinstance(grad_clip, Real)
                                  or not math.isfinite(grad_clip) or grad_clip <= 0)):
        raise ValueError('grad_clip must be finite and positive, or None')
    inputs = batch['input']
    if (not torch.is_tensor(inputs) or inputs.ndim != 4 or inputs.shape[1] != 1
            or not inputs.is_floating_point() or not bool(torch.isfinite(inputs).all())
            or bool(((inputs < 0) | (inputs > 1)).any())):
        raise ValueError('Region input must be finite floating B1HW probability')
    modes = [(module, module.training) for module in model.modules()]
    optimizer.zero_grad(set_to_none=True)
    try:
        model.train()
        logits = model.forward_logits(inputs.detach())
        if logits.shape != inputs.shape:
            raise ValueError('Generator logits and region input shape differ')
        loss, audit = region_loss(logits, batch, settings)
        result = {**audit, 'grad_norm_g': 0.0, 'nonzero_grad_g': False,
                  'generator_updated': False, 'optimizer_steps': 0}
        if not audit['supervised']:
            return result
        loss.backward()
        active = [parameter for parameter in model.parameters() if parameter.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(active, float('inf') if grad_clip is None else float(grad_clip),
                                            error_if_nonfinite=True) if active else torch.tensor(0.)
        value = float(norm.detach())
        result['grad_norm_g'] = value
        result['nonzero_grad_g'] = value > 0
        if value > 0:
            optimizer.step()
            result.update(generator_updated=True, optimizer_steps=1)
        return result
    finally:
        for module, mode in modes:
            module.training = mode
