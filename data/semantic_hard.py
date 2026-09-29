# -*- coding: utf-8 -*-
"""训练源的平滑光度困难视图，以及独立灰度区域诊断。"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from data.semantic_consistency import _range, _unit, _valid_mask
from models.semantic_lora import semantic_features


def hard_appearance(image, valid_content, config, seed, mode):
    """正斜率仿射的全局/局部版本；不移动像素，不修改监督目标。

    局部场是一个大尺度椭圆高斯，三通道共享，截断超过预算时整体回退。
    """
    if mode not in ('global', 'local') or image.ndim != 4 or image.shape[:2] != (1, 3):
        raise ValueError('Expected one RGB image and global/local mode')
    if not image.is_floating_point() or not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
        raise ValueError('RGB must be finite and within [0,1]')
    valid = _valid_mask(valid_content, image)
    if not bool(valid.any()):
        raise ValueError('Empty content mask')
    contrast = _range(config, 'contrast', [.25, .60], 0, 1, positive=True)
    offset = _range(config, 'offset', [.12, .30], -1, 1)
    budget = _unit(config.get('max_clipped_fraction', .01), 'max_clipped_fraction')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20260927, 83]))
    a, b = float(rng.uniform(*contrast)), float(rng.uniform(*offset))
    height, width = image.shape[-2:]
    if mode == 'local':
        yy, xx = torch.meshgrid(torch.linspace(0, 1, height, device=image.device),
                                torch.linspace(0, 1, width, device=image.device), indexing='ij')
        cx, cy = rng.uniform(.15, .85, size=2)
        sx, sy = rng.uniform(.25, .55, size=2)
        field = torch.exp(-.5 * (((xx - cx) / sx)**2 + ((yy - cy) / sy)**2))[None, None]
        field = field / field[valid].max()
        geometry = dict(center=[float(cx), float(cy)], scale=[float(sx), float(sy)])
    else:
        field = image.new_ones((1, 1, height, width))
        geometry = None
    delta = field * ((a - 1) * image + b)
    def clipped(scale):
        value = image + scale * delta
        return float((((value < 0) | (value > 1)).any(1, keepdim=True) & valid).sum() / valid.sum())
    scale = 1.
    if clipped(scale) > budget:
        low, high = 0., 1.
        for _ in range(16):
            middle = (low + high) / 2
            if clipped(middle) <= budget:
                low = middle
            else:
                high = middle
        scale = low
    effective_a = 1 + scale * field * (a - 1)
    return (image + scale * delta).clamp(0, 1), {
        'mode': mode, 'seed': int(seed), 'contrast': a, 'offset': b,
        'field': geometry, 'scale': scale, 'clipped_fraction': clipped(scale),
        'minimum_slope': float(effective_a[valid].min()),
        'maximum_slope': float(effective_a[valid].max())}


def gray_regions(target, mask, minimum_pixels=32):
    """仅由清晰图先验产生连通内部区域；不是晶粒实例或人工真值。"""
    if target.shape != mask.shape or target.shape[:2] != (1, 1) or mask.dtype != torch.bool:
        raise ValueError('Expected matching B1HW prior and boolean mask')
    if minimum_pixels < 1:
        raise ValueError('minimum_pixels must be positive')
    labels = target.detach().cpu().numpy()[0, 0] > .5
    accepted = mask.detach().cpu().numpy()[0, 0]
    regions, classes = np.zeros(labels.shape, np.int32), []
    for cls in (0, 1):
        count, connected, stats, _ = cv2.connectedComponentsWithStats(
            (accepted & (labels == cls)).astype(np.uint8), connectivity=8)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= minimum_pixels:
                classes.append(cls)
                regions[connected == index] = len(classes)
    return torch.from_numpy(regions).to(target.device)[None, None], classes


def region_probabilities(logits, regions, classes):
    if logits.shape != regions.shape:
        raise ValueError('Region and semantic grid differ')
    probability = logits.float().sigmoid()
    # 每张通常数十个区域；scatter_add在CUDA确定性模式下不受支持，直接逐区平均。
    return torch.stack([probability[regions == i].mean() for i in range(1, len(classes) + 1)]) if classes else probability.reshape(-1)[:0]


def region_difficulty(logits, target, mask, regions, classes, minimum_confidence=.9):
    """按类别分别看最困难的至多4个先验区域；仅选视图，不生成目标或反传。"""
    probability = region_probabilities(logits, regions, classes)
    if not classes:
        if not bool(mask.any()):
            return 0.
        p = logits.float().sigmoid()
        confidence = torch.where(target > .5, p, 1 - p)[mask]
        return float((minimum_confidence - confidence).clamp_min(0).mean())
    labels = torch.tensor(classes, device=logits.device, dtype=torch.bool)
    true_probability = torch.where(labels, probability, 1 - probability)
    deficits = (minimum_confidence - true_probability).clamp_min(0)
    parts = [deficits[labels == cls].topk(min(4, int((labels == cls).sum()))).values.mean()
             for cls in (False, True) if bool((labels == cls).any())]
    return float(torch.stack(parts).mean())


def harder_index(scores):
    """并列时沿用原视图；不为制造变化而强制选择新增强。"""
    if not scores or not all(np.isfinite(s) and s >= 0 for s in scores):
        raise ValueError('Expected finite nonnegative view scores')
    return int(np.argmax(scores))


def gradient_budget_weight(nominal, supervised_norm, prior_norm, maximum_ratio):
    """约束附加梯度范数，保留原GT梯度；不保证二者方向一致。"""
    values = [nominal, supervised_norm, prior_norm, maximum_ratio]
    if not all(np.isfinite(v) and v >= 0 for v in values) or maximum_ratio == 0:
        raise ValueError('Invalid gradient budget')
    if prior_norm == 0:
        return float(nominal)
    return float(min(nominal, maximum_ratio * supervised_norm / prior_norm))


def hard_gray_forward(model, restorer, raw, target, mask, config, seed, replay=None, scale_spec=None):
    """只改变无标签视图：原增强+同一光度类型的两次候选，按区域困难度选一张。"""
    from data.semantic_consistency import apply_acquisition_appearance
    from models.backend_adaptation import restore_first
    hcfg, ucfg = config['semantic_hard'], config['semantic_consistency']
    regions, classes = gray_regions(target, mask, hcfg['minimum_region_pixels'])
    base, appearance = apply_acquisition_appearance(
        raw['strong_image'], raw['valid_content'], ucfg['appearance'], seed + 3000000)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20260927, 97]))
    keep_base = bool(rng.random() < hcfg['keep_base_probability'])
    draw = int(raw['global_draw'])
    family = ('global', 'local')[draw % 2]
    modes = ['base'] + ([] if keep_base else [family] * int(hcfg['variants_per_family']))
    if replay is not None:
        if (replay['family'] != family or replay['keep_base_draw'] != keep_base
                or [v['mode'] for v in replay['views']] != modes
                or not 0 <= replay['selected_index'] < len(modes)):
            raise RuntimeError('Replayed hard-view family/draw differs from gray4')
    training = model.semantic_decoder.training
    candidates, scores, best = [], [], None
    try:
        model.semantic_decoder.eval()
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            for index, mode in enumerate(modes):
                transformed, stats = (base, appearance) if mode == 'base' else hard_appearance(
                    base, raw['valid_content'], hcfg, seed + 7000000 + (index - 1) * 100003, mode)
                restored = restore_first(restorer, transformed, seed + 4000000)
                features = semantic_features(model, restored.float())
                logits = model.semantic_decoder(features, restored)
                score = region_difficulty(logits, target, mask, regions, classes)
                diagnostic = region_diagnostics(logits, target, mask, regions, classes)
                diagnostic.pop('region_scores')
                candidates.append({'mode': mode, 'variant': index, 'difficulty': score,
                                   'appearance': stats, 'diagnostic': diagnostic})
                if replay is not None and stats != replay['views'][index]['appearance']:
                    raise RuntimeError('Replayed hard-view appearance differs from gray4')
                scores.append(score)
                selected_here = replay['selected_index'] == index if replay is not None else harder_index(scores) == index
                if selected_here:
                    best = (score, index, restored, features)
    finally:
        model.semantic_decoder.train(training)
    _, selected, restored, features = best
    local_view = scale_spec is not None and scale_spec['mode'] == 'local'
    if local_view:
        # 先按原全图回放光度视图和D5a，再裁语义视野；不重建局部或重选困难增强。
        from data.semantic_scale import apply_scale_image
        restored = apply_scale_image(restored, scale_spec)
    adapter = getattr(model, 'semantic_lora', None)
    if local_view or adapter is not None and any(p.requires_grad for p in adapter.parameters()):
        with torch.autocast('cuda', enabled=False):
            features = semantic_features(model, restored.float())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model.semantic_decoder(features, restored)
    selection = {
        'family': family, 'keep_base_draw': keep_base, 'selected_index': selected,
        'selected_mode': candidates[selected]['mode'], 'views': candidates,
        'target_source': 'unchanged_clean_raw_prior', 'selection_only_no_target_relabeling': True}
    if replay is not None:
        selection['selection_replayed_from_gray4'] = True
    return logits, restored, appearance, selection


@torch.no_grad()
def hard_monitor(model, restorer, dataset, config, output, epoch):
    """固定训练难例的过程观察；样本仍参与训练，不作为验证或checkpoint选择。"""
    from pathlib import Path
    from torch.utils.data import default_collate
    from data.semantic_consistency import apply_acquisition_appearance
    from data.semantic_gray import build_gray_targets, save_prior_preview
    from models.backend_adaptation import restore_first
    from train_backend_adaptation import write_json
    device = next(model.parameters()).device
    hcfg = config['semantic_hard']
    old_epoch = dataset.epoch
    model.eval()
    results = []
    try:
        with torch.random.fork_rng(devices=[device.index or 0]):
            for item in hcfg['monitor']:
                draw, mode = int(item['draw']), item['mode']
                dataset.set_epoch(draw // len(dataset) + 1)
                raw = default_collate([dataset[draw % len(dataset)]])
                target, mask, _ = build_gray_targets(raw['weak_image'], raw['valid_content'],
                    raw['known_gt_mask'], config['semantic_gray'])
                target, mask = target.to(device), mask.to(device)
                regions, classes = gray_regions(target, mask, hcfg['minimum_region_pixels'])
                seed = config['semantic_adaptation']['seed'] + 10000 + draw
                base, _ = apply_acquisition_appearance(raw['strong_image'].to(device), raw['valid_content'],
                    config['semantic_consistency']['appearance'], seed + 3000000)
                for condition in ['base', mode]:
                    image = base if condition == 'base' else hard_appearance(
                        base, raw['valid_content'], hcfg, seed + 7000000, mode)[0]
                    restored = restore_first(restorer, image, seed + 4000000)
                    logits = model.semantic_decoder(semantic_features(model, restored.float()), restored)
                    path = Path(output) / 'hard_monitor' / f'epoch_{epoch:03d}' / f'draw_{draw:04d}_{condition}.png'
                    save_prior_preview(path, raw['weak_image'], restored, target, mask, logits, raw['valid_content'])
                    results.append({'draw': draw, 'mode': condition,
                        **region_diagnostics(logits, target, mask, regions, classes)})
            write_json(Path(output) / 'hard_monitor' / f'epoch_{epoch:03d}' / 'summary.json', results)
    finally:
        dataset.set_epoch(old_epoch)


@torch.no_grad()
def region_diagnostics(logits, target, mask, regions, classes):
    p = logits.float().sigmoid()
    wrong = ((p > .5) != (target > .5)) & mask
    scores = region_probabilities(logits, regions, classes).cpu().tolist()
    rows = [{'region': i + 1, 'prior_class': int(cls), 'area': int((regions == i + 1).sum()),
             'ferrite_probability': float(score), 'disagrees': int(score > .5) != cls}
            for i, (cls, score) in enumerate(zip(classes, scores))]
    result = dict(accepted=int(mask.sum()), disagree_pixels=int(wrong.sum()),
        ferrite_to_pearlite_pixels=int((wrong & (target > .5)).sum()),
        pearlite_to_ferrite_pixels=int((wrong & (target <= .5)).sum()),
        regions=len(rows), disagree_regions=sum(r['disagrees'] for r in rows),
        ferrite_to_pearlite_regions=sum(r['disagrees'] and r['prior_class'] == 1 for r in rows),
        pearlite_to_ferrite_regions=sum(r['disagrees'] and r['prior_class'] == 0 for r in rows))
    result['maximum_wrong_component'] = 0
    for cls in (0, 1):
        part = (wrong & ((target > .5) == bool(cls)))[0, 0].cpu().numpy().astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(part, connectivity=8)
        if count > 1:
            result['maximum_wrong_component'] = max(result['maximum_wrong_component'], int(stats[1:, cv2.CC_STAT_AREA].max()))
    result['region_scores'] = rows
    return result
