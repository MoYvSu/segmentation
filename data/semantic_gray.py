# -*- coding: utf-8 -*-
"""清晰训练原图的独立明度软监督；不读取网络预测，不生成实例标签。"""
from __future__ import annotations

from pathlib import Path
import math

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from data.direct_dual_head_dataset import _spatial_transform
from data.semantic_consistency import SemanticConsistencyDataset


class GrayPriorDataset(SemanticConsistencyDataset):
    """保持1000源循环抽样；人工已知GT区域不再接受灰度伪监督。"""
    def __init__(self, *args, manual_dataset, **kwargs):
        super().__init__(*args, **kwargs)
        from tools.analyze_semantic_domain import verify_cohorts
        self.manual_mapping = verify_cohorts(self.samples, manual_dataset.data_dir, len(manual_dataset))
        by_name = {path.name: i for i, path in enumerate(manual_dataset.samples)}
        self.manual_dataset = manual_dataset
        self.manual_indices = {row['pool_name']: by_name[row['manual_name']] for row in self.manual_mapping}
        self.known_cache = {}

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        name = sample['name']
        if name in self.manual_indices:
            if name not in self.known_cache:
                manual = self.manual_dataset[self.manual_indices[name]]
                source = self.base_dataset[sample['source_index']]
                if not torch.equal(manual['image'], source['image']):
                    raise RuntimeError('Content-matched manual source preprocessing differs')
                self.known_cache[name] = manual['semantic_valid_content'].bool()
            known = _spatial_transform(self.known_cache[name], sample['horizontal_flip'],
                                       sample['vertical_flip'], sample['rotation_k'])
        else:
            known = torch.zeros_like(sample['valid_content'], dtype=torch.bool)
        sample['known_gt_mask'] = known
        return sample


def _fit_brightness_modes(values, cfg):
    """Otsu仅初始化两组；均值分离和直方图谷共同检查是否有可用明暗证据。"""
    if values.size < int(cfg['min_region_pixels']):
        return None
    quantized = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    threshold, _ = cv2.threshold(quantized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = quantized <= threshold
    fraction = float(low.mean())
    if min(fraction, 1 - fraction) < float(cfg['min_mode_fraction']):
        return None
    dark, bright = values[low], values[~low]
    lo, hi = float(dark.mean()), float(bright.mean())
    delta = hi - lo
    separation = delta / max(float(np.sqrt((dark.var() + bright.var()) / 2)), 1 / 255)
    if delta < float(cfg['min_lightness_gap']) or separation < float(cfg['min_separation']):
        return None
    histogram = np.bincount(quantized, minlength=256).astype(np.float32)
    histogram = cv2.GaussianBlur(histogram[None], (0, 0), 1.5).ravel()
    split = int(threshold)
    peak_low = int(np.argmax(histogram[:split + 1]))
    peak_high = split + 1 + int(np.argmax(histogram[split + 1:]))
    if peak_high - peak_low < 4:
        return None
    valley = float(histogram[peak_low + 1:peak_high].min())
    valley_ratio = valley / max(float(min(histogram[peak_low], histogram[peak_high])), 1e-6)
    if valley_ratio > float(cfg['max_valley_ratio']):
        return None
    return {'dark_mean': lo, 'bright_mean': hi, 'lightness_gap': delta,
            'separation': separation, 'valley_ratio': valley_ratio,
            'dark_fraction': fraction}


def _brightness_probability(lightness, modes, cfg):
    middle = (modes['dark_mean'] + modes['bright_mean']) / 2
    scale = max(float(cfg['transition_fraction']) * modes['lightness_gap'], 1 / 255)
    p = 1 / (1 + np.exp(-np.clip((lightness - middle) / scale, -20, 20)))
    return p.clip(.005, .995).astype(np.float32)


def _starts(length, window, stride):
    if length <= window:
        return [0]
    return sorted(set([*range(0, length - window + 1, stride), length - window]))


def _extend_gray_coverage(lightness, valid, known, old_votes, global_probability,
                          target, accepted, cfg):
    """只填补小窗口零支持处；已有局部冲突、GT和旧标签均不覆盖。"""
    ccfg = cfg['coverage']
    window, stride = int(ccfg['window']), int(ccfg['stride'])
    local_sum, votes = np.zeros_like(target), np.zeros_like(target)
    local_min, local_max = np.ones_like(target), np.zeros_like(target)
    info = dict(version='larger_window_v1', base_accepted_pixels=int(accepted.sum()),
                unsupported_pixels=int((valid & ~known & (old_votes == 0)).sum()),
                large_windows=0, accepted_large_windows=0)
    for y in _starts(target.shape[0], window, stride):
        for x in _starts(target.shape[1], window, stride):
            region = np.s_[y:y + window, x:x + window]
            local_valid = valid[region]
            info['large_windows'] += 1
            modes = _fit_brightness_modes(lightness[region][local_valid], cfg)
            if modes is None:
                continue
            info['accepted_large_windows'] += 1
            p = _brightness_probability(lightness[region], modes, cfg)
            local_sum[region] += p * local_valid
            votes[region] += local_valid
            local_min[region] = np.where(local_valid, np.minimum(local_min[region], p), local_min[region])
            local_max[region] = np.where(local_valid, np.maximum(local_max[region], p), local_max[region])
    proposed = (global_probability + local_sum / np.maximum(votes, 1)) / 2
    confidence = float(cfg['confidence'])
    agreement = ((global_probability >= confidence) & (local_min >= confidence)) | (
        (global_probability <= 1 - confidence) & (local_max <= 1 - confidence))
    candidate = valid & (old_votes == 0) & (votes > 0) & agreement & (
        np.maximum(global_probability, local_max) - np.minimum(global_probability, local_min)
        <= float(cfg['max_probability_gap']))
    added = np.zeros_like(accepted)
    radius = int(cfg['erosion_radius'])
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    for label in (False, True):
        region = candidate & ((proposed > .5) == label)
        if radius:
            region = cv2.erode(region.astype(np.uint8), kernel,
                               borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        added |= region
    added &= ~known & ~accepted
    target[added] = proposed[added]
    accepted |= added
    info.update(added_candidate_pixels=int((candidate & ~known).sum()), added_pixels=int(added.sum()),
                added_ferrite=int((added & (proposed > .5)).sum()),
                added_pearlite=int((added & (proposed <= .5)).sum()))
    return info


@torch.no_grad()
def build_gray_targets(clean_image, valid_content, known_gt_mask, cfg, grid=256):
    """原图Lab L*→全图/重叠局部共识→内部软目标。目标独立于学生及强退化。

    以CPU/OpenCV计算，返回同设备的B1gridgrid；仅支持现有batch=1实验。
    known_gt_mask只用于排除已有GT覆盖，不从GT选择阈值或校准先验。
    """
    if clean_image.ndim != 4 or clean_image.shape[:2] != (1, 3):
        raise ValueError('Expected one clean RGB image')
    if not bool(torch.isfinite(clean_image).all()) or bool(((clean_image < 0) | (clean_image > 1)).any()):
        raise ValueError('Clean RGB must be finite and within [0,1]')
    if valid_content.shape != known_gt_mask.shape or valid_content.shape != (1, 1, *clean_image.shape[-2:]):
        raise ValueError('Full-resolution content and known-GT masks must match RGB')
    if valid_content.dtype != torch.bool or known_gt_mask.dtype != torch.bool:
        raise ValueError('Content and known-GT masks must be boolean')
    if not .5 < float(cfg['confidence']) < 1 or int(cfg['erosion_radius']) < 0:
        raise ValueError('Invalid confidence or erosion radius')
    coverage = cfg.get('coverage', {}).get('enabled', False)
    if coverage:
        ccfg = cfg['coverage']
        if (ccfg.get('version') != 'larger_window_v1'
                or not int(cfg['local_window']) < int(ccfg['window']) < grid
                or not 1 <= int(ccfg['stride']) <= int(ccfg['window'])):
            raise ValueError('Coverage requires a larger local window smaller than the global grid')
    device = clean_image.device
    rgb = clean_image[0].detach().float().cpu().permute(1, 2, 0).contiguous().numpy()
    lightness = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0] / 100
    lightness = cv2.resize(lightness, (grid, grid), interpolation=cv2.INTER_AREA)
    valid = F.interpolate(valid_content.float().cpu(), (grid, grid), mode='area')[0, 0].numpy() >= 1
    known = F.interpolate(known_gt_mask.float().cpu(), (grid, grid), mode='area')[0, 0].numpy() > 0
    target = np.full((grid, grid), .5, dtype=np.float32)
    accepted = np.zeros((grid, grid), dtype=bool)
    counts = {'valid_pixels': int(valid.sum()), 'known_gt_excluded_pixels': int((valid & known).sum()),
              'candidate_pixels': 0, 'local_regions': 0, 'accepted_local_regions': 0,
              'target_source': 'clean_raw_lab_lightness', 'network_gating': False}
    global_modes = _fit_brightness_modes(lightness[valid], cfg)
    counts['global_modes'] = global_modes
    if global_modes is not None:
        global_probability = _brightness_probability(lightness, global_modes, cfg)
        local_sum = np.zeros_like(target)
        votes = np.zeros_like(target)
        local_min = np.ones_like(target)
        local_max = np.zeros_like(target)
        window, stride = int(cfg['local_window']), int(cfg['local_stride'])
        if window < 4 or stride < 1 or stride > window:
            raise ValueError('Invalid local window/stride')
        for y in _starts(grid, window, stride):
            for x in _starts(grid, window, stride):
                region = np.s_[y:y + window, x:x + window]
                local_valid = valid[region]
                counts['local_regions'] += 1
                modes = _fit_brightness_modes(lightness[region][local_valid], cfg)
                if modes is None:
                    continue
                counts['accepted_local_regions'] += 1
                p = _brightness_probability(lightness[region], modes, cfg)
                local_sum[region] += p * local_valid
                votes[region] += local_valid
                local_min[region] = np.where(local_valid, np.minimum(local_min[region], p), local_min[region])
                local_max[region] = np.where(local_valid, np.maximum(local_max[region], p), local_max[region])
        local_probability = local_sum / np.maximum(votes, 1)
        target = ((global_probability + local_probability) / 2).astype(np.float32)
        confidence = float(cfg['confidence'])
        agreement = ((global_probability >= confidence) & (local_min >= confidence)) | (
            (global_probability <= 1 - confidence) & (local_max <= 1 - confidence))
        candidate = valid & (votes > 0) & agreement & (
            np.maximum(global_probability, local_max) - np.minimum(global_probability, local_min)
            <= float(cfg['max_probability_gap']))
        counts['candidate_pixels'] = int(candidate.sum())
        radius = int(cfg['erosion_radius'])
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
        for label in (False, True):
            region = candidate & ((target > .5) == label)
            if radius:
                region = cv2.erode(region.astype(np.uint8), kernel,
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            accepted |= region
        accepted &= ~known
        if coverage:
            counts['coverage'] = _extend_gray_coverage(lightness, valid, known, votes,
                global_probability, target, accepted, cfg)
    if coverage and 'coverage' not in counts:
        counts['coverage'] = dict(version='larger_window_v1', base_accepted_pixels=0,
            unsupported_pixels=0, large_windows=0, accepted_large_windows=0,
            added_candidate_pixels=0, added_pixels=0, added_ferrite=0, added_pearlite=0)
    # 无局部支持区域不带有伪造目标；其mask始终为False。
    target[~accepted] = .5
    counts.update(accepted_pixels=int(accepted.sum()),
                  accepted_fraction=float(accepted.sum() / max(1, valid.sum())),
                  accepted_ferrite=int((accepted & (target > .5)).sum()),
                  accepted_pearlite=int((accepted & (target <= .5)).sum()))
    return (torch.from_numpy(target)[None, None].to(device),
            torch.from_numpy(accepted)[None, None].to(device), counts)


def gray_prior_loss(logits, target, mask, *, mode='soft_bce', minimum_confidence=.9,
                    maximum_normalization_gain=1., return_info=False):
    """两类独立计量；单向约束可限制幅度地按活动位置平均。

    one_sided使用截断的硬类别BCE，减去置信门槛处的常数以保持损失连续。
    分母为max(活动数, 接受数/最大增益)，没有活动位置的类别仍保留零项。
    默认最大增益1，沿用原mean操作，保持历史目标、数值与梯度不变。
    """
    if logits.shape != target.shape or logits.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError('Gray supervision shapes/mask differ')
    if not all(bool(torch.isfinite(t).all()) for t in (logits, target)):
        raise ValueError('Non-finite gray supervision')
    if bool(((target < 0) | (target > 1)).any()):
        raise ValueError('Gray target outside [0,1]')
    if mode not in ('soft_bce', 'one_sided'):
        raise ValueError(f'Unknown gray prior loss: {mode}')
    if not .5 < float(minimum_confidence) < 1:
        raise ValueError('Gray minimum confidence must be between 0.5 and 1')
    gain = float(maximum_normalization_gain)
    if not math.isfinite(gain) or gain < 1 or (gain != 1 and mode != 'one_sided'):
        raise ValueError('Normalization gain must be finite and >=1; amplification requires one_sided')
    labels = target.detach() > .5
    if mode == 'one_sided':
        signed_logits = logits.float() * torch.where(labels, 1., -1.)
        margin = signed_logits.new_tensor(math.log(minimum_confidence / (1 - minimum_confidence)))
        pixel_loss = F.relu(F.softplus(-signed_logits) - F.softplus(-margin))
        active = mask & (signed_logits.detach() < margin)
    else:
        pixel_loss = F.binary_cross_entropy_with_logits(logits.float(), target.detach().float(), reduction='none')
        active = mask
    terms, normalization = [], {}
    for label in (False, True):
        selected = mask & (labels == label)
        accepted_count = int(selected.sum())
        active_count = int((selected & active).sum())
        denominator = max(active_count, accepted_count / gain) if accepted_count else 0.
        if accepted_count:
            terms.append(pixel_loss[selected].mean() if gain == 1 else pixel_loss[selected].sum() / denominator)
        normalization['ferrite' if label else 'pearlite'] = {
            'accepted': accepted_count, 'active': active_count, 'denominator': denominator,
            'gain': accepted_count / denominator if denominator else 0.}
    loss = torch.stack(terms).mean() if terms else logits.float().sum() * 0
    if not return_info:
        return loss
    accepted, count = int(mask.sum()), int(active.sum())
    return loss, {
        'loss_mode': mode,
        'minimum_confidence': float(minimum_confidence) if mode == 'one_sided' else None,
        'maximum_normalization_gain': gain,
        'class_normalization': normalization,
        'active_pixels': count, 'satisfied_pixels': accepted - count,
        'active_fraction': count / max(1, accepted),
        'active_ferrite': int((active & labels).sum()),
        'active_pearlite': int((active & ~labels).sum()),
        'confidence_only_active_pixels': int((active & ((logits.detach() > 0) == labels)).sum()),
    }


def save_prior_preview(path, clean, strong, target, mask, logits, valid):
    """训练源诊断缩略图：可看出先验接受位置及与当前学生的冲突。"""
    valid_array = valid[0, 0].detach().cpu().numpy().astype(bool)
    yy, xx = np.nonzero(valid_array)
    crop = np.s_[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    p = logits.detach().float().sigmoid()[0, 0].cpu().numpy()
    q, accepted = target[0, 0].cpu().numpy(), mask[0, 0].cpu().numpy()
    selection = np.full((*accepted.shape, 3), 65, np.uint8)
    selection[accepted] = (235, 235, 235)
    selection[accepted & ((p > .5) != (q > .5))] = (225, 45, 65)
    def rgb(t):
        return np.rint(t[0].detach().float().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)[crop]
    def heat(value):
        return cv2.cvtColor(cv2.applyColorMap(np.rint(value * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    panels = [('Clean source', rgb(clean)), ('Strong -> D5a', rgb(strong))]
    for title, body in [('Gray prior P(F)', heat(q)), ('Current student P(F)', heat(p)),
                        ('Accepted; red=conflict', selection)]:
        body = cv2.resize(body, (valid_array.shape[1], valid_array.shape[0]), interpolation=cv2.INTER_NEAREST)[crop]
        panels.append((title, body))
    frames = []
    for title, body in panels:
        thumb = cv2.resize(body, (224, max(1, round(body.shape[0] * 224 / body.shape[1]))), interpolation=cv2.INTER_AREA)
        thumb = cv2.copyMakeBorder(thumb, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(thumb, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .37, (0, 0, 0), 1, cv2.LINE_AA)
        frames.append(thumb)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)):
        raise IOError(f'Unable to save gray-prior preview: {path.name}')


def save_coverage_preview(path, clean, strong, target, base_mask, mask, logits, valid):
    """固定色标展示原接受域和新增域，明确将所有忽略位置标灰。"""
    valid_array = valid[0, 0].detach().cpu().numpy().astype(bool)
    yy, xx = np.nonzero(valid_array)
    crop = np.s_[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    q = target[0, 0].detach().cpu().numpy()
    old = base_mask[0, 0].detach().cpu().numpy().astype(bool)
    accepted = mask[0, 0].detach().cpu().numpy().astype(bool)
    added = accepted & ~old
    def rgb(t):
        return np.rint(t[0].detach().float().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)[crop]
    def marked(selection):
        result = np.full((*q.shape, 3), 75, np.uint8)
        result[selection & (q > .5)] = (245, 185, 45)
        result[selection & (q <= .5)] = (60, 145, 230)
        return result
    panels = [('Clean source', rgb(clean)),
              ('Input -> D5a' if logits is not None else 'Clean (rule check)', rgb(strong))]
    small = [('Old prior: F gold / P blue', marked(old)), ('Added prior; gray=ignore', marked(added))]
    if logits is not None:
        p = logits[0, 0].detach().float().sigmoid().cpu().numpy()
        heat = cv2.cvtColor(cv2.applyColorMap(np.rint(p * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        conflict = marked(added)
        conflict[added & ((p > .5) != (q > .5))] = (235, 30, 60)
        small += [('Student P(F): fixed 0..1', heat), ('Added; red=disagreement', conflict)]
    for title, body in small:
        panels.append((title, cv2.resize(body, (valid_array.shape[1], valid_array.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)[crop]))
    frames = []
    for title, body in panels:
        thumb = cv2.resize(body, (224, max(1, round(body.shape[0] * 224 / body.shape[1]))), interpolation=cv2.INTER_AREA)
        thumb = cv2.copyMakeBorder(thumb, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(thumb, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .34, (0, 0, 0), 1, cv2.LINE_AA)
        frames.append(thumb)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)):
        raise IOError(f'Unable to save coverage preview: {path.name}')
