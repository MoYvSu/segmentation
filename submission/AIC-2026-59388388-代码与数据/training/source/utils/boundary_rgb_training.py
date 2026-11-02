# -*- coding: utf-8 -*-
"""RGB条件NAF的容差边界监督与独立GAN更新。

不要求输出命中GT的同一像素：GT每个可信线点沿局部法向的短截面出现响应即可。
截面横跨边界而不是沿边界走，限制二维邻域峰值的宽松匹配；默认逐点目标仍可
被法向错位的亮点串满足。coherent3额外约束三个真实GT邻点共享法向偏移。
GT附近的歧义带不作内部负例，远离GT的可信内部（包括旧输入线）允许删除；
未知标签不参与方向、正负损失或判别器可靠感受野。
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch.nn import functional as F


def _nonnegative(value, name):
    if (not isinstance(value, Real) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        raise ValueError(name + " must be finite and nonnegative")
    return float(value)


def _validate_target(target, valid):
    if (not torch.is_tensor(target) or target.ndim != 4 or target.shape[1] != 1
            or min(target.shape) < 1 or not target.is_floating_point()
            or not torch.is_tensor(valid) or valid.shape != target.shape
            or valid.device != target.device):
        raise ValueError("Target/valid must be aligned nonempty floating B1HW and binary B1HW")
    if valid.is_floating_point() and not bool(torch.isfinite(valid).all()):
        raise ValueError("Valid must be finite")
    if bool(((valid != 0) & (valid != 1)).any()):
        raise ValueError("Valid must be binary")
    valid = valid.bool()
    known = target[valid]
    if not bool(torch.isfinite(known).all()) or bool(((known != 0) & (known != 1)).any()):
        raise ValueError("Known boundary GT must be binary")
    safe = torch.where(valid, target.detach(), torch.zeros_like(target))
    return safe, valid


def _erode_valid(valid, radius):
    # 显式把图外当不可靠，max_pool的隐式padding不能冒充有效区域。
    size = 2 * radius + 1
    padded = F.pad(valid.float(), (radius,) * 4, value=0)
    return F.avg_pool2d(padded, size, stride=1) == 1


@torch.no_grad()
def build_tolerant_targets(target, valid, *, tolerance=2, orientation_radius=3):
    """生成固定GT集合；方向由局部GT坐标协方差得到，无模型依赖的筛选。

    normal_index 0/1/2/3分别是水平、右下、垂直、右上法向；方向在45度网格量化。
    容差统一采用L-infinity距离（横纵坐标各最多偏移tolerance像素），因此斜向最远
    欧氏距离为sqrt(2)*tolerance；负集合使用同一方形带。各向同性/点状GT方向不可靠时，
    显式回退到同半径二维邻域覆盖并记录数量，不悄悄丢弃该GT正点。
    所有可信GT线点进入positive，可靠性只排除图外或unknown邻域。
    """
    if type(tolerance) is not int or tolerance < 0:
        raise ValueError("tolerance must be a nonnegative integer")
    if type(orientation_radius) is not int or orientation_radius < 1:
        raise ValueError("orientation_radius must be a positive integer")
    target, valid = _validate_target(target, valid)
    radius = max(tolerance, orientation_radius)
    trusted = _erode_valid(valid, radius)
    kernel_size = 2 * orientation_radius + 1
    axis = torch.arange(-orientation_radius, orientation_radius + 1,
                        device=target.device, dtype=target.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernels = torch.stack((torch.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy))[:, None]
    moments = F.conv2d(target, kernels, padding=orientation_radius)
    count = moments[:, 0:1].clamp_min(1)
    mean_x, mean_y = moments[:, 1:2] / count, moments[:, 2:3] / count
    covariance_xx = (moments[:, 3:4] / count - mean_x.square()).clamp_min(0)
    covariance_yy = (moments[:, 4:5] / count - mean_y.square()).clamp_min(0)
    covariance_xy = moments[:, 5:6] / count - mean_x * mean_y
    tangent_angle = .5 * torch.atan2(2 * covariance_xy, covariance_xx - covariance_yy)
    normal_angle = tangent_angle + math.pi / 2
    # floor(x+.5)避免round的偶数舍入把同类45度中点产生不必要的奇偶差异。
    normal_index = torch.floor(normal_angle / (math.pi / 4) + .5).long().remainder(4)
    positive = (target > .5) & trusted
    eigen_gap = ((covariance_xx - covariance_yy).square() + 4 * covariance_xy.square()).sqrt()
    anisotropy = eigen_gap / (covariance_xx + covariance_yy).clamp_min(1e-8)
    fallback = positive & ((anisotropy < .25) | (count < 3))
    expanded = F.max_pool2d(target, 2 * tolerance + 1, stride=1, padding=tolerance)
    negative = (expanded == 0) & trusted
    return {"positive": positive, "negative": negative, "trusted": trusted,
            "normal_index": normal_index, "fallback": fallback,
            "safe_target": target, "valid": valid,
            "tolerance": tolerance, "orientation_radius": orientation_radius}


def _shift(value, dy, dx, radius):
    padded = F.pad(value, (radius,) * 4, value=float("-inf"))
    height, width = value.shape[-2:]
    return padded[..., radius + dy:radius + dy + height,
                  radius + dx:radius + dx + width]


def directional_coverage_logits(logits, normal_index, valid, tolerance, fallback=None):
    """仅在GT法向短截面找最大logit，unknown输出不能替邻近GT完成覆盖。"""
    safe = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
    directions = ((0, 1), (1, 1), (1, 0), (-1, 1))
    values = []
    for dy, dx in directions:
        offsets = [_shift(safe, dy * delta, dx * delta, tolerance)
                   for delta in range(-tolerance, tolerance + 1)]
        values.append(torch.stack(offsets).amax(0))
    stacked = torch.cat(values, dim=1)
    result = torch.gather(stacked, 1, normal_index)
    if fallback is not None:
        isotropic = F.max_pool2d(safe, 2 * tolerance + 1, stride=1, padding=tolerance)
        result = torch.where(fallback, isotropic, result)
    return result


def _coverage_mode(value):
    if value not in ('pointwise', 'coherent3'):
        raise ValueError('coverage_mode must be pointwise or coherent3')
    return value


def coherent3_coverage(logits, prepared):
    """相邻三个真实GT点共用一个法向偏移；不能用单个强点抵消另外的空点。

    仅在三点均可信、非fallback且量化法向一致时替换逐点覆盖。其余GT仍由原
    pointwise目标负责。先逐点softplus，再对三点平均，最后在共同偏移中取最小值。
    common_probability是同一偏移下三点的最弱响应，再选最佳偏移，只作诊断。
    """
    positive = prepared['positive']
    normal = prepared['normal_index']
    supported = positive & ~prepared['fallback']
    eligible = torch.zeros_like(positive)
    losses = torch.zeros_like(logits)
    probabilities = torch.zeros_like(logits)
    radius = prepared['tolerance']
    height, width = logits.shape[-2:]
    # (法向dy,dx, 切向dy,dx)，切向正负仅交换邻点顺序，不影响目标。
    directions = ((0, 1, 1, 0), (1, 1, 1, -1),
                  (1, 0, 0, 1), (-1, 1, 1, 1))
    for index, (ny, nx, ty, tx) in enumerate(directions):
        candidates = supported & (normal == index)
        batch, channel, y, x = candidates.nonzero(as_tuple=True)
        if not batch.numel():
            continue
        # 不使用roll环绕，也不根据缺失GT推断邻点。每个邻点自身都必须在trusted内。
        inside = ((y + ty >= 0) & (y + ty < height) & (y - ty >= 0) & (y - ty < height)
                  & (x + tx >= 0) & (x + tx < width) & (x - tx >= 0) & (x - tx < width))
        batch, channel, y, x = (v[inside] for v in (batch, channel, y, x))
        agreed = (candidates[batch, channel, y + ty, x + tx]
                  & candidates[batch, channel, y - ty, x - tx])
        batch, channel, y, x = (v[agreed] for v in (batch, channel, y, x))
        if not batch.numel():
            continue
        eligible[batch, channel, y, x] = True
        # trusted已按max(orientation_radius,tolerance)腐蚀，且三点各自可信；
        # 因而下面所有采样位置均有效，不读取unknown像素。
        samples = torch.stack([torch.stack([
            logits[batch, channel, y + step * ty + delta * ny, x + step * tx + delta * nx]
            for step in (-1, 0, 1)], dim=0)
            for delta in range(-radius, radius + 1)], dim=0)
        losses[batch, channel, y, x] = F.softplus(-samples).mean(dim=1).amin(dim=0)
        probabilities[batch, channel, y, x] = samples.detach().sigmoid().amin(dim=1).amax(dim=0)
    return dict(eligible=eligible, losses=losses, probabilities=probabilities)


def tolerant_boundary_loss(logits, target, valid, *, tolerance=2, orientation_radius=3,
                           coverage_weight=1., interior_weight=1., prepared=None,
                           denominators=None, coverage_mode='pointwise'):
    """返回容差集合损失和计数；可传全有效batch分母以保持microbatch等价。

    正负分开归一，不因背景数量大而冲淡每条GT线的补线要求。
    这是位置容差的截面监督，不是精确GT像素BCE，也不保证最终分水岭一定闭合。
    """
    _nonnegative(coverage_weight, "coverage_weight")
    _nonnegative(interior_weight, "interior_weight")
    _coverage_mode(coverage_mode)
    if (not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[1] != 1
            or not logits.is_floating_point() or logits.shape != target.shape
            or logits.device != target.device or not bool(torch.isfinite(logits).all())):
        raise ValueError("Logits must be finite floating B1HW aligned with GT")
    if prepared is None:
        prepared = build_tolerant_targets(target, valid, tolerance=tolerance,
                                         orientation_radius=orientation_radius)
    positive, negative = prepared["positive"], prepared["negative"]
    local_positive, local_negative = int(positive.sum()), int(negative.sum())
    if denominators is None:
        total_positive, total_negative = local_positive, local_negative
    else:
        total_positive, total_negative = denominators
        if total_positive < local_positive or total_negative < local_negative:
            raise ValueError("Global denominators cannot be smaller than micro counts")
    zero = logits.sum() * 0
    coverage = zero
    covered = 0
    coverage_probability_sum = 0.
    coherent = None
    if local_positive:
        peaks = directional_coverage_logits(logits, prepared["normal_index"],
            prepared["valid"], prepared["tolerance"], prepared["fallback"])
        coverage = F.softplus(-peaks[positive]).sum() / total_positive
        probabilities = peaks[positive].detach().sigmoid()
        covered = int((probabilities > .65).sum())
        coverage_probability_sum = float(probabilities.sum())
        if coverage_mode == 'coherent3':
            pointwise_coverage = coverage
            coherent = coherent3_coverage(logits, prepared)
            eligible = coherent['eligible']
            if bool(eligible.any()):
                # 分母仍是全部可信GT正点数；不丢掉不能形成三点的GT。
                coverage = (F.softplus(-peaks[positive & ~eligible]).sum()
                            + coherent['losses'][eligible].sum()) / total_positive
    interior = (F.softplus(logits[negative]).sum() / total_negative) if local_negative else zero
    loss = coverage_weight * coverage + interior_weight * interior
    stats = {"coverage": coverage, "interior": interior,
                  "positive_points": local_positive, "negative_pixels": local_negative,
                  "trusted_pixels": int(prepared["trusted"].sum()),
                  "valid_pixels": int(prepared["valid"].sum()),
                  "invalid_pixels": int((~prepared["trusted"]).sum()),
                  "invalid_gt_points": int(((prepared["safe_target"] > .5)
                                             & ~prepared["trusted"]).sum()),
                  "fallback_gt_points": int(prepared["fallback"].sum()),
                  "covered_gt_points": covered,
                  "coverage_probability_sum": coverage_probability_sum,
                  "exact_pixel_BCE": False, "sparse_cut_selection": False}
    if coverage_mode == 'coherent3':
        eligible_count = int(coherent['eligible'].sum()) if coherent is not None else 0
        common_probabilities = (coherent['probabilities'][coherent['eligible']]
                                if eligible_count else logits.detach().new_empty(0))
        common_covered = int((common_probabilities > .65).sum())
        stats.update(coverage_mode=coverage_mode,
            pointwise_coverage=pointwise_coverage if local_positive else zero,
            pointwise_covered_gt_points=covered,
            pointwise_coverage_probability_sum=coverage_probability_sum,
            coherent3_eligible_gt_points=eligible_count,
            coherent3_eligible_gt_fraction=eligible_count / max(1, local_positive),
            coherent3_covered_gt_points=common_covered,
            coherent3_covered_gt_fraction=common_covered / max(1, eligible_count),
            coherent3_coverage_probability_sum=float(common_probabilities.sum()))
    return loss, stats


def _gradient_norm(parameters, threshold):
    active = [value for value in parameters if value.grad is not None]
    if not active:
        return 0.
    norm = torch.nn.utils.clip_grad_norm_(active, float("inf") if threshold is None else threshold,
                                         error_if_nonfinite=True)
    return float(norm.detach())


def rgb_gan_update(generator, discriminator, g_optimizer, d_optimizer, batch, *,
                   micro_batch_size=16, coverage_weight=1., interior_weight=1.,
                   adversarial_weight=.01, tolerance=2, orientation_radius=3, grad_clip=1.,
                   coverage_mode='pointwise', shift_consistency=None, consistency_step=None,
                   ridge_supervision=None, boundary_input_channel=None):
    """容差集合恢复＋低权PatchGAN；只更新独立G/D，支持RGBT或T-only。

    谱归一化每个D更新仅迭代一次；G更新不改变D参数/缓冲。
    G/D模式与requires_grad在正常或异常退出时恢复。全unknown不动优化器或缓冲。
    """
    if type(micro_batch_size) is not int or micro_batch_size < 1:
        raise ValueError("micro_batch_size must be a positive integer")
    shift_cfg = dict(enabled=False)
    if shift_consistency is not None:
        from utils.boundary_shift_consistency import (
            build_shift_masks, shift_consistency_loss, shift_consistency_settings,
            shift_for_step, shift_integer_reflect,
        )
        shift_cfg = shift_consistency_settings(shift_consistency)
    shifted_mode = shift_cfg['enabled']
    shift = shift_for_step(shift_cfg, consistency_step) if shifted_mode else None
    # 缺省/关闭分支不导入新目标，也不增加零项、回执字段或随机数消耗。
    if ridge_supervision is not None and (not isinstance(ridge_supervision, dict)
            or type(ridge_supervision.get('enabled', False)) is not bool):
        raise ValueError('ridge_supervision必须是显式配置对象，enabled必须为布尔值')
    ridge_mode = isinstance(ridge_supervision, dict) and ridge_supervision.get('enabled') is True
    ridge_cfg, ridge_targets, ridge_denominator = None, None, 0
    if ridge_mode:
        from utils.boundary_ridge_loss import (
            build_ridge_targets, ridge_supervision_loss, ridge_supervision_settings,
        )
        ridge_cfg = ridge_supervision_settings(ridge_supervision)
        if shifted_mode:
            raise ValueError('本轮法向单连通监督不能同时启用shift_consistency')
        if coverage_mode != 'pointwise':
            raise ValueError('法向单连通监督仅追加于原pointwise覆盖目标')
    _coverage_mode(coverage_mode)
    for name, value in (("coverage_weight", coverage_weight), ("interior_weight", interior_weight),
                        ("adversarial_weight", adversarial_weight)):
        _nonnegative(value, name)
    if grad_clip is not None and (not isinstance(grad_clip, Real)
                                 or not math.isfinite(grad_clip) or grad_clip <= 0):
        raise ValueError("grad_clip must be positive finite or None")
    input_map, target, valid = batch["input"], batch["target"], batch["valid"]
    # 旧接口默认仍只接受T/RGBT。双输入实验须显式声明第0路为二值T8，
    # 不把连续B误作missing/support诊断的二值依据；三项训练目标保持原样。
    dual_input = type(boundary_input_channel) is int and boundary_input_channel == 0
    if boundary_input_channel is not None and not dual_input:
        raise ValueError('Explicit dual input requires boundary_input_channel=0')
    allowed_channels = (2,) if dual_input else (1, 4)
    if (not torch.is_tensor(input_map) or input_map.ndim != 4 or input_map.shape[1] not in allowed_channels
            or min(input_map.shape) < 1 or not input_map.is_floating_point()
            or input_map.shape[0] != target.shape[0] or input_map.shape[-2:] != target.shape[-2:]
            or input_map.device != target.device or not bool(torch.isfinite(input_map).all())
            or bool(((input_map < 0) | (input_map > 1)).any())):
        raise ValueError("Input must be finite [0,1], with explicit legacy or dual channel contract")
    if dual_input and not bool(((input_map[:, :1] == 0) | (input_map[:, :1] == 1)).all()):
        raise ValueError('Dual channel0 must be binary T8')
    prepared = build_tolerant_targets(target, valid, tolerance=tolerance,
                                     orientation_radius=orientation_radius)
    if ridge_mode:
        # 输入已经统一校验为B1/B4；该目标只依赖同坐标logits/GT/实例，RGB不改变其定义。
        if 'ridge_instances' not in batch:
            raise ValueError('法向单连通监督缺少同坐标的ridge_instances')
        ridge_targets = build_ridge_targets(prepared, batch['ridge_instances'], ridge_cfg)
        ridge_points = int(ridge_targets['eligible'].sum())
        ridge_denominator = ridge_points * (2 * tolerance + 1)
    shift_masks, shift_denominators, shift_active = None, None, False
    if shifted_mode:
        if input_map.shape[1] != 1:
            raise ValueError('Shift-consistency experiment requires single-channel input')
        # Sequential recomputation is gradient-equivalent only for a stateless G.
        # The deployed NAF G has no BN/dropout/spectral buffers; D is not translated.
        from torch import nn
        if any(isinstance(module, (nn.modules.batchnorm._BatchNorm,
                                   nn.modules.dropout._DropoutNd))
               or hasattr(module, 'weight_u') for module in generator.modules()):
            raise ValueError('Shift recomputation requires stateless deterministic generator')
        shift_masks = build_shift_masks(prepared, shift_cfg, *shift)
        shift_denominators = {key: int(shift_masks[key].sum()) for key in ['band', 'interior']}
        shift_active = any(shift_denominators.values())
    count = target.shape[0]
    denominators = int(prepared["positive"].sum()), int(prepared["negative"].sum())
    boundary_input = input_map[:, :1] if dual_input else input_map[:, -1:]
    near_input = F.max_pool2d(boundary_input.detach(), 2 * tolerance + 1,
                             stride=1, padding=tolerance) >= .5
    missing_gt = prepared["positive"] & ~near_input
    if adversarial_weight > 0 and min(target.shape[-2:]) < discriminator.minimum_input_size:
        raise ValueError("Adversarial crop is too small")
    if min(target.shape[-2:]) >= discriminator.minimum_input_size:
        patch_mask = discriminator.valid_patch_mask(prepared["valid"])
    else:
        patch_mask = torch.zeros((count, 1, 1, 1), device=target.device, dtype=torch.bool)
    patch_count = int(patch_mask.sum())
    adversarial_active = bool(adversarial_weight > 0 and patch_count)
    stats = {"loss_g": 0., "loss_d": 0., "coverage": 0., "interior": 0., "adv_g": 0.,
             "adv_d": 0., "grad_norm_g": 0., "grad_norm_d": 0., "samples": count,
             "valid_pixels": int(prepared["valid"].sum()),
             "trusted_pixels": int(prepared["trusted"].sum()),
             "positive_points": denominators[0], "negative_pixels": denominators[1],
             "invalid_pixels": int((~prepared["trusted"]).sum()),
             "invalid_gt_points": int(((prepared["safe_target"] > .5)
                                         & ~prepared["trusted"]).sum()),
             "fallback_gt_points": int(prepared["fallback"].sum()),
             "coverage_probability_mean": 0., "covered_gt_points": 0,
             "covered_gt_fraction": 0., "missing_gt_points": int(missing_gt.sum()),
             "missing_covered_gt_points": 0, "missing_covered_gt_fraction": 0.,
             "adversarial_patches": patch_count, "adversarial_active": adversarial_active,
             "spectral_power_iterations": 0, "nonzero_grad_g": False, "nonzero_grad_d": False,
             "generator_updated": False, "discriminator_updated": False,
             "exact_pixel_BCE": False, "sparse_cut_selection": False,
             "tolerance": tolerance, "distance_contract": "L_infinity",
             "orientation_radius": orientation_radius}
    if coverage_mode == 'coherent3':
        stats.update(coverage_mode=coverage_mode, pointwise_coverage=0.,
            coherent3_eligible_gt_points=0, coherent3_eligible_gt_fraction=0.,
            coherent3_covered_gt_points=0, coherent3_covered_gt_fraction=0.,
            coherent3_coverage_probability_mean=0., coherent3_missing_covered_gt_points=0,
            coherent3_missing_covered_gt_fraction=0.)
    if shifted_mode:
        stats.update(shift_consistency_enabled=True, shift_consistency_active=shift_active,
            shift_consistency_loss=0., shift_consistency_weighted=0.,
            shift_band_loss=0., shift_interior_loss=0.,
            shift_band_pixels=shift_denominators['band'],
            shift_interior_pixels=shift_denominators['interior'],
            shift_dy=shift[0], shift_dx=shift[1], consistency_step=consistency_step,
            shift_recompute_max_error=0., shift_second_branch_probe_grad_norm=0.,
            shift_value_counted_once=True, shift_symmetric_recomputed=True)
    if ridge_mode:
        stats.update(ridge_supervision_enabled=True, ridge_loss=0., ridge_weighted=0.,
            ridge_eligible_points=int(ridge_targets['eligible'].sum()),
            ridge_positions=ridge_denominator, ridge_gap_points=0, ridge_gap_profiles=0,
            ridge_multitrack_points=0, ridge_gap_fraction=0., ridge_multitrack_fraction=0.)
    generator_mode, discriminator_mode = generator.training, discriminator.training
    flags = [parameter.requires_grad for parameter in discriminator.parameters()]
    g_optimizer.zero_grad(set_to_none=True)
    d_optimizer.zero_grad(set_to_none=True)
    try:
        if adversarial_active:
            discriminator.train()
            with torch.no_grad():
                discriminator(prepared["safe_target"][:1])
            stats["spectral_power_iterations"] = 1
            discriminator.eval()
            generator.eval()
            for start in range(0, count, micro_batch_size):
                stop = min(start + micro_batch_size, count)
                selected = patch_mask[start:stop]
                if not bool(selected.any()):
                    continue
                with torch.no_grad():
                    fake = generator(input_map[start:stop])
                real_logits = discriminator(prepared["safe_target"][start:stop])
                fake_logits = discriminator(fake.detach())
                d_loss = .5 * (F.softplus(-real_logits[selected]).sum()
                               + F.softplus(fake_logits[selected]).sum()) / patch_count
                if not bool(torch.isfinite(d_loss)):
                    raise FloatingPointError("Nonfinite discriminator loss")
                d_loss.backward()
                stats["adv_d"] += float(d_loss.detach())
            stats["grad_norm_d"] = _gradient_norm(discriminator.parameters(), grad_clip)
            stats["nonzero_grad_d"] = stats["grad_norm_d"] > 0
            d_optimizer.step()
            stats["discriminator_updated"] = True
            stats["loss_d"] = stats["adv_d"]
        if not (coverage_weight * denominators[0] or interior_weight * denominators[1]
                or adversarial_active or shift_active or (ridge_mode and ridge_denominator)):
            return stats
        generator.train()
        discriminator.eval()
        discriminator.requires_grad_(False)
        for start in range(0, count, micro_batch_size):
            stop = min(start + micro_batch_size, count)
            micro = {key: value[start:stop] if torch.is_tensor(value) else value
                     for key, value in prepared.items()}
            logits = generator.forward_logits(input_map[start:stop])
            loss, audit = tolerant_boundary_loss(logits, target[start:stop], valid[start:stop],
                coverage_weight=coverage_weight, interior_weight=interior_weight,
                prepared=micro, denominators=denominators, coverage_mode=coverage_mode)
            adv = logits.sum() * 0
            selected = patch_mask[start:stop]
            if adversarial_active and bool(selected.any()):
                fake_logits = discriminator(logits.sigmoid())
                adv = F.softplus(-fake_logits[selected]).sum() / patch_count
            total = loss + adversarial_weight * adv
            if ridge_mode:
                local_ridge = {key: value[start:stop] if torch.is_tensor(value) else value
                               for key, value in ridge_targets.items()}
                ridge_loss, ridge_audit = ridge_supervision_loss(
                    logits, local_ridge, denominator=ridge_denominator)
                total = total + ridge_cfg['weight'] * ridge_loss
                stats['ridge_loss'] += float(ridge_loss.detach())
                for key in ('ridge_gap_points', 'ridge_gap_profiles', 'ridge_multitrack_points'):
                    stats[key] += int(ridge_audit.get(key, 0))
            local_shift_masks, shifted_input, shifted_reference, original_reference = None, None, None, None
            if shift_active:
                local_shift_masks = {key: value[start:stop] for key, value in shift_masks.items()}
                local_active = bool(local_shift_masks['band'].any() or local_shift_masks['interior'].any())
                if local_active:
                    original_probability = logits.sigmoid()
                    original_reference = original_probability.detach()
                    shifted_input = shift_integer_reflect(input_map[start:stop], *shift)
                    with torch.no_grad():
                        shifted_reference = generator(shifted_input)
                        aligned_reference = shift_integer_reflect(shifted_reference, -shift[0], -shift[1])
                    consistency, ca = shift_consistency_loss(original_probability, aligned_reference,
                        local_shift_masks, denominators=shift_denominators)
                    total = total + shift_cfg['weight'] * consistency
                    stats['shift_consistency_loss'] += float(consistency.detach())
                    stats['shift_band_loss'] += float(ca['band'].detach())
                    stats['shift_interior_loss'] += float(ca['interior'].detach())
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Nonfinite generator loss")
            total.backward()
            if shift_active and shifted_input is not None:
                # Original graph is already released. Recompute only shifted branch with gradients.
                # Both derivatives get the full weight: L(a,sg(b)) + L(sg(a),b) gives dL/da+dL/db.
                # The scalar objective and statistics are recorded only by the first branch above.
                shifted_probability = generator(shifted_input)
                error = float((shifted_probability.detach() - shifted_reference).abs().max())
                stats['shift_recompute_max_error'] = max(stats['shift_recompute_max_error'], error)
                if error > 1e-6:
                    raise RuntimeError('Shift reference and gradient recomputation differ')
                aligned_probability = shift_integer_reflect(shifted_probability, -shift[0], -shift[1])
                second, _ = shift_consistency_loss(original_reference, aligned_probability,
                    local_shift_masks, denominators=shift_denominators)
                probe = next((p for p in generator.parameters() if p.requires_grad), None)
                before_probe = (probe.grad.detach().clone() if probe is not None and probe.grad is not None
                                else torch.zeros_like(probe) if probe is not None else None)
                (shift_cfg['weight'] * second).backward()
                if probe is not None and probe.grad is not None:
                    stats['shift_second_branch_probe_grad_norm'] += float((probe.grad.detach() - before_probe).norm())
                del shifted_probability, aligned_probability, second, before_probe
            stats["loss_g"] += float(total.detach())
            stats["coverage"] += float(audit["coverage"].detach())
            stats["interior"] += float(audit["interior"].detach())
            stats["adv_g"] += float(adv.detach())
            stats["covered_gt_points"] += audit["covered_gt_points"]
            stats["coverage_probability_mean"] += audit["coverage_probability_sum"] / max(1, denominators[0])
            if coverage_mode == 'coherent3':
                stats['pointwise_coverage'] += float(audit['pointwise_coverage'].detach())
                stats['coherent3_eligible_gt_points'] += audit['coherent3_eligible_gt_points']
                stats['coherent3_covered_gt_points'] += audit['coherent3_covered_gt_points']
                stats['coherent3_coverage_probability_mean'] += audit['coherent3_coverage_probability_sum']
            if bool(missing_gt[start:stop].any()):
                with torch.no_grad():
                    peaks = directional_coverage_logits(logits.detach(), micro["normal_index"],
                        micro["valid"], micro["tolerance"], micro["fallback"])
                    probability = peaks.sigmoid()
                    stats['missing_covered_gt_points'] += int(
                        (probability[missing_gt[start:stop]] > .65).sum())
                    if coverage_mode == 'coherent3':
                        common = coherent3_coverage(logits.detach(), micro)
                        probability = torch.where(common['eligible'], common['probabilities'], probability)
                        stats['coherent3_missing_covered_gt_points'] += int(
                            (probability[missing_gt[start:stop]] > .65).sum())
        stats["covered_gt_fraction"] = stats["covered_gt_points"] / max(1, denominators[0])
        stats["missing_covered_gt_fraction"] = stats["missing_covered_gt_points"] / max(1, stats["missing_gt_points"])
        if shifted_mode:
            stats['shift_consistency_weighted'] = shift_cfg['weight'] * stats['shift_consistency_loss']
        if ridge_mode:
            stats['ridge_weighted'] = ridge_cfg['weight'] * stats['ridge_loss']
            stats['ridge_gap_fraction'] = stats['ridge_gap_points'] / max(1, ridge_denominator)
            stats['ridge_multitrack_fraction'] = (stats['ridge_multitrack_points']
                                                 / max(1, stats['ridge_eligible_points']))
        if coverage_mode == 'coherent3':
            eligible_count = stats['coherent3_eligible_gt_points']
            stats['coherent3_eligible_gt_fraction'] = eligible_count / max(1, denominators[0])
            stats['coherent3_covered_gt_fraction'] = stats['coherent3_covered_gt_points'] / max(1, eligible_count)
            stats['coherent3_coverage_probability_mean'] /= max(1, eligible_count)
            stats['pointwise_covered_gt_points'] = stats['covered_gt_points']
            stats['pointwise_covered_gt_fraction'] = stats['covered_gt_fraction']
            stats['pointwise_coverage_probability_mean'] = stats['coverage_probability_mean']
            stats['coherent3_missing_covered_gt_fraction'] = (
                stats['coherent3_missing_covered_gt_points'] / max(1, stats['missing_gt_points']))
        stats["grad_norm_g"] = _gradient_norm(generator.parameters(), grad_clip)
        stats["nonzero_grad_g"] = stats["grad_norm_g"] > 0
        g_optimizer.step()
        stats["generator_updated"] = True
        return stats
    finally:
        generator.train(generator_mode)
        discriminator.train(discriminator_mode)
        for parameter, flag in zip(discriminator.parameters(), flags):
            parameter.requires_grad_(flag)
