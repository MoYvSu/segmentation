# -*- coding: utf-8 -*-
"""NAF补边器的独立GAN更新：全有效batch分母、未知域隔离与可选拓扑项。"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch.nn import functional as F


def _validate_batch(batch):
    values = [batch[key] for key in ("input", "target", "valid")]
    input_map, target, valid = values
    if (not all(torch.is_tensor(value) for value in values) or input_map.ndim != 4
            or input_map.shape[1] != 1 or min(input_map.shape) < 1
            or target.shape != input_map.shape or valid.shape != input_map.shape):
        raise ValueError("GAN batch input/target/valid must have equal nonempty B1HW shape")
    if any(value.device != input_map.device for value in values):
        raise ValueError("GAN batch tensors must be on the same device")
    if not input_map.is_floating_point() or not target.is_floating_point():
        raise ValueError("GAN input and target must be floating point")
    if not bool(torch.isfinite(input_map).all()) or bool(((input_map < 0) | (input_map > 1)).any()):
        raise ValueError("GAN boundary input must be finite and within [0,1]")
    if valid.is_floating_point() and not bool(torch.isfinite(valid).all()):
        raise ValueError("GAN valid mask must be finite")
    if bool(((valid != 0) & (valid != 1)).any()):
        raise ValueError("GAN valid mask must contain only zero and one")
    valid = valid.bool()
    known = target[valid]
    if not bool(torch.isfinite(known).all()) or bool(((known < 0) | (known > 1)).any()):
        raise ValueError("Known GT targets must be finite and within [0,1]")
    # 未知target允许-1等哨兵；先选择再计算BCE，避免先算出NaN再乘0。
    safe_target = torch.where(valid, target.detach(), torch.zeros_like(target))
    return input_map, safe_target, valid


def _gradient_norm(parameters, grad_clip):
    active = [value for value in parameters if value.grad is not None]
    if not active:
        return 0.0
    norm = torch.nn.utils.clip_grad_norm_(
        active, float("inf") if grad_clip is None else float(grad_clip), error_if_nonfinite=True)
    return float(norm.detach())


def _sum_audit_counts(audits):
    """保留原始逐micro审计；仅顶层以count/points/pairs/patches结尾的计数求和。"""
    totals = {}
    for item in audits:
        for key, value in item["audit"].items():
            if (isinstance(value, Real) and not isinstance(value, bool)
                    and key.endswith(("count", "counts", "points", "pairs", "patches", "pixels"))):
                totals[key] = totals.get(key, 0) + value
    return totals


def gan_update(generator, discriminator, g_optimizer, d_optimizer, batch, *,
               micro_batch_size=16, bce_weight=1.0, adversarial_weight=1.0,
               topology_fn=None, topology_weight=0.0, grad_clip=1.0):
    """完成一次D更新和一次G更新，并返回可JSON序列化的诊断。

    BCE按整个有效batch的已知像素归一，GAN按整个batch的可靠判别格归一。
    ``topology_fn(probability, safe_target, valid_bool)``返回一个micro的样本均值
    loss及审计字典；此处按该micro样本数/总样本数加权。拓扑选择不可把未知当背景。

    谱归一化每个D更新仅执行一次power iteration：先以首个真实样本no_grad热身，
    之后D.eval做所有梯度前向。因此micro数量不改变u/v更新次数。更新G时额外冻结D
    参数，仍允许判别器对生成图反传；不更新任何旧后端。没有可靠D格时明确跳过D。
    """
    if type(micro_batch_size) is not int or micro_batch_size < 1:
        raise ValueError("micro_batch_size must be a positive integer")
    for key, value in (("bce_weight", bce_weight), ("adversarial_weight", adversarial_weight),
                       ("topology_weight", topology_weight)):
        if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if grad_clip is not None and (not math.isfinite(grad_clip) or grad_clip <= 0):
        raise ValueError("grad_clip must be finite and positive, or None")
    if topology_weight > 0 and topology_fn is None:
        raise ValueError("A positive topology weight requires topology_fn")
    input_map, target, valid = _validate_batch(batch)
    if min(input_map.shape[-2:]) < discriminator.minimum_input_size:
        raise ValueError("Training crop is too small for the discriminator")
    count = input_map.shape[0]
    valid_pixels = int(valid.sum().item())
    patch_mask = discriminator.valid_patch_mask(valid)
    patch_count = int(patch_mask.sum().item())
    adversarial_active = bool(adversarial_weight > 0 and patch_count > 0)
    stats = {
        "loss_g": 0.0, "loss_d": 0.0, "bce": 0.0, "adv_g": 0.0, "adv_d": 0.0,
        "grad_norm_g": 0.0, "grad_norm_d": 0.0, "valid_pixels": valid_pixels,
        "adversarial_patches": patch_count, "adversarial_active": adversarial_active,
        "topology_loss": 0.0, "topology_audit": [], "topology_counts": {},
        "nonzero_grad_g": False, "nonzero_grad_d": False,
        "generator_updated": False, "discriminator_updated": False,
        "samples": count, "micro_batches": (count + micro_batch_size - 1) // micro_batch_size,
        "spectral_power_iterations": 0,
    }
    generator_mode, discriminator_mode = generator.training, discriminator.training
    d_flags = [parameter.requires_grad for parameter in discriminator.parameters()]
    g_optimizer.zero_grad(set_to_none=True)
    d_optimizer.zero_grad(set_to_none=True)
    try:
        if adversarial_active:
            # 仅这一次train前向更新各层谱归一化缓冲，次数与micro拆分无关。
            discriminator.train()
            with torch.no_grad():
                discriminator(target[:1])
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
                real_logits = discriminator(target[start:stop])
                fake_logits = discriminator(fake.detach())
                real_loss = F.binary_cross_entropy_with_logits(
                    real_logits[selected], torch.ones_like(real_logits[selected]), reduction="sum")
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_logits[selected], torch.zeros_like(fake_logits[selected]), reduction="sum")
                d_loss = 0.5 * (real_loss + fake_loss) / patch_count
                if not bool(torch.isfinite(d_loss)):
                    raise FloatingPointError("Nonfinite discriminator loss")
                d_loss.backward()
                stats["adv_d"] += float(d_loss.detach())
            stats["grad_norm_d"] = _gradient_norm(discriminator.parameters(), grad_clip)
            stats["nonzero_grad_d"] = stats["grad_norm_d"] > 0
            d_optimizer.step()
            stats["discriminator_updated"] = True
            stats["loss_d"] = stats["adv_d"]

        # 未知域既无BCE，也不可能有有效D感受野；整批为空时不走Adam动量/权重衰减。
        if valid_pixels == 0 or not (bce_weight > 0 or adversarial_active or topology_weight > 0):
            return stats

        generator.train()
        discriminator.eval()
        discriminator.requires_grad_(False)
        for start in range(0, count, micro_batch_size):
            stop = min(start + micro_batch_size, count)
            micro_input, micro_target = input_map[start:stop], target[start:stop]
            micro_valid, selected = valid[start:stop], patch_mask[start:stop]
            logits = generator.forward_logits(micro_input)
            if logits.shape != micro_target.shape or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Generator logits are invalid or misaligned")
            probability = logits.sigmoid()
            bce = F.binary_cross_entropy_with_logits(logits, micro_target, reduction="none")
            bce = bce[micro_valid].sum() / valid_pixels
            adv = logits.sum() * 0.0
            if adversarial_active and bool(selected.any()):
                fake_logits = discriminator(probability)
                adv = F.binary_cross_entropy_with_logits(
                    fake_logits[selected], torch.ones_like(fake_logits[selected]), reduction="sum") / patch_count
            topology = logits.sum() * 0.0
            if topology_weight > 0:
                topology_mean, audit = topology_fn(probability, micro_target, micro_valid)
                if not torch.is_tensor(topology_mean) or topology_mean.ndim != 0 or not isinstance(audit, dict):
                    raise ValueError("topology_fn must return scalar tensor loss and audit dict")
                topology = topology_mean * ((stop - start) / count)
                stats["topology_audit"].append({"samples": stop - start, "audit": audit})
            loss = bce_weight * bce + adversarial_weight * adv + topology_weight * topology
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite generator loss")
            loss.backward()
            stats["loss_g"] += float(loss.detach())
            stats["bce"] += float(bce.detach())
            stats["adv_g"] += float(adv.detach())
            stats["topology_loss"] += float(topology.detach())
        stats["grad_norm_g"] = _gradient_norm(generator.parameters(), grad_clip)
        stats["nonzero_grad_g"] = stats["grad_norm_g"] > 0
        g_optimizer.step()
        stats["generator_updated"] = True
        stats["topology_counts"] = _sum_audit_counts(stats["topology_audit"])
        return stats
    finally:
        generator.train(generator_mode)
        discriminator.train(discriminator_mode)
        for parameter, flag in zip(discriminator.parameters(), d_flags):
            parameter.requires_grad_(flag)
