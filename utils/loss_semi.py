# -*- coding: utf-8 -*-
"""
半监督一致性损失（边界预测版本 - Mean Teacher + Stage-1 锚点）
=============================================================
Mean Teacher 框架的无监督一致性损失。

双路一致性：
1. 教师路径（无梯度）：干净图像 -> 教师模型 -> 伪标签（语义+边界概率图）
2. 学生路径（有梯度）：强增强图像 -> 学生模型 -> 预测概率图
3. 一致性损失：
   - 语义通道：MSE（学生预测 vs 教师伪标签），Patch Masking 加权
   - 边界通道：Stage-1 锚点 + EMA 教师渐进混合 + BCE 软目标 + pos_weight

边界锚点机制（替代硬门控）：
  - Stage-1 模型用 GT 训练，边界预测稳定锐利，不会随训练漂移
  - 边界伪标签 = anchor_alpha * stage1_prob + (1-anchor_alpha) * teacher_prob
  - anchor_alpha 从 1.0 渐进衰减到 anchor_floor（如 0.3），始终保留锚点
   - 使用 BCE 软目标（混合概率作为连续 target），温度锐化后接近二值
   - pos_weight 放大正样本梯度，解决类别不平衡（边界像素仅占 2~5%）
   - 温度锐化：将软目标推向 0/1 极端，防止背景概率膨胀
   - 掩码区域降权：遮挡区域权重降低，防止 Patch Masking 圆斑过拟合

教师模型通过学生模型权重的 EMA（Exponential Moving Average）动态更新。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_unsupervised_loss(
    student_model: nn.Module,
    teacher_model: nn.Module,
    img_weak: torch.Tensor,
    img_strong: torch.Tensor,
    patch_mask: torch.Tensor,
    output_size: Optional[Tuple[int, int]] = None,
    boundary_anchor_cfg: Optional[dict] = None,
    ref_model: Optional[nn.Module] = None,
    anchor_alpha: float = 1.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    计算 Mean Teacher 无监督一致性损失。

    教师模型（EMA）对干净图像生成语义伪标签，学生模型对强增强图像生成预测。

    语义通道：MSE 一致性损失，Patch Masking 加权。
    边界通道：
      - 如果 boundary_anchor_cfg 启用且有 ref_model -> Stage-1 锚点 + EMA 渐进混合 + BCE 软目标
      - 否则 -> 回退到 MSE 一致性损失

    Args:
        student_model: 学生模型（有梯度）
        teacher_model: 教师模型（EMA，无梯度）
        img_weak: [B, 3, H, W] 干净无增强图像
        img_strong: [B, 3, H, W] 强增强 + Patch Masking 图像
        patch_mask: [B, 1, H, W] Patch Masking 掩码（1=遮挡, 0=保留）
        output_size: 可选，输出尺寸 (H, W)
        boundary_anchor_cfg: 边界锚点配置，包含:
            - enabled: bool
            - pos_weight: float（正样本重加权因子，负样本保持 1.0）
            - sharpen_temp: float（温度锐化参数，<1 锐化，1=不锐化）
            - mask_region_weight: float（掩码区域权重，1.0=不降权，0=完全忽略）
        ref_model: Stage-1 冻结参考模型（提供稳定边界伪标签）
        anchor_alpha: Stage-1 锚点权重（1.0=纯 Stage-1, 0.0=纯 EMA 教师）

    Returns:
        total_loss: 标量张量，总一致性损失
        loss_seg_val: float，语义通道一致性损失
        loss_boundary_val: float，边界通道一致性损失
    """
    device = next(student_model.parameters()).device

    img_weak = img_weak.to(device)
    img_strong = img_strong.to(device)
    patch_mask = patch_mask.to(device)

    # 如果指定了 output_size，将 patch_mask 对齐到该尺寸
    if output_size is not None:
        patch_mask = torch.nn.functional.interpolate(
            patch_mask, size=output_size, mode="nearest"
        )

    # ---- 教师路径（无梯度）----
    with torch.no_grad():
        teacher_output = teacher_model(img_weak, output_size=output_size)
        teacher_seg_prob = torch.sigmoid(teacher_output[:, 0])
        teacher_boundary_prob = torch.sigmoid(teacher_output[:, 1])

    # ---- 学生路径（有梯度）----
    student_output = student_model(img_strong, output_size=output_size)
    student_seg_prob = torch.sigmoid(student_output[:, 0])
    student_boundary_prob = torch.sigmoid(student_output[:, 1])

    # Patch Masking 掩码
    pm = patch_mask[:, 0]  # [B, H, W]

    # 语义通道权重：被遮挡区域权重加倍（语义通道保持原逻辑）
    seg_weight_map = 1.0 + pm  # 遮挡区域=2.0, 非遮挡=1.0

    # ---- 语义一致性损失（MSE，保持不变）----
    seg_mse = (student_seg_prob - teacher_seg_prob) ** 2
    loss_seg = (seg_mse * seg_weight_map).mean()

    # ---- 边界一致性损失 ----
    if boundary_anchor_cfg is not None and ref_model is not None and boundary_anchor_cfg.get("enabled", False):
        pos_weight_val = boundary_anchor_cfg.get("pos_weight", 3.0)
        sharpen_temp = boundary_anchor_cfg.get("sharpen_temp", 0.5)
        mask_region_weight = boundary_anchor_cfg.get("mask_region_weight", 0.3)

        # Stage-1 锚点：冻结参考模型提供稳定边界概率
        with torch.no_grad():
            ref_output = ref_model(img_weak, output_size=output_size)
            ref_boundary_prob = torch.sigmoid(ref_output[:, 1])

        # 渐进混合：anchor_alpha * stage1 + (1-anchor_alpha) * ema_teacher
        mixed_boundary_prob = (
            anchor_alpha * ref_boundary_prob
            + (1.0 - anchor_alpha) * teacher_boundary_prob
        )

        # 温度锐化：将软目标推向 0/1 极端，防止背景概率膨胀
        # p_sharp = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
        if sharpen_temp < 1.0:
            p = mixed_boundary_prob.clamp(1e-6, 1.0 - 1e-6)
            p_sharp = p.pow(1.0 / sharpen_temp)
            mixed_boundary_prob = p_sharp / (p_sharp + (1.0 - p).pow(1.0 / sharpen_temp))

        # BCE 软目标（锐化后的混合概率作为 target）+ pos_weight 重加权
        student_boundary_logits = student_output[:, 1]  # [B, H, W]
        bce = F.binary_cross_entropy_with_logits(
            student_boundary_logits, mixed_boundary_prob, reduction="none"
        )  # [B, H, W]

        # pos_weight 矩阵：正样本（混合概率 > 0.5）放大，负样本保持 1.0
        weight_matrix = torch.ones_like(mixed_boundary_prob)
        weight_matrix[mixed_boundary_prob > 0.5] = pos_weight_val

        # 边界通道权重：掩码区域降权，防止 Patch Masking 圆斑过拟合
        # 遮挡区域=mask_region_weight, 非遮挡=1.0
        bnd_weight_map = 1.0 + pm * (mask_region_weight - 1.0)

        loss_boundary = (weight_matrix * bce * bnd_weight_map).mean()
    else:
        # 回退到 MSE 一致性损失
        boundary_mse = (student_boundary_prob - teacher_boundary_prob) ** 2
        loss_boundary = (boundary_mse * seg_weight_map).mean()

    total_loss = loss_seg + loss_boundary

    return total_loss, loss_seg.item(), loss_boundary.item()


def update_ema(teacher_model: nn.Module, student_model: nn.Module, ema_decay: float):
    """
    更新教师模型的 EMA 权重。

    teacher_param = ema_decay * teacher_param + (1 - ema_decay) * student_param

    Args:
        teacher_model: 教师模型（将被原地更新）
        student_model: 学生模型（提供新权重）
        ema_decay: EMA 衰减系数（如 0.999）
    """
    with torch.no_grad():
        teacher_params = dict(teacher_model.named_parameters())
        student_params = dict(student_model.named_parameters())

        for name, teacher_param in teacher_params.items():
            if name in student_params:
                student_param = student_params[name]
                teacher_param.data.mul_(ema_decay).add_(
                    student_param.data, alpha=1.0 - ema_decay
                )

        # 同步更新 buffers（如 BatchNorm 的 running_mean/var）
        teacher_buffers = dict(teacher_model.named_buffers())
        student_buffers = dict(student_model.named_buffers())
        for name, teacher_buffer in teacher_buffers.items():
            if name in student_buffers:
                teacher_buffer.data.copy_(student_buffers[name].data)