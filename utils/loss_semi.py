# -*- coding: utf-8 -*-
"""
半监督一致性损失（边界预测版本 - Mean Teacher）
================================================
Mean Teacher 框架的无监督一致性损失。

双路一致性：
1. 教师路径（无梯度）：干净图像 -> 教师模型 -> 伪标签（语义+边界概率图）
2. 学生路径（有梯度）：强增强图像 -> 学生模型 -> 预测概率图
3. 一致性损失：MSE（学生预测 vs 教师伪标签），重点惩罚被 Patch Masking 遮挡的区域

教师模型通过学生模型权重的 EMA（Exponential Moving Average）动态更新。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def compute_unsupervised_loss(
    student_model: nn.Module,
    teacher_model: nn.Module,
    img_weak: torch.Tensor,
    img_strong: torch.Tensor,
    patch_mask: torch.Tensor,
    output_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, float, float]:
    """
    计算 Mean Teacher 无监督一致性损失。

    教师模型（EMA）对干净图像生成伪标签，学生模型对强增强图像生成预测，
    两者之间的 MSE 损失作为一致性约束。被 Patch Masking 遮挡的区域
    权重加倍，迫使网络从上下文插值出缺失的晶界。

    Args:
        student_model: 学生模型（有梯度）
        teacher_model: 教师模型（EMA，无梯度）
        img_weak: [B, 3, H, W] 干净无增强图像
        img_strong: [B, 3, H, W] 强增强 + Patch Masking 图像
        patch_mask: [B, 1, H, W] Patch Masking 掩码（1=遮挡, 0=保留）
        output_size: 可选，输出尺寸 (H, W)

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
    # （有标签数据经过 crop 后 output_size 可能小于无标签数据的原始尺寸）
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

    # ---- 一致性损失（MSE）----
    seg_mse = (student_seg_prob - teacher_seg_prob) ** 2
    boundary_mse = (student_boundary_prob - teacher_boundary_prob) ** 2

    # Patch Masking 权重：被遮挡区域权重加倍
    pm = patch_mask[:, 0]  # [B, H, W]
    weight_map = 1.0 + pm  # 遮挡区域=2.0, 非遮挡=1.0

    loss_seg = (seg_mse * weight_map).mean()
    loss_boundary = (boundary_mse * weight_map).mean()

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