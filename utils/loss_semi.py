# -*- coding: utf-8 -*-
"""
半监督一致性损失（边界预测版本 - Mean Teacher）
================================================
Mean Teacher 框架的无监督一致性损失。

双路一致性：
1. 教师路径（无梯度）：干净图像 -> 教师模型 -> 伪标签（语义+边界概率图）
2. 学生路径（有梯度）：强增强图像 -> 学生模型 -> 预测概率图
3. 一致性损失：
   - 语义通道：MSE（学生预测 vs 教师伪标签），Patch Masking 加权
   - 边界通道：硬门控截断 + BCE + alpha 小样本加权（防止雾状区域冲刷边界头）

边界硬门控机制：
  - teacher_boundary_prob > positive_threshold -> 伪标签=1（高置信度正样本）
  - teacher_boundary_prob < negative_threshold -> 伪标签=0（高置信度负样本）
  - 处于 [negative_threshold, positive_threshold] 之间的"雾状区域"被屏蔽，
    不贡献梯度，防止海量不确定像素冲刷边界预测头。

注意：不使用 Focal Loss 聚焦（gamma），因为硬门控已将教师连续概率截断为二值
标签，Focal Loss 的难样本放大效应会破坏模型流形的连续性。改为 BCE + alpha
小样本加权即可实现正负样本平衡。

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
    boundary_gate_cfg: Optional[dict] = None,
) -> Tuple[torch.Tensor, float, float]:
    """
    计算 Mean Teacher 无监督一致性损失。

    教师模型（EMA）对干净图像生成伪标签，学生模型对强增强图像生成预测。

    语义通道：MSE 一致性损失，Patch Masking 加权。
    边界通道：
      - 如果 boundary_gate_cfg 启用 -> 硬门控截断 + BCE + alpha 小样本加权
      - 否则 -> 回退到 MSE 一致性损失

    Args:
        student_model: 学生模型（有梯度）
        teacher_model: 教师模型（EMA，无梯度）
        img_weak: [B, 3, H, W] 干净无增强图像
        img_strong: [B, 3, H, W] 强增强 + Patch Masking 图像
        patch_mask: [B, 1, H, W] Patch Masking 掩码（1=遮挡, 0=保留）
        output_size: 可选，输出尺寸 (H, W)
        boundary_gate_cfg: 边界硬门控配置，包含:
            - enabled: bool
            - positive_threshold: float（正样本阈值）
            - negative_threshold: float（负样本阈值）
            - alpha: float（正样本权重，用于小样本加权平衡）

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

    # Patch Masking 权重：被遮挡区域权重加倍
    pm = patch_mask[:, 0]  # [B, H, W]
    weight_map = 1.0 + pm  # 遮挡区域=2.0, 非遮挡=1.0

    # ---- 语义一致性损失（MSE，保持不变）----
    seg_mse = (student_seg_prob - teacher_seg_prob) ** 2
    loss_seg = (seg_mse * weight_map).mean()

    # ---- 边界一致性损失 ----
    gate_enabled = (
        boundary_gate_cfg is not None
        and boundary_gate_cfg.get("enabled", False)
    )

    if gate_enabled and boundary_gate_cfg is not None:
        pos_thresh = boundary_gate_cfg.get("positive_threshold", 0.7)
        neg_thresh = boundary_gate_cfg.get("negative_threshold", 0.1)
        alpha = boundary_gate_cfg.get("alpha", 0.75)

        # 硬门控截断：生成二值伪标签 + 有效区域掩码
        target_b = (teacher_boundary_prob > pos_thresh).float()
        loss_mask = (
            (teacher_boundary_prob > pos_thresh)
            | (teacher_boundary_prob < neg_thresh)
        ).float()

        # BCE + alpha 小样本加权（不用 Focal Loss 聚焦，避免破坏流形连续性）
        student_boundary_logits = student_output[:, 1]  # [B, H, W]
        bce = F.binary_cross_entropy_with_logits(
            student_boundary_logits, target_b, reduction="none"
        )  # [B, H, W]

        alpha_t = alpha * target_b + (1.0 - alpha) * (1.0 - target_b)

        # 叠加 alpha 加权 + loss_mask（屏蔽雾状区域）+ patch_mask 权重
        loss_boundary = (
            alpha_t * bce * loss_mask * weight_map
        ).sum() / (loss_mask.sum() + 1e-6)
    else:
        # 回退到原 MSE 一致性损失
        boundary_mse = (student_boundary_prob - teacher_boundary_prob) ** 2
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