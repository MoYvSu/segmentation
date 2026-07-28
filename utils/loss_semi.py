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

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def skeleton_filter_boundary(
    boundary_prob: torch.Tensor,
    threshold: float = 0.5,
    dilate_width: int = 1,
    blur_sigma: float = 1.0,
) -> torch.Tensor:
    """
    对教师边界概率图施加骨架过滤（形态学先验精炼伪标签）。

    流程（逐样本）：
      1. 概率图 > threshold -> 二值边界
      2. 骨架化（Zhang-Suen / skimage fallback）-> 1px 中心线
      3. 膨胀 dilate_width -> 可配置宽度边界带
      4. 高斯模糊（blur_sigma > 0 时）-> 软概率 [0, 1]

    边界情况：边界像素过少或骨架化为空时，返回原始概率图（跳过过滤）。

    Args:
        boundary_prob: [B, H, W] 教师边界概率图（sigmoid 后，值域 [0, 1]）
        threshold: 二值化阈值
        dilate_width: 骨架膨胀宽度（最终边界宽度 = 2*w+1 px）
        blur_sigma: 高斯模糊 sigma（0=硬二值目标，>0=软目标）

    Returns:
        filtered_prob: [B, H, W] 过滤后的边界概率图
    """
    device = boundary_prob.device
    B, H, W = boundary_prob.shape

    # 转 CPU numpy 逐样本处理
    prob_np = boundary_prob.detach().cpu().numpy()
    filtered_np = np.zeros_like(prob_np)

    for i in range(B):
        p = prob_np[i]  # [H, W]

        # Step 1: 二值化
        binary = (p > threshold).astype(np.uint8) * 255

        # 边界像素过少 -> 跳过
        if cv2.countNonZero(binary) < 10:
            filtered_np[i] = p
            continue

        # Step 2: 骨架化
        try:
            skeleton = cv2.ximgproc.thinning(
                binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
        except (AttributeError, cv2.error):
            from skimage.morphology import skeletonize as sk_skeletonize
            skeleton = (sk_skeletonize(binary > 0) * 255).astype(np.uint8)

        # 骨架化为空 -> 跳过
        if cv2.countNonZero(skeleton) < 5:
            filtered_np[i] = p
            continue

        # Step 3: 膨胀
        if dilate_width > 0:
            kernel_size = 2 * dilate_width + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            belt = cv2.dilate(skeleton, kernel)
        else:
            belt = skeleton

        # Step 4: 转回概率图
        belt_float = (belt > 0).astype(np.float32)

        if blur_sigma > 0:
            ksize = int(2 * round(2 * blur_sigma) + 1)
            if ksize < 3:
                ksize = 3
            belt_float = cv2.GaussianBlur(
                belt_float, (ksize, ksize), blur_sigma
            )
            # 归一化到 [0, 1]
            mx = belt_float.max()
            if mx > 0:
                belt_float = belt_float / mx

        filtered_np[i] = belt_float

    return torch.from_numpy(filtered_np).to(device=device, dtype=boundary_prob.dtype)


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
    skeleton_filter_cfg: Optional[dict] = None,
    freeze_seg: bool = False,
    freeze_boundary: bool = False,
    seg_mask_region_weight: float = 2.0,
    boundary_mask_region_weight: float = 0.3,
) -> Tuple[torch.Tensor, float, float]:
    """
    计算 Mean Teacher 无监督一致性损失。

    教师模型（EMA）对干净图像生成语义伪标签，学生模型对强增强图像生成预测。

    语义通道：MSE 一致性损失，Patch Masking 加权。
    边界通道：
      - 如果 boundary_anchor_cfg 启用且有 ref_model -> Stage-1 锚点 + EMA 渐进混合 + BCE 软目标
      - 否则 -> 回退到 MSE 一致性损失

    骨架过滤（可选）：在教师边界概率输出后、混合/锐化前，施加形态学骨架过滤，
    去除弥散噪声和拓扑毛刺，精炼边界伪标签。

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
        ref_model: Stage-1 冻结参考模型（提供稳定边界伪标签）
        anchor_alpha: Stage-1 锚点权重（1.0=纯 Stage-1, 0.0=纯 EMA 教师）
        skeleton_filter_cfg: 骨架过滤配置，包含:
            - enabled: bool
            - threshold: float（二值化阈值）
            - dilate_width: int（骨架膨胀宽度，最终边界宽度=2*w+1 px）
            - blur_sigma: float（高斯模糊 sigma，0=硬二值，>0=软目标）
        freeze_seg: 冻结语义分支时为 True，跳过语义一致性损失。
        freeze_boundary: 冻结边界分支时为 True，跳过边界一致性损失。
        seg_mask_region_weight: 语义通道掩码区域权重（1.0=不降权，2.0=加倍）。
        boundary_mask_region_weight: 边界通道掩码区域权重（1.0=不降权，0.3=降权）。

    Returns:
        total_loss: 标量张量，总一致性损失
        loss_seg_val: float，语义通道一致性损失
        loss_boundary_val: float，边界通道一致性损失

    注意：当 freeze_seg=True 时 loss_seg=0（无梯度），
          当 freeze_boundary=True 时 loss_boundary=0（无梯度）。
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

        # 骨架过滤：精炼教师边界伪标签（去除弥散噪声和拓扑毛刺）
        if skeleton_filter_cfg is not None and skeleton_filter_cfg.get("enabled", False):
            teacher_boundary_prob = skeleton_filter_boundary(
                teacher_boundary_prob,
                threshold=skeleton_filter_cfg.get("threshold", 0.5),
                dilate_width=skeleton_filter_cfg.get("dilate_width", 1),
                blur_sigma=skeleton_filter_cfg.get("blur_sigma", 1.0),
            )

    # ---- 学生路径（有梯度）----
    student_output = student_model(img_strong, output_size=output_size)
    student_seg_prob = torch.sigmoid(student_output[:, 0])
    student_boundary_prob = torch.sigmoid(student_output[:, 1])

    # Patch Masking 掩码
    pm = patch_mask[:, 0]  # [B, H, W]

    # 语义通道权重：被遮挡区域权重可配置（低频面状特征，默认加倍鼓励从上下文推断）
    seg_weight_map = 1.0 + pm * (seg_mask_region_weight - 1.0)

    # ---- 语义一致性损失（MSE）----
    if freeze_seg:
        loss_seg = torch.tensor(0.0, device=device, requires_grad=False)
    else:
        seg_mse = (student_seg_prob - teacher_seg_prob) ** 2
        loss_seg = (seg_mse * seg_weight_map).mean()

    # ---- 边界一致性损失 ----
    if freeze_boundary:
        loss_boundary = torch.tensor(0.0, device=device, requires_grad=False)
    elif boundary_anchor_cfg is not None and ref_model is not None and boundary_anchor_cfg.get("enabled", False):
        pos_weight_val = boundary_anchor_cfg.get("pos_weight", 3.0)
        sharpen_temp = boundary_anchor_cfg.get("sharpen_temp", 0.5)

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
        # 遮挡区域=boundary_mask_region_weight, 非遮挡=1.0
        bnd_weight_map = 1.0 + pm * (boundary_mask_region_weight - 1.0)

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

    跳过 requires_grad=False 的参数（冻结分支的学生参数不变，教师无需 EMA 跟随）。

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
                # 跳过冻结参数（requires_grad=False），学生未更新，教师也无需更新
                if not student_param.requires_grad:
                    continue
                teacher_param.data.mul_(ema_decay).add_(
                    student_param.data, alpha=1.0 - ema_decay
                )

        # 同步更新 buffers（如 BatchNorm 的 running_mean/var）
        # 冻结分支的 buffers 仍需同步（GroupNorm 无可学习参数但有运行统计）
        teacher_buffers = dict(teacher_model.named_buffers())
        student_buffers = dict(student_model.named_buffers())
        for name, teacher_buffer in teacher_buffers.items():
            if name in student_buffers:
                teacher_buffer.data.copy_(student_buffers[name].data)