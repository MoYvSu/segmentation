# -*- coding: utf-8 -*-
"""
半监督一致性损失（边界预测版本 - 多模式伪标签源）
=====================================================
支持多种边界伪标签源模式的无监督一致性损失。

双路一致性：
1. 伪标签路径（无梯度）：干净图像 -> 伪标签源 -> 伪标签（语义+边界概率图）
2. 学生路径（有梯度）：强增强图像 -> 学生模型 -> 预测概率图
3. 一致性损失：
   - 语义通道：MSE（学生预测 vs 教师伪标签），Patch Masking 加权
   - 边界通道：MSE + Sobel 梯度一致性 + 各向异性 TV + 背景抑制
     （伪标签源由 boundary_teacher_mode 决定）

边界伪标签源模式（boundary_teacher_mode）：
  - "ema": EMA 教师 + Stage-1 锚点渐进混合（默认）
    边界伪标签 = anchor_alpha * stage1_prob + (1-anchor_alpha) * teacher_prob
  - "stage1_direct": Stage-1 冻结模型直接提供（无 EMA 滞后）
    边界伪标签 = ref_model(img_weak) 边界通道输出
  - "self_consistency": 学生弱增强预测 stop-gradient（无 EMA 依赖）
    边界伪标签 = student_model(img_weak).detach() 边界通道输出

骨架过滤时序：
  - 骨架过滤在伪标签生成之后施加，对最终目标伪标签过滤
  - 这样无论 ref_model、教师还是学生引入的弥散噪声都会被清除

梯度感知损失设计动机：
  - 旧的 BCE 软目标 + 温度锐化 + pos_weight 链路像素级独立，缺乏空间结构约束
  - 经历多轮训练后边界头展现雾状热力图（弥散低概率响应）
  - Sobel 梯度一致性约束学生与教师在空间梯度结构上一致
  - 各向异性 TV 在非边界区强惩罚（去雾），边界区弱惩罚（保锐）
  - 背景抑制损失直接将非边界区学生预测压向零

EMA 教师模型（仅 ema 模式需要）通过学生模型权重的 EMA 动态更新。
"""

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def skeleton_filter_boundary(
    boundary_prob: torch.Tensor,
    threshold: float = 0.5,
    dilate_width: int = 1,
    blur_sigma: float = 1.0,
) -> torch.Tensor:
    """
    对边界概率图施加骨架过滤（形态学先验精炼伪标签）。

    流程（逐样本）：
      1. 概率图 > threshold -> 二值边界
      2. 骨架化（Zhang-Suen / skimage fallback）-> 1px 中心线
      3. 膨胀 dilate_width -> 可配置宽度边界带
      4. 高斯模糊（blur_sigma > 0 时）-> 软概率 [0, 1]

    边界情况：边界像素过少或骨架化为空时，返回原始概率图（跳过过滤），
    并记录 debug 日志以便监控跳过率。

    Args:
        boundary_prob: [B, H, W] 边界概率图（sigmoid 后，值域 [0, 1]）
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

    skip_count = 0

    for i in range(B):
        p = prob_np[i]  # [H, W]

        # Step 1: 二值化
        binary = (p > threshold).astype(np.uint8) * 255

        nonzero_count = cv2.countNonZero(binary)

        # 边界像素过少 -> 跳过
        if nonzero_count < 10:
            filtered_np[i] = p
            skip_count += 1
            continue

        # Step 2: 骨架化
        try:
            skeleton = cv2.ximgproc.thinning(
                binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
        except (AttributeError, cv2.error):
            from skimage.morphology import skeletonize as sk_skeletonize
            skeleton = (sk_skeletonize(binary > 0) * 255).astype(np.uint8)

        skel_count = cv2.countNonZero(skeleton)

        # 骨架化为空 -> 跳过
        if skel_count < 5:
            filtered_np[i] = p
            skip_count += 1
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

    if skip_count > 0:
        logger.debug(
            f"skeleton_filter: {skip_count}/{B} samples skipped "
            f"(threshold={threshold}, too few boundary pixels or empty skeleton)"
        )

    return torch.from_numpy(filtered_np).to(device=device, dtype=boundary_prob.dtype)


def sobel_gradient_consistency(
    pred_prob: torch.Tensor,
    target_prob: torch.Tensor,
) -> torch.Tensor:
    """
    Sobel 梯度一致性损失。

    对学生预测和教师伪标签分别施加 Sobel 算子，计算梯度幅值图的 L1 差异。
    约束学生不仅在像素值上与教师一致，更在空间梯度结构上一致，
    鼓励学生在边界位置产生与教师相同的锐利梯度响应。

    Args:
        pred_prob: [B, H, W] 学生边界概率（sigmoid 后）
        target_prob: [B, H, W] 教师伪标签概率（无梯度）

    Returns:
        标量损失
    """
    # Sobel 卷积核
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=pred_prob.dtype, device=pred_prob.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=pred_prob.dtype, device=pred_prob.device,
    ).view(1, 1, 3, 3)

    # 添加通道维度用于 conv2d
    pred = pred_prob.unsqueeze(1)        # [B, 1, H, W]
    target = target_prob.unsqueeze(1)    # [B, 1, H, W]

    # 计算梯度幅值
    pred_gx = F.conv2d(pred, sobel_x, padding=1)
    pred_gy = F.conv2d(pred, sobel_y, padding=1)
    pred_grad = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-8)

    with torch.no_grad():
        target_gx = F.conv2d(target, sobel_x, padding=1)
        target_gy = F.conv2d(target, sobel_y, padding=1)
        target_grad = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-8)

    # L1 损失
    return (pred_grad - target_grad).abs().mean()


def anisotropic_tv(
    pred_prob: torch.Tensor,
    target_prob: torch.Tensor,
    dilate_radius: int = 3,
    bg_weight: float = 1.0,
    boundary_weight: float = 0.1,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    各向异性总变差正则化。

    非边界区域施加强 TV 权重（抑制雾状噪声→干净归零），
    边界区域施加弱 TV 权重（保留锐利边缘）。

    边界/非边界区域通过教师伪标签的二值化+膨胀掩码区分。

    Args:
        pred_prob: [B, H, W] 学生边界概率
        target_prob: [B, H, W] 教师伪标签概率（用于生成边界掩码）
        dilate_radius: 边界区域膨胀半径（px）
        bg_weight: 非边界区域 TV 权重
        boundary_weight: 边界区域 TV 权重
        threshold: 伪标签二值化阈值

    Returns:
        标量 TV 损失
    """
    # 生成边界区域掩码（二值化 + 膨胀）
    with torch.no_grad():
        boundary_binary = (target_prob > threshold).float().unsqueeze(1)  # [B, 1, H, W]
        if dilate_radius > 0:
            kernel_size = 2 * dilate_radius + 1
            boundary_dilated = F.max_pool2d(
                boundary_binary, kernel_size=kernel_size, stride=1, padding=dilate_radius
            ).squeeze(1)  # [B, H, W]
        else:
            boundary_dilated = boundary_binary.squeeze(1)  # [B, H, W]

    # 权重图：边界区域 = boundary_weight, 非边界 = bg_weight
    weight_map = boundary_dilated * (boundary_weight - bg_weight) + bg_weight  # [B, H, W]

    # TV = |dx| + |dy|
    dx = (pred_prob[:, :, 1:] - pred_prob[:, :, :-1]).abs()  # [B, H, W-1]
    dy = (pred_prob[:, 1:, :] - pred_prob[:, :-1, :]).abs()  # [B, H-1, W]

    # 对齐权重图维度
    wx = weight_map[:, :, :-1]  # [B, H, W-1]
    wy = weight_map[:, :-1, :]  # [B, H-1, W]

    tv = (dx * wx).mean() + (dy * wy).mean()
    return tv


def background_suppression_loss(
    pred_prob: torch.Tensor,
    target_prob: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    背景抑制损失。

    对非边界区域（目标伪标签 < threshold）的学生预测施加 L1 惩罚，
    强力将背景区推向零，抑制弥散雾状响应。

    与 TV 正则化互补：TV 抑制像素间变化（平滑），背景抑制直接压低绝对值。

    Args:
        pred_prob: [B, H, W] 学生边界概率
        target_prob: [B, H, W] 教师伪标签概率（用于区分背景区）
        threshold: 低于此值视为背景区域

    Returns:
        标量损失
    """
    with torch.no_grad():
        bg_mask = (target_prob < threshold).float()  # [B, H, W], 1=背景

    # 背景区域学生预测的 L1 惩罚
    return (pred_prob * bg_mask).mean()


def boundary_margin_loss(
    pred_prob: torch.Tensor,
    target_prob: torch.Tensor,
    margin: float = 0.4,
    pos_threshold: float = 0.5,
    bg_threshold: float = 0.05,
) -> torch.Tensor:
    """边界-背景差值 margin 损失。

    对每个样本，约束学生输出中"边界像素均值 − 背景像素均值 ≥ margin"：
      L = mean_i relu(margin - gap_i) * has_pos_i
    直接针对"边界与背景差值过小、输出区间被压缩"的问题，
    与像素级 MSE 互补（MSE 管逐像素对齐，margin 管对比度/区间）。

    Args:
        pred_prob: [B, H, W] 学生边界概率（sigmoid 后，有梯度）
        target_prob: [B, H, W] 目标伪标签（划分边界/背景区域，无梯度）
        margin: 期望的最小边界-背景差值
        pos_threshold: 目标高于此值视为边界像素
        bg_threshold: 目标低于此值视为背景像素

    Returns:
        标量损失（无正样本的样本不参与）
    """
    with torch.no_grad():
        pos_mask = (target_prob > pos_threshold).float()
        bg_mask = (target_prob < bg_threshold).float()

    pos_cnt = pos_mask.sum(dim=(1, 2)).clamp(min=1.0)
    bg_cnt = bg_mask.sum(dim=(1, 2)).clamp(min=1.0)
    bnd_mean = (pred_prob * pos_mask).sum(dim=(1, 2)) / pos_cnt
    bg_mean = (pred_prob * bg_mask).sum(dim=(1, 2)) / bg_cnt
    gap = bnd_mean - bg_mean  # [B]

    has_pos = (pos_mask.sum(dim=(1, 2)) > 0).float()
    denom = has_pos.sum().clamp(min=1.0)
    return (torch.relu(margin - gap) * has_pos).sum() / denom


def compute_unsupervised_loss(
    student_model: nn.Module,
    img_weak: torch.Tensor,
    img_strong: torch.Tensor,
    patch_mask: torch.Tensor,
    output_size: Optional[Tuple[int, int]] = None,
    teacher_model: Optional[nn.Module] = None,
    boundary_teacher_mode: str = "ema",
    boundary_anchor_cfg: Optional[dict] = None,
    ref_model: Optional[nn.Module] = None,
    anchor_alpha: float = 1.0,
    skeleton_filter_cfg: Optional[dict] = None,
    freeze_seg: bool = False,
    freeze_boundary: bool = False,
    seg_mask_region_weight: float = 2.0,
    boundary_mask_region_weight: float = 0.3,
    sobel_weight: float = 1.0,
    tv_weight: float = 0.1,
    tv_dilate_radius: int = 3,
    tv_bg_weight: float = 1.0,
    tv_boundary_weight: float = 0.1,
    bg_suppress_weight: float = 0.5,
    bg_suppress_threshold: float = 0.1,
    cached_boundary_target: Optional[torch.Tensor] = None,
    pos_weight: float = 5.0,
    margin_loss_weight: float = 0.0,
    margin: float = 0.4,
    rate_regularizer_weight: float = 0.0,
    rate_slack: float = 0.05,
) -> Tuple[torch.Tensor, float, float, Dict[str, float]]:
    """
    计算无监督一致性损失（支持多种边界伪标签源模式）。

    语义通道：MSE 一致性损失，Patch Masking 加权。
    边界通道：MSE + Sobel 梯度一致性 + 各向异性 TV + 背景抑制
      - 伪标签源由 boundary_teacher_mode 决定：
        * "ema": EMA 教师 + Stage-1 锚点混合（默认）
        * "stage1_direct": Stage-1 冻结模型直接提供（无 EMA 滞后）
        * "self_consistency": 学生弱增强预测 stop-gradient（无 EMA 依赖）
      - 骨架过滤在伪标签生成之后施加（对最终目标伪标签过滤）
      - MSE 提供像素级一致性基础
      - Sobel 梯度一致性约束空间梯度结构一致
      - 各向异性 TV 抑制非边界区雾状噪声，保留边界锐利度
      - 背景抑制直接将非边界区学生预测压向零

    Args:
        student_model: 学生模型（有梯度）
        img_weak: [B, 3, H, W] 干净无增强图像
        img_strong: [B, 3, H, W] 强增强 + Patch Masking 图像
        patch_mask: [B, 1, H, W] Patch Masking 掩码（1=遮挡, 0=保留）
        output_size: 可选，输出尺寸 (H, W)
        teacher_model: EMA 教师模型（无梯度）。当 boundary_teacher_mode != "ema"
            且 freeze_seg=True 时可为 None。
        boundary_teacher_mode: 边界伪标签源模式
            "ema": EMA 教师伪标签 + Stage-1 锚点混合（默认）
            "stage1_direct": Stage-1 冻结模型直接提供
            "self_consistency": 学生弱增强预测 stop-gradient
        boundary_anchor_cfg: 边界锚点配置（仅 ema 模式生效），包含:
            - enabled: bool
            - anchor_floor: float（锚点权重下限）
            - anchor_ramp_epochs: int（衰减 epoch 数）
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
        sobel_weight: Sobel 梯度一致性损失权重。
        tv_weight: 各向异性 TV 正则化权重。
        tv_dilate_radius: TV 中边界区域膨胀半径（px）。
        tv_bg_weight: 非边界区域 TV 权重。
        tv_boundary_weight: 边界区域 TV 权重。
        bg_suppress_weight: 背景抑制损失权重。
        bg_suppress_threshold: 背景抑制阈值，低于此值视为背景区域。
        cached_boundary_target: 离线预计算的 Stage-1 边界目标（[B,1,H,W] 概率图）。
            仅 stage1_direct 模式使用；提供时跳过 ref_model 前向。
        pos_weight: 目标 > 0.5 像素的一致性损失放大权重（稀疏正样本重平衡）。
        margin_loss_weight: 边界-背景 margin 损失权重。
        margin: margin 损失的目标差值。
        rate_regularizer_weight: 预测正样本占比上限正则权重。
            当学生预测均值超过"目标占比 + rate_slack"时产生 hinge 惩罚，
            直接阻止边界概率空间扩散（>0.5 占比膨胀到 30%+ 的失效模式）。
        rate_slack: 预测占比允许超出目标占比的余量。

    Returns:
        total_loss: 标量张量，总一致性损失
        loss_seg_val: float，语义通道一致性损失
        loss_boundary_val: float，边界通道一致性损失
        bnd_stats: dict，边界输出统计（max / >0.5 占比 / 边界-背景差值），
            用于训练日志观察输出区间是否被拉开

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

    # ---- 语义伪标签路径（仅当语义分支未冻结时需要 EMA 教师）----
    teacher_seg_prob = None
    if not freeze_seg:
        if teacher_model is None:
            raise ValueError(
                "teacher_model is required when freeze_seg=False "
                "(语义通道需要 EMA 教师提供伪标签)"
            )
        with torch.no_grad():
            teacher_output = teacher_model(img_weak, output_size=output_size)
            teacher_seg_prob = torch.sigmoid(teacher_output[:, 0])

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
        bnd_stats = {"bnd_max": 0.0, "bnd_pos_frac": 0.0, "bnd_gap": 0.0}
    else:
        # ---- 根据 boundary_teacher_mode 选择边界伪标签源 ----
        if boundary_teacher_mode == "ema":
            # 模式 1: EMA 教师 + Stage-1 锚点混合（默认行为）
            if teacher_model is None:
                raise ValueError(
                    "teacher_model is required for boundary_teacher_mode='ema'"
                )
            with torch.no_grad():
                teacher_output = teacher_model(img_weak, output_size=output_size)
                teacher_boundary_prob = torch.sigmoid(teacher_output[:, 1])

            # Stage-1 锚点混合（如果启用）
            if (
                boundary_anchor_cfg is not None
                and ref_model is not None
                and boundary_anchor_cfg.get("enabled", False)
            ):
                with torch.no_grad():
                    ref_output = ref_model(img_weak, output_size=output_size)
                    ref_boundary_prob = torch.sigmoid(ref_output[:, 1])
                target_boundary_prob = (
                    anchor_alpha * ref_boundary_prob
                    + (1.0 - anchor_alpha) * teacher_boundary_prob
                )
            else:
                target_boundary_prob = teacher_boundary_prob

        elif boundary_teacher_mode == "stage1_direct":
            # 模式 2: Stage-1 冻结模型直接提供伪标签（无 EMA 滞后）
            if cached_boundary_target is not None:
                # 离线预计算缓存（tools/precompute_pseudo_labels.py 生成）：
                # 免去每 step 的 ref_model 前向，目标为 1024 letterbox 概率图
                target_boundary_prob = cached_boundary_target.to(
                    device=device, dtype=torch.float32
                )
                if target_boundary_prob.shape[-2:] != student_boundary_prob.shape[-2:]:
                    target_boundary_prob = F.interpolate(
                        target_boundary_prob,
                        size=student_boundary_prob.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                # 缓存为 [B, 1, H, W]，统一为与其它模式一致的 [B, H, W]
                target_boundary_prob = target_boundary_prob[:, 0]
            else:
                if ref_model is None:
                    raise ValueError(
                        "ref_model (Stage-1) is required for "
                        "boundary_teacher_mode='stage1_direct'"
                    )
                with torch.no_grad():
                    ref_output = ref_model(img_weak, output_size=output_size)
                    target_boundary_prob = torch.sigmoid(ref_output[:, 1])

        elif boundary_teacher_mode == "self_consistency":
            # 模式 3: 学生弱增强预测 stop-gradient 作为伪标签
            with torch.no_grad():
                student_weak_output = student_model(img_weak, output_size=output_size)
                target_boundary_prob = torch.sigmoid(student_weak_output[:, 1])

        elif boundary_teacher_mode == "anchor_self":
            # 模式 4: 学生弱增强预测(stop-grad)与 Stage-1 锚点混合
            # 自一致性恢复弱边界 recall（学生经微调后弱边界比 Stage-1 更强），
            # 锚点锚定几何结构防漂移；anchor_alpha 从 1.0 退火到 anchor_floor，
            # 训练中逐步把主导权交给学生。
            with torch.no_grad():
                student_weak_output = student_model(img_weak, output_size=output_size)
                student_weak_boundary = torch.sigmoid(student_weak_output[:, 1])

            if cached_boundary_target is not None:
                anchor_boundary = cached_boundary_target.to(
                    device=device, dtype=torch.float32
                )
                if anchor_boundary.shape[-2:] != student_boundary_prob.shape[-2:]:
                    anchor_boundary = F.interpolate(
                        anchor_boundary,
                        size=student_boundary_prob.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                anchor_boundary = anchor_boundary[:, 0]
            else:
                if ref_model is None:
                    raise ValueError(
                        "ref_model (Stage-1) is required for "
                        "boundary_teacher_mode='anchor_self'"
                    )
                with torch.no_grad():
                    ref_output = ref_model(img_weak, output_size=output_size)
                    anchor_boundary = torch.sigmoid(ref_output[:, 1])

            target_boundary_prob = (
                anchor_alpha * anchor_boundary
                + (1.0 - anchor_alpha) * student_weak_boundary
            )

        else:
            raise ValueError(
                f"Unknown boundary_teacher_mode: '{boundary_teacher_mode}'. "
                f"Supported modes: 'ema', 'stage1_direct', 'self_consistency'"
            )

        # 骨架过滤：在伪标签生成之后施加，对最终目标伪标签过滤
        # 这样无论 ref_model、教师还是学生引入的弥散噪声都会被清除
        with torch.no_grad():
            if skeleton_filter_cfg is not None and skeleton_filter_cfg.get("enabled", False):
                target_boundary_prob = skeleton_filter_boundary(
                    target_boundary_prob,
                    threshold=skeleton_filter_cfg.get("threshold", 0.5),
                    dilate_width=skeleton_filter_cfg.get("dilate_width", 1),
                    blur_sigma=skeleton_filter_cfg.get("blur_sigma", 1.0),
                )

        # 掩码区域降权 + 正样本加权（目标 > 0.5 像素放大，扭转稀疏正样本梯度劣势）
        bnd_weight_map = 1.0 + pm * (boundary_mask_region_weight - 1.0)
        if pos_weight > 0:
            with torch.no_grad():
                pos_map = (target_boundary_prob > 0.5).float()
            bnd_weight_map = bnd_weight_map * (1.0 + pos_weight * pos_map)

        # 1. MSE 像素一致性（基础项）
        boundary_mse = (student_boundary_prob - target_boundary_prob) ** 2
        loss_mse = (boundary_mse * bnd_weight_map).mean()

        # 2. Sobel 梯度一致性
        loss_sobel = torch.tensor(0.0, device=device)
        if sobel_weight > 0:
            loss_sobel = sobel_gradient_consistency(
                student_boundary_prob, target_boundary_prob
            )

        # 3. 各向异性 TV 正则化
        loss_tv = torch.tensor(0.0, device=device)
        if tv_weight > 0:
            tv_threshold = (
                skeleton_filter_cfg.get("threshold", 0.5)
                if skeleton_filter_cfg
                else 0.5
            )
            loss_tv = anisotropic_tv(
                student_boundary_prob,
                target_boundary_prob,
                dilate_radius=tv_dilate_radius,
                bg_weight=tv_bg_weight,
                boundary_weight=tv_boundary_weight,
                threshold=tv_threshold,
            )

        # 4. 背景抑制损失
        loss_bg = torch.tensor(0.0, device=device)
        if bg_suppress_weight > 0:
            loss_bg = background_suppression_loss(
                student_boundary_prob,
                target_boundary_prob,
                threshold=bg_suppress_threshold,
            )

        # 5. 边界-背景 margin 损失（直接拉开学生输出差值）
        loss_margin = torch.tensor(0.0, device=device)
        if margin_loss_weight > 0:
            loss_margin = boundary_margin_loss(
                student_boundary_prob,
                target_boundary_prob,
                margin=margin,
                pos_threshold=0.5,
                bg_threshold=bg_suppress_threshold,
            )

        # 6. 预测正样本占比上限正则（阻止边界带扩散/背景膨胀）
        loss_rate = torch.tensor(0.0, device=device)
        if rate_regularizer_weight > 0:
            with torch.no_grad():
                target_rate = target_boundary_prob.mean(dim=(1, 2))
                ceiling = target_rate + rate_slack
            pred_rate = student_boundary_prob.mean(dim=(1, 2))
            loss_rate = torch.relu(pred_rate - ceiling).mean()

        loss_boundary = (
            loss_mse
            + sobel_weight * loss_sobel
            + tv_weight * loss_tv
            + bg_suppress_weight * loss_bg
            + margin_loss_weight * loss_margin
            + rate_regularizer_weight * loss_rate
        )

        # 边界输出统计（训练日志用，观察输出区间是否被拉开）
        bnd_stats = {}
        with torch.no_grad():
            bnd_stats["bnd_max"] = float(student_boundary_prob.max())
            bnd_stats["bnd_pred_rate"] = float(student_boundary_prob.mean())
            bnd_stats["bnd_pos_frac"] = float(
                (student_boundary_prob > 0.5).float().mean()
            )
            pos_cnt = (target_boundary_prob > 0.5).sum(dim=(1, 2)).clamp(min=1.0)
            bg_cnt = (target_boundary_prob < bg_suppress_threshold).sum(
                dim=(1, 2)
            ).clamp(min=1.0)
            bnd_mean = (
                student_boundary_prob * (target_boundary_prob > 0.5).float()
            ).sum(dim=(1, 2)) / pos_cnt
            bg_mean = (
                student_boundary_prob
                * (target_boundary_prob < bg_suppress_threshold).float()
            ).sum(dim=(1, 2)) / bg_cnt
            bnd_stats["bnd_gap"] = float((bnd_mean - bg_mean).mean())

    total_loss = loss_seg + loss_boundary

    return total_loss, loss_seg.item(), loss_boundary.item(), bnd_stats


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
