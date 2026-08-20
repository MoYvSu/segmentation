# -*- coding: utf-8 -*-
"""
边界预测损失函数
================
语义、边界和中心热图预测损失：
  - 通道 0（语义）：BCEWithLogitsLoss
  - 通道 1（边界）：Focal Loss × W_boundary（EDT 权重调制）
  - 通道 2（中心）：稀疏 Gaussian 峰的 focal BCE

L_sup = BCEWithLogitsLoss(pred[:,0], target[:,0])
      + alpha * FocalLoss(pred[:,1], target[:,1]) * W_boundary

其中 W_boundary 基于边界掩码的 EDT 计算，边界附近权重高。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """
    双通道边界预测损失（梯度过载保护版本）。

    语义分支：标准 BCEWithLogitsLoss
    边界分支：Focal Loss × clamp(EDT 边界权重图, 1.0, 4.0)

    改造说明：
    - alpha_boundary 调低至 0.1，截断边界损失的梯度过载
    - EDT 权重图在 loss 内部 clamp 到 [1.0, 4.0]，防止稀疏边界区域
      释放毁灭性的宏观梯度
    - 确保语义损失与 alpha*边界损失在同一量级

    pred[:, 0] 为语义 logits，target[:, 0] 为二值语义掩码 (0/1)。
    pred[:, 1] 为边界 logits，target[:, 1] 为二值边界掩码 (0/1)。
    weight[:, 0] 为 EDT 边界权重图。
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha_boundary: float = 0.1,
        alpha_focal: float = 0.75,
        seg_dice_weight: float = 0.0,
        weight_clamp_min: float = 1.0,
        weight_clamp_max: float = 4.0,
        eps: float = 1e-6,
        freeze_seg: bool = False,
        freeze_boundary: bool = False,
        peak_weight: float = 0.0,
        peak_logit: float = 2.0,
        hard_negative_weight: float = 0.0,
        hard_negative_logit: float = -1.5,
        ridge_weight: float = 0.0,
        ridge_positive_logit: float = 1.0,
        ridge_negative_logit: float = -1.5,
        ridge_core_threshold: float = 0.5,
        ridge_background_threshold: float = 0.05,
        ridge_tolerance: int = 1,
        ridge_ring_radius: int = 5,
        ridge_ring_weight: float = 1.0,
        ridge_mode: str = "absolute",
        ridge_margin: float = 1.5,
        center_weight: float = 1.0,
        center_gamma: float = 2.0,
    ):
        """
        Args:
            gamma: Focal Loss 聚焦参数。
            alpha_boundary: 边界损失总权重系数（推荐 0.05~0.2）。
            alpha_focal: Focal Loss 平衡参数（正样本权重）。
            weight_clamp_min: EDT 权重截断下限。
            weight_clamp_max: EDT 权重截断上限。
            eps: 数值稳定常数。
            freeze_seg: 冻结语义分支时为 True，跳过语义损失项。
            freeze_boundary: 冻结边界分支时为 True，跳过边界损失项。
            peak_weight: 边界正样本高置信度峰值约束权重。
            peak_logit: 正边界期望达到的最小 logit（2.0≈0.88 概率）。
            hard_negative_weight: 对负样本高响应区域的抑制权重。
            hard_negative_logit: 负样本 logit 超过该值时开始抑制。
            ridge_weight: 局部边界脊线约束权重。
            ridge_positive_logit: GT 核心附近的局部峰值最小 logit。
            ridge_negative_logit: GT 外环背景允许的最大 logit。
            ridge_core_threshold: 软边界中视为高质量核心的阈值。
            ridge_background_threshold: 软边界中视为背景的阈值。
            ridge_tolerance: 核心峰值允许的像素定位误差。
            ridge_ring_radius: 在 GT 核心外构造背景抑制环的半径。
            ridge_ring_weight: 外环背景项相对于核心命中项的权重。
            ridge_mode: `absolute` 使用独立明暗下限；`relative` 只约束局部对比。
            ridge_margin: relative 模式中边界峰值超过邻近背景的 logit 间隔。
            center_weight: 中心热图损失权重；仅在 pred/target 存在第 3 通道时生效。
            center_gamma: 中心热图 focal BCE 的聚焦参数。
        """
        super(BoundaryLoss, self).__init__()
        self.gamma = gamma
        self.alpha_boundary = alpha_boundary
        self.alpha_focal = alpha_focal
        self.seg_dice_weight = seg_dice_weight
        self.weight_clamp_min = weight_clamp_min
        self.weight_clamp_max = weight_clamp_max
        self.eps = eps
        self.freeze_seg = freeze_seg
        self.freeze_boundary = freeze_boundary
        self.peak_weight = peak_weight
        self.peak_logit = peak_logit
        self.hard_negative_weight = hard_negative_weight
        self.hard_negative_logit = hard_negative_logit
        self.ridge_weight = ridge_weight
        self.ridge_positive_logit = ridge_positive_logit
        self.ridge_negative_logit = ridge_negative_logit
        self.ridge_core_threshold = ridge_core_threshold
        self.ridge_background_threshold = ridge_background_threshold
        self.ridge_tolerance = int(ridge_tolerance)
        self.ridge_ring_radius = int(ridge_ring_radius)
        self.ridge_ring_weight = ridge_ring_weight
        self.ridge_mode = str(ridge_mode).lower()
        self.ridge_margin = ridge_margin
        self.center_weight = center_weight
        self.center_gamma = center_gamma

        if self.ridge_tolerance < 0 or self.ridge_ring_radius < 0:
            raise ValueError("ridge tolerance and ring radius must be non-negative")
        if self.ridge_mode not in {"absolute", "relative"}:
            raise ValueError("ridge_mode must be 'absolute' or 'relative'")

    def boundary_ridge_loss(self, boundary_logits, boundary_target):
        """Favor a confident, narrow ridge near GT without demanding exact pixels.

        Polygon-derived targets can be slightly displaced or doubled.  The positive
        term therefore asks for one local peak inside a small tolerance window,
        while the ring term suppresses elevated responses in the nearby true
        background.  Together they distinguish a useful thin ridge from diffuse
        haze without globally driving every negative pixel down.
        """
        target = boundary_target.detach().clamp(0.0, 1.0)
        core_mask = (target >= self.ridge_core_threshold).to(boundary_logits.dtype)
        core_mass = core_mask * target.pow(2)

        tolerance_kernel = 2 * self.ridge_tolerance + 1
        local_peak = F.max_pool2d(
            boundary_logits.unsqueeze(1),
            kernel_size=tolerance_kernel,
            stride=1,
            padding=self.ridge_tolerance,
        ).squeeze(1)
        ring_kernel = 2 * self.ridge_ring_radius + 1
        dilated_core = F.max_pool2d(
            core_mask.unsqueeze(1),
            kernel_size=ring_kernel,
            stride=1,
            padding=self.ridge_ring_radius,
        ).squeeze(1)
        ring_mask = (
            (dilated_core > 0)
            & (target <= self.ridge_background_threshold)
        ).to(boundary_logits.dtype)

        if self.ridge_mode == "relative":
            # 只对比每个 GT 核心附近的边界峰值和最强背景。
            # 两者同时加上任意常数时损失不变，因此无法通过全图
            # 统一变亮或变暗来降低损失。
            masked_ring_logits = torch.where(
                ring_mask > 0,
                boundary_logits,
                torch.full_like(boundary_logits, -1.0e4),
            )
            nearby_ring_peak = F.max_pool2d(
                masked_ring_logits.unsqueeze(1),
                kernel_size=ring_kernel,
                stride=1,
                padding=self.ridge_ring_radius,
            ).squeeze(1)
            nearby_ring_exists = F.max_pool2d(
                ring_mask.unsqueeze(1),
                kernel_size=ring_kernel,
                stride=1,
                padding=self.ridge_ring_radius,
            ).squeeze(1)
            valid_core_mass = core_mass * (nearby_ring_exists > 0).to(core_mass.dtype)
            contrast_hinge = F.relu(
                self.ridge_margin - local_peak + nearby_ring_peak
            ).pow(2)
            return (contrast_hinge * valid_core_mass).sum() / (
                valid_core_mass.sum().clamp_min(1.0)
            )

        missed_core = F.relu(self.ridge_positive_logit - local_peak).pow(2)
        core_loss = (missed_core * core_mass).sum() / core_mass.sum().clamp_min(1.0)
        raised_ring = F.relu(boundary_logits - self.ridge_negative_logit).pow(2)
        ring_loss = (raised_ring * ring_mask).sum() / ring_mask.sum().clamp_min(1.0)
        return core_loss + self.ridge_ring_weight * ring_loss

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: [B, 2/3, H, W] 模型输出
                  - pred[:, 0] 为语义 logits
                  - pred[:, 1] 为边界 logits
            target: [B, 2/3, H, W] 目标
                    - target[:, 0] 为二值语义掩码 (0/1)
                    - target[:, 1] 为二值边界掩码 (0/1)
            weight: [B, 1, H, W] 或 [B, H, W] EDT 边界权重图（可选）

        Returns:
            tuple: (total_loss, loss_seg_value, loss_boundary_value)

        注意：当 freeze_seg=True 时 loss_seg=0（无梯度），
              当 freeze_boundary=True 时 loss_boundary=0（无梯度）。
              冻结分支返回的 loss 值仍为 0.0 以保持接口一致。
        """
        seg_logits = pred[:, 0]           # [B, H, W] 语义 logits
        seg_target = target[:, 0]         # [B, H, W] 二值语义掩码
        boundary_logits = pred[:, 1]      # [B, H, W] 边界 logits
        boundary_target = target[:, 1]    # [B, H, W] 二值边界掩码

        # ---- 语义 BCE Loss ----
        if self.freeze_seg:
            loss_seg = torch.tensor(0.0, device=pred.device, requires_grad=False)
        else:
            loss_seg = F.binary_cross_entropy_with_logits(
                seg_logits, seg_target, reduction="mean"
            )
            if self.seg_dice_weight > 0:
                # 语义 Dice：块状低频结构，BCE 在类不平衡下易钝；
                # Dice 对掩码边界更敏感。两类平均 Dice (DSC_0+DSC_1)/2：
                # 原实现只算 class-1（铁素体前景）单类 Dice，在两类面积可比
                # （无显著小样本特性）时会对珠光体不公平，改为对称处理。
                seg_prob = torch.sigmoid(seg_logits)
                eps = self.eps
                dice0 = 1.0 - (
                    2.0 * ((1.0 - seg_prob) * (1.0 - seg_target)).sum() + eps
                ) / ((1.0 - seg_prob).sum() + (1.0 - seg_target).sum() + eps)
                dice1 = 1.0 - (
                    2.0 * (seg_prob * seg_target).sum() + eps
                ) / (seg_prob.sum() + seg_target.sum() + eps)
                dice = 0.5 * (dice0 + dice1)
                loss_seg = loss_seg + self.seg_dice_weight * dice

        # ---- 边界 Focal Loss × EDT 权重 ----
        if self.freeze_boundary:
            loss_boundary = torch.tensor(0.0, device=pred.device, requires_grad=False)
        else:
            bce_boundary = F.binary_cross_entropy_with_logits(
                boundary_logits, boundary_target, reduction="none"
            )  # [B, H, W]

            p_t = torch.exp(-bce_boundary)
            focal_weight = (1.0 - p_t) ** self.gamma
            alpha_t = (
                self.alpha_focal * boundary_target
                + (1.0 - self.alpha_focal) * (1.0 - boundary_target)
            )

            if weight is not None:
                # 权重图可能是 [B, 1, H, W] 或 [B, H, W]
                if weight.dim() == 4 and weight.shape[1] == 1:
                    w = weight[:, 0]  # [B, H, W]
                else:
                    w = weight  # [B, H, W]
                # 梯度过载保护：截断 EDT 权重到 [1.0, 4.0]
                w = torch.clamp(w, min=self.weight_clamp_min, max=self.weight_clamp_max)
                focal_loss = alpha_t * focal_weight * w * bce_boundary
            else:
                focal_loss = alpha_t * focal_weight * bce_boundary

            loss_boundary = focal_loss.mean()

            # 监督目标允许为 [0, 1] 的软边界。对目标质量最高的边界像素
            # 单独施加峰值约束，解决“面积大致正确但整张热力图发灰”的问题。
            if self.peak_weight > 0:
                positive_mass = boundary_target.detach().clamp(0.0, 1.0).pow(2)
                positive_den = positive_mass.sum().clamp_min(1.0)
                peak_hinge = F.relu(self.peak_logit - boundary_logits).pow(2)
                peak_loss = (peak_hinge * positive_mass).sum() / positive_den
                loss_boundary = loss_boundary + self.peak_weight * peak_loss

            # 只抑制明显高于背景的负样本，不对普通低概率背景施加额外压力，
            # 以减少圆斑/雾状响应而不牺牲细边界召回。
            if self.hard_negative_weight > 0:
                negative_mass = (boundary_target.detach() < 0.05).float()
                hard_negative = F.relu(
                    boundary_logits - self.hard_negative_logit
                ).pow(2)
                negative_den = negative_mass.sum().clamp_min(1.0)
                hard_negative_loss = (hard_negative * negative_mass).sum() / negative_den
                loss_boundary = loss_boundary + (
                    self.hard_negative_weight * hard_negative_loss
                )

            if self.ridge_weight > 0:
                loss_boundary = loss_boundary + self.ridge_weight * (
                    self.boundary_ridge_loss(boundary_logits, boundary_target)
                )

        # ---- 中心热图 focal BCE（可选第三通道） ----
        loss_center = torch.tensor(0.0, device=pred.device, requires_grad=False)
        if pred.shape[1] >= 3 and target.shape[1] >= 3 and self.center_weight > 0:
            center_logits = pred[:, 2]
            center_target = target[:, 2].clamp(0.0, 1.0)
            bce_center = F.binary_cross_entropy_with_logits(
                center_logits, center_target, reduction="none"
            )
            p_t_center = torch.exp(-bce_center)
            focal_center = (1.0 - p_t_center).pow(self.center_gamma) * bce_center
            # Gaussian 峰中心承载主要监督，背景仍保留但不额外放大。
            center_weight_map = 1.0 + 4.0 * center_target.detach().pow(2)
            loss_center = (focal_center * center_weight_map).mean()

        # ---- 总损失 ----
        total_loss = (
            loss_seg
            + self.alpha_boundary * loss_boundary
            + self.center_weight * loss_center
        )

        return total_loss, loss_seg.item(), loss_boundary.item()
