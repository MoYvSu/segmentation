# -*- coding: utf-8 -*-
"""
边界预测损失函数
================
双通道边界预测损失：
  - 通道 0（语义）：BCEWithLogitsLoss
  - 通道 1（边界）：Focal Loss × W_boundary（EDT 权重调制）

L_sup = BCEWithLogitsLoss(pred[:,0], target[:,0])
      + alpha * FocalLoss(pred[:,1], target[:,1]) * W_boundary

其中 W_boundary 基于边界掩码的 EDT 计算，边界附近权重高。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """
    双通道边界预测损失。

    语义分支：标准 BCEWithLogitsLoss
    边界分支：Focal Loss × EDT 边界权重图

    pred[:, 0] 为语义 logits，target[:, 0] 为二值语义掩码 (0/1)。
    pred[:, 1] 为边界 logits，target[:, 1] 为二值边界掩码 (0/1)。
    weight[:, 0] 为 EDT 边界权重图。
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha_boundary: float = 1.0,
        alpha_focal: float = 0.75,
        eps: float = 1e-6,
    ):
        """
        Args:
            gamma: Focal Loss 聚焦参数。
            alpha_boundary: 边界损失总权重系数。
            alpha_focal: Focal Loss 平衡参数（正样本权重）。
            eps: 数值稳定常数。
        """
        super(BoundaryLoss, self).__init__()
        self.gamma = gamma
        self.alpha_boundary = alpha_boundary
        self.alpha_focal = alpha_focal
        self.eps = eps

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: [B, 2, H, W] 模型输出
                  - pred[:, 0] 为语义 logits
                  - pred[:, 1] 为边界 logits
            target: [B, 2, H, W] 目标
                    - target[:, 0] 为二值语义掩码 (0/1)
                    - target[:, 1] 为二值边界掩码 (0/1)
            weight: [B, 1, H, W] 或 [B, H, W] EDT 边界权重图（可选）

        Returns:
            tuple: (total_loss, loss_seg_value, loss_boundary_value)
        """
        seg_logits = pred[:, 0]           # [B, H, W] 语义 logits
        seg_target = target[:, 0]         # [B, H, W] 二值语义掩码
        boundary_logits = pred[:, 1]      # [B, H, W] 边界 logits
        boundary_target = target[:, 1]    # [B, H, W] 二值边界掩码

        # ---- 语义 BCE Loss ----
        loss_seg = F.binary_cross_entropy_with_logits(
            seg_logits, seg_target, reduction="mean"
        )

        # ---- 边界 Focal Loss × EDT 权重 ----
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
            loss_boundary = (alpha_t * focal_weight * w * bce_boundary).mean()
        else:
            loss_boundary = (alpha_t * focal_weight * bce_boundary).mean()

        # ---- 总损失 ----
        total_loss = loss_seg + self.alpha_boundary * loss_boundary

        return total_loss, loss_seg.item(), loss_boundary.item()