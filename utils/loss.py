# -*- coding: utf-8 -*-
"""
Focal Loss + 距离场双任务损失
=============================
手工构建标准 Focal Loss（分类分支）+ MSE（回归分支）。

任务设计：
- 通道 0（分类分支）：原始 logits，送入 Focal Loss（gamma=2.0, alpha=0.95）
- 通道 1（回归分支）：经 Sigmoid 的预测距离场，送入 MSELoss

双任务刚性绑定 = FocalLoss + 10.0 * MSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDistanceFieldLoss(nn.Module):
    """
    Focal Loss + 距离场 MSE 双任务损失。

    pred[:, 0] 为分类 logits，target[:, 0] 为二值掩码 (0/1)。
    pred[:, 1] 为经 Sigmoid 的预测距离场 [0,1]，target[:, 1] 为真实距离场 [0,1]。

    分类分支（Focal Loss）：
    - gamma=2.0：聚焦参数，压制易分类样本梯度
    - alpha=0.95：平衡参数，放大稀疏铁素体正样本损失
    - 数值稳定实现：利用 BCEWithLogits 底层算子

    回归分支：
    - loss_dist = MSE(pred[:, 1], target[:, 1])

    总损失 = FocalLoss + 10.0 * loss_dist
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.95, eps: float = 1e-6):
        """
        Args:
            gamma: Focal Loss 聚焦参数，越大越聚焦于难分类样本。
            alpha: Focal Loss 平衡参数，正样本权重。
            eps: 数值稳定常数。
        """
        super(FocalDistanceFieldLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 2, H, W] 模型输出
                  - pred[:, 0] 为分类 logits
                  - pred[:, 1] 为经 Sigmoid 的距离场预测
            target: [B, 2, H, W] 目标
                    - target[:, 0] 为二值掩码 (0/1)
                    - target[:, 1] 为归一化距离场 [0,1]

        Returns:
            tuple: (total_loss, loss_seg_value, loss_dist_value)
        """
        seg_logits = pred[:, 0]   # [B, H, W] 分类 logits
        seg_target = target[:, 0]  # [B, H, W] 二值掩码

        # ---- Focal Loss（数值稳定实现）----
        # 利用 BCEWithLogits 计算 -log(p_t)，避免手动 Sigmoid 的数值溢出
        bce = F.binary_cross_entropy_with_logits(
            seg_logits, seg_target, reduction="none"
        )  # [B, H, W]，值 = -log(p_t)

        # p_t = 正确类别的预测概率
        p_t = torch.exp(-bce)  # 因为 bce = -log(p_t)

        # 聚焦权重：(1 - p_t)^gamma，难分类样本权重接近 1，易分类样本权重趋近 0
        focal_weight = (1.0 - p_t) ** self.gamma

        # alpha 平衡：正样本 alpha=0.95，负样本 (1-alpha)=0.05
        alpha_t = self.alpha * seg_target + (1.0 - self.alpha) * (1.0 - seg_target)

        loss_focal = (alpha_t * focal_weight * bce).mean()

        # ---- 距离场回归损失 ----
        loss_dist = self.mse(pred[:, 1], target[:, 1])

        # ---- 双任务线性加权和 ----
        total_loss = loss_focal + 10.0 * loss_dist

        return total_loss, loss_focal.item(), loss_dist.item()