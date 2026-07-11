# -*- coding: utf-8 -*-
"""
自适应距离场双任务损失
======================
批次自适应反比加权 BCE + 二分类 Dice + 连续距离场 MSE 回归。

任务设计：
- 通道 0（分类分支）：原始 logits，送入动态加权 BCE + Dice
- 通道 1（回归分支）：经 Sigmoid 的预测距离场，送入 MSELoss

分类总损失 = loss_bce + 2.0 * loss_dice
双任务刚性绑定 = loss_seg + 5.0 * loss_dist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveDistanceFieldLoss(nn.Module):
    """
    批次自适应反比加权 + Dice 混合的双任务损失。

    pred[:, 0] 为分类 logits，target[:, 0] 为二值掩码 (0/1)。
    pred[:, 1] 为经 Sigmoid 的预测距离场 [0,1]，target[:, 1] 为真实距离场 [0,1]。

    分类分支：
    - 动态计算当前 Batch 内正负样本比例，生成 pos_weight 因子
    - pos_weight = clamp(num_neg / num_pos, 1.0, 15.0)
    - loss_seg = loss_bce(pos_weight) + 2.0 * loss_dice

    回归分支：
    - loss_dist = MSE(pred[:, 1], target[:, 1])

    总损失 = loss_seg + 5.0 * loss_dist
    """

    def __init__(self, eps=1e-6):
        """
        Args:
            eps: 数值稳定常数，防止分母为零。
        """
        super(AdaptiveDistanceFieldLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

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
        # pred[:, 0]: 分类 Logits, target[:, 0]: 二值掩码 (0/1)
        # pred[:, 1]: 经过 Sigmoid 的预测距离场, target[:, 1]: 真实距离场 (0~1)

        # 1. 动态计算当前 Batch 的自适应正样本权重
        num_neg = (target[:, 0] == 0).sum().float()
        num_pos = (target[:, 0] == 1).sum().float()

        # 计算背景与正样本的像素比率，以对抗样本间的面积比例波动
        pos_weight_factor = (num_neg + self.eps) / (num_pos + self.eps)

        # 实施数值截断限制，防止极端稀疏样本下权重过大引发数值不稳定
        pos_weight_factor = torch.clamp(pos_weight_factor, min=1.0, max=15.0)

        # 2. 计算动态加权二元交叉熵损失
        loss_bce = F.binary_cross_entropy_with_logits(
            pred[:, 0],
            target[:, 0],
            pos_weight=pos_weight_factor.to(pred.device)
        )

        # 3. 计算区域尺度不变的二分类 Dice 损失
        pred_seg_prob = torch.sigmoid(pred[:, 0])
        intersection = (pred_seg_prob * target[:, 0]).sum()
        union = pred_seg_prob.sum() + target[:, 0].sum()
        loss_dice = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)

        # 聚合分类分支总损失
        loss_seg = loss_bce + 2.0 * loss_dice

        # 4. 计算回归分支损失
        loss_dist = self.mse(pred[:, 1], target[:, 1])

        # 双任务刚性绑定
        total_loss = loss_seg + 5.0 * loss_dist

        return total_loss, loss_seg.item(), loss_dist.item()