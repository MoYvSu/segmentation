# -*- coding: utf-8 -*-
"""
距离场双任务损失
================
二分类掩码 BCE 损失 + 连续距离场 MSE 损失的静态加权组合。

任务设计：
- 通道 0（分类分支）：原始 logits，送入 BCEWithLogitsLoss
- 通道 1（回归分支）：经 Sigmoid 的预测距离场，送入 MSELoss

静态权重 1:5（回归损失通常数值偏小，给予更大权重）。
"""

import torch
import torch.nn as nn


class DistanceFieldLoss(nn.Module):
    """
    双任务静态混合损失：BCE 分类 + MSE 距离场回归。

    pred[:, 0] 为分类 logits，target[:, 0] 为二值掩码 (0/1)。
    pred[:, 1] 为经 Sigmoid 的预测距离场 [0,1]，target[:, 1] 为真实距离场 [0,1]。

    总损失 = loss_seg + dist_weight * loss_dist
    """

    def __init__(self, dist_weight: float = 5.0):
        """
        Args:
            dist_weight: 距离场回归损失的权重（默认 5.0）。
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.dist_weight = dist_weight

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
            dict: {
                "total": 总损失,
                "seg": 分类损失,
                "dist": 距离场回归损失,
            }
        """
        loss_seg = self.bce(pred[:, 0], target[:, 0])
        loss_dist = self.mse(pred[:, 1], target[:, 1])

        total = loss_seg + self.dist_weight * loss_dist

        return {
            "total": total,
            "seg": loss_seg.item(),
            "dist": loss_dist.item(),
        }