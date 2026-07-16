# -*- coding: utf-8 -*-
"""
双轨空间权重 Focal Loss + TV 平滑 + 边界加权向量场回归 + 不确定性多任务平衡
=============================================================================
向量场版本（Spatial Embedding）：

1. 双轨空间权重
   - W_space_internal（基于 EDT）：晶粒中心权重 1.0，晶界附近权重 0.3
     * 用于 loss_focal 和 loss_tv，维持晶粒内部平滑，抑制过分割
   - W_space_boundary（基于 EDT 反转）：晶界权重 10.0，中心权重 1.0
     * 用于 loss_vec，强制模型在晶界处精确回归向量场，抑制欠分割
2. 空间全变差平滑（TV Loss），乘以 W_space_internal 使平滑约束集中于晶粒内部
3. 边界加权向量场回归：element-wise MSE × W_space_boundary
4. 同方差不确定性加权（Homoscedastic Uncertainty, Kendall et al. 2018）
   - 注册 log_var_cls / log_var_vec 两个可学习参数
   - 自动平衡分类与回归任务的梯度贡献

任务设计：
- 通道 0（分类分支）：原始 logits，送入加权 Focal Loss
- 通道 1-2（回归分支）：经 Tanh 的向量场预测 [-1,1] (Vx, Vy)，送入边界加权 MSE

总损失 = precision_cls × (loss_focal + tv_weight × loss_tv)
       + precision_vec × loss_vec
       + log_var_cls + log_var_vec
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class FocalDistanceFieldLoss(nn.Module):
    """
    双轨空间权重 Focal Loss + TV 平滑 + 边界加权向量场 MSE + 不确定性多任务平衡。

    pred[:, 0] 为分类 logits，target[:, 0] 为二值掩码 (0/1)。
    pred[:, 1:3] 为经 Tanh 的向量场预测 [-1,1] (Vx, Vy)，
    target[:, 1:3] 为真实向量场 [-1,1] (Vx, Vy)。

    分类分支（加权 Focal Loss）：
    - gamma=2.0：聚焦参数
    - alpha=0.95：平衡参数
    - W_space_internal：基于 EDT 的空间权重，晶粒内部高权，晶界低权

    TV 平滑分支：
    - 对 Sigmoid 概率图计算全变差，乘以 W_space_internal

    回归分支（边界加权向量场 MSE）：
    - W_space_boundary = 1.0 + 9.0 × (1.0 - dist_norm_edt)
    - 晶界处权重 10.0，中心权重 1.0
    - loss_vec = mean( (pred - target)^2 × W_space_boundary )

    不确定性加权：
    - log_var_cls / log_var_vec 初始化为 0.0
    - total = precision_cls × (focal + tv_weight × tv) + precision_vec × vec + log_var_cls + log_var_vec
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.95,
        eps: float = 1e-6,
        tv_weight: float = 0.05,
        edt_weight_floor: float = 0.3,
        edt_scale_factor: float = 10.0,
    ):
        """
        Args:
            gamma: Focal Loss 聚焦参数。
            alpha: Focal Loss 平衡参数。
            eps: 数值稳定常数。
            tv_weight: TV 平滑损失权重系数（默认 0.05）。
            edt_weight_floor: W_space_internal 权重下限（默认 0.3）。
            edt_scale_factor: EDT 归一化缩放因子（默认 10.0）。
        """
        super(FocalDistanceFieldLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.tv_weight = tv_weight
        self.edt_weight_floor = edt_weight_floor
        self.edt_scale_factor = edt_scale_factor

        # 同方差不确定性参数（Kendall et al. 2018）
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self.log_var_vec = nn.Parameter(torch.tensor(0.0))

    def _compute_w_space_internal(self, seg_target: torch.Tensor) -> torch.Tensor:
        """
        基于 EDT 计算内部空间权重矩阵 W_space_internal。

        晶粒中心 -> 权重 1.0，晶界附近 -> 权重 edt_weight_floor。
        用于 loss_focal 和 loss_tv，维持晶粒内部平滑，抑制过分割。
        """
        B, H, W = seg_target.shape
        device = seg_target.device
        dtype = seg_target.dtype

        masks_np = seg_target.detach().cpu().numpy()
        w_space = np.full((B, H, W), self.edt_weight_floor, dtype=np.float32)

        for i in range(B):
            mask = masks_np[i] > 0.5
            if mask.sum() == 0:
                continue
            dist = distance_transform_edt(mask)
            dist_norm = dist / (dist + self.edt_scale_factor)
            dist_norm = np.maximum(dist_norm, self.edt_weight_floor)
            w_space[i] = dist_norm

        return torch.from_numpy(w_space).to(device=device, dtype=dtype)

    def _compute_w_space_boundary(self, seg_target: torch.Tensor) -> torch.Tensor:
        """
        基于 EDT 计算边界空间权重矩阵 W_space_boundary。

        公式: W_space_boundary = 1.0 + 9.0 * (1.0 - dist_norm_edt)
        - 晶界处（dist_norm -> 0）-> 权重 10.0
        - 晶粒中心（dist_norm -> 1）-> 权重 1.0

        用于 loss_vec，强制模型在晶界处精确回归向量场。

        Args:
            seg_target: [B, H, W] 二值掩码 (0=珠光体, 1=铁素体)

        Returns:
            W_space_boundary: [B, H, W] 权重矩阵，范围 [1.0, 10.0]
        """
        B, H, W = seg_target.shape
        device = seg_target.device
        dtype = seg_target.dtype

        masks_np = seg_target.detach().cpu().numpy()
        w_boundary = np.ones((B, H, W), dtype=np.float32)

        for i in range(B):
            mask = masks_np[i] > 0.5
            if mask.sum() == 0:
                continue
            dist = distance_transform_edt(mask)
            dist_norm = dist / (dist + self.edt_scale_factor)
            # 珠光体区域 dist_norm=0 -> boundary weight = 10.0
            # 铁素体中心 dist_norm->1 -> boundary weight = 1.0
            w_boundary[i] = 1.0 + 9.0 * (1.0 - dist_norm)

        return torch.from_numpy(w_boundary).to(device=device, dtype=dtype)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 3, H, W] 模型输出
                  - pred[:, 0] 为分类 logits
                  - pred[:, 1:3] 为经 Tanh 的向量场预测 [-1,1]
            target: [B, 3, H, W] 目标
                    - target[:, 0] 为二值掩码 (0/1)
                    - target[:, 1:3] 为真实向量场 [-1,1]

        Returns:
            tuple: (total_loss, loss_seg_value, loss_vec_value)
        """
        seg_logits = pred[:, 0]      # [B, H, W] 分类 logits
        seg_target = target[:, 0]    # [B, H, W] 二值掩码
        vec_target = target[:, 1:3]  # [B, 2, H, W] 真实向量场 [-1,1]
        vec_pred = pred[:, 1:3]      # [B, 2, H, W] 预测向量场 [-1,1]

        # ---- 双轨空间权重 ----
        w_space_internal = self._compute_w_space_internal(seg_target)   # [B, H, W]
        w_space_boundary = self._compute_w_space_boundary(seg_target)   # [B, H, W]

        # ---- 加权 Focal Loss（使用 W_space_internal）----
        bce = F.binary_cross_entropy_with_logits(
            seg_logits, seg_target, reduction="none"
        )
        p_t = torch.exp(-bce)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha * seg_target + (1.0 - self.alpha) * (1.0 - seg_target)
        loss_focal = (alpha_t * focal_weight * w_space_internal * bce).mean()

        # ---- TV 平滑损失（使用 W_space_internal）----
        prob = torch.sigmoid(seg_logits)
        diff_h = torch.abs(prob[:, 1:, :] - prob[:, :-1, :])
        diff_w = torch.abs(prob[:, :, 1:] - prob[:, :, :-1])
        w_h = w_space_internal[:, :-1, :]
        w_w = w_space_internal[:, :, :-1]
        loss_tv = (diff_h * w_h).mean() + (diff_w * w_w).mean()

        # ---- 边界加权向量场回归损失（使用 W_space_boundary）----
        # element-wise squared error × boundary weight
        sq_error = (vec_pred - vec_target) ** 2  # [B, 2, H, W]
        # 对通道维取均值（Vx 和 Vy 的平均），对空间维加权
        sq_error_mean = sq_error.mean(dim=1)  # [B, H, W]
        loss_vec = (sq_error_mean * w_space_boundary).mean()

        # ---- 不确定性加权 ----
        precision_cls = torch.exp(-self.log_var_cls)
        precision_vec = torch.exp(-self.log_var_vec)

        total_loss = (
            precision_cls * (loss_focal + self.tv_weight * loss_tv)
            + precision_vec * loss_vec
            + self.log_var_cls
            + self.log_var_vec
        )

        return total_loss, loss_focal.item(), loss_vec.item()