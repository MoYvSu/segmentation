# -*- coding: utf-8 -*-
"""
加权 Focal Loss + TV 平滑 + 距离场双任务损失
=============================================
在 Focal Loss 基础上引入：
1. 基于欧氏距离变换（EDT）的内部高权惩罚 W_space
   - 晶粒中心（深水区）权重趋近 1.0，晶界附近权重趋近下限
   - 珠光体区域权重 = edt_weight_floor（非零，保证监督不缺失）
2. 空间全变差平滑（TV Loss），乘以 W_space 使平滑约束集中于晶粒内部
3. 距离场 MSE 回归（不变）

任务设计：
- 通道 0（分类分支）：原始 logits，送入加权 Focal Loss
- 通道 1（回归分支）：经 Sigmoid 的预测距离场，送入 MSELoss

总损失 = L_weighted_focal + tv_weight * L_tv + 10.0 * L_mse
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class FocalDistanceFieldLoss(nn.Module):
    """
    加权 Focal Loss + TV 平滑 + 距离场 MSE 双任务损失。

    pred[:, 0] 为分类 logits，target[:, 0] 为二值掩码 (0/1)。
    pred[:, 1] 为经 Sigmoid 的预测距离场 [0,1]，target[:, 1] 为真实距离场 [0,1]。

    分类分支（加权 Focal Loss）：
    - gamma=2.0：聚焦参数，压制易分类样本梯度
    - alpha=0.95：平衡参数，放大稀疏铁素体正样本损失
    - W_space：基于 EDT 的空间权重，晶粒内部高权，晶界低权
    - 数值稳定实现：利用 BCEWithLogits 底层算子

    TV 平滑分支：
    - 对 Sigmoid 概率图计算全变差（相邻像素绝对差分）
    - 乘以 W_space，使平滑约束集中于晶粒内部，不模糊真实晶界

    回归分支：
    - loss_dist = MSE(pred[:, 1], target[:, 1])

    总损失 = L_weighted_focal + tv_weight * L_tv + 10.0 * loss_dist
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
            gamma: Focal Loss 聚焦参数，越大越聚焦于难分类样本。
            alpha: Focal Loss 平衡参数，正样本权重。
            eps: 数值稳定常数。
            tv_weight: TV 平滑损失权重系数（默认 0.05）。
            edt_weight_floor: W_space 权重下限，确保珠光体区域仍有非零监督（默认 0.3）。
            edt_scale_factor: EDT 归一化缩放因子（与距离场一致的 scale_factor，默认 10.0）。
        """
        super(FocalDistanceFieldLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.tv_weight = tv_weight
        self.edt_weight_floor = edt_weight_floor
        self.edt_scale_factor = edt_scale_factor
        self.mse = nn.MSELoss()

    def _compute_w_space(self, seg_target: torch.Tensor) -> torch.Tensor:
        """
        基于 EDT 计算空间权重矩阵 W_space。

        物理效果：
        - 铁素体晶粒中心（深水区）→ 权重趋近 1.0
        - 铁素体晶界附近 → 权重趋近 edt_weight_floor
        - 珠光体区域 → 权重 = edt_weight_floor（非零，保证监督不缺失）

        流程：
        1. 对铁素体掩码执行 distance_transform_edt，得到每个铁素体像素
           到最近非铁素体像素的欧氏距离
        2. 非线性归一化: dist / (dist + scale_factor) → [0, 1]
        3. 应用权重下限: max(dist_norm, edt_weight_floor)

        Args:
            seg_target: [B, H, W] 二值掩码 (0=珠光体, 1=铁素体)

        Returns:
            W_space: [B, H, W] 权重矩阵，范围 [edt_weight_floor, 1.0]
        """
        B, H, W = seg_target.shape
        device = seg_target.device
        dtype = seg_target.dtype

        # 转移到 CPU numpy 计算 EDT（scipy 不支持 GPU）
        masks_np = seg_target.detach().cpu().numpy()

        w_space = np.full((B, H, W), self.edt_weight_floor, dtype=np.float32)

        for i in range(B):
            mask = masks_np[i] > 0.5  # 铁素体区域
            if mask.sum() == 0:
                # 无铁素体区域，全部使用权重下限
                continue

            # 计算铁素体像素到最近非铁素体像素的欧氏距离
            dist = distance_transform_edt(mask)

            # 非线性归一化: dist / (dist + scale_factor)
            dist_norm = dist / (dist + self.edt_scale_factor)

            # 应用权重下限
            dist_norm = np.maximum(dist_norm, self.edt_weight_floor)

            w_space[i] = dist_norm

        return torch.from_numpy(w_space).to(device=device, dtype=dtype)

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
                   - total_loss = weighted_focal + tv_weight * tv + 10.0 * mse
                   - loss_seg_value 为 weighted_focal（不含 TV，用于监控）
                   - loss_dist_value 为 MSE（用于监控）
        """
        seg_logits = pred[:, 0]    # [B, H, W] 分类 logits
        seg_target = target[:, 0]  # [B, H, W] 二值掩码

        # ---- 计算空间权重矩阵 W_space ----
        w_space = self._compute_w_space(seg_target)  # [B, H, W]

        # ---- 加权 Focal Loss（数值稳定实现）----
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

        # 像素级加权 Focal Loss：focal_weight * alpha_t * W_space * bce
        loss_focal = (alpha_t * focal_weight * w_space * bce).mean()

        # ---- TV 平滑损失（加权集中于晶粒内部）----
        prob = torch.sigmoid(seg_logits)  # [B, H, W] 概率图

        # 高度方向相邻像素绝对差分
        diff_h = torch.abs(prob[:, 1:, :] - prob[:, :-1, :])  # [B, H-1, W]
        # 宽度方向相邻像素绝对差分
        diff_w = torch.abs(prob[:, :, 1:] - prob[:, :, :-1])  # [B, H, W-1]

        # W_space 对齐到差分尺寸（取每对像素中第一个的权重）
        w_h = w_space[:, :-1, :]  # [B, H-1, W]
        w_w = w_space[:, :, :-1]  # [B, H, W-1]

        # 加权 TV 损失
        loss_tv = (diff_h * w_h).mean() + (diff_w * w_w).mean()

        # ---- 距离场回归损失 ----
        loss_dist = self.mse(pred[:, 1], target[:, 1])

        # ---- 联合总损失 ----
        total_loss = loss_focal + self.tv_weight * loss_tv + 10.0 * loss_dist

        return total_loss, loss_focal.item(), loss_dist.item()