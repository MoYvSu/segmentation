# -*- coding: utf-8 -*-
"""
多任务半监督损失模块
====================
第二阶段半监督微调的无监督一致性损失。

双路解耦一致性：
1. 分类一致性：img_weak 的分类概率 -> 伪标签（置信度 > threshold）-> 与 img_strong_appearance 的预测计算 BCE
2. 回归几何一致性：img_weak 的距离场预测 -> 施加几何变换 T -> 与 img_strong_geometric 的预测计算 MSE

非侵入式设计：不修改第一阶段 utils/loss.py。
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset_semi import apply_transform_batch


def compute_stage2_unsupervised_loss(
    model: nn.Module,
    img_weak: torch.Tensor,
    img_strong_appearance: torch.Tensor,
    img_strong_geometric: torch.Tensor,
    T_list: List[Dict],
    confidence_threshold: float = 0.90,
    dist_weight: float = 10.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    计算第二阶段无监督一致性损失。

    前向传播计划（3 次编码器前向）：
    - 第 1 次：img_weak -> 获取分类概率 + 距离场预测（教师预测源）
    - 第 2 次：img_strong_appearance -> 获取分类 logits（学生外观增强预测）
    - 第 3 次：img_strong_geometric -> 获取距离场预测（学生几何增强预测）

    Args:
        model: SegmentationModel（冻结 encoder + decoder 主体，仅 cls_branch/reg_branch 可训练）
        img_weak: [B, 3, H, W] 无增强图像
        img_strong_appearance: [B, 3, H, W] 外观增强图像
        img_strong_geometric: [B, 3, H, W] 几何增强图像
        T_list: 长度为 B 的几何变换元数据列表
        confidence_threshold: 伪标签置信度阈值（默认 0.90）
        dist_weight: 距离场回归损失权重（默认 10.0）

    Returns:
        total_loss: 标量张量，总无监督损失
        loss_cls_val: float，分类一致性损失值
        loss_reg_val: float，回归几何一致性损失值
    """
    device = next(model.parameters()).device

    img_weak = img_weak.to(device)
    img_strong_appearance = img_strong_appearance.to(device)
    img_strong_geometric = img_strong_geometric.to(device)

    # ---- 第 1 次前向：img_weak ----
    out_weak = model(img_weak)  # [B, 2, H, W]
    cls_logits_weak = out_weak[:, 0]  # [B, H, W] 分类 logits
    dist_pred_weak = out_weak[:, 1]   # [B, H, W] 距离场预测 [0,1]

    # 分类概率
    cls_prob_weak = torch.sigmoid(cls_logits_weak)  # [B, H, W]

    # ================================================================
    # 分类一致性：img_weak 伪标签 vs img_strong_appearance 预测
    # ================================================================

    # 第 2 次前向：img_strong_appearance
    out_strong_app = model(img_strong_appearance)  # [B, 2, H, W]
    cls_logits_strong_app = out_strong_app[:, 0]   # [B, H, W]

    # 筛选高置信度区域作为伪标签
    # p > confidence_threshold -> 伪标签 1（高置信度铁素体）
    # p < (1 - confidence_threshold) -> 伪标签 0（高置信度珠光体）
    high_conf_mask = (
        (cls_prob_weak > confidence_threshold)
        | (cls_prob_weak < (1.0 - confidence_threshold))
    )  # [B, H, W] bool

    # 生成伪标签
    pseudo_label = (cls_prob_weak > 0.5).float()  # [B, H, W]

    # 计算区域 BCE 损失（仅在高置信度区域）
    if high_conf_mask.sum() > 0:
        bce_map = F.binary_cross_entropy_with_logits(
            cls_logits_strong_app, pseudo_label, reduction="none"
        )  # [B, H, W]
        loss_cls = (bce_map * high_conf_mask.float()).sum() / (
            high_conf_mask.float().sum() + 1e-7
        )
    else:
        # 无高置信度区域时，损失为 0（但仍保持梯度图连接）
        loss_cls = cls_logits_strong_app.sum() * 0.0

    # ================================================================
    # 回归几何一致性：img_weak 距离场 -> 施加 T -> vs img_strong_geometric 预测
    # ================================================================

    # 第 3 次前向：img_strong_geometric
    out_strong_geo = model(img_strong_geometric)  # [B, 2, H, W]
    dist_pred_strong_geo = out_strong_geo[:, 1]   # [B, H, W]

    # 对 img_weak 的距离场预测施加相同的几何变换 T
    # 使其对齐到 img_strong_geometric 的坐标系
    dist_pred_weak_expanded = dist_pred_weak.unsqueeze(1)  # [B, 1, H, W]
    dist_pred_weak_transformed = apply_transform_batch(
        dist_pred_weak_expanded, T_list
    )  # [B, 1, H, W]
    dist_pred_weak_transformed = dist_pred_weak_transformed.squeeze(1)  # [B, H, W]

    # 计算像素级 MSE 损失
    loss_reg = F.mse_loss(dist_pred_strong_geo, dist_pred_weak_transformed)

    # ---- 总无监督损失 ----
    total_loss = loss_cls + dist_weight * loss_reg

    return total_loss, loss_cls.item(), loss_reg.item()