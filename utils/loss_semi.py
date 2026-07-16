# -*- coding: utf-8 -*-
"""
多任务半监督损失模块（向量场版本）
================================
第二阶段半监督微调的无监督一致性损失。

双路解耦一致性：
1. 分类一致性：img_weak 的分类概率 -> 伪标签（置信度 > threshold）-> 与 img_strong_appearance 的预测计算 BCE
2. 回归几何一致性：img_weak 的向量场预测 -> 施加几何变换 T（含分量变换）-> 与 img_strong_geometric 的预测计算 MSE

优化：冻结层前向 no_grad 化（方案 A）
- encoder + decoder 冻结部分（lateral_convs / residual_blocks / semantic_head）在 torch.no_grad() 下运行
- 仅 cls_branch + reg_branch 在梯度图内运行
- 减少 3 次无监督前向中冻结层的中间激活值内存存储和反向传播开销
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset_semi import apply_transform_batch


def _forward_frozen_to_semantic(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    冻结层前向：encoder -> lateral_convs -> residual_blocks -> semantic_head
    在 torch.no_grad() 下运行，不保留中间激活值。

    Args:
        model: SegmentationModel
        x: [B, 3, H, W] 输入图像

    Returns:
        semantic: [B, fpn_channels, H/4, W/4] 语义特征图
    """
    decoder = model.decoder
    with torch.no_grad():
        # Encoder 前向（冻结）
        features = model.encoder(x)

        # Decoder 冻结部分前向：lateral_convs + residual_blocks + top-down + semantic_head
        laterals = []
        for i in range(decoder.num_stages):
            proj = decoder.lateral_convs[i](features[i])
            laterals.append(decoder.residual_blocks[i](proj))

        top_down = laterals[-1]
        for i in range(decoder.num_stages - 2, -1, -1):
            top_down = F.interpolate(
                top_down,
                size=laterals[i].shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            top_down = top_down + laterals[i]

        semantic = decoder.semantic_head(top_down)

    return semantic


def _forward_heads(model: nn.Module, semantic: torch.Tensor) -> torch.Tensor:
    """
    可训练头前向：cls_branch + reg_branch（在梯度图内运行）。

    Args:
        model: SegmentationModel
        semantic: [B, fpn_channels, H/4, W/4] 语义特征图

    Returns:
        output: [B, 3, H/4, W/4]
            - output[:, 0] 为分类 logits
            - output[:, 1:3] 为经 Tanh 的向量场预测 [-1,1] (Vx, Vy)
    """
    decoder = model.decoder
    seg_logits = decoder.cls_branch(semantic)
    vec_pred = decoder.reg_branch(semantic)
    output = torch.cat([seg_logits, vec_pred], dim=1)
    return output


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
    计算第二阶段无监督一致性损失（向量场版本）。

    优化后的前向传播计划（3 次编码器前向，但冻结层在 no_grad 下）：
    - 第 1 次：img_weak -> 冻结层 no_grad -> 可训练头 -> 分类概率 + 向量场（教师预测源）
    - 第 2 次：img_strong_appearance -> 冻结层 no_grad -> 可训练头 -> 分类 logits（学生外观增强）
    - 第 3 次：img_strong_geometric -> 冻结层 no_grad -> 可训练头 -> 向量场预测（学生几何增强）

    Args:
        model: SegmentationModel（冻结 encoder + decoder 主体，仅 cls_branch/reg_branch 可训练）
        img_weak: [B, 3, H, W] 无增强图像
        img_strong_appearance: [B, 3, H, W] 外观增强图像
        img_strong_geometric: [B, 3, H, W] 几何增强图像
        T_list: 长度为 B 的几何变换元数据列表
        confidence_threshold: 伪标签置信度阈值（默认 0.90）
        dist_weight: 向量场回归损失权重（默认 10.0）

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
    # 冻结层 no_grad + 可训练头梯度
    semantic_weak = _forward_frozen_to_semantic(model, img_weak)
    # detach semantic 以切断冻结部分的梯度图（虽然 no_grad 已经不记录，但确保安全）
    out_weak = _forward_heads(model, semantic_weak.detach())
    cls_logits_weak = out_weak[:, 0]       # [B, H, W] 分类 logits
    vec_pred_weak = out_weak[:, 1:3]       # [B, 2, H, W] 向量场预测 [-1,1]

    # 分类概率
    cls_prob_weak = torch.sigmoid(cls_logits_weak)  # [B, H, W]

    # ================================================================
    # 分类一致性：img_weak 伪标签 vs img_strong_appearance 预测
    # ================================================================

    # 第 2 次前向：img_strong_appearance
    semantic_strong_app = _forward_frozen_to_semantic(model, img_strong_appearance)
    out_strong_app = _forward_heads(model, semantic_strong_app.detach())
    cls_logits_strong_app = out_strong_app[:, 0]   # [B, H, W]

    # 筛选高置信度区域作为伪标签
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
    # 回归几何一致性：img_weak 向量场 -> 施加 T -> vs img_strong_geometric 预测
    # ================================================================

    # 第 3 次前向：img_strong_geometric
    semantic_strong_geo = _forward_frozen_to_semantic(model, img_strong_geometric)
    out_strong_geo = _forward_heads(model, semantic_strong_geo.detach())
    vec_pred_strong_geo = out_strong_geo[:, 1:3]   # [B, 2, H, W]

    # 对 img_weak 的向量场预测施加相同的几何变换 T（含分量变换）
    vec_pred_weak_transformed = apply_transform_batch(
        vec_pred_weak, T_list, is_vector=True
    )  # [B, 2, H, W]

    # 计算像素级 MSE 损失
    loss_reg = F.mse_loss(vec_pred_strong_geo, vec_pred_weak_transformed)

    # ---- 总无监督损失 ----
    total_loss = loss_cls + dist_weight * loss_reg

    return total_loss, loss_cls.item(), loss_reg.item()