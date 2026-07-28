# -*- coding: utf-8 -*-
"""
独立双 FPN 解码头（语义 + 边界各自独立）- 边界预测版本
=====================================================
完全随机初始化的特征金字塔解码头，禁止加载任何 SAM 2 原生 Mask Decoder 权重。

设计：
1. 语义分支和边界分支各自拥有独立的 FPN backbone（lateral_convs + residual_blocks + top-down fusion）。
2. 每个 FPN backbone 独立从 encoder 的四个尺度特征中提取信息，消除语义/边界梯度干扰。
3. 每个 FPN backbone 后接独立的输出头（两层 3x3 Conv + Norm + ReLU + Dropout + 1x1 Conv）。
4. 使用 GroupNorm 替代 BatchNorm（对小 batch size 更稳定，不依赖 batch 统计）。

输入约定：features = [feat_s1, feat_s2, feat_s3, feat_s4]
    - feat_s1: 最高分辨率, 112ch
    - feat_s2: 224ch
    - feat_s3: 448ch
    - feat_s4: 最低分辨率, 896ch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def _make_group_norm(channels: int) -> nn.GroupNorm:
    """创建 GroupNorm，num_groups = min(32, channels) 且保证整除。"""
    num_groups = min(32, channels)
    while channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels)


class ResidualBlock(nn.Module):
    """
    标准残差卷积块：两层 3*3 Conv + GroupNorm + ReLU + Identity Shortcut。
    """

    def __init__(self, channels: int, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = _make_group_norm

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_layer(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out


class FPNBackbone(nn.Module):
    """
    FPN backbone：横向投影 + 残差块 + 自上而下融合。

    将 encoder 的四个尺度特征统一对齐到 fpn_channels 维，
    经过残差块增强后，自上而下融合为单一特征图。
    """

    def __init__(
        self,
        in_channels: List[int],
        fpn_channels: int,
        num_stages: int = 4,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = _make_group_norm

        self.num_stages = num_stages

        # 1*1 横向连接卷积：将各 Stage 通道对齐到 fpn_channels
        self.lateral_convs = nn.ModuleList()
        for ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(ch, fpn_channels, kernel_size=1, bias=False)
            )

        # 残差块：每个尺度投影后挂载一个 ResidualBlock
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_stages):
            self.residual_blocks.append(
                ResidualBlock(fpn_channels, norm_layer=norm_layer)
            )

    def forward(self, features):
        """
        前向传播：横向投影 + 残差块 + 自上而下融合。

        Args:
            features: [feat_s1, feat_s2, feat_s3, feat_s4]

        Returns:
            top_down: 融合后的最高分辨率特征图 [B, fpn_channels, H, W]
        """
        assert len(features) == self.num_stages, (
            f"期望 {self.num_stages} 个尺度特征，实际得到 {len(features)}"
        )

        # Step 1: 横向 1*1 卷积对齐通道 + 残差块
        laterals = []
        for i in range(self.num_stages):
            proj = self.lateral_convs[i](features[i])
            laterals.append(self.residual_blocks[i](proj))

        # Step 2: 自上而下融合（双线性插值 + 逐元素相加）
        top_down = laterals[-1]
        for i in range(self.num_stages - 2, -1, -1):
            top_down = F.interpolate(
                top_down,
                size=laterals[i].shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            top_down = top_down + laterals[i]

        return top_down


class FPNDecoder(nn.Module):
    """
    独立双 FPN 解码头（语义 + 边界各自独立）- 边界预测版本。

    语义分支和边界分支各自拥有独立的 FPN backbone 和输出头，
    从 encoder 特征中独立提取信息，消除梯度干扰。

    输出 2 通道：
    - 通道 0：语义 logits（铁素体 vs 珠光体）
    - 通道 1：边界 logits（晶界 vs 非晶界）

    参数量约 10.8M（fpn_channels=256，双 FPN）。
    """

    def __init__(
        self,
        in_channels: Optional[List[int]] = None,
        fpn_channels: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_bn: bool = True,
    ):
        """
        Args:
            in_channels: 各 Stage 输入通道数列表，从高分辨率到低分辨率。
                         默认 [112, 224, 448, 896]（Hiera base+）。
            fpn_channels: FPN 统一通道数（默认 256）。
            num_classes: 输出通道数（固定为 2：语义 + 边界）。
            dropout: dropout 概率。
            use_bn: 是否使用归一化层（True=GroupNorm, False=Identity）。
        """
        super().__init__()
        if in_channels is None:
            in_channels = [112, 224, 448, 896]

        assert len(in_channels) == 4, "FPN 解码头需要 4 个尺度的输入特征"
        assert num_classes == 2, "边界预测版本固定输出 2 通道（语义 + 边界）"

        self.in_channels = in_channels
        self.fpn_channels = fpn_channels
        self.num_classes = num_classes
        self.num_stages = len(in_channels)

        # GroupNorm 对小 batch size 更稳定（不依赖 batch 统计）
        if use_bn:
            norm_layer = _make_group_norm
        else:
            norm_layer = nn.Identity

        # 独立双 FPN backbone
        self.seg_fpn = FPNBackbone(
            in_channels, fpn_channels, self.num_stages, norm_layer
        )
        self.boundary_fpn = FPNBackbone(
            in_channels, fpn_channels, self.num_stages, norm_layer
        )

        # 语义分支输出头：聚焦低频面状特征
        half_channels = fpn_channels // 2
        self.seg_branch = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, half_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(half_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(half_channels, 1, kernel_size=1, bias=True),
        )

        # 边界分支输出头：聚焦高频线状特征
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, half_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(half_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(half_channels, 1, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        """使用 Kaiming 初始化（随机初始化，不加载预训练权重）。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features, output_size=None):
        """
        前向传播：双 FPN 独立提取 + 各自输出头 + 拼接。

        Args:
            features: [feat_s1, feat_s2, feat_s3, feat_s4]
            output_size: 可选，最终输出的 (H, W) 尺寸。

        Returns:
            output: [B, 2, H, W]
                - output[:, 0] 为语义 logits
                - output[:, 1] 为边界 logits
        """
        assert len(features) == self.num_stages, (
            f"期望 {self.num_stages} 个尺度特征，实际得到 {len(features)}"
        )

        # 独立双 FPN 提取
        seg_feat = self.seg_fpn(features)
        boundary_feat = self.boundary_fpn(features)

        # 各自输出头
        seg_logits = self.seg_branch(seg_feat)            # [B, 1, H, W]
        boundary_logits = self.boundary_branch(boundary_feat)  # [B, 1, H, W]
        output = torch.cat([seg_logits, boundary_logits], dim=1)  # [B, 2, H, W]

        # 动态上采样到指定尺寸（推理时对齐原图）
        if output_size is not None:
            output = F.interpolate(
                output,
                size=output_size,
                mode="bilinear",
                align_corners=True,
            )

        return output

    def freeze_seg_branch(self):
        """冻结语义分支（seg_fpn + seg_branch），仅训练边界分支时使用。"""
        for param in self.seg_fpn.parameters():
            param.requires_grad = False
        for param in self.seg_branch.parameters():
            param.requires_grad = False
        logger_freeze_info = (
            f"Semantic branch FROZEN (seg_fpn + seg_branch, "
            f"{sum(p.numel() for p in self.seg_fpn.parameters()) + sum(p.numel() for p in self.seg_branch.parameters())} params)"
        )
        print(logger_freeze_info)

    def freeze_boundary_branch(self):
        """冻结边界分支（boundary_fpn + boundary_branch），仅训练语义分支时使用。"""
        for param in self.boundary_fpn.parameters():
            param.requires_grad = False
        for param in self.boundary_branch.parameters():
            param.requires_grad = False
        logger_freeze_info = (
            f"Boundary branch FROZEN (boundary_fpn + boundary_branch, "
            f"{sum(p.numel() for p in self.boundary_fpn.parameters()) + sum(p.numel() for p in self.boundary_branch.parameters())} params)"
        )
        print(logger_freeze_info)

    def param_count(self):
        """返回解码头的参数总量。"""
        return sum(p.numel() for p in self.parameters())


class SegmentationModel(nn.Module):
    """
    完整分割模型：冻结 SAM 2 Encoder + 随机初始化双 FPN Decoder（边界预测版本）。
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, output_size=None):
        """
        前向传播。

        Args:
            x: 输入图像 [B, 3, 1024, 1024]（Letterbox 后）
            output_size: 可选，推理时对齐原图尺寸 (H, W)。

        Returns:
            output: [B, 2, H, W]
                - output[:, 0] 为语义 logits
                - output[:, 1] 为边界 logits
        """
        features = self.encoder(x)
        output = self.decoder(features, output_size=output_size)
        return output

    def total_param_count(self):
        """返回各部分参数量统计。"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total = encoder_params + decoder_params
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "total": total,
            "total_M": total / 1e6,
            "constraint_passed": total < 500e6,
        }