# -*- coding: utf-8 -*-
"""
自研轻量级 FPN 解码头
====================
完全随机初始化的特征金字塔解码头，禁止加载任何 SAM 2 原生 Mask Decoder 权重。

设计：
1. 通过 1*1 卷积将 SAM 2 trunk 提取的四个尺度特征统一对齐到相同通道数（如 128 或 256）。
2. 自上而下通过双线性插值融合（从最低分辨率 Stage 4 逐步上采样与高层分辨率融合）。
3. 在最高分辨率特征图上通过 3*3 卷积输出通道数固定为 3 的 logits
   （0=珠光体, 1=铁素体核, 2=晶界）。

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


class FPNDecoder(nn.Module):
    """
    轻量级全卷积 FPN 解码头（随机初始化）。

    参数量极小（远小于 SAM 2 原生 Mask Decoder），确保总参数量 < 500M。
    """

    def __init__(
        self,
        in_channels: Optional[List[int]] = None,
        fpn_channels: int = 128,
        num_classes: int = 3,
        dropout: float = 0.1,
        use_bn: bool = True,
    ):
        """
        Args:
            in_channels: 各 Stage 输入通道数列表，从高分辨率到低分辨率。
                         默认 [112, 224, 448, 896]（Hiera base+）。
            fpn_channels: FPN 统一通道数。
            num_classes: 输出类别数（三分类：0=珠光体, 1=铁素体核, 2=晶界）。
            dropout: dropout 概率。
            use_bn: 是否使用 BatchNorm。
        """
        super().__init__()
        if in_channels is None:
            in_channels = [112, 224, 448, 896]

        assert len(in_channels) == 4, "FPN 解码头需要 4 个尺度的输入特征"

        self.in_channels = in_channels
        self.fpn_channels = fpn_channels
        self.num_classes = num_classes
        self.num_stages = len(in_channels)

        norm_layer = nn.BatchNorm2d if use_bn else nn.Identity

        # 1*1 横向连接卷积：将各 Stage 通道对齐到 fpn_channels
        self.lateral_convs = nn.ModuleList()
        for ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, fpn_channels, kernel_size=1, bias=False),
                    norm_layer(fpn_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # 3*3 平滑卷积（融合后平滑）
        self.smooth_convs = nn.ModuleList()
        for _ in range(self.num_stages - 1):
            self.smooth_convs.append(
                nn.Sequential(
                    nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                    norm_layer(fpn_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # 输出头：在最高分辨率特征图上输出 num_classes 通道 logits
        self.output_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1, bias=True),
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
        前向传播：FPN 自上而下融合 + 输出三分类 logits。

        Args:
            features: [feat_s1, feat_s2, feat_s3, feat_s4]
                - feat_s1: 最高分辨率
                - feat_s4: 最低分辨率
            output_size: 可选，最终输出的 (H, W) 尺寸。

        Returns:
            logits: [B, num_classes, H, W] 三分类 logits
        """
        assert len(features) == self.num_stages, (
            f"期望 {self.num_stages} 个尺度特征，实际得到 {len(features)}"
        )

        # Step 1: 横向 1*1 卷积对齐通道
        laterals = [self.lateral_convs[i](features[i]) for i in range(self.num_stages)]

        # Step 2: 自上而下融合
        top_down = laterals[-1]
        for i in range(self.num_stages - 2, -1, -1):
            top_down = F.interpolate(
                top_down,
                size=laterals[i].shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            fused = top_down + laterals[i]
            smooth_idx = self.num_stages - 2 - i
            top_down = self.smooth_convs[smooth_idx](fused)

        # Step 3: 输出头
        logits = self.output_head(top_down)

        # Step 4: 动态上采样到指定尺寸（推理时对齐原图）
        if output_size is not None:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=True,
            )

        return logits

    def param_count(self):
        """返回解码头的参数总量。"""
        return sum(p.numel() for p in self.parameters())


class SegmentationModel(nn.Module):
    """
    完整分割模型：冻结 SAM 2 Encoder + 随机初始化 FPN Decoder。

    用于训练和推理的统一入口。
    """

    def __init__(self, encoder, decoder):
        """
        Args:
            encoder: 冻结的 SAM 2 Image Encoder 提取器。
            decoder: 随机初始化的 FPN 解码头。
        """
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
            logits: [B, num_classes, H, W]
        """
        features = self.encoder(x)
        logits = self.decoder(features, output_size=output_size)
        return logits

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