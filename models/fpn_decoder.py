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

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .edge_prior import EdgePriorResidualFusion

logger = logging.getLogger(__name__)


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


class BoundaryRefineHead(nn.Module):
    """高分辨率边界细化头：256->512->1024 渐进上采样 + 图像多尺度跳连。

    设计动机（"梨形"）：经典分割网络在下采样路径各尺度有对应的高分辨率特征，
    上采样路径逐级跳连。SAM2 trunk 最小 stride 为 4（无 512/1024 特征），
    因此本头以【输入图像金字塔】充当高分辨率编码侧，每个上采样阶段拼接
    该分辨率下的图像浅层特征——细化头不需要"无中生有"地猜高频细节，
    只需学会把图像细节映射到边界位置，深度需求保持轻量（每级 2 层卷积）。

    尺度自适应：输出尺寸 = 输入 boundary_feat 的 4 倍（1024 输入 -> 1024 输出；
    512 裁剪 -> 512 输出）。
    """

    def __init__(self, fpn_channels: int = 256, img_ch: int = 3, hidden: int = 96):
        super().__init__()
        self.img_stem = nn.ModuleList([
            self._make_stem(img_ch, 32),   # 256 级
            self._make_stem(img_ch, 24),   # 512 级
            self._make_stem(img_ch, 16),   # 1024 级
        ])
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(fpn_channels + 32, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(16, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(16, hidden), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden + 24, hidden // 2, 3, padding=1, bias=False),
            nn.GroupNorm(16, hidden // 2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, hidden // 2, 3, padding=1, bias=False),
            nn.GroupNorm(16, hidden // 2), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden // 2 + 16, 1, 1)

    @staticmethod
    def _make_stem(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True),
        )

    def forward(
        self,
        boundary_feat: torch.Tensor,
        image: torch.Tensor,
        semantic_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """boundary_feat: [B, C, H, W]；image: [B, 3, H_img, W_img]"""
        # 256 级：拼接该分辨率图像特征
        img_cur = F.interpolate(image, size=boundary_feat.shape[-2:],
                                mode="bilinear", align_corners=True)
        image_feat = self.img_stem[0](img_cur)
        if semantic_logits is None:
            gate = torch.ones(
                image_feat.shape[0], 1, image_feat.shape[2], image_feat.shape[3],
                device=image_feat.device, dtype=image_feat.dtype
            )
        else:
            semantic_prob = torch.sigmoid(semantic_logits)
            local_mean = F.avg_pool2d(semantic_prob, kernel_size=5, stride=1, padding=2)
            semantic_contrast = (semantic_prob - local_mean).abs()
            gate = torch.sigmoid(
                self.semantic_gate(torch.cat([semantic_prob, semantic_contrast], dim=1))
            )
            gate = 0.25 + 0.75 * gate
        x = torch.cat([boundary_feat, image_feat * gate], dim=1)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.up1(x)  # 512 级
        # 512 级：拼接 512 图像特征
        img_cur = F.interpolate(image, size=x.shape[-2:],
                                mode="bilinear", align_corners=True)
        x = torch.cat([x, self.img_stem[1](img_cur)], dim=1)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.up2(x)  # 1024 级
        # 1024 级：拼接 1024 图像特征
        img_cur = F.interpolate(image, size=x.shape[-2:],
                                mode="bilinear", align_corners=True)
        x = torch.cat([x, self.img_stem[2](img_cur)], dim=1)
        return self.out(x)


class SemanticResidualAdapter(nn.Module):
    """Geometry-isolated semantic correction with adaptive photometric cues.

    The historical V6 semantic FPN/head remains untouched.  This adapter sees
    its feature map plus per-image normalized luminance and local contrast, and
    predicts a bounded residual logit.  A zero-initialized output makes epoch 0
    exactly reproduce V6 while giving the semantic path additional capacity for
    dim or unevenly illuminated ferrite.
    """

    def __init__(
        self,
        feature_channels: int = 256,
        hidden_channels: int = 64,
        color_channels: int = 16,
        use_photometric_cues: bool = True,
        max_logit_delta: float = 2.0,
    ):
        super().__init__()
        hidden_channels = int(hidden_channels)
        color_channels = int(color_channels)
        if hidden_channels <= 0 or color_channels <= 0:
            raise ValueError("semantic residual channels must be positive")
        if float(max_logit_delta) <= 0:
            raise ValueError("semantic residual max_logit_delta must be positive")
        self.use_photometric_cues = bool(use_photometric_cues)
        self.max_logit_delta = float(max_logit_delta)
        self.feature_path = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 3, padding=1, bias=False),
            _make_group_norm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            _make_group_norm(hidden_channels),
            nn.ReLU(inplace=True),
        )
        fusion_channels = hidden_channels
        self.photometric_path = None
        if self.use_photometric_cues:
            self.photometric_path = nn.Sequential(
                nn.Conv2d(2, color_channels, 3, padding=1, bias=False),
                _make_group_norm(color_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(color_channels, color_channels, 3, padding=1, bias=False),
                _make_group_norm(color_channels),
                nn.ReLU(inplace=True),
            )
            fusion_channels += color_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, hidden_channels, 3, padding=1, bias=False),
            _make_group_norm(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden_channels, 1, 1, bias=True)

    @staticmethod
    def adaptive_photometric_cues(
        image: torch.Tensor, output_size,
    ) -> torch.Tensor:
        """Return illumination-normalized luminance and local contrast."""
        resized = F.interpolate(
            image, size=output_size, mode="bilinear", align_corners=False
        )
        # Input tensors are RGB in [0, 1].  This is a differentiable luminance
        # proxy; per-image normalization avoids a fixed global gray threshold.
        weights = resized.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
        luminance = (resized * weights).sum(dim=1, keepdim=True)
        mean = luminance.mean(dim=(-2, -1), keepdim=True)
        std = luminance.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(0.03)
        normalized = ((luminance - mean) / std).clamp(-3.0, 3.0) / 3.0
        local_mean = F.avg_pool2d(
            F.pad(luminance, (7, 7, 7, 7), mode="reflect"),
            kernel_size=15,
            stride=1,
        )
        local_contrast = ((luminance - local_mean) / std).clamp(-2.0, 2.0) / 2.0
        return torch.cat([normalized, local_contrast], dim=1)

    def reset_output(self) -> None:
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feature: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        values = [self.feature_path(feature)]
        if self.photometric_path is not None:
            cues = self.adaptive_photometric_cues(image, feature.shape[-2:])
            values.append(self.photometric_path(cues))
        raw_delta = self.out(self.fusion(torch.cat(values, dim=1)))
        return self.max_logit_delta * torch.tanh(raw_delta)


class SemanticHighResolutionResidualAdapter(nn.Module):
    """Learn semantic evidence at image resolution without changing V6 features.

    SAM2's finest encoder feature is stride four.  The old semantic residual
    therefore made its decision on that grid and only interpolated the logits.
    This adapter progressively restores the two missing spatial scales and
    injects shallow RGB/photometric evidence at each scale.  The coarse V6
    probability and its local variation are *features*, not a confidence lock:
    the head may correct a confident but locally inconsistent coarse decision.
    """

    def __init__(
        self,
        feature_channels: int = 256,
        hidden_channels: int = 64,
        color_channels: int = 16,
        use_photometric_cues: bool = True,
        max_logit_delta: float = 0.75,
        half_channels: int = 48,
        full_channels: int = 24,
    ):
        super().__init__()
        hidden_channels = int(hidden_channels)
        color_channels = int(color_channels)
        half_channels = int(half_channels)
        full_channels = int(full_channels)
        if min(hidden_channels, color_channels, half_channels, full_channels) <= 0:
            raise ValueError("semantic high-resolution channels must be positive")
        if float(max_logit_delta) <= 0:
            raise ValueError("semantic residual max_logit_delta must be positive")
        self.use_photometric_cues = bool(use_photometric_cues)
        self.max_logit_delta = float(max_logit_delta)

        self.feature_path = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 3, padding=1, bias=False),
            _make_group_norm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            _make_group_norm(hidden_channels),
            nn.ReLU(inplace=True),
        )
        # RGB + coarse probability + coarse local variation; normalized
        # luminance/local contrast are appended when photometric cues are on.
        cue_channels = 7 if self.use_photometric_cues else 5
        self.half_image_path = nn.Sequential(
            nn.Conv2d(cue_channels, color_channels, 3, padding=1, bias=False),
            _make_group_norm(color_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(color_channels, color_channels, 3, padding=1, bias=False),
            _make_group_norm(color_channels),
            nn.ReLU(inplace=True),
        )
        self.up_half = nn.Sequential(
            nn.Conv2d(
                hidden_channels + color_channels,
                half_channels,
                3,
                padding=1,
                bias=False,
            ),
            _make_group_norm(half_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(half_channels, half_channels, 3, padding=1, bias=False),
            _make_group_norm(half_channels),
            nn.ReLU(inplace=True),
        )
        self.full_image_path = nn.Sequential(
            nn.Conv2d(cue_channels, color_channels, 3, padding=1, bias=False),
            _make_group_norm(color_channels),
            nn.ReLU(inplace=True),
        )
        self.up_full = nn.Sequential(
            nn.Conv2d(
                half_channels + color_channels,
                full_channels,
                3,
                padding=1,
                bias=False,
            ),
            _make_group_norm(full_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(full_channels, full_channels, 3, padding=1, bias=False),
            _make_group_norm(full_channels),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(full_channels, 1, 1, bias=True)

    def _image_cues(
        self,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
        output_size,
    ) -> torch.Tensor:
        rgb = F.interpolate(
            image, size=output_size, mode="bilinear", align_corners=False
        )
        coarse_probability = torch.sigmoid(
            F.interpolate(
                coarse_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        )
        local_mean = F.avg_pool2d(
            coarse_probability, kernel_size=5, stride=1, padding=2
        )
        coarse_variation = (coarse_probability - local_mean).abs()
        values = [rgb, coarse_probability, coarse_variation]
        if self.use_photometric_cues:
            values.append(
                SemanticResidualAdapter.adaptive_photometric_cues(
                    image, output_size
                )
            )
        return torch.cat(values, dim=1)

    def reset_output(self) -> None:
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        feature: torch.Tensor,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> torch.Tensor:
        half_size = tuple(int(value * 2) for value in feature.shape[-2:])
        full_size = tuple(int(value) for value in image.shape[-2:])
        x = F.interpolate(
            self.feature_path(feature),
            size=half_size,
            mode="bilinear",
            align_corners=False,
        )
        half_cues = self.half_image_path(
            self._image_cues(image, coarse_logits, half_size)
        )
        x = self.up_half(torch.cat([x, half_cues], dim=1))
        x = F.interpolate(
            x, size=full_size, mode="bilinear", align_corners=False
        )
        full_cues = self.full_image_path(
            self._image_cues(image, coarse_logits, full_size)
        )
        raw_delta = self.out(self.up_full(torch.cat([x, full_cues], dim=1)))
        return self.max_logit_delta * torch.tanh(raw_delta)


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
        boundary_refine: bool = False,
        boundary_refine_version: str = "legacy_lowres",
        center_head: bool = False,
        edge_prior_fusion: bool = False,
        edge_prior_fusion_hidden: int = 32,
        edge_prior_max_logit_delta: float = 1.0,
        semantic_residual: bool = False,
        semantic_residual_version: str = "lowres_v1",
        semantic_residual_hidden: int = 64,
        semantic_residual_color_channels: int = 16,
        semantic_residual_use_photometric: bool = True,
        semantic_residual_max_logit_delta: float = 2.0,
        semantic_residual_half_channels: int = 48,
        semantic_residual_full_channels: int = 24,
    ):
        """
        Args:
            in_channels: 各 Stage 输入通道数列表，从高分辨率到低分辨率。
                         默认 [112, 224, 448, 896]（Hiera base+）。
            fpn_channels: FPN 统一通道数（默认 256）。
            num_classes: 输出通道数（固定为 2：语义 + 边界）。
            dropout: dropout 概率。
            use_bn: 是否使用归一化层（True=GroupNorm, False=Identity）。
            boundary_refine: 启用高分辨率边界细化头（256->1024 + 原图引导）。
                边界通道原生输出 1024，语义保持 256 上采样。
            boundary_refine_version: ``legacy_lowres`` 保持旧 checkpoint 行为；
                ``v2_fullres_isolated`` 使用原始分辨率图像并切断语义梯度。
            center_head: 增加轻量中心热图头，复用 boundary FPN。
            edge_prior_fusion: 启用冻结 G0b 结构先验的有界残差融合。
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

        # 边界分支输出头：聚焦高频线状特征（boundary_refine 启用时保留权重兼容但旁路）
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
        self.boundary_refine = boundary_refine
        valid_refine_versions = {"legacy_lowres", "v2_fullres_isolated"}
        if boundary_refine_version not in valid_refine_versions:
            raise ValueError(
                f"Unsupported boundary_refine_version={boundary_refine_version!r}; "
                f"expected one of {sorted(valid_refine_versions)}"
            )
        self.boundary_refine_version = boundary_refine_version
        self.center_head_enabled = bool(center_head)
        self.edge_prior_fusion_enabled = bool(edge_prior_fusion)
        self.semantic_residual_enabled = bool(semantic_residual)
        self.semantic_residual_version = str(semantic_residual_version)
        self.center_branch = None
        if self.center_head_enabled:
            self.center_branch = nn.Sequential(
                nn.Conv2d(fpn_channels, half_channels, 3, padding=1, bias=False),
                norm_layer(half_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(half_channels, half_channels // 2, 3, padding=1, bias=False),
                norm_layer(half_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                nn.Conv2d(half_channels // 2, 1, 1, bias=True),
            )
        if boundary_refine:
            self.boundary_refine_head = BoundaryRefineHead(
                fpn_channels=fpn_channels, img_ch=3, hidden=96
            )
        self.edge_prior_fusion = None
        if self.edge_prior_fusion_enabled:
            self.edge_prior_fusion = EdgePriorResidualFusion(
                hidden_channels=edge_prior_fusion_hidden,
                max_logit_delta=edge_prior_max_logit_delta,
            )
        self.semantic_residual = None
        if self.semantic_residual_enabled:
            if self.semantic_residual_version == "lowres_v1":
                self.semantic_residual = SemanticResidualAdapter(
                    feature_channels=fpn_channels,
                    hidden_channels=semantic_residual_hidden,
                    color_channels=semantic_residual_color_channels,
                    use_photometric_cues=semantic_residual_use_photometric,
                    max_logit_delta=semantic_residual_max_logit_delta,
                )
            elif self.semantic_residual_version == "highres_v1":
                self.semantic_residual = SemanticHighResolutionResidualAdapter(
                    feature_channels=fpn_channels,
                    hidden_channels=semantic_residual_hidden,
                    color_channels=semantic_residual_color_channels,
                    use_photometric_cues=semantic_residual_use_photometric,
                    max_logit_delta=semantic_residual_max_logit_delta,
                    half_channels=semantic_residual_half_channels,
                    full_channels=semantic_residual_full_channels,
                )
            else:
                raise ValueError(
                    "semantic_residual_version must be lowres_v1 or highres_v1; "
                    f"got {self.semantic_residual_version!r}"
                )

        self._init_weights()
        if self.boundary_refine:
            # Residual refinement starts as an identity path over the coarse
            # boundary head, so a new high-resolution head cannot saturate it.
            nn.init.zeros_(self.boundary_refine_head.out.weight)
            nn.init.zeros_(self.boundary_refine_head.out.bias)
        if self.edge_prior_fusion is not None:
            self.edge_prior_fusion.reset_output()
        if self.semantic_residual is not None:
            self.semantic_residual.reset_output()

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

    @staticmethod
    def _reset_modules(modules) -> None:
        """Kaiming-reset selected decoder modules without touching siblings."""
        visited = set()
        for module in modules:
            if module is None:
                continue
            for layer in module.modules():
                identifier = id(layer)
                if identifier in visited:
                    continue
                visited.add(identifier)
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(
        self, features, output_size=None, image=None, boundary_features=None,
        edge_prior_raw=None,
    ):
        """
        前向传播：双 FPN 独立提取 + 各自输出头 + 拼接。

        Args:
            features: [feat_s1, feat_s2, feat_s3, feat_s4]
            boundary_features: 可选的边界专用多尺度特征。为空时与
                ``features`` 相同；GDA 实验仅改写该路径，语义路径保持原样。
            output_size: 可选，最终输出的 (H, W) 尺寸。
            image: 原始输入图像 [B, 3, H, W]（boundary_refine 启用时提供，
                   作为高分辨率边界的高频引导）。
            edge_prior_raw: 冻结 G0b 输出的边缘/方向原始张量。

        Returns:
            output: [B, 2, H, W]
                - output[:, 0] 为语义 logits
                - output[:, 1] 为边界 logits
        """
        assert len(features) == self.num_stages, (
            f"期望 {self.num_stages} 个尺度特征，实际得到 {len(features)}"
        )
        if boundary_features is None:
            boundary_features = features
        assert len(boundary_features) == self.num_stages, (
            f"期望 {self.num_stages} 个边界尺度特征，"
            f"实际得到 {len(boundary_features)}"
        )

        # 独立双 FPN 提取
        seg_feat = self.seg_fpn(features)
        boundary_feat = self.boundary_fpn(boundary_features)

        # 各自输出头
        seg_logits = self.seg_branch(seg_feat)            # [B, 1, 256, 256]
        if self.semantic_residual is not None:
            if image is None:
                raise ValueError("semantic_residual=True requires image")
            if self.semantic_residual_version == "highres_v1":
                semantic_delta = self.semantic_residual(
                    seg_feat, image, coarse_logits=seg_logits
                )
                seg_logits = F.interpolate(
                    seg_logits,
                    size=semantic_delta.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                ) + semantic_delta
            else:
                seg_logits = seg_logits + self.semantic_residual(seg_feat, image)
        coarse_boundary_logits = self.boundary_branch(boundary_feat)
        if self.boundary_refine:
            # 高分辨率边界细化头：原生输出 1024
            if image is None:
                raise ValueError("boundary_refine=True 时 forward 需传入 image")
            if self.boundary_refine_version == "v2_fullres_isolated":
                # B2：直接传入原始分辨率图像。旧实现先降到 FPN 分辨率再
                # 重新放大，使 512/1024 两级 image stem 看不到高频细节。
                # semantic logits 仅作为只读门控上下文。
                refine_image = image
                refine_semantic = F.interpolate(
                    seg_logits.detach(),
                    size=boundary_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                # 保留旧 center/refine checkpoint 的历史行为，避免同一权重
                # 在代码升级后静默改变推理语义。
                refine_image = F.interpolate(
                    image, size=seg_feat.shape[-2:],
                    mode="bilinear", align_corners=True,
                )
                refine_semantic = F.interpolate(
                    seg_logits,
                    size=boundary_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            refine_logits = self.boundary_refine_head(
                boundary_feat, refine_image, semantic_logits=refine_semantic
            )
            coarse_boundary_logits = F.interpolate(
                coarse_boundary_logits, size=refine_logits.shape[-2:],
                mode="bilinear", align_corners=True
            )
            boundary_logits = coarse_boundary_logits + 0.5 * refine_logits
        else:
            boundary_logits = coarse_boundary_logits  # [B, 1, 256, 256]

        center_logits = (
            self.center_branch(boundary_feat)
            if self.center_branch is not None
            else None
        )

        # 动态上采样到指定尺寸（语义与边界分别处理：边界已 1024 时不再插值）
        if output_size is not None:
            seg_logits = F.interpolate(
                seg_logits, size=output_size, mode="bilinear", align_corners=True
            )
            if boundary_logits.shape[-2:] != output_size:
                boundary_logits = F.interpolate(
                    boundary_logits, size=output_size, mode="bilinear", align_corners=True
                )
            if center_logits is not None and center_logits.shape[-2:] != output_size:
                center_logits = F.interpolate(
                    center_logits, size=output_size, mode="bilinear", align_corners=True
                )
        elif seg_logits.shape[-2:] != boundary_logits.shape[-2:]:
            # 无指定输出尺寸时保留两条路径中的较高原生分辨率。旧 B2 是
            # boundary=fullres；E9 highres_v1 则是 semantic=fullres。
            seg_area = int(seg_logits.shape[-2] * seg_logits.shape[-1])
            boundary_area = int(
                boundary_logits.shape[-2] * boundary_logits.shape[-1]
            )
            target_size = (
                seg_logits.shape[-2:]
                if seg_area >= boundary_area
                else boundary_logits.shape[-2:]
            )
            if seg_logits.shape[-2:] != target_size:
                seg_logits = F.interpolate(
                    seg_logits,
                    size=target_size,
                    mode="bilinear",
                    align_corners=True,
                )
            if boundary_logits.shape[-2:] != target_size:
                boundary_logits = F.interpolate(
                    boundary_logits,
                    size=target_size,
                    mode="bilinear",
                    align_corners=True,
                )
        if center_logits is not None and center_logits.shape[-2:] != boundary_logits.shape[-2:]:
            center_logits = F.interpolate(
                center_logits, size=boundary_logits.shape[-2:],
                mode="bilinear", align_corners=True,
            )
        if self.edge_prior_fusion is not None:
            if edge_prior_raw is None:
                raise ValueError(
                    "decoder.edge_prior_fusion=True requires edge_prior_raw"
                )
            boundary_logits = boundary_logits + self.edge_prior_fusion(
                boundary_logits, edge_prior_raw
            )
        outputs = [seg_logits, boundary_logits]
        if center_logits is not None:
            outputs.append(center_logits)
        output = torch.cat(outputs, dim=1)  # [B, 2/3, H, W]
        return output

    def freeze_seg_branch(self):
        """冻结语义分支（seg_fpn + seg_branch），仅训练边界分支时使用。"""
        for param in self.seg_fpn.parameters():
            param.requires_grad = False
        for param in self.seg_branch.parameters():
            param.requires_grad = False
        if self.semantic_residual is not None:
            for param in self.semantic_residual.parameters():
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
        if self.boundary_refine:
            for param in self.boundary_refine_head.parameters():
                param.requires_grad = False
        if self.center_branch is not None:
            for param in self.center_branch.parameters():
                param.requires_grad = False
        if self.edge_prior_fusion is not None:
            for param in self.edge_prior_fusion.parameters():
                param.requires_grad = False
        logger_freeze_info = (
            f"Boundary branch FROZEN (boundary_fpn + boundary_branch"
            f"{' + boundary_refine_head' if self.boundary_refine else ''}, "
            f"{sum(p.numel() for p in self.boundary_fpn.parameters()) + sum(p.numel() for p in self.boundary_branch.parameters())} params)"
        )
        print(logger_freeze_info)

    def set_semantic_residual_only(self):
        """Train only the isolated residual; preserve V6 semantics and geometry."""
        if self.semantic_residual is None:
            raise ValueError("semantic residual-only mode requires semantic_residual=True")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.semantic_residual.parameters():
            parameter.requires_grad_(True)
        count = sum(p.numel() for p in self.semantic_residual.parameters())
        print(f"Semantic residual ONLY ({count} trainable params)")

    def reset_semantic_branch(self) -> None:
        """Cold-start the complete semantic decoder while preserving geometry.

        The semantic FPN, coarse classifier and optional high-resolution module
        are reset together.  The high-resolution output starts at zero so the
        randomly initialized coarse path first establishes phase semantics,
        then learns image-guided full-resolution corrections.
        """
        modules = [self.seg_fpn, self.seg_branch]
        if self.semantic_residual is not None:
            modules.append(self.semantic_residual)
        self._reset_modules(modules)
        if self.semantic_residual is not None:
            self.semantic_residual.reset_output()
        count = sum(
            parameter.numel()
            for module in modules
            for parameter in module.parameters()
        )
        print(f"Semantic branch RESET (seg_fpn + seg_branch + highres, {count} params)")

    def set_semantic_cold_start_only(self) -> None:
        """Train the complete semantic decoder and freeze all geometry heads."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        modules = [self.seg_fpn, self.seg_branch]
        if self.semantic_residual is not None:
            modules.append(self.semantic_residual)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        count = sum(
            parameter.numel()
            for module in modules
            for parameter in module.parameters()
        )
        print(
            "Semantic cold-start ONLY "
            f"(seg_fpn + seg_branch + highres, {count} trainable params)"
        )

    def set_boundary_base_trainable(self, trainable: bool):
        """切换 V6 coarse boundary 基座，保持 refine head 状态不变。

        B2 训练前段只更新 ``boundary_refine_head``；稳定后再以较低学习率
        解冻 ``boundary_fpn + boundary_branch``。该接口不会触碰语义分支、
        center head 或 refine head，避免多任务梯度重新污染 V6 表征。
        """
        for module in (self.boundary_fpn, self.boundary_branch):
            for param in module.parameters():
                param.requires_grad_(trainable)
        state = "TRAINABLE" if trainable else "FROZEN"
        count = sum(
            param.numel()
            for module in (self.boundary_fpn, self.boundary_branch)
            for param in module.parameters()
        )
        print(f"Boundary base {state} (boundary_fpn + boundary_branch, {count} params)")

    def set_edge_prior_fusion_only(self):
        """Freeze the historical model and calibrate only the new fusion head."""
        if self.edge_prior_fusion is None:
            raise ValueError("edge-prior fusion-only mode requires fusion enabled")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.edge_prior_fusion.parameters():
            parameter.requires_grad_(True)
        count = sum(p.numel() for p in self.edge_prior_fusion.parameters())
        print(f"Edge-prior fusion ONLY ({count} trainable params)")

    def reset_boundary_branch(self):
        """重新初始化边界路径，保留语义路径和 encoder 权重。"""
        modules = [self.boundary_fpn, self.boundary_branch]
        if self.boundary_refine:
            modules.append(self.boundary_refine_head)
        if self.center_branch is not None:
            modules.append(self.center_branch)
        if self.edge_prior_fusion is not None:
            modules.append(self.edge_prior_fusion)
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        if self.edge_prior_fusion is not None:
            self.edge_prior_fusion.reset_output()
        print("Boundary branch RESET (boundary FPN/head reinitialized)")
    def param_count(self):
        """返回解码头的参数总量。"""
        return sum(p.numel() for p in self.parameters())


def load_decoder_state(decoder: nn.Module, state_dict: dict, tag: str = "decoder"):
    """宽容加载 decoder 权重（支持架构演进）。

    新增层（如高分辨率边界细化头）在旧 checkpoint 中不存在 -> 保持随机初始化；
    旧架构层（如被旁路的输出头）在新模型中存在但 checkpoint 无对应 -> 忽略。
    缺失/多余键都会打日志，便于排查。
    """
    missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
    if missing:
        logger.info(f"{tag}: 新增层缺失 {len(missing)} 个键（保持随机初始化）: "
                    f"{list(missing)[:4]}...")
    if unexpected:
        logger.info(f"{tag}: 旧架构多余 {len(unexpected)} 个键（忽略）: "
                    f"{list(unexpected)[:4]}...")
    return missing, unexpected


class SegmentationModel(nn.Module):
    """
    完整分割模型：冻结 SAM 2 Encoder + 随机初始化双 FPN Decoder（边界预测版本）。
    """

    def __init__(self, encoder, decoder, boundary_adapter=None, edge_prior=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.boundary_adapter = boundary_adapter
        self.edge_prior = edge_prior

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
        boundary_features = features
        if self.boundary_adapter is not None:
            boundary_features = self.boundary_adapter(features, gated=True)
        edge_prior_raw = None
        if self.edge_prior is not None:
            self.edge_prior.eval()
            with torch.no_grad():
                edge_prior_raw = self.edge_prior(features[:2], x.shape[-2:])
        decoder_kwargs = {
            "output_size": output_size,
            "image": x,
            "boundary_features": boundary_features,
        }
        if edge_prior_raw is not None:
            decoder_kwargs["edge_prior_raw"] = edge_prior_raw
        output = self.decoder(features, **decoder_kwargs)
        return output

    def total_param_count(self):
        """返回各部分参数量统计。"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        adapter_params = (
            sum(p.numel() for p in self.boundary_adapter.parameters())
            if self.boundary_adapter is not None else 0
        )
        edge_prior_params = (
            sum(p.numel() for p in self.edge_prior.parameters())
            if self.edge_prior is not None else 0
        )
        total = encoder_params + decoder_params + adapter_params + edge_prior_params
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "boundary_adapter": adapter_params,
            "edge_prior": edge_prior_params,
            "total": total,
            "total_M": total / 1e6,
            "constraint_passed": total < 500e6,
        }
