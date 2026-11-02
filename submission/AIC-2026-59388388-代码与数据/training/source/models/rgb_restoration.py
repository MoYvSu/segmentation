# -*- coding: utf-8 -*-
"""独立 RGB 残差修复器；不构建 SAM2，不改变输入的空间尺寸。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


CHECKPOINT_FORMAT = "rgb_restoration_v1"
CHECKPOINT_FORMAT_NAFNET = "rgb_restoration_nafnet_v1"


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.body(x)


class RGBRestorer(nn.Module):
    """输入/输出均为 B×3×H×W、RGB、[0, 1]；放在 encoder 归一化之前。

    两次降采样的小型 U-Net，输出有界修正量。末层零初始化，初始输出逐值
    等于输入；strength=0 可显式旁路。这里的恒等性不代表部署链已经验证。
    """

    checkpoint_format = CHECKPOINT_FORMAT

    def __init__(self, width: int = 24, max_residual: float = 0.5):
        super().__init__()
        if width < 4 or not 0 < max_residual <= 1:
            raise ValueError("width must be >= 4 and max_residual in (0, 1]")
        self.model_config = {"width": int(width), "max_residual": float(max_residual)}
        self.max_residual = float(max_residual)
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.encoder1 = ResidualBlock(width)
        self.down1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.encoder2 = ResidualBlock(width * 2)
        self.down2 = nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(ResidualBlock(width * 4), ResidualBlock(width * 4))
        self.up2 = nn.Conv2d(width * 4, width * 2, 3, padding=1)
        self.decoder2 = ResidualBlock(width * 2)
        self.up1 = nn.Conv2d(width * 2, width, 3, padding=1)
        self.decoder1 = ResidualBlock(width)
        self.correction = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("expected nonempty B x 3 x H x W RGB tensor")
        if not 0 <= strength <= 1:
            raise ValueError("strength must be in [0, 1]")
        if strength == 0:
            return x
        h, w = x.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        mode = "reflect" if h > ph and w > pw else "replicate"
        features = F.pad(x, (0, pw, 0, ph), mode=mode) if ph or pw else x
        skip1 = self.encoder1(self.stem(features))
        skip2 = self.encoder2(self.down1(skip1))
        features = self.bottleneck(self.down2(skip2))
        features = self.decoder2(self.up2(F.interpolate(
            features, size=skip2.shape[-2:], mode="bilinear", align_corners=False,
        )) + skip2)
        features = self.decoder1(self.up1(F.interpolate(
            features, size=skip1.shape[-2:], mode="bilinear", align_corners=False,
        )) + skip1)
        delta = self.correction(features)[..., :h, :w].float().tanh()
        return (x + strength * self.max_residual * delta).clamp(0, 1)


# 以下 NAF 模块参考官方 NAFNet，并改为 PyTorch 原生通道 LayerNorm、reflect
# padding 和本项目的有界 RGB 修正输出；不包含 BasicSR 或预训练权重。
# https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py
# MIT License
# Copyright (c) 2022 megvii-model
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class ChannelLayerNorm(nn.Module):
    """每个空间位置沿通道归一化；交给原生算子处理反向传播与混合精度。"""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, eps=1e-6,
        ).permute(0, 3, 1, 2)


class NAFBlock(nn.Module):
    """乘法门控、深度卷积和通道注意力组成的两段残差块。"""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = ChannelLayerNorm(channels)
        self.expand = nn.Conv2d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels)
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm2 = ChannelLayerNorm(channels)
        self.ffn_expand = nn.Conv2d(channels, 2 * channels, 1)
        self.ffn_project = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def gate(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.gate(self.depthwise(self.expand(self.norm1(x))))
        branch = self.project(branch * self.attention(branch))
        intermediate = x + self.beta * branch
        branch = self.ffn_project(self.gate(self.ffn_expand(self.norm2(intermediate))))
        return intermediate + self.gamma * branch


class NAFRGBRestorer(nn.Module):
    """NAFNet 小型多尺度骨干，保留独立 3→3 修复器的输出及旁路契约。"""

    checkpoint_format = CHECKPOINT_FORMAT_NAFNET

    def __init__(
        self,
        width: int = 32,
        max_residual: float = 0.5,
        encoder_blocks: tuple[int, ...] | list[int] = (1, 2, 4),
        middle_blocks: int = 4,
        decoder_blocks: tuple[int, ...] | list[int] = (2, 2, 1),
    ):
        super().__init__()
        if not isinstance(width, int) or width < 4 or not 0 < max_residual <= 1:
            raise ValueError("width must be an integer >= 4 and max_residual in (0, 1]")
        encoder_blocks, decoder_blocks = tuple(encoder_blocks), tuple(decoder_blocks)
        if not encoder_blocks or len(encoder_blocks) != len(decoder_blocks):
            raise ValueError("encoder_blocks and decoder_blocks must have the same nonzero length")
        if any(not isinstance(count, int) or count < 1
               for count in (*encoder_blocks, middle_blocks, *decoder_blocks)):
            raise ValueError("all block counts must be positive integers")
        self.model_config = {
            "architecture": "nafnet", "width": width, "max_residual": float(max_residual),
            "encoder_blocks": list(encoder_blocks), "middle_blocks": middle_blocks,
            "decoder_blocks": list(decoder_blocks),
        }
        self.max_residual = float(max_residual)
        self.padding_multiple = 2 ** len(encoder_blocks)
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        channels = width
        for count in encoder_blocks:
            self.encoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, stride=2))
            channels *= 2
        self.middle = nn.Sequential(*(NAFBlock(channels) for _ in range(middle_blocks)))
        for count in decoder_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, 2 * channels, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
        self.correction = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("expected nonempty B x 3 x H x W RGB tensor")
        if not 0 <= strength <= 1:
            raise ValueError("strength must be in [0, 1]")
        if strength == 0:
            return x
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.padding_multiple, (-w) % self.padding_multiple
        mode = "reflect" if h > ph and w > pw else "replicate"
        features = F.pad(x, (0, pw, 0, ph), mode=mode) if ph or pw else x
        features = self.stem(features)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        features = self.middle(features)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            features = decoder(up(features) + skip)
        delta = self.correction(features)[..., :h, :w].float().tanh()
        return (x + strength * self.max_residual * delta).clamp(0, 1)


def build_rgb_restorer(model_config: dict) -> nn.Module:
    """旧配置默认使用原结构；新结构必须显式选择，拼错选项时立即报错。"""
    options = dict(model_config)
    architecture = options.pop("architecture", "rgb_restorer")
    if architecture == "rgb_restorer":
        return RGBRestorer(**options)
    if architecture == "nafnet":
        return NAFRGBRestorer(**options)
    if architecture == "pixel_diffusion_nafnet":
        # 延迟导入避免扩散模块复用 NAFBlock 时出现循环导入。
        from models.rgb_diffusion_restoration import PixelDiffusionRGBRestorer
        return PixelDiffusionRGBRestorer(**options)
    raise ValueError(f"unknown RGB restoration architecture: {architecture!r}")


def load_rgb_restorer(path: str, device: str | torch.device = "cpu") -> nn.Module:
    """显式版本、严格加载；不接受分割模型或其他重建头的 checkpoint。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_format = checkpoint.get("format")
    formats = {
        CHECKPOINT_FORMAT: "rgb_restorer", CHECKPOINT_FORMAT_NAFNET: "nafnet",
        "rgb_restoration_diffusion_v1": "pixel_diffusion_nafnet",
    }
    if checkpoint_format not in formats:
        raise ValueError(f"unsupported RGB restoration checkpoint format: {checkpoint_format!r}")
    config = checkpoint["model_config"]
    expected_architecture = formats[checkpoint_format]
    if config.get("architecture", "rgb_restorer") != expected_architecture:
        raise ValueError("checkpoint format and model_config architecture do not match")
    model = build_rgb_restorer(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()
