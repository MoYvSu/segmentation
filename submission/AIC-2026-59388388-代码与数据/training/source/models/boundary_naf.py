# -*- coding: utf-8 -*-
"""独立NAFNet边界生成器与Patch判别器，不读取SAM2内部特征。

NAFBlock与U形骨干据官方NAFNet实现重写为原生PyTorch通道LayerNorm：
https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py
MIT License, Copyright (c) 2022 megvii-model.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Dual-CGB公开正文给出块数但未在此处核到完整原源码。width48按其17.63M参数
量重建；不能仅因四舍五入相同宣称完整复现。分类接口显式使用全局残差logit和
sigmoid；不套用旧补边器零初始化、保持损失或修复后回退规则。
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


FORMAT = "boundary_naf_v1"
DISCRIMINATOR_FORMAT = "boundary_naf_discriminator_v1"
UPSAMPLING_MODES = ("pixelshuffle", "project_bilinear_v1")


def _positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_boundary(value):
    if (not torch.is_tensor(value) or value.ndim != 4 or value.shape[1] != 1
            or min(value.shape) < 1 or not value.is_floating_point()):
        raise ValueError("Expected nonempty floating B1HW boundary input")
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError("Boundary probabilities must be finite and within [0,1]")


class BoundaryChannelLayerNorm(nn.Module):
    """逐位置沿通道归一化；不混合相邻像素或不同样本。"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, value):
        value = value.permute(0, 2, 3, 1)
        return F.layer_norm(value, (value.shape[-1],), self.weight, self.bias, 1e-6).permute(0, 3, 1, 2)


class BoundaryNAFBlock(nn.Module):
    """原NAF的SimpleGate、通道注意力与两段可学习残差。"""

    def __init__(self, channels):
        super().__init__()
        self.norm1 = BoundaryChannelLayerNorm(channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.depthwise = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm2 = BoundaryChannelLayerNorm(channels)
        self.ffn_expand = nn.Conv2d(channels, channels * 2, 1)
        self.ffn_project = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def gate(value):
        first, second = value.chunk(2, dim=1)
        return first * second

    def forward(self, value):
        branch = self.gate(self.depthwise(self.expand(self.norm1(value))))
        branch = self.project(branch * self.attention(branch))
        value = value + self.beta * branch
        branch = self.ffn_project(self.gate(self.ffn_expand(self.norm2(value))))
        return value + self.gamma * branch


class BoundaryNAFNet(nn.Module):
    """单通道NAF生成器，输入为高值表示边界的概率图。

    ``forward_logits``专供BCEWithLogits；``forward``返回同尺寸概率；
    ``residual``只供旧分块接口拼接有符号差值。随机e0并非恒等映射，是否启用
    应由外层显式开关控制，不能把随机初态当作原部署的等价结果。
    """

    checkpoint_format = FORMAT

    def __init__(self, width=48, encoder_blocks=(1, 1, 2, 4), middle_blocks=2,
                 decoder_blocks=(1, 1, 1, 1), seed=20261026,
                 upsampling="pixelshuffle"):
        super().__init__()
        _positive_integer(width, "width", 2)
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if upsampling not in UPSAMPLING_MODES:
            raise ValueError("upsampling must be pixelshuffle or project_bilinear_v1")
        encoder_blocks, decoder_blocks = tuple(encoder_blocks), tuple(decoder_blocks)
        if not encoder_blocks or len(encoder_blocks) != len(decoder_blocks):
            raise ValueError("Encoder and decoder must have the same nonzero number of stages")
        for value in (*encoder_blocks, middle_blocks, *decoder_blocks):
            _positive_integer(value, "block count")
        self.architecture = {
            "width": width, "encoder_blocks": list(encoder_blocks),
            "middle_blocks": middle_blocks, "decoder_blocks": list(decoder_blocks), "seed": seed,
        }
        self.output_contract = {
            "channels": 1, "polarity": "high_is_boundary", "range": [0, 1],
            "logits": "ending_features_plus_input_probability", "initial_identity": False,
            "padding": "reflect_or_replicate_to_stage_multiple", "post_restoration_revert": False,
        }
        self.padding_multiple = 2 ** len(encoder_blocks)
        with torch.random.fork_rng(devices=[]):
            # 仅设CPU初始化流；torch.manual_seed还会改CUDA流，而此fork只封存CPU。
            torch.random.default_generator.manual_seed(seed)
            self.intro = nn.Conv2d(1, width, 3, padding=1)
            self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
            self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
            channels = width
            for count in encoder_blocks:
                self.encoders.append(nn.Sequential(*(BoundaryNAFBlock(channels) for _ in range(count))))
                self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
                channels *= 2
            self.middle = nn.Sequential(*(BoundaryNAFBlock(channels) for _ in range(middle_blocks)))
            for count in decoder_blocks:
                self.ups.append(nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)))
                channels //= 2
                self.decoders.append(nn.Sequential(*(BoundaryNAFBlock(channels) for _ in range(count))))
            self.ending = nn.Conv2d(width, 1, 3, padding=1)
        self.upsampling = upsampling
        if upsampling == "project_bilinear_v1":
            # 先完成旧模型的全部抽样，使encoder/decoder/ending等公共初态完全相同。
            # 新up再用独立CPU流随机初始化，不读取任何已训练权重。
            projection_seed = seed + 20261029
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(projection_seed)
                projected_ups = nn.ModuleList()
                for previous in self.ups:
                    channels = previous[0].in_channels
                    projected_ups.append(nn.Sequential(
                        nn.Conv2d(channels, channels // 2, 1, bias=False),
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)))
                self.ups = projected_ups
            self.architecture["upsampling"] = upsampling
            self.output_contract["upsampling"] = {
                "version": upsampling,
                "projection": "low_resolution_1x1_bias_false",
                "resize": "bilinear_x2_align_corners_false",
                "initialization": "independent_random_after_shared_legacy_construction",
                "initialization_seed": projection_seed,
            }
        # 默认pixelshuffle不写新增元数据，旧architecture/checkpoint合同保持。
        # 配置若使用默认路径也应省略upsampling键，便于旧导出器逐字典核对。

    def forward_logits(self, boundary):
        _validate_boundary(boundary)
        height, width = boundary.shape[-2:]
        ph, pw = (-height) % self.padding_multiple, (-width) % self.padding_multiple
        mode = "reflect" if height > ph and width > pw else "replicate"
        padded = F.pad(boundary, (0, pw, 0, ph), mode=mode) if ph or pw else boundary
        value = self.intro(padded)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            value = encoder(value)
            skips.append(value)
            value = down(value)
        value = self.middle(value)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            value = decoder(up(value) + skip)
        return (self.ending(value) + padded)[..., :height, :width]

    def forward(self, boundary):
        return self.forward_logits(boundary).sigmoid()

    def residual(self, boundary):
        return self(boundary) - boundary

    def parameter_summary(self):
        return {
            "parameters": sum(value.numel() for value in self.parameters()),
            "architecture": copy.deepcopy(self.architecture),
            "output_contract": copy.deepcopy(self.output_contract),
        }


class BoundaryDiscriminator(nn.Module):
    """五层谱归一化Patch判别器，输出未sigmoid的真假logit。

    kernels=4，strides=2/2/2/1/1，感受野70、输出格距8。可靠监督掩码必须覆盖
    一格输出依赖的全部输入像素；``valid_patch_mask``不将卷积padding算作可靠。
    更新G时使用D.eval()并冻结D参数，避免谱归一化u/v缓冲在G步中变化。
    """

    checkpoint_format = DISCRIMINATOR_FORMAT
    receptive_field = 70
    output_stride = 8
    minimum_input_size = 24
    strides = (2, 2, 2, 1, 1)

    def __init__(self, width=64, seed=20261026):
        super().__init__()
        _positive_integer(width, "width", 2)
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.architecture = {"width": width, "seed": seed}
        channels = (1, width, width * 2, width * 4, width * 8, 1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.convs = nn.ModuleList([
                spectral_norm(nn.Conv2d(channels[i], channels[i + 1], 4, stride=stride, padding=1))
                for i, stride in enumerate(self.strides)
            ])

    def forward(self, boundary):
        _validate_boundary(boundary)
        if min(boundary.shape[-2:]) < self.minimum_input_size:
            raise ValueError(f"Discriminator input must be at least {self.minimum_input_size} pixels per side")
        value = boundary
        for index, conv in enumerate(self.convs):
            value = conv(value)
            if index != len(self.convs) - 1:
                value = F.leaky_relu(value, negative_slope=0.2, inplace=False)
        return value

    @classmethod
    def valid_patch_mask(cls, valid):
        """返回B1H'W'布尔掩码；任一依赖像素unknown或图外即不监督。"""
        if (not torch.is_tensor(valid) or valid.ndim != 4 or valid.shape[1] != 1
                or min(valid.shape) < 1):
            raise ValueError("Expected nonempty B1HW valid mask")
        if min(valid.shape[-2:]) < cls.minimum_input_size:
            raise ValueError("Valid mask is too small for discriminator")
        if valid.is_floating_point() and not bool(torch.isfinite(valid).all()):
            raise ValueError("Valid mask must be finite")
        if bool(((valid != 0) & (valid != 1)).any()):
            raise ValueError("Valid mask must contain only zero and one")
        value = valid.to(dtype=torch.float32)
        kernel = torch.ones(1, 1, 4, 4, device=value.device, dtype=value.dtype)
        for stride in cls.strides:
            value = (F.conv2d(value, kernel, stride=stride, padding=1) == 16).float()
        return value.bool()
