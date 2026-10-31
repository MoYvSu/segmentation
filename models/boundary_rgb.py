# -*- coding: utf-8 -*-
"""RGB条件NAF边界恢复；独立版本，不改变旧单通道checkpoint合同。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.boundary_naf import BoundaryNAFNet


FORMAT = 'boundary_rgb_naf_v1'
INPUT_MODES = ('boundary', 'rgb_boundary')


class BoundaryRGBNAFNet(BoundaryNAFNet):
    """输入T或[RGB,T]，输出一通道边界概率；全局跳连仅使用T。

    两组从同seed重新构建公共骨干，RGB首层为零，边界首层使用同一随机值。
    因而数学上的初态函数相同；没有载入任何训练过的恢复器权重。
    RGB首层仍可从首步获得梯度，并非冻结或永久忽略RGB。
    """

    checkpoint_format = FORMAT

    def __init__(self, width=48, encoder_blocks=(1, 1, 2, 4), middle_blocks=2,
                 decoder_blocks=(1, 1, 1, 1), seed=20261026,
                 input_mode='rgb_boundary', upsampling='pixelshuffle'):
        if input_mode not in INPUT_MODES:
            raise ValueError('input_mode must be boundary or rgb_boundary')
        super().__init__(width, encoder_blocks, middle_blocks, decoder_blocks, seed,
                         upsampling=upsampling)
        self.input_mode = input_mode
        self.input_channels = 4 if input_mode == 'rgb_boundary' else 1
        self.boundary_channel = 3 if input_mode == 'rgb_boundary' else 0
        if self.input_channels == 4:
            old_intro = self.intro
            with torch.random.fork_rng(devices=[]):
                # 新增层的构建不改变调用方CPU/CUDA随机流。
                torch.random.default_generator.manual_seed(seed)
                self.intro = nn.Conv2d(4, width, 3, padding=1)
            with torch.no_grad():
                self.intro.weight.zero_()
                self.intro.weight[:, 3:4].copy_(old_intro.weight)
                self.intro.bias.copy_(old_intro.bias)
        self.architecture.update(input_mode=input_mode)
        self.output_contract.update(
            input_channels=self.input_channels,
            input_order=['R', 'G', 'B', 'binary_boundary'] if self.input_channels == 4 else ['binary_boundary'],
            input_range=[0, 1], input_boundary_binary=True,
            logits='ending_features_plus_boundary_input_only',
            rgb_initialization='zero_intro_channels; shared_random_boundary_and_backbone',
            allows_addition=True, allows_deletion=True)

    def forward_logits(self, inputs):
        if (not torch.is_tensor(inputs) or inputs.ndim != 4
                or inputs.shape[1] != self.input_channels or min(inputs.shape) < 1
                or not inputs.is_floating_point()):
            raise ValueError(f'Expected nonempty floating B{self.input_channels}HW input')
        if not bool(torch.isfinite(inputs).all()) or bool(((inputs < 0) | (inputs > 1)).any()):
            raise ValueError('RGB and boundary inputs must be finite and within [0,1]')
        boundary = inputs[:, self.boundary_channel:self.boundary_channel + 1]
        if bool(((boundary != 0) & (boundary != 1)).any()):
            raise ValueError('Boundary input must be binary T, not continuous affinity or skeleton adapter')
        height, width = inputs.shape[-2:]
        ph, pw = (-height) % self.padding_multiple, (-width) % self.padding_multiple
        mode = 'reflect' if height > ph and width > pw else 'replicate'
        padded = F.pad(inputs, (0, pw, 0, ph), mode=mode) if ph or pw else inputs
        value = self.intro(padded)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            value = encoder(value)
            skips.append(value)
            value = down(value)
        value = self.middle(value)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            value = decoder(up(value) + skip)
        baseline = padded[:, self.boundary_channel:self.boundary_channel + 1]
        return (self.ending(value) + baseline)[..., :height, :width]

    def residual(self, inputs):
        return self(inputs) - inputs[:, self.boundary_channel:self.boundary_channel + 1]
