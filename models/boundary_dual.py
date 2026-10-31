# -*- coding: utf-8 -*-
"""T8二值线图与连续N1双输入NAF；旧单通道模型合同保持不变。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.boundary_rgb import BoundaryRGBNAFNet


FORMAT = 'boundary_dual_naf_v1'
INPUT_MODE = 'boundary_continuous'


class BoundaryDualNAFNet(BoundaryRGBNAFNet):
    """输入[N,2,H,W]依次为T8和连续N1，输出一通道边界概率。

    先原样构建单通道网络，再添加独立零初始化的连续输入卷积；公共参数
    及调用方随机流不受新增层影响。全局logit跳连仍只使用T8。
    ``use_continuous=False``显式走原单通道计算路径，供工程兼容核验。
    """

    checkpoint_format = FORMAT
    # 在创建逐元素检查临时张量前拒绝误传的大图/批次；正常batch16×2×256²为2M。
    max_input_elements = 16_777_216

    def __init__(self, width=48, encoder_blocks=(1, 1, 2, 4), middle_blocks=2,
                 decoder_blocks=(1, 1, 1, 1), seed=20261026,
                 input_mode=INPUT_MODE, upsampling='pixelshuffle'):
        if input_mode != INPUT_MODE:
            raise ValueError(f'input_mode must be {INPUT_MODE}')
        super().__init__(width, encoder_blocks, middle_blocks, decoder_blocks, seed,
                         input_mode='boundary', upsampling=upsampling)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 20261031)
            self.continuous_intro = nn.Conv2d(1, width, 3, padding=1, bias=False)
        with torch.no_grad():
            self.continuous_intro.weight.zero_()
        self.input_mode = INPUT_MODE
        self.input_channels = 2
        self.boundary_channel = 0
        self.architecture.update(input_mode=INPUT_MODE)
        self.output_contract.pop('rgb_initialization', None)
        self.output_contract.update(
            input_channels=2,
            input_order=['binary_boundary_T8', 'continuous_N1_boundary'],
            input_range=[0, 1], input_boundary_binary=True,
            continuous_input_range=[0, 1], continuous_input_binarized=False,
            logits='ending_features_plus_binary_boundary_input_only',
            input_fusion='legacy_intro_T8_plus_separate_continuous_intro',
            continuous_initialization='zero_conv3x3_no_bias_after_shared_legacy_construction',
            continuous_initialization_seed=seed + 20261031,
            disabled_continuous_path='legacy_single_channel_forward_logits',
        )

    @classmethod
    def _validate_inputs(cls, inputs):
        if (not torch.is_tensor(inputs) or inputs.ndim != 4
                or inputs.shape[1] != 2 or min(inputs.shape) < 1
                or not inputs.is_floating_point()):
            raise ValueError('Expected nonempty floating B2HW [T8, continuous_N1] input')
        if inputs.numel() > cls.max_input_elements:
            raise ValueError(f'Dual input exceeds bounded {cls.max_input_elements}-element batch')
        if not bool(torch.isfinite(inputs).all()) or bool(((inputs < 0) | (inputs > 1)).any()):
            raise ValueError('T8 and continuous N1 inputs must be finite and within [0,1]')
        boundary = inputs[:, :1]
        if bool(((boundary != 0) & (boundary != 1)).any()):
            raise ValueError('First input channel must be binary T8')

    def forward_logits(self, inputs, *, use_continuous=True):
        self._validate_inputs(inputs)
        if type(use_continuous) is not bool:
            raise ValueError('use_continuous must be bool')
        # 保持原intro看到的张量布局；不将旧卷积扩成两通道。
        boundary = inputs[:, :1].contiguous()
        if not use_continuous:
            # 父类会按self.input_channels检查，不能直接调用其双通道入口。
            return self._legacy_logits(boundary)
        height, width = boundary.shape[-2:]
        ph, pw = (-height) % self.padding_multiple, (-width) % self.padding_multiple
        mode = 'reflect' if height > ph and width > pw else 'replicate'
        continuous = inputs[:, 1:2].contiguous()
        if ph or pw:
            boundary = F.pad(boundary, (0, pw, 0, ph), mode=mode)
            continuous = F.pad(continuous, (0, pw, 0, ph), mode=mode)
        value = self.intro(boundary) + self.continuous_intro(continuous)
        return self._decode_logits(value, boundary)[..., :height, :width]

    def _legacy_logits(self, boundary):
        """原单通道计算次序，不增添零分支相加；供关闭路径桥接。"""
        height, width = boundary.shape[-2:]
        ph, pw = (-height) % self.padding_multiple, (-width) % self.padding_multiple
        mode = 'reflect' if height > ph and width > pw else 'replicate'
        padded = F.pad(boundary, (0, pw, 0, ph), mode=mode) if ph or pw else boundary
        value = self.intro(padded)
        return self._decode_logits(value, padded)[..., :height, :width]

    def _decode_logits(self, value, boundary):
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            value = encoder(value)
            skips.append(value)
            value = down(value)
        value = self.middle(value)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            value = decoder(up(value) + skip)
        return self.ending(value) + boundary

    def forward(self, inputs, *, use_continuous=True):
        return self.forward_logits(inputs, use_continuous=use_continuous).sigmoid()

    def residual(self, inputs, *, use_continuous=True):
        return self(inputs, use_continuous=use_continuous) - inputs[:, :1]
