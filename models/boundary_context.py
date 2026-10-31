# -*- coding: utf-8 -*-
"""补边器单变量 RGB 条件对照：共同层来自旧随机 e0，新增输入通道从零开始。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.boundary_restore import BoundaryRestoreNet, FORMAT as BASE_FORMAT, state_sha256

FORMAT = 'boundary_context_v1'


class BoundaryContextNet(BoundaryRestoreNet):
    """输入保持 heat 与 RGB 两个参数；损失与输出仍是单通道边界概率。"""
    checkpoint_format = FORMAT

    def __init__(self, width=16, max_residual=1., seed=20261025):
        super().__init__(width=width, max_residual=max_residual, seed=seed)
        self.shared_initial_state_sha256 = self.initial_state_sha256
        self.base_architecture = dict(self.architecture)
        old_stem = self.stem
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.stem = nn.Conv2d(4, width, 3, padding=1)
            with torch.no_grad():
                self.stem.weight.zero_()
                self.stem.weight[:, :1].copy_(old_stem.weight)
                self.stem.bias.copy_(old_stem.bias)
        self.architecture = {**self.base_architecture, 'channels': 4, 'heat_channels': 1,
            'context_channels': 3, 'context_scaling': 'RGB uint8 / 255',
            'context_injection': 'concatenate heat then RGB before stem; RGB weights zero at e0'}
        self.initial_state_sha256 = state_sha256(self)
        self.initialization_receipt = dict(source='reconstructed_same_seed_random_e0',
            shared_initial_state_sha256=self.shared_initial_state_sha256,
            initial_state_sha256=self.initial_state_sha256,
            parameters=sum(p.numel() for p in self.parameters()),
            new_rgb_parameters=int(self.stem.weight[:, 1:].numel()),
            shared_tensors_exact=True, heat_stem_exact=True, rgb_stem_zero=True,
            output_correction_zero=True, trained_weights_inherited=False)

    def initialize_from_e0(self, checkpoint):
        """正式实验必须绑定封存旧 e0；拒绝 e5、伪装的步数或不兼容架构。"""
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (payload.get('format') != BASE_FORMAT or payload.get('epoch') != 0
                or payload.get('global_step') != 0):
            raise ValueError('Only an old boundary_restore random e0 with zero updates is allowed')
        if payload.get('architecture') != self.base_architecture:
            raise ValueError('Old e0 architecture differs from the shared architecture')
        base = BoundaryRestoreNet(**{k: self.base_architecture[k] for k in ('width','max_residual','seed')})
        base.load_state_dict(payload['model_state_dict'], strict=True)
        actual = state_sha256(base)
        if (actual != self.shared_initial_state_sha256 or actual != payload.get('model_sha256')
                or actual != payload.get('initial_state_sha256')):
            raise ValueError('Old e0 weights are not the exact shared random initialization')
        state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        old_weight = state['stem.weight']
        state['stem.weight'] = torch.zeros_like(self.stem.weight, device='cpu')
        state['stem.weight'][:, :1] = old_weight
        self.load_state_dict(state, strict=True)
        if state_sha256(self) != self.initial_state_sha256:
            raise RuntimeError('Shared e0 conversion does not match the context initialization')
        self.initialization_receipt = {**self.initialization_receipt, 'source': 'sealed_old_random_e0',
            'checkpoint': str(Path(checkpoint)),
            'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
        return dict(self.initialization_receipt)

    @staticmethod
    def validate_context(boundary, context_rgb):
        BoundaryRestoreNet.validate_input(boundary)
        if (not torch.is_tensor(context_rgb) or context_rgb.ndim != 4 or context_rgb.shape[1] != 3
                or context_rgb.shape[0] != boundary.shape[0] or context_rgb.shape[-2:] != boundary.shape[-2:]
                or context_rgb.device != boundary.device or not context_rgb.is_floating_point()):
            raise ValueError('Context must be floating B3HW RGB aligned to boundary on the same device')
        if not bool(torch.isfinite(context_rgb).all()) or bool(((context_rgb < 0) | (context_rgb > 1)).any()):
            raise ValueError('Context RGB must be finite and within [0,1]')

    def residual(self, boundary, context_rgb):
        self.validate_context(boundary, context_rgb)
        h, w = boundary.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        mode = 'reflect' if h > ph and w > pw else 'replicate'
        value = torch.cat((boundary.float(), context_rgb.float()), dim=1)
        if ph or pw:
            value = F.pad(value, (0, pw, 0, ph), mode=mode)
        first = self.encoder1(self.stem(value))
        second = self.encoder2(self.down1(first))
        value = self.bottleneck(self.down2(second))
        value = self.decoder2(self.up2(F.interpolate(value, size=second.shape[-2:],
                                                   mode='bilinear', align_corners=False)) + second)
        value = self.decoder1(self.up1(F.interpolate(value, size=first.shape[-2:],
                                                   mode='bilinear', align_corners=False)) + first)
        return self.max_residual * self.correction(value)[..., :h, :w].tanh()

    def forward(self, boundary, context_rgb):
        return (boundary + self.residual(boundary, context_rgb)).clamp(0, 1)


def load_boundary_context(checkpoint, device='cpu'):
    """独立版本、完整架构和严格键校验；不会把旧单通道权重当新模型吞入。"""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('format') != FORMAT:
        raise ValueError('Unsupported boundary context checkpoint format')
    architecture = payload['architecture']
    model = BoundaryContextNet(**{k: architecture[k] for k in ('width','max_residual','seed')})
    if model.architecture != architecture:
        raise ValueError('Checkpoint architecture differs from this RGB context implementation')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if state_sha256(model) != payload['model_sha256']:
        raise ValueError('Boundary context state digest differs')
    if payload.get('initial_state_sha256') != model.initial_state_sha256:
        raise ValueError('Boundary context initialization digest differs')
    return model.to(device).eval().requires_grad_(False)
