# -*- coding: utf-8 -*-
"""单通道连续边界补全器：随机小 U-Net，零初始化残差，初始严格恒等。"""
from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'boundary_restore_v1'


def state_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class BoundaryBlock(nn.Module):
    def __init__(self, width, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(4, width), nn.SiLU())

    def forward(self, value):
        return value + self.body(value)


class BoundaryRestoreNet(nn.Module):
    """只读一张高值表示边界的概率图；不读取 RGB、语义、实例 ID 或 GT。"""
    checkpoint_format = FORMAT

    def __init__(self, width=16, max_residual=1.0, seed=20261025):
        super().__init__()
        if type(width) is not int or width < 4 or width % 4:
            raise ValueError('Width must be an integer multiple of 4, at least 4')
        if not 0 < float(max_residual) <= 1:
            raise ValueError('max_residual must be within (0,1]')
        self.max_residual = float(max_residual)
        self.architecture = dict(width=width, max_residual=self.max_residual, seed=int(seed),
                                 channels=1, residual='tanh_probability',
                                 normalization='GroupNorm4', bottleneck_dilations=[2, 4])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.stem = nn.Conv2d(1, width, 3, padding=1)
            self.encoder1 = BoundaryBlock(width)
            self.down1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
            self.encoder2 = BoundaryBlock(width * 2)
            self.down2 = nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1)
            self.bottleneck = nn.Sequential(BoundaryBlock(width * 4, 2), BoundaryBlock(width * 4, 4))
            self.up2 = nn.Conv2d(width * 4, width * 2, 3, padding=1)
            self.decoder2 = BoundaryBlock(width * 2)
            self.up1 = nn.Conv2d(width * 2, width, 3, padding=1)
            self.decoder1 = BoundaryBlock(width)
            self.correction = nn.Conv2d(width, 1, 3, padding=1)
            nn.init.zeros_(self.correction.weight)
            nn.init.zeros_(self.correction.bias)
        self.initial_state_sha256 = state_sha256(self)

    @staticmethod
    def validate_input(value):
        if (not torch.is_tensor(value) or value.ndim != 4 or value.shape[1] != 1
                or min(value.shape) < 1 or not value.is_floating_point()):
            raise ValueError('Boundary input must be a nonempty floating B1HW tensor')
        if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
            raise ValueError('Continuous boundary input must be finite and within [0,1]')

    def residual(self, boundary):
        """返回有符号修正量；分块推理应先融合此量，再在全图统一 clamp。"""
        self.validate_input(boundary)
        h, w = boundary.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        mode = 'reflect' if h > ph and w > pw else 'replicate'
        value = F.pad(boundary.float(), (0, pw, 0, ph), mode=mode) if ph or pw else boundary.float()
        first = self.encoder1(self.stem(value))
        second = self.encoder2(self.down1(first))
        value = self.bottleneck(self.down2(second))
        value = self.decoder2(self.up2(F.interpolate(value, size=second.shape[-2:],
                                                   mode='bilinear', align_corners=False)) + second)
        value = self.decoder1(self.up1(F.interpolate(value, size=first.shape[-2:],
                                                   mode='bilinear', align_corners=False)) + first)
        return self.max_residual * self.correction(value)[..., :h, :w].tanh()

    def forward(self, boundary):
        return (boundary + self.residual(boundary)).clamp(0, 1)


def region_masks(batch, preserve_threshold=.2):
    """各区域独立分母；保留项约束未被擦除的真边界。"""
    valid = batch['valid'].bool()
    gap = batch['gap_mask'].bool()
    noise = batch['noise_mask'].bool()
    identity = batch['identity'].reshape(-1).bool()[:, None, None, None]
    return dict(valid=valid, gap=gap & valid, noise=noise & valid,
                preserve=valid & (batch['target'] > float(preserve_threshold)) & ~gap,
                identity=valid & identity, unchanged=valid & ~(gap | noise))


def boundary_restore_loss(prediction, batch, *, valid_weight=1., gap_weight=1., noise_weight=.5,
                          preserve_weight=.25, preserve_threshold=.2, normalizers=None):
    """L1 各区域分别归一再加权；可传有效 batch 的分母进行等价梯度累积。"""
    required = ('input', 'target', 'valid', 'gap_mask', 'noise_mask', 'identity')
    if not all(k in batch for k in required):
        raise ValueError(f'Missing loss keys: {required}')
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError('Prediction must be B1HW')
    for name in required[:-1]:
        value = batch[name]
        if not torch.is_tensor(value) or value.shape != prediction.shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Invalid {name} tensor shape or values')
    for name in ('valid', 'gap_mask', 'noise_mask'):
        if bool(((batch[name] != 0) & (batch[name] != 1)).any()):
            raise ValueError(f'{name} must be binary')
    valid = batch['valid'].bool()
    gaps, noise = batch['gap_mask'].bool(), batch['noise_mask'].bool()
    if bool(((gaps | noise) & ~valid).any()):
        raise ValueError('Corruption masks must stay inside reliable valid support')
    if bool((valid.sum(dim=(1, 2, 3)) == 0).any()):
        raise ValueError('Every training sample needs reliable supervision')
    identity = batch['identity']
    if (identity.numel() != prediction.shape[0]
            or bool(((identity != 0) & (identity != 1)).any())):
        raise ValueError('Identity must have one boolean per sample')
    identity = identity.reshape(-1).bool()
    identity_pixels = valid & identity[:, None, None, None]
    if bool(((gaps | noise) & identity_pixels).any()) or not torch.equal(
            batch['input'][identity_pixels], batch['target'][identity_pixels]):
        raise ValueError('Identity samples must have unchanged targets and no corruption masks')
    if (not 0 < float(valid_weight) < float('inf')
            or not all(0 <= float(v) < float('inf') for v in (gap_weight, noise_weight, preserve_weight))
            or not 0 <= float(preserve_threshold) <= 1):
        raise ValueError('Invalid region loss weights or preservation threshold')
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError('Non-finite boundary prediction')
    if bool(((batch['target'][valid] < 0) | (batch['target'][valid] > 1)).any()):
        raise ValueError('Reliable boundary targets must be within [0,1]')
    error = (prediction - batch['target']).abs()
    masks = region_masks(batch, preserve_threshold)
    coefficients = dict(valid=float(valid_weight), gap=float(gap_weight), noise=float(noise_weight),
                        preserve=float(preserve_weight))
    loss = prediction.sum() * 0.
    terms = {}
    for name, coefficient in coefficients.items():
        count = int(masks[name].sum())
        denominator = count if normalizers is None else normalizers[name]
        if type(denominator) is not int or denominator < count:
            raise ValueError('Effective-batch region denominator cannot be below local support')
        term = error[masks[name]].sum() / denominator if denominator else prediction.sum() * 0.
        loss = loss + coefficient * term
        terms[name] = dict(coefficient=coefficient, local_pixels=count, denominator=denominator,
                           contribution=float((coefficient * term).detach()))
    def measure(mask):
        count = int(mask.sum())
        return dict(pixels=count, absolute_error=float(error[mask].sum().detach()),
                    mae=float(error[mask].mean().detach()) if count else None)
    stats = dict(loss=float(loss.detach()), regions={name: measure(mask) for name, mask in masks.items()},
                 terms=terms, identity_samples=int(identity.sum()),
                 unknown_pixels=int((~valid).sum()), samples=int(prediction.shape[0]))
    return loss, stats


def load_boundary_restorer(checkpoint, device='cpu'):
    """仅加载本实验格式，严格校验实际网络张量；不宽松吞掉错误键。"""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('format') != FORMAT:
        raise ValueError('Unsupported boundary restoration checkpoint format')
    arch = payload['architecture']
    model = BoundaryRestoreNet(width=arch['width'], max_residual=arch['max_residual'], seed=arch['seed'])
    if model.architecture != arch:
        raise ValueError('Checkpoint architecture differs from this implementation')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if state_sha256(model) != payload['model_sha256']:
        raise ValueError('Boundary model state digest differs')
    return model.to(device).eval().requires_grad_(False)
