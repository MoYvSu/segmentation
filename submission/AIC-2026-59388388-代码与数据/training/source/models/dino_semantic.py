# -*- coding: utf-8 -*-
"""冻结官方 DINOv3 通用骨干，随机初始化轻量语义头；不包含几何或任务预训练权重。"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(module):
    """包括 buffer 的实际张量摘要，避免只检查 requires_grad 而漏掉状态变化。"""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_dino_backbone(source_dir, checkpoint, expected_sha256):
    """直接导入指定官方源码；严格读取本地裸 state_dict，无下载、hub 或用户缓存。"""
    source = Path(source_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not (source / 'dinov3/hub/backbones.py').is_file():
        raise FileNotFoundError(f'Official local DINOv3 source missing: {source}')
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError('An explicit SHA256 for the approved backbone is required')
    actual_sha = file_sha256(checkpoint)
    if actual_sha != expected_sha256:
        raise ValueError('DINO backbone checkpoint SHA256 differs')
    # 已导入的另一个 dinov3 副本不能悄悄遮蔽指定源码。
    for name, module in list(sys.modules.items()):
        if name == 'dinov3' or name.startswith('dinov3.'):
            filename = getattr(module, '__file__', None)
            if filename and not Path(filename).resolve().is_relative_to(source):
                raise RuntimeError(f'Another DINOv3 source is already imported: {filename}')
    sys.path.insert(0, str(source))
    try:
        factory = importlib.import_module('dinov3.hub.backbones')
        if not Path(factory.__file__).resolve().is_relative_to(source):
            raise RuntimeError('DINOv3 factory did not resolve inside the specified source')
        backbone = factory.dinov3_vitb16(pretrained=False)
    finally:
        sys.path.remove(str(source))
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(weights, dict) or not weights or not all(torch.is_tensor(v) for v in weights.values()):
        raise ValueError('Expected official bare backbone state_dict, not a task checkpoint')
    backbone.load_state_dict(weights, strict=True)
    backbone.float().eval().requires_grad_(False)
    files = {p.relative_to(source).as_posix(): file_sha256(p)
             for p in sorted((source / 'dinov3').rglob('*.py'))}
    source_sha = hashlib.sha256()
    for name, sha in files.items():
        source_sha.update(name.encode())
        source_sha.update(bytes.fromhex(sha))
    identity = dict(architecture='dinov3_vitb16', checkpoint_sha256=actual_sha,
                    source_sha256=source_sha.hexdigest(), source_files=files,
                    backbone_state_sha256=state_sha256(backbone),
                    official_source='https://github.com/facebookresearch/dinov3',
                    task_weights_loaded=False, strict_load=True)
    return backbone, identity


class DinoSemanticHead(nn.Module):
    """四层同尺度特征先独立投影再融合；不引入原 RGB 修正支路。"""
    def __init__(self, embed_dim=768, width=64, layers=4):
        super().__init__()
        if width < 8 or width % 8 or layers < 1:
            raise ValueError('Head width must be a positive multiple of 8')
        self.embed_dim, self.layers = int(embed_dim), int(layers)
        self.projections = nn.ModuleList(nn.Conv2d(embed_dim, width, 1) for _ in range(layers))
        self.fusion = nn.Sequential(
            nn.Conv2d(width * layers, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(), nn.Conv2d(width, 1, 1))

    def forward(self, features, output_size):
        if len(features) != self.layers:
            raise ValueError('Wrong number of DINO intermediate feature maps')
        shape = features[0].shape
        if len(shape) != 4 or shape[1] != self.embed_dim or any(f.shape != shape for f in features):
            raise ValueError('DINO features must be aligned BCHW maps of the configured width')
        logits = self.fusion(torch.cat([proj(f.float()) for proj, f in zip(self.projections, features)], 1))
        return F.interpolate(logits, size=output_size, mode='bilinear', align_corners=False)


class FrozenDinoSemantic(nn.Module):
    def __init__(self, backbone, layers=(2, 5, 8, 11), head_width=64, head_seed=20261025):
        super().__init__()
        self.backbone = backbone.float().eval().requires_grad_(False)
        self.layers = tuple(layers)
        if self.layers != (2, 5, 8, 11):
            raise ValueError('This experiment fixes DINO layers [2,5,8,11]')
        self.embed_dim = int(backbone.embed_dim)
        self.patch_size = int(backbone.patch_size)
        if self.patch_size != 16:
            raise ValueError('DINO ViT-B/16 patch size must be 16')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(head_seed))
            self.head = DinoSemanticHead(self.embed_dim, int(head_width), len(self.layers))
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.initial_head_sha256 = state_sha256(self.head)
        self.architecture = dict(backbone='dinov3_vitb16', layers=list(self.layers),
                                 head_width=int(head_width), embed_dim=self.embed_dim,
                                 head_seed=int(head_seed), normalization='imagenet_rgb01',
                                 direct_rgb_residual=False, head_initialization='random')

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval().requires_grad_(False)
        return self

    def forward(self, rgb):
        if (rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point()
                or any(n % self.patch_size for n in rgb.shape[-2:])):
            raise ValueError('Expected RGB01 BCHW input with height/width divisible by 16')
        if not bool(torch.isfinite(rgb).all()) or bool((rgb < 0).any()) or bool((rgb > 1).any()):
            raise ValueError('RGB must contain finite values in [0,1]')
        self.backbone.eval()
        if any(p.requires_grad for p in self.backbone.parameters()):
            raise RuntimeError('The DINO backbone must stay frozen')
        # no_grad 而非 inference_mode：头的反向需要保存这些冻结特征。
        with torch.no_grad(), torch.autocast(device_type=rgb.device.type, enabled=False):
            normalized = (rgb.float() - self.rgb_mean) / self.rgb_std
            features = self.backbone.get_intermediate_layers(normalized, n=self.layers,
                                                             reshape=True, norm=True)
        expected = (rgb.shape[0], self.embed_dim, rgb.shape[-2] // 16, rgb.shape[-1] // 16)
        if len(features) != 4 or any(tuple(f.shape) != expected for f in features):
            raise ValueError(f'Unexpected official DINO intermediate feature shapes; expected {expected}')
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            return self.head(features, rgb.shape[-2:])
