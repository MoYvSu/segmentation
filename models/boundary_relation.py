# -*- coding: utf-8 -*-
"""相邻区域关系分类：冻结图像特征作输入，A/B交换严格不改变判定。"""
from __future__ import annotations

import hashlib
import json
import re

import torch
from torch import nn
from torch.nn import functional as F


FORMAT = 'boundary_relation_v1'
INITIAL_FORMAT = 'boundary_relation_initial_v1'


def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf8'))
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class BoundaryRelationNet(nn.Module):
    """input=[RGB(-1..1),N1,NAF]；masks=[A,B,shared_seam,valid]。

    编码器不接收区域命名。两侧共享特征池化后，只保留和及绝对差，
    因而交换A/B不会改变keep logit；模型不生成或修改实例掩码。
    """

    def __init__(self, widths=(24, 48, 96), hidden=128, seed=20261030):
        super().__init__()
        widths = tuple(widths)
        if (len(widths) != 3 or any(type(w) is not int or w < 8 or w % 8 for w in widths)
                or type(hidden) is not int or hidden < 1 or type(seed) is not int):
            raise ValueError('三层width须为8的正倍数，hidden/seed须为整数')
        self.architecture = dict(widths=list(widths), hidden=hidden, seed=seed,
            input_channels=['R', 'G', 'B', 'N1', 'NAF'],
            masks=['A', 'B', 'shared_seam', 'valid'],
            normalization='GroupNorm8', activation='SiLU', strides=[2, 2, 2],
            pooling='area_downsample(mask*valid); normalized weighted mean',
            relation_features=['poolA+poolB', 'abs(poolA-poolB)', 'poolSeam', 'poolUnion'],
            positive_label='keep', pretrained_weights=False)
        with torch.random.fork_rng(devices=[]):
            # manual_seed会同时触碰CUDA generator；CPU构造不得改已存在的GPU随机流。
            torch.random.default_generator.manual_seed(seed)
            layers = []
            previous = 5
            for width in widths:
                layers.extend([nn.Conv2d(previous, width, 3, stride=2, padding=1, bias=False),
                               nn.GroupNorm(8, width), nn.SiLU(),
                               nn.Conv2d(width, width, 3, padding=1, bias=False),
                               nn.GroupNorm(8, width), nn.SiLU()])
                previous = width
            self.encoder = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.Linear(widths[-1] * 4, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.initial_state_sha256 = state_sha256(self)

    @staticmethod
    def validate_inputs(value, masks, *, check_values=False):
        if (not torch.is_tensor(value) or value.ndim != 4 or value.shape[1] != 5
                or not value.is_floating_point() or min(value.shape[0], *value.shape[-2:]) < 1
                or not torch.is_tensor(masks) or masks.ndim != 4 or masks.shape[1] != 4
                or masks.shape[0] != value.shape[0] or masks.shape[-2:] != value.shape[-2:]
                or masks.device != value.device):
            raise ValueError('关系模型需要同设备B5HW输入和B4HW masks')
        if check_values:
            if (not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(masks).all())
                    or bool(((value[:, :3] < -1) | (value[:, :3] > 1)).any())
                    or bool(((value[:, 3:] < 0) | (value[:, 3:] > 1)).any())
                    or bool(((masks != 0) & (masks != 1)).any())):
                raise ValueError('输入范围或binary masks错误')
            if bool((masks[:, :3] > masks[:, 3:4]).any()) or bool((masks[:, 0] * masks[:, 1] > 0).any()):
                raise ValueError('两侧不能重叠，且masks须位于valid内')
            if bool((masks.float().sum(dim=(-2, -1)) == 0).any()):
                raise ValueError('每例须含两侧、共享分隔及原图有效像素')

    def forward(self, value, masks):
        self.validate_inputs(value, masks)
        feature = self.encoder(value)
        a, b, seam, valid = masks.float().split(1, dim=1)
        # 先在原格点乘valid，再面积下采样，保留细缝及部分有效cell的实际质量。
        support = torch.cat((a, b, seam, torch.maximum(a, b)), dim=1) * valid
        support = F.interpolate(support, size=feature.shape[-2:], mode='area')
        denominator = support.sum(dim=(-2, -1)).clamp_min(1e-12)
        pooled = torch.einsum('bchw,bkhw->bkc', feature, support) / denominator[..., None]
        pa, pb, ps, pu = pooled.unbind(dim=1)
        relation = torch.cat((pa + pb, (pa - pb).abs(), ps, pu), dim=1)
        return self.head(relation).squeeze(1)


def model_from_architecture(architecture):
    model = BoundaryRelationNet(**{key: architecture[key] for key in ('widths', 'hidden', 'seed')})
    if model.architecture != architecture:
        raise ValueError('关系分类checkpoint架构/版本不一致')
    return model


def load_boundary_relation(checkpoint, device='cpu'):
    """严格加载本模块格式；拒绝旧SAM/NAF/补边器或宽松键匹配。"""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('format') != FORMAT:
        raise ValueError('不是关系分类器checkpoint')
    model = model_from_architecture(saved['architecture'])
    recorded_initial = saved.get('initial_state_sha256')
    if not isinstance(recorded_initial, str) or re.fullmatch(r'[0-9a-f]{64}', recorded_initial) is None:
        raise ValueError('checkpoint缺少合法的来源初态SHA')
    identity = saved.get('identity')
    if identity is not None:
        # 已训练推理只核来源记录自洽；不同CPU的随机初始化末位可不同。
        if (not isinstance(identity, dict) or identity.get('format') != FORMAT
                or identity.get('architecture') != saved['architecture']
                or identity.get('initial_state_sha256') != recorded_initial):
            raise ValueError('checkpoint初态/架构与训练身份记录不一致')
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()
        if saved.get('identity_sha256') != digest:
            raise ValueError('checkpoint训练身份SHA不一致')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    if state_sha256(model) != saved['model_sha256']:
        raise ValueError('关系模型状态SHA不一致')
    model.constructor_initial_state_sha256 = model.initial_state_sha256
    model.initial_state_sha256 = recorded_initial
    return model.to(device).eval().requires_grad_(False)
