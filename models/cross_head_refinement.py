# -*- coding: utf-8 -*-
"""冻结 E10a+G4b 上的等参数 self/cross 小幅修正。"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.fused_deployment import load_fused_deployment_model
from utils.config import project_path


CROSS_HEAD_FORMAT = "mainline_cross_head_v1"


def configure_mainline_precision():
    # 与既有 run_fused_affinity_submission.py 的 PyTorch 默认设置一致。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _resize(value, size):
    if value.shape[-2:] == tuple(size):
        return value
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


def _conv_block(in_channels, out_channels, kernel=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
    )


class _TaskCorrection(nn.Module):
    def __init__(self, hidden, output_channels):
        super().__init__()
        # 两条非线性分支独立学习；self 组没有闲置或只乘零的参数。
        self.own = _conv_block(hidden, hidden)
        self.context = _conv_block(hidden, hidden)
        self.mix = _conv_block(2 * hidden, hidden)
        self.delta = nn.Conv2d(hidden, output_channels, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, own, context):
        return self.delta(self.mix(torch.cat([self.own(own), self.context(context)], 1)))


class CrossHeadRefiner(nn.Module):
    def __init__(
        self, *, semantic_feature_channels=256, affinity_feature_channels=128,
        hidden_channels=32, max_logit_delta=1.0, context_mode="cross",
    ):
        super().__init__()
        if context_mode not in {"self", "cross"}:
            raise ValueError("context_mode must be self or cross")
        if hidden_channels < 4 or hidden_channels % 4:
            raise ValueError("hidden_channels must be a positive multiple of four")
        if not math.isfinite(max_logit_delta) or max_logit_delta <= 0:
            raise ValueError("max_logit_delta must be finite and positive")
        self.context_mode = context_mode
        self.max_logit_delta = float(max_logit_delta)
        self.architecture = dict(
            semantic_feature_channels=semantic_feature_channels,
            affinity_feature_channels=affinity_feature_channels,
            hidden_channels=hidden_channels, max_logit_delta=max_logit_delta,
        )
        self.semantic_projection = _conv_block(
            semantic_feature_channels + 1, hidden_channels, kernel=1
        )
        self.affinity_projection = _conv_block(
            affinity_feature_channels + 8, hidden_channels, kernel=1
        )
        self.semantic = _TaskCorrection(hidden_channels, 1)
        self.affinity = _TaskCorrection(hidden_channels, 8)

    def forward(self, frozen):
        sem_logits = frozen["semantic_logits"]
        aff_logits = frozen["affinity_logits"]
        sem_feature = frozen["semantic_feature"]
        aff_feature = frozen["affinity_feature"]
        sem = self.semantic_projection(torch.cat([
            sem_feature, _resize(sem_logits.sigmoid(), sem_feature.shape[-2:]),
        ], 1))
        aff = self.affinity_projection(torch.cat([
            aff_feature, _resize(aff_logits.sigmoid(), aff_feature.shape[-2:]),
        ], 1))
        grid = aff_logits.shape[-2:]
        sem, aff = _resize(sem, grid), _resize(aff, grid)
        sem_context, aff_context = (aff, sem) if self.context_mode == "cross" else (sem, aff)
        sem_delta = self.max_logit_delta * self.semantic(sem, sem_context).tanh()
        aff_delta = self.max_logit_delta * self.affinity(aff, aff_context).tanh()
        # 只放大语义修正；主线 logits 不重采样，affinity 仍在原 512 网格融合。
        return {
            "semantic_logits": sem_logits + _resize(sem_delta, sem_logits.shape[-2:]),
            "affinity_logits": aff_logits + aff_delta,
        }


class RefinedMainline(nn.Module):
    def __init__(self, base, refiner):
        super().__init__()
        if getattr(base, "geometry_highres_refiner", None) is not None:
            raise ValueError("cross-head v1 requires the unrefined G4b output grid")
        if getattr(base.encoder, "input_normalization", "legacy_none") != "legacy_none":
            raise ValueError("cross-head v1 requires the legacy_none E10a+G4b base")
        self.base = base.requires_grad_(False).eval()
        self.refiner = refiner

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, image):
        # FP32 与正式 fused 部署一致；不在 autocast 中缓存/复用冻结算子。
        with torch.no_grad(), torch.autocast(image.device.type, enabled=False):
            frozen = self.base(image.float(), return_features=True)
        return self.refiner(frozen)

    def parameter_summary(self):
        base = sum(p.numel() for p in self.base.parameters())
        trainable = sum(p.numel() for p in self.refiner.parameters())
        return {"base": base, "trainable": trainable, "total": base + trainable,
                "total_M": (base + trainable) / 1e6,
                "constraint_passed": base + trainable < 500_000_000}


def build_refined_from_checkpoint(payload, base, base_sha256, device):
    if payload.get("format") != CROSS_HEAD_FORMAT:
        raise ValueError("unsupported cross-head checkpoint format")
    if payload["base_sha256"] != base_sha256:
        raise ValueError("cross-head checkpoint requires its exact frozen base")
    refiner = CrossHeadRefiner(
        **payload["architecture"], context_mode=payload["context_mode"]
    ).to(device)
    refiner.load_state_dict(payload["refiner_state_dict"], strict=True)
    return RefinedMainline(base, refiner).to(device).eval()


def load_refined_mainline(checkpoint_path, config, device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != CROSS_HEAD_FORMAT:
        raise ValueError("unsupported cross-head checkpoint format")
    base_path = project_path(config, payload["base_checkpoint"])
    digest = file_digest(base_path)
    if digest != payload["base_sha256"]:
        raise ValueError("frozen base file digest mismatch")
    base, _ = load_fused_deployment_model(base_path, config, device)
    return build_refined_from_checkpoint(payload, base, digest, device), payload
