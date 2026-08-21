# -*- coding: utf-8 -*-
"""Generative domain adapters and masked-image reconstruction components.

The temporary reconstruction decoder is used only during G0 pretraining.  The
small multi-scale adapters are retained and can later be routed only into the
boundary branch, leaving the proven E1 semantic path bit-for-bit unchanged when
their downstream gates are zero.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(16, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class GenerativeDomainAdapter(nn.Module):
    """A lightweight residual adapter for one Hiera feature scale."""

    def __init__(self, channels: int, bottleneck_ratio: int = 8):
        super().__init__()
        hidden = max(16, channels // max(1, bottleneck_ratio))
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            _group_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            _group_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        # A small non-zero initialization is required during generative
        # pretraining.  Exact E1 identity downstream is provided by zero gates.
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class GenerativeDomainAdapterPyramid(nn.Module):
    """Four-scale GDA pyramid with zero-initialized downstream gates."""

    def __init__(
        self,
        channels: Sequence[int] = (112, 224, 448, 896),
        bottleneck_ratio: int = 8,
    ):
        super().__init__()
        self.channels = tuple(int(ch) for ch in channels)
        self.adapters = nn.ModuleList(
            GenerativeDomainAdapter(ch, bottleneck_ratio) for ch in self.channels
        )
        self.gates = nn.Parameter(torch.zeros(len(self.channels)))

    def deltas(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) != len(self.adapters):
            raise ValueError(
                f"expected {len(self.adapters)} feature scales, got {len(features)}"
            )
        return [adapter(feature) for adapter, feature in zip(self.adapters, features)]

    def forward(
        self,
        features: Sequence[torch.Tensor],
        *,
        gated: bool = False,
    ) -> List[torch.Tensor]:
        deltas = self.deltas(features)
        if gated:
            # tanh keeps a learned gate bounded while gate=0 is an exact identity.
            scales = torch.tanh(self.gates)
            return [
                feature + scales[i] * delta
                for i, (feature, delta) in enumerate(zip(features, deltas))
            ]
        return [feature + delta for feature, delta in zip(features, deltas)]

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        return self.adapters.parameters()


def load_pretrained_gda(
    checkpoint_path: str,
    *,
    channels: Sequence[int] = (112, 224, 448, 896),
    bottleneck_ratio: int = 8,
    map_location="cpu",
    freeze_adapters: bool = True,
    train_gates: bool = True,
) -> GenerativeDomainAdapterPyramid:
    """Build the retained GDA pyramid and load a G0 or downstream checkpoint.

    G0 checkpoints and Stage-2 checkpoints intentionally share the
    ``gda_state_dict`` key.  Adapter weights are frozen by default downstream;
    only the four zero-initialized gates are optimized in G1.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    state_dict = checkpoint.get("gda_state_dict")
    if not state_dict:
        raise KeyError(f"checkpoint has no gda_state_dict: {checkpoint_path}")

    gda = GenerativeDomainAdapterPyramid(channels, bottleneck_ratio)
    gda.load_state_dict(state_dict, strict=True)
    for parameter in gda.adapters.parameters():
        parameter.requires_grad_(not freeze_adapters)
    gda.gates.requires_grad_(bool(train_gates))
    return gda


class MIMReconstructionDecoder(nn.Module):
    """Temporary FPN decoder reconstructing RGB from adapted Hiera features."""

    def __init__(
        self,
        in_channels: Sequence[int] = (112, 224, 448, 896),
        hidden_channels: int = 96,
    ):
        super().__init__()
        self.laterals = nn.ModuleList(
            nn.Conv2d(ch, hidden_channels, 1, bias=False) for ch in in_channels
        )
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                _group_norm(hidden_channels),
                nn.GELU(),
            )
            for _ in in_channels
        )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            _group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            _group_norm(hidden_channels // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 3, 1),
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: Sequence[int],
    ) -> torch.Tensor:
        laterals = [
            smooth(lateral(feature))
            for smooth, lateral, feature in zip(self.smooth, self.laterals, features)
        ]
        fused = laterals[-1]
        for lateral in reversed(laterals[:-1]):
            fused = F.interpolate(
                fused, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            fused = fused + lateral
        logits = self.reconstruct(fused)
        logits = F.interpolate(
            logits, size=tuple(output_size), mode="bilinear", align_corners=False
        )
        return torch.sigmoid(logits)


class GDAMaskedAutoencoder(nn.Module):
    """Frozen E1 encoder + trainable GDA pyramid + temporary RGB decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        channels: Sequence[int] = (112, 224, 448, 896),
        bottleneck_ratio: int = 8,
        decoder_channels: int = 96,
    ):
        super().__init__()
        self.encoder = encoder
        self.gda = GenerativeDomainAdapterPyramid(channels, bottleneck_ratio)
        self.decoder = MIMReconstructionDecoder(channels, decoder_channels)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        if hasattr(self.encoder, "trainable_lora"):
            self.encoder.trainable_lora = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, masked_image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(masked_image)
        adapted = self.gda(features, gated=False)
        return self.decoder(adapted, masked_image.shape[-2:])

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.gda.adapter_parameters()
        yield from self.decoder.parameters()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def _sobel(image: torch.Tensor):
    gray = (
        0.299 * image[:, 0:1]
        + 0.587 * image[:, 1:2]
        + 0.114 * image[:, 2:3]
    )
    kernel_x = image.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    kernel_y = kernel_x.transpose(-1, -2)
    return F.conv2d(gray, kernel_x, padding=1), F.conv2d(gray, kernel_y, padding=1)


def gda_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    rgb_weight: float = 0.50,
    low_frequency_weight: float = 0.25,
    gradient_weight: float = 0.25,
):
    """Masked cross-view reconstruction loss with low/high-frequency terms."""
    charbonnier = torch.sqrt((prediction - target).square() + 1e-6)
    rgb_loss = _masked_mean(charbonnier, mask)

    pred_low = F.avg_pool2d(prediction, 9, stride=1, padding=4)
    target_low = F.avg_pool2d(target, 9, stride=1, padding=4)
    low_loss = _masked_mean((pred_low - target_low).abs(), mask)

    pred_gx, pred_gy = _sobel(prediction)
    target_gx, target_gy = _sobel(target)
    gradient_error = (pred_gx - target_gx).abs() + (pred_gy - target_gy).abs()
    gradient_mask = F.max_pool2d(mask, 5, stride=1, padding=2)
    gradient_loss = _masked_mean(gradient_error, gradient_mask)

    total = (
        rgb_weight * rgb_loss
        + low_frequency_weight * low_loss
        + gradient_weight * gradient_loss
    )
    return total, {
        "rgb": float(rgb_loss.detach()),
        "low_frequency": float(low_loss.detach()),
        "gradient": float(gradient_loss.detach()),
    }
