# -*- coding: utf-8 -*-
"""Retained generative edge prior for metallographic boundary segmentation.

Unlike the G0 RGB reconstruction decoder, this entire module is retained for
downstream inference.  It learns to recover stable multi-scale structure from
masked and physically degraded views of the allowed unlabeled images.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(16, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        _group_norm(out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        _group_norm(out_channels),
        nn.GELU(),
    )


class GenerativeEdgePrior(nn.Module):
    """Shallow-feature decoder producing edge magnitude and orientation."""

    def __init__(
        self,
        in_channels: Sequence[int] = (112, 224),
        hidden_channels: int = 64,
    ):
        super().__init__()
        if len(in_channels) != 2:
            raise ValueError("GenerativeEdgePrior expects two shallow scales")
        self.in_channels = tuple(int(value) for value in in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lateral_high = nn.Conv2d(self.in_channels[0], hidden_channels, 1)
        self.lateral_low = nn.Conv2d(self.in_channels[1], hidden_channels, 1)
        self.fuse = _conv_block(hidden_channels * 2, hidden_channels)
        self.up_half = _conv_block(hidden_channels, hidden_channels // 2)
        self.up_full = _conv_block(hidden_channels // 2, hidden_channels // 2)
        self.out = nn.Conv2d(hidden_channels // 2, 3, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: Sequence[int],
    ) -> torch.Tensor:
        if len(features) < 2:
            raise ValueError("edge prior requires at least two feature scales")
        high = self.lateral_high(features[0])
        low = self.lateral_low(features[1])
        low = F.interpolate(
            low, size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = self.fuse(torch.cat([high, low], dim=1))
        half_size = tuple(max(1, int(value) // 2) for value in output_size)
        fused = F.interpolate(
            fused, size=half_size, mode="bilinear", align_corners=False
        )
        fused = self.up_half(fused)
        fused = F.interpolate(
            fused, size=tuple(output_size), mode="bilinear", align_corners=False
        )
        return self.out(self.up_full(fused))

    @staticmethod
    def decode(raw: torch.Tensor):
        return torch.sigmoid(raw[:, 0:1]), torch.tanh(raw[:, 1:3])


class FrozenEncoderEdgePrior(nn.Module):
    """Frozen E1 SAM2 encoder plus the fully retained edge-prior decoder."""

    def __init__(self, encoder: nn.Module, prior: GenerativeEdgePrior):
        super().__init__()
        self.encoder = encoder
        self.prior = prior
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        if hasattr(self.encoder, "trainable_lora"):
            self.encoder.trainable_lora = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(image)
        return self.prior(features[:2], image.shape[-2:])

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.prior.parameters()


def _box_blur(gray: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    padded = F.pad(gray, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size, stride=1)


def _sobel(gray: torch.Tensor):
    kernel_x = gray.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernel_x), F.conv2d(padded, kernel_y)


@torch.no_grad()
def build_structural_edge_target(image: torch.Tensor) -> torch.Tensor:
    """Generate a label-free, multi-scale-stable edge/orientation target.

    Fine pearlite lamellae tend to disappear after wider smoothing, while phase
    boundaries remain present.  The minimum response across three scales and
    the orientation agreement term therefore favour persistent structure without
    using external labels.
    """
    image = image.float()
    gray = (
        0.299 * image[:, 0:1]
        + 0.587 * image[:, 1:2]
        + 0.114 * image[:, 2:3]
    )
    gradients = []
    for kernel_size in (3, 7, 15):
        gx, gy = _sobel(_box_blur(gray, kernel_size))
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
        gradients.append((gx, gy, magnitude))

    # Use one reference scale for every smoothing level.  Normalizing each
    # level independently would amplify tiny residual responses of periodic
    # pearlite lamellae after wide smoothing and defeat texture suppression.
    reference = (
        gradients[0][2].mean(dim=(-2, -1), keepdim=True).clamp_min(1e-5) * 4.0
    )
    normalized = [
        (gradient[2] / reference).clamp(0.0, 1.0)
        for gradient in gradients
    ]

    persistent = torch.minimum(
        torch.minimum(normalized[0], normalized[1]), normalized[2]
    )
    gx0, gy0, mag0 = gradients[0]
    gx2, gy2, mag2 = gradients[2]
    agreement = (
        (gx0 * gx2 + gy0 * gy2).abs()
        / (mag0 * mag2 + 1e-6)
    ).clamp(0.0, 1.0)
    edge = (persistent * agreement.sqrt()).clamp(0.0, 1.0)

    gx, gy, magnitude = gradients[1]
    orientation = torch.cat(
        [gx / (magnitude + 1e-5), gy / (magnitude + 1e-5)], dim=1
    ).clamp(-1.0, 1.0)
    return torch.cat([edge, orientation], dim=1)


def edge_prior_loss(
    raw_prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_weight: float = 4.0,
    masked_region_weight: float = 1.5,
    dice_weight: float = 0.5,
    orientation_weight: float = 0.25,
    multiscale_weight: float = 0.25,
    background_weight: float = 0.10,
):
    edge_prediction, orientation_prediction = GenerativeEdgePrior.decode(
        raw_prediction
    )
    edge_target = target[:, 0:1]
    orientation_target = target[:, 1:3]
    region_weight = 1.0 + masked_region_weight * mask
    edge_weight = region_weight * (1.0 + positive_weight * edge_target)

    edge_l1 = (
        (edge_prediction - edge_target).abs() * edge_weight
    ).sum() / edge_weight.sum().clamp_min(1e-6)
    intersection = (edge_prediction * edge_target * region_weight).sum()
    denominator = (
        (edge_prediction + edge_target) * region_weight
    ).sum().clamp_min(1e-6)
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)

    orientation_weight_map = region_weight * edge_target
    orientation_loss = (
        (orientation_prediction - orientation_target).abs()
        * orientation_weight_map
    ).sum() / (
        orientation_weight_map.sum().clamp_min(1e-6)
        * orientation_prediction.shape[1]
    )

    multiscale_loss = raw_prediction.new_tensor(0.0)
    for scale in (2, 4):
        pred_small = F.avg_pool2d(edge_prediction, scale, stride=scale)
        target_small = F.avg_pool2d(edge_target, scale, stride=scale)
        multiscale_loss = multiscale_loss + F.l1_loss(pred_small, target_small)
    multiscale_loss = multiscale_loss / 2.0
    background_loss = (
        edge_prediction * (1.0 - edge_target) * region_weight
    ).sum() / region_weight.sum().clamp_min(1e-6)

    total = (
        edge_l1
        + dice_weight * dice_loss
        + orientation_weight * orientation_loss
        + multiscale_weight * multiscale_loss
        + background_weight * background_loss
    )
    return total, {
        "edge_l1": float(edge_l1.detach()),
        "dice": float(dice_loss.detach()),
        "orientation": float(orientation_loss.detach()),
        "multiscale": float(multiscale_loss.detach()),
        "background": float(background_loss.detach()),
        "predicted_edge_mean": float(edge_prediction.detach().mean()),
        "target_edge_mean": float(edge_target.detach().mean()),
    }
