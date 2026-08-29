# -*- coding: utf-8 -*-
"""Independent local-affinity decoder for bottom-up instance partitioning."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_decoder import FPNBackbone, _make_group_norm


class AffinityGeometryDecoder(nn.Module):
    def __init__(
        self,
        in_channels: List[int] | None = None,
        affinity_channels: int = 8,
        fpn_channels: int = 256,
        up_channels: int = 128,
        output_grid: int = 512,
    ):
        super().__init__()
        in_channels = in_channels or [112, 224, 448, 896]
        self.output_grid = int(output_grid)
        self.affinity_channels = int(affinity_channels)
        self.geometry_fpn = FPNBackbone(in_channels, fpn_channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(fpn_channels, up_channels, 3, padding=1, bias=False),
            _make_group_norm(up_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(up_channels, up_channels, 3, padding=1, bias=False),
            _make_group_norm(up_channels),
            nn.ReLU(inplace=True),
        )
        hidden = max(32, up_channels // 2)
        self.affinity_head = nn.Sequential(
            nn.Conv2d(up_channels, hidden, 3, padding=1, bias=False),
            _make_group_norm(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.affinity_channels, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
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
        output = self.affinity_head[-1]
        nn.init.normal_(output.weight, std=0.001)
        nn.init.zeros_(output.bias)

    def initialize_fpn_from_boundary(self, boundary_fpn: nn.Module):
        self.geometry_fpn.load_state_dict(boundary_fpn.state_dict(), strict=True)

    def forward(self, features: Iterable[torch.Tensor]) -> Dict[str, torch.Tensor]:
        feature = self.geometry_fpn(list(features))
        feature = F.interpolate(
            feature,
            size=(self.output_grid, self.output_grid),
            mode="bilinear",
            align_corners=False,
        )
        feature = self.upsample(feature)
        return {
            "affinity_logits": self.affinity_head(feature),
            # Expose the trained G4b feature without changing its checkpoint.
            # A separate frozen-base refiner may consume this tensor; existing
            # callers can continue to use affinity_logits only.
            "affinity_feature": feature,
        }

    def trainable_param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class HighResolutionShortAffinityResidual(nn.Module):
    """Full-resolution residual for the four unit-offset affinity channels.

    The proven G4b decoder remains the coarse, frozen geometry model.  This
    module compresses its 512-grid feature, restores the image grid, and uses
    shallow RGB evidence to correct only the four distance-1 logits.  The
    distance-2/4 channels are merely interpolated and cannot be trained here.

    The final convolution is zero initialized, so the learnable correction is
    exactly zero at initialization.  This is an architectural stability anchor,
    not probability distillation from G4b.
    """

    def __init__(
        self,
        feature_channels: int = 128,
        short_channels: int = 4,
        feature_hidden: int = 32,
        image_hidden: int = 16,
        fusion_hidden: int = 32,
        max_logit_delta: float = 1.0,
    ):
        super().__init__()
        values = {
            "feature_channels": feature_channels,
            "short_channels": short_channels,
            "feature_hidden": feature_hidden,
            "image_hidden": image_hidden,
            "fusion_hidden": fusion_hidden,
        }
        if any(int(value) <= 0 for value in values.values()):
            raise ValueError(f"high-resolution affinity channels must be positive: {values}")
        if float(max_logit_delta) <= 0:
            raise ValueError("max_logit_delta must be positive")
        self.short_channels = int(short_channels)
        self.max_logit_delta = float(max_logit_delta)
        self.feature_path = nn.Sequential(
            nn.Conv2d(feature_channels, feature_hidden, 1, bias=False),
            _make_group_norm(feature_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_hidden, feature_hidden, 3, padding=1, bias=False),
            _make_group_norm(feature_hidden),
            nn.ReLU(inplace=True),
        )
        # RGB and a local RGB residual preserve weak interfaces while making no
        # ferrite/pearlite class assumption.
        self.image_path = nn.Sequential(
            nn.Conv2d(6, image_hidden, 3, padding=1, bias=False),
            _make_group_norm(image_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_hidden, image_hidden, 3, padding=1, bias=False),
            _make_group_norm(image_hidden),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                feature_hidden + image_hidden + self.short_channels,
                fusion_hidden,
                3,
                padding=1,
                bias=False,
            ),
            _make_group_norm(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_hidden, fusion_hidden, 3, padding=1, bias=False),
            _make_group_norm(fusion_hidden),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(fusion_hidden, self.short_channels, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
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
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _image_cues(image: torch.Tensor) -> torch.Tensor:
        local_mean = F.avg_pool2d(
            F.pad(image, (2, 2, 2, 2), mode="reflect"),
            kernel_size=5,
            stride=1,
        )
        return torch.cat([image, image - local_mean], dim=1)

    def forward(
        self,
        coarse_feature: torch.Tensor,
        coarse_logits: torch.Tensor,
        image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if coarse_logits.ndim != 4 or coarse_logits.shape[1] < self.short_channels:
            raise ValueError(
                "coarse affinity logits do not contain the requested short channels"
            )
        output_size = tuple(int(value) for value in image.shape[-2:])
        coarse_native = F.interpolate(
            coarse_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        feature_native = F.interpolate(
            self.feature_path(coarse_feature),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        image_feature = self.image_path(self._image_cues(image))
        raw_delta = self.out(
            self.fusion(
                torch.cat(
                    [
                        feature_native,
                        image_feature,
                        coarse_native[:, : self.short_channels],
                    ],
                    dim=1,
                )
            )
        )
        delta = self.max_logit_delta * torch.tanh(raw_delta)
        refined = torch.cat(
            [
                coarse_native[:, : self.short_channels] + delta,
                coarse_native[:, self.short_channels :],
            ],
            dim=1,
        )
        return {
            "affinity_logits": refined,
            "coarse_affinity_logits": coarse_logits,
            "short_affinity_delta": delta,
        }
