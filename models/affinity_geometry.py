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
        return {"affinity_logits": self.affinity_head(feature)}

    def trainable_param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
