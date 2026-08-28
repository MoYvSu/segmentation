# -*- coding: utf-8 -*-
"""Independent global center-offset geometry decoder."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_decoder import FPNBackbone, _make_group_norm


class CenterOffsetGeometryDecoder(nn.Module):
    """Four-scale FPN followed by a learned stride-2 geometry upsampler."""

    def __init__(
        self,
        in_channels: List[int] | None = None,
        fpn_channels: int = 256,
        up_channels: int = 128,
        output_grid: int = 512,
    ):
        super().__init__()
        in_channels = in_channels or [112, 224, 448, 896]
        self.output_grid = int(output_grid)
        self.geometry_fpn = FPNBackbone(in_channels, fpn_channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(fpn_channels, up_channels, 3, padding=1, bias=False),
            _make_group_norm(up_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(up_channels, up_channels, 3, padding=1, bias=False),
            _make_group_norm(up_channels),
            nn.ReLU(inplace=True),
        )
        head_hidden = max(32, up_channels // 2)
        self.center_head = nn.Sequential(
            nn.Conv2d(up_channels, head_hidden, 3, padding=1, bias=False),
            _make_group_norm(head_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_hidden, 1, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(up_channels, head_hidden, 3, padding=1, bias=False),
            _make_group_norm(head_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_hidden, 2, 1),
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
        center_out = self.center_head[-1]
        nn.init.normal_(center_out.weight, std=0.001)
        nn.init.constant_(center_out.bias, -2.1972246)  # sigmoid=0.1
        offset_out = self.offset_head[-1]
        nn.init.normal_(offset_out.weight, std=0.001)
        nn.init.zeros_(offset_out.bias)

    def initialize_fpn_from_boundary(self, boundary_fpn: nn.Module):
        """Copy only the V6 boundary FPN feature extractor, never its output head."""
        self.geometry_fpn.load_state_dict(boundary_fpn.state_dict(), strict=True)

    def forward(self, features: Iterable[torch.Tensor]) -> Dict[str, torch.Tensor]:
        feature = self.geometry_fpn(list(features))
        feature = F.interpolate(
            feature, size=(self.output_grid, self.output_grid),
            mode="bilinear", align_corners=False,
        )
        feature = self.upsample(feature)
        return {
            "center_logits": self.center_head(feature),
            "offsets": self.offset_head(feature),
        }

    def trainable_param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class FrozenSemanticGeometrySystem(nn.Module):
    """Frozen, reproducible V6 reference plus an independent geometry decoder."""

    def __init__(
        self,
        reference_model: nn.Module,
        geometry_decoder: nn.Module,
        geometry_feature_adapter: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.reference_model = reference_model
        self.geometry_decoder = geometry_decoder
        self.geometry_feature_adapter = geometry_feature_adapter
        self.freeze_reference()

    def freeze_reference(self):
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)
        self.reference_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.reference_model.eval()
        self.geometry_decoder.train(mode)
        if self.geometry_feature_adapter is not None:
            self.geometry_feature_adapter.train(mode)
        return self

    def geometry_forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.reference_model.eval()
        with torch.no_grad():
            features = self.reference_model.encoder(image)
        features = [feature.detach() for feature in features]
        if self.geometry_feature_adapter is not None:
            features = self.geometry_feature_adapter(features, gated=True)
        return self.geometry_decoder(features)

    def geometry_trainable_parameters(self):
        yield from self.geometry_decoder.parameters()
        if self.geometry_feature_adapter is not None:
            yield from self.geometry_feature_adapter.parameters()

    @torch.no_grad()
    def semantic_logits(self, image: torch.Tensor) -> torch.Tensor:
        self.reference_model.eval()
        return self.reference_model(image)[:, :1]

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.geometry_forward(image)


def semantic_state_digest(reference_model: nn.Module) -> str:
    """Hash V6 semantic FPN/head and LoRA tensors for the freeze contract."""
    digest = hashlib.sha256()
    selected = []
    for name, tensor in reference_model.decoder.state_dict().items():
        if (
            name.startswith("seg_fpn.")
            or name.startswith("seg_branch.")
            or name.startswith("semantic_residual.")
        ):
            selected.append((f"decoder.{name}", tensor))
    for name, tensor in reference_model.encoder.trunk.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            selected.append((f"encoder.trunk.{name}", tensor))
    for name, tensor in sorted(selected, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    if not selected:
        raise ValueError("semantic digest selected no tensors")
    return digest.hexdigest()


def geometry_feature_state_digest(reference_model: nn.Module) -> str:
    """Hash only trainable encoder adapters that affect geometry features.

    A semantic-decoder-only update is compatible with a frozen affinity
    decoder.  LoRA changes are not, because affinity consumes encoder features.
    """

    digest = hashlib.sha256()
    selected = []
    for name, tensor in reference_model.encoder.trunk.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            selected.append((f"encoder.trunk.{name}", tensor))
    for name, tensor in sorted(selected, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    if not selected:
        raise ValueError("geometry feature digest selected no LoRA tensors")
    return digest.hexdigest()


def lora_state_dict_digest(state_dict: dict) -> str:
    """Hash a checkpoint ``lora_state_dict`` for override compatibility."""

    selected = [
        (str(name), tensor)
        for name, tensor in state_dict.items()
        if "lora_A" in str(name) or "lora_B" in str(name)
    ]
    if not selected:
        raise ValueError("checkpoint contains no LoRA tensors")
    digest = hashlib.sha256()
    for name, tensor in sorted(selected, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
