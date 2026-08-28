# -*- coding: utf-8 -*-
"""Load a decoder-only semantic challenger without changing geometry features."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.fpn_decoder import (
    SemanticHighResolutionResidualAdapter,
    SemanticResidualAdapter,
)
from models.offset_geometry import lora_state_dict_digest
from utils.config import project_path


class SemanticChallenger(nn.Module):
    """Semantic-only decoder copied from a compatible Stage-2 checkpoint."""

    def __init__(
        self,
        reference_decoder: nn.Module,
        semantic_residual: nn.Module | None = None,
        semantic_residual_version: str = "none",
    ):
        super().__init__()
        self.seg_fpn = copy.deepcopy(reference_decoder.seg_fpn)
        self.seg_branch = copy.deepcopy(reference_decoder.seg_branch)
        self.semantic_residual = copy.deepcopy(semantic_residual)
        self.semantic_residual_version = str(semantic_residual_version)

    def forward(self, features, image=None):
        semantic_feature = self.seg_fpn(features)
        logits = self.seg_branch(semantic_feature)
        if self.semantic_residual is None:
            return logits
        if image is None:
            raise ValueError("high-resolution semantic challenger requires image")
        if self.semantic_residual_version != "highres_v1":
            return logits + self.semantic_residual(semantic_feature, image)
        delta = self.semantic_residual(
            semantic_feature, image, coarse_logits=logits
        )
        return F.interpolate(
            logits,
            size=delta.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ) + delta


def _build_checkpoint_semantic_residual(checkpoint, reference_decoder):
    """Construct only the challenger's optional residual from its own config.

    The fixed V6 reference must keep its native architecture so strict
    checkpoint validation remains meaningful.  A challenger may nevertheless
    contain a newer high-resolution semantic path; construct that path here
    instead of adding it to the reference decoder.
    """
    decoder_cfg = checkpoint.get("config", {}).get("decoder", {})
    if not bool(decoder_cfg.get("semantic_residual", False)):
        raise ValueError(
            "semantic challenger has semantic_residual tensors but its "
            "checkpoint lacks matching decoder configuration"
        )
    if "semantic_residual_max_logit_delta" not in decoder_cfg:
        raise ValueError(
            "semantic challenger checkpoint must record "
            "decoder.semantic_residual_max_logit_delta"
        )
    version = str(decoder_cfg.get("semantic_residual_version", "lowres_v1"))
    common = {
        "feature_channels": int(reference_decoder.fpn_channels),
        "hidden_channels": int(decoder_cfg.get("semantic_residual_hidden", 64)),
        "color_channels": int(
            decoder_cfg.get("semantic_residual_color_channels", 16)
        ),
        "use_photometric_cues": bool(
            decoder_cfg.get("semantic_residual_use_photometric", True)
        ),
        "max_logit_delta": float(
            decoder_cfg["semantic_residual_max_logit_delta"]
        ),
    }
    if version == "lowres_v1":
        return SemanticResidualAdapter(**common), version
    if version == "highres_v1":
        return (
            SemanticHighResolutionResidualAdapter(
                **common,
                half_channels=int(
                    decoder_cfg.get("semantic_residual_half_channels", 48)
                ),
                full_channels=int(
                    decoder_cfg.get("semantic_residual_full_channels", 24)
                ),
            ),
            version,
        )
    raise ValueError(f"unsupported semantic residual version: {version!r}")


def build_semantic_challenger(config, system, reference_path, device):
    """Build an optional classifier-only challenger and verify LoRA identity."""

    challenger_cfg = config.get("semantic_challenger", {})
    if not bool(challenger_cfg.get("enabled", False)):
        return None, None

    challenger_path = Path(project_path(config, challenger_cfg["checkpoint"]))
    if not challenger_path.is_file():
        raise FileNotFoundError(challenger_path)
    reference_path = Path(reference_path)
    reference_checkpoint = torch.load(
        reference_path, map_location="cpu", weights_only=False
    )
    challenger_checkpoint = torch.load(
        challenger_path, map_location="cpu", weights_only=False
    )
    reference_lora = lora_state_dict_digest(
        reference_checkpoint.get("lora_state_dict", {})
    )
    challenger_lora = lora_state_dict_digest(
        challenger_checkpoint.get("lora_state_dict", {})
    )
    if reference_lora != challenger_lora:
        raise RuntimeError(
            "Semantic challenger changed encoder LoRA tensors; a classifier-only "
            "challenger must share the exact geometry feature extractor"
        )

    decoder_state = challenger_checkpoint.get("decoder_state_dict")
    if not decoder_state:
        raise KeyError(f"decoder_state_dict missing from {challenger_path}")
    include_highres = any(
        key.startswith("semantic_residual.") for key in decoder_state
    )
    semantic_residual = None
    semantic_residual_version = "none"
    if include_highres:
        semantic_residual, semantic_residual_version = (
            _build_checkpoint_semantic_residual(
                challenger_checkpoint, system.reference_model.decoder
            )
        )
    semantic_prefixes = ["seg_fpn.", "seg_branch."]
    if include_highres:
        semantic_prefixes.append("semantic_residual.")
    semantic_state = {
        key: value
        for key, value in decoder_state.items()
        if any(key.startswith(prefix) for prefix in semantic_prefixes)
    }
    challenger = SemanticChallenger(
        system.reference_model.decoder,
        semantic_residual=semantic_residual,
        semantic_residual_version=semantic_residual_version,
    )
    challenger.load_state_dict(semantic_state, strict=True)
    challenger.to(device).eval()
    for parameter in challenger.parameters():
        parameter.requires_grad_(False)
    return challenger, {
        "checkpoint": str(challenger_path.resolve()),
        "epoch": int(challenger_checkpoint.get("epoch", -1)),
        "best_score": challenger_checkpoint.get(
            "best_composite_score", challenger_checkpoint.get("best_val_iou")
        ),
        "contract": "shared_encoder_lora_classifier_only",
        "high_resolution": bool(include_highres),
    }
