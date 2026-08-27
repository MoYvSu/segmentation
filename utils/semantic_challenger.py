# -*- coding: utf-8 -*-
"""Load a decoder-only semantic challenger without changing geometry features."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from models.offset_geometry import lora_state_dict_digest
from utils.config import project_path


class SemanticChallenger(nn.Module):
    """Only the semantic FPN/head copied from a compatible Stage-2 decoder."""

    def __init__(self, reference_decoder: nn.Module):
        super().__init__()
        self.seg_fpn = copy.deepcopy(reference_decoder.seg_fpn)
        self.seg_branch = copy.deepcopy(reference_decoder.seg_branch)

    def forward(self, features):
        return self.seg_branch(self.seg_fpn(features))


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
    semantic_state = {
        key: value
        for key, value in decoder_state.items()
        if key.startswith("seg_fpn.") or key.startswith("seg_branch.")
    }
    challenger = SemanticChallenger(system.reference_model.decoder)
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
    }
