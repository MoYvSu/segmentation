# -*- coding: utf-8 -*-
"""One-backbone semantic + affinity deployment model and bundle loader."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from models.affinity_geometry import (
    AffinityGeometryDecoder,
    HighResolutionShortAffinityResidual,
)
from models.fpn_decoder import FPNDecoder
from models.gda_mim import GenerativeDomainAdapterPyramid
from models.lora import inject_trunk_lora
from models.sam2_encoder import SAM2Encoder
from utils.semantic_challenger import (
    SemanticChallenger,
    _build_checkpoint_semantic_residual,
)


FUSED_DEPLOYMENT_FORMAT = "phase_affinity_fused_v1"


class FusedPhaseAffinityModel(nn.Module):
    """Run one shared encoder and two independent task decoders."""

    def __init__(
        self,
        encoder: nn.Module,
        semantic_decoder: nn.Module,
        affinity_decoder: nn.Module,
        geometry_feature_adapter: nn.Module | None = None,
        geometry_highres_refiner: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.semantic_decoder = semantic_decoder
        self.affinity_decoder = affinity_decoder
        self.geometry_feature_adapter = geometry_feature_adapter
        self.geometry_highres_refiner = geometry_highres_refiner

    def forward(self, image: torch.Tensor):
        features = self.encoder(image)
        semantic_logits = self.semantic_decoder(features, image)
        geometry_features = features
        if self.geometry_feature_adapter is not None:
            geometry_features = self.geometry_feature_adapter(
                geometry_features, gated=True
            )
        affinity_output = self.affinity_decoder(geometry_features)
        affinity_logits = affinity_output["affinity_logits"]
        if self.geometry_highres_refiner is not None:
            affinity_logits = self.geometry_highres_refiner(
                affinity_output["affinity_feature"], affinity_logits, image
            )["affinity_logits"]
        return {
            "semantic_logits": semantic_logits,
            "affinity_logits": affinity_logits,
        }

    def parameter_summary(self):
        values = {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "semantic_decoder": sum(
                p.numel() for p in self.semantic_decoder.parameters()
            ),
            "affinity_decoder": sum(
                p.numel() for p in self.affinity_decoder.parameters()
            ),
            "geometry_feature_adapter": (
                sum(p.numel() for p in self.geometry_feature_adapter.parameters())
                if self.geometry_feature_adapter is not None else 0
            ),
            "geometry_highres_refiner": (
                sum(p.numel() for p in self.geometry_highres_refiner.parameters())
                if self.geometry_highres_refiner is not None else 0
            ),
        }
        values["total"] = sum(values.values())
        values["total_M"] = values["total"] / 1.0e6
        values["constraint_passed"] = values["total"] < 500_000_000
        return values


def _semantic_scaffold(decoder_cfg, in_channels):
    return FPNDecoder(
        in_channels=list(in_channels),
        fpn_channels=int(decoder_cfg.get("fpn_channels", 256)),
        num_classes=int(decoder_cfg.get("num_classes", 2)),
        dropout=float(decoder_cfg.get("dropout", 0.1)),
        use_bn=bool(decoder_cfg.get("use_bn", True)),
        boundary_refine=False,
        center_head=False,
        semantic_residual=False,
    )


def build_fused_model_from_bundle(bundle, config, device):
    """Construct a fused model without loading any source checkpoint."""
    if bundle.get("format") != FUSED_DEPLOYMENT_FORMAT:
        raise ValueError(
            f"unsupported fused checkpoint format: {bundle.get('format')!r}"
        )
    architecture = bundle["architecture"]
    sam2_cfg = architecture["sam2"]
    paths_cfg = config["paths"]
    encoder = SAM2Encoder(
        config_file=str(sam2_cfg["config_file"]),
        ckpt_path=None,
        device=device,
        freeze=True,
        sam2_repo_path=str(
            Path(paths_cfg["project_root"]) / sam2_cfg["sam2_repo_path"]
        ),
    )
    lora_cfg = architecture["lora"]
    inject_trunk_lora(
        encoder,
        rank=int(lora_cfg["rank"]),
        alpha=float(lora_cfg["alpha"]),
        target_layers=lora_cfg.get("target_layers"),
        use_grad_checkpoint=False,
    )

    decoder_cfg = architecture["semantic_decoder"]
    scaffold = _semantic_scaffold(decoder_cfg, encoder.get_stage_channels())
    residual = None
    residual_version = "none"
    if bool(decoder_cfg.get("semantic_residual", False)):
        residual, residual_version = _build_checkpoint_semantic_residual(
            {"config": {"decoder": decoder_cfg}}, scaffold
        )
    semantic_decoder = SemanticChallenger(
        scaffold,
        semantic_residual=residual,
        semantic_residual_version=residual_version,
    )
    del scaffold

    affinity_cfg = architecture["affinity_decoder"]
    affinity_decoder = AffinityGeometryDecoder(
        in_channels=encoder.get_stage_channels(),
        affinity_channels=int(affinity_cfg["affinity_channels"]),
        fpn_channels=int(affinity_cfg["fpn_channels"]),
        up_channels=int(affinity_cfg["up_channels"]),
        output_grid=int(affinity_cfg["output_grid"]),
    )
    refiner_cfg = architecture.get("geometry_highres_refiner")
    highres_refiner = None
    if refiner_cfg is not None:
        highres_refiner = HighResolutionShortAffinityResidual(
            feature_channels=int(affinity_cfg["up_channels"]),
            short_channels=int(refiner_cfg.get("short_channels", 4)),
            feature_hidden=int(refiner_cfg.get("feature_hidden", 32)),
            image_hidden=int(refiner_cfg.get("image_hidden", 16)),
            fusion_hidden=int(refiner_cfg.get("fusion_hidden", 32)),
            max_logit_delta=float(refiner_cfg.get("max_logit_delta", 1.0)),
        )
    adapter_cfg = architecture.get("geometry_feature_adapter")
    adapter = None
    if adapter_cfg is not None:
        adapter = GenerativeDomainAdapterPyramid(
            channels=encoder.get_stage_channels(),
            bottleneck_ratio=int(adapter_cfg.get("bottleneck_ratio", 8)),
            gate_mode=str(adapter_cfg.get("gate_mode", "scalar")),
            active_scales=adapter_cfg.get("active_scales"),
        )

    model = FusedPhaseAffinityModel(
        encoder,
        semantic_decoder,
        affinity_decoder,
        geometry_feature_adapter=adapter,
        geometry_highres_refiner=highres_refiner,
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_fused_deployment_model(bundle_path, config, device):
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    return build_fused_model_from_bundle(bundle, config, device), bundle
