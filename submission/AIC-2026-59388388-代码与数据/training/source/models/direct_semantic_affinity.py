# -*- coding: utf-8 -*-
"""Direct SSL-LoRA semantic + affinity model construction and phase control."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from models.affinity_geometry import AffinityGeometryDecoder
from models.fpn_decoder import FPNDecoder
from models.fused_deployment import FusedPhaseAffinityModel
from models.lora import inject_trunk_lora, load_lora_state_dict
from models.sam2_encoder import SAM2Encoder
from utils.config import project_path
from utils.semantic_challenger import SemanticChallenger


DIRECT_SEMANTIC_AFFINITY_FORMAT = "direct_semantic_affinity_v1"


def _load_complete_lora_state(model, state: dict) -> int:
    """Load exactly the LoRA tensors required by the configured trunk."""
    current = {
        key: value
        for key, value in model.encoder.trunk.state_dict().items()
        if "lora_A" in key or "lora_B" in key
    }
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    if missing or unexpected:
        raise RuntimeError(
            "direct LoRA architecture mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    shape_mismatch = [
        key
        for key in current
        if tuple(current[key].shape) != tuple(state[key].shape)
    ]
    if shape_mismatch:
        raise RuntimeError(
            f"direct LoRA shape mismatch: {shape_mismatch[:5]}"
        )
    loaded = load_lora_state_dict(model.encoder, state)
    if loaded != len(current):
        raise RuntimeError(
            f"loaded {loaded}/{len(current)} direct LoRA tensors"
        )
    return loaded


def _checkpoint_architecture_config(payload: dict, runtime_config: dict) -> dict:
    """Use checkpoint architecture while resolving machine paths at runtime."""
    stored = payload.get("config")
    if not isinstance(stored, dict):
        raise RuntimeError("direct checkpoint does not contain its build config")
    config = copy.deepcopy(stored)
    for key in ("project_root", "weights_dir", "sam2_ckpt"):
        config["paths"][key] = runtime_config["paths"][key]
    config["sam2"]["sam2_repo_path"] = runtime_config["sam2"][
        "sam2_repo_path"
    ]
    return config


def _build_semantic_decoder(config: dict, in_channels):
    cfg = config["direct_semantic_affinity"]["semantic_decoder"]
    scaffold = FPNDecoder(
        in_channels=list(in_channels),
        fpn_channels=int(cfg.get("fpn_channels", 256)),
        num_classes=2,
        dropout=float(cfg.get("dropout", 0.1)),
        use_bn=bool(cfg.get("use_group_norm", True)),
        boundary_refine=False,
        center_head=False,
        semantic_residual=bool(cfg.get("highres", True)),
        semantic_residual_version="highres_v1",
        semantic_residual_hidden=int(cfg.get("highres_hidden", 64)),
        semantic_residual_color_channels=int(cfg.get("color_channels", 16)),
        semantic_residual_use_photometric=bool(
            cfg.get("use_photometric", True)
        ),
        semantic_residual_max_logit_delta=float(
            cfg.get("max_logit_delta", 2.0)
        ),
        semantic_residual_half_channels=int(cfg.get("half_channels", 48)),
        semantic_residual_full_channels=int(cfg.get("full_channels", 24)),
    )
    residual = scaffold.semantic_residual
    version = "highres_v1" if residual is not None else "none"
    decoder = SemanticChallenger(
        scaffold,
        semantic_residual=residual,
        semantic_residual_version=version,
    )
    del scaffold
    return decoder


def _build_direct_architecture(config: dict, device):
    """Build the direct architecture without loading task or SSL weights."""
    sam2_cfg = config["sam2"]
    paths_cfg = config["paths"]
    lora_cfg = config["lora"]
    direct_cfg = config["direct_semantic_affinity"]
    checkpoint_path = project_path(
        config, paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"]
    )
    encoder = SAM2Encoder(
        config_file=sam2_cfg["config_file"],
        ckpt_path=checkpoint_path,
        device=str(device),
        freeze=True,
        sam2_repo_path=project_path(config, sam2_cfg["sam2_repo_path"]),
    )
    injected = inject_trunk_lora(
        encoder,
        rank=int(lora_cfg.get("rank", 16)),
        alpha=float(lora_cfg.get("alpha", 32.0)),
        target_layers=lora_cfg.get("target_layers"),
        use_grad_checkpoint=bool(lora_cfg.get("gradient_checkpointing", True)),
    )
    if injected <= 0:
        raise RuntimeError("direct dual-head training injected no LoRA layers")
    channels = encoder.get_stage_channels()
    semantic_decoder = _build_semantic_decoder(config, channels)
    affinity_cfg = direct_cfg["affinity_decoder"]
    affinity_decoder = AffinityGeometryDecoder(
        in_channels=channels,
        affinity_channels=int(affinity_cfg.get("channels", 8)),
        fpn_channels=int(affinity_cfg.get("fpn_channels", 256)),
        up_channels=int(affinity_cfg.get("up_channels", 128)),
        output_grid=int(direct_cfg.get("affinity_grid", 512)),
    )
    model = FusedPhaseAffinityModel(
        encoder, semantic_decoder, affinity_decoder
    ).to(device)
    summary = model.parameter_summary()
    if not summary["constraint_passed"]:
        raise RuntimeError(
            f"direct dual-head model has {summary['total_M']:.2f}M parameters"
        )
    return model


def build_direct_semantic_affinity_model(config: dict, device):
    """Build a random dual decoder on top of the SSL-initialized LoRA trunk."""
    model = _build_direct_architecture(config, device)
    ssl_path = Path(project_path(config, config["lora"]["init_from"]))
    if not ssl_path.is_file():
        raise FileNotFoundError(ssl_path)
    ssl_state = torch.load(ssl_path, map_location="cpu", weights_only=False)
    if isinstance(ssl_state, dict) and "lora_state_dict" in ssl_state:
        ssl_state = ssl_state["lora_state_dict"]
    loaded = _load_complete_lora_state(model, ssl_state)
    return model, {
        "ssl_lora_path": str(ssl_path.resolve()),
        "loaded_tensors": loaded,
    }


def load_direct_semantic_affinity_model(checkpoint_path, config: dict, device):
    """Load a deployable model without requiring the original SSL LoRA file."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != DIRECT_SEMANTIC_AFFINITY_FORMAT:
        raise RuntimeError(
            f"unsupported direct checkpoint format: {payload.get('format')!r}"
        )
    architecture_config = _checkpoint_architecture_config(payload, config)
    architecture_config["lora"]["gradient_checkpointing"] = False
    model = _build_direct_architecture(architecture_config, device)
    model.semantic_decoder.load_state_dict(
        payload["semantic_state_dict"], strict=True
    )
    model.affinity_decoder.load_state_dict(
        payload["affinity_state_dict"], strict=True
    )
    _load_complete_lora_state(model, payload["lora_state_dict"])
    model.eval()
    model.encoder.trainable_lora = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def configure_direct_training_phase(model, *, train_lora: bool):
    """Keep SAM2 frozen; train both heads and optionally the injected LoRA."""
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.encoder.trunk.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parameter.requires_grad_(bool(train_lora))
    model.encoder.trainable_lora = bool(train_lora)
    for parameter in model.semantic_decoder.parameters():
        parameter.requires_grad_(True)
    for parameter in model.affinity_decoder.parameters():
        parameter.requires_grad_(True)
    model.geometry_feature_adapter = None
    model.geometry_highres_refiner = None


def direct_parameter_groups(model, learning_rates: dict):
    groups = [
        {
            "name": "semantic",
            "params": [
                value
                for value in model.semantic_decoder.parameters()
                if value.requires_grad
            ],
            "lr": float(learning_rates["semantic"]),
        },
        {
            "name": "affinity",
            "params": [
                value
                for value in model.affinity_decoder.parameters()
                if value.requires_grad
            ],
            "lr": float(learning_rates["affinity"]),
        },
    ]
    lora = [
        value
        for name, value in model.encoder.trunk.named_parameters()
        if ("lora_A" in name or "lora_B" in name) and value.requires_grad
    ]
    if lora:
        groups.append(
            {"name": "lora", "params": lora, "lr": float(learning_rates["lora"])}
        )
    if any(not group["params"] for group in groups):
        raise RuntimeError("direct dual-head optimizer contains an empty head group")
    return groups
