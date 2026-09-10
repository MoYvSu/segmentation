# -*- coding: utf-8 -*-
"""Direct SSL-LoRA semantic + affinity model construction and phase control."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from models.affinity_geometry import AffinityGeometryDecoder
from models.fpn_decoder import FPNDecoder
from models.fused_deployment import FusedPhaseAffinityModel
from models.lora import inject_trunk_lora, load_lora_state_dict
from models.sam2_encoder import (
    SAM2Encoder,
    SAM2_INPUT_NORMALIZATION_NONE,
    canonical_sam2_input_normalization,
)
from utils.config import project_path
from utils.semantic_challenger import SemanticChallenger


DIRECT_SEMANTIC_AFFINITY_FORMAT = "direct_semantic_affinity_v1"


def direct_input_normalization(config: dict) -> str:
    """读取 direct/SSL 共用的 Hiera 输入合同，旧配置默认兼容原路径。"""
    return canonical_sam2_input_normalization(
        config.get("sam2", {}).get(
            "input_normalization", SAM2_INPUT_NORMALIZATION_NONE
        )
    )


def _ssl_artifact_input_normalization(path: Path, payload) -> str | None:
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        return direct_input_normalization(payload["config"])
    meta_path = path.parent / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "input_normalization" not in meta:
        return None
    return canonical_sam2_input_normalization(meta["input_normalization"])


def load_direct_ssl_lora_state(path: str | Path, config: dict):
    """加载 SSL-LoRA，并拒绝同形状但输入合同不同的静默错配。"""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = direct_input_normalization(config)
    actual = _ssl_artifact_input_normalization(path, payload)
    if actual is None:
        if expected != SAM2_INPUT_NORMALIZATION_NONE:
            raise RuntimeError(
                "normalized direct training requires SSL input_normalization "
                f"metadata: {path}"
            )
        actual = SAM2_INPUT_NORMALIZATION_NONE
    if actual != expected:
        raise RuntimeError(
            "SSL/direct input_normalization mismatch: "
            f"ssl={actual}, direct={expected}, path={path}"
        )
    state = payload.get("lora_state_dict") if isinstance(payload, dict) else None
    if state is None:
        state = payload
    if not isinstance(state, dict):
        raise RuntimeError(f"invalid SSL LoRA artifact: {path}")
    return state, actual


def assert_direct_checkpoint_input_contract(model, payload: dict):
    """在 resume/recovery 前验证模型与 checkpoint 的输入语义一致。"""
    stored = payload.get("config")
    if not isinstance(stored, dict):
        raise RuntimeError("direct checkpoint does not contain its build config")
    checkpoint_mode = direct_input_normalization(stored)
    model_mode = canonical_sam2_input_normalization(
        getattr(model.encoder, "input_normalization", SAM2_INPUT_NORMALIZATION_NONE)
    )
    if checkpoint_mode != model_mode:
        raise RuntimeError(
            "direct checkpoint input_normalization mismatch: "
            f"checkpoint={checkpoint_mode}, model={model_mode}"
        )


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
        input_normalization=direct_input_normalization(config),
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
    ssl_state, ssl_contract = load_direct_ssl_lora_state(ssl_path, config)
    loaded = _load_complete_lora_state(model, ssl_state)
    return model, {
        "ssl_lora_path": str(ssl_path.resolve()),
        "loaded_tensors": loaded,
        "input_normalization": ssl_contract,
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
    assert_direct_checkpoint_input_contract(model, payload)
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


def configure_direct_semantic_recovery(model):
    """冻结 encoder/LoRA/affinity，仅允许语义解码头继续学习。"""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.encoder.trainable_lora = False
    for parameter in model.semantic_decoder.parameters():
        parameter.requires_grad_(True)
    model.encoder.eval()
    model.affinity_decoder.eval()
    model.geometry_feature_adapter = None
    model.geometry_highres_refiner = None


def configure_direct_affinity_recovery(model):
    """冻结 encoder/LoRA/semantic，仅允许 affinity 解码头继续学习。"""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.encoder.trainable_lora = False
    for parameter in model.affinity_decoder.parameters():
        parameter.requires_grad_(True)
    model.encoder.eval()
    model.semantic_decoder.eval()
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
