# -*- coding: utf-8 -*-
"""复赛后端适配：严格复用 S-align/V，冻结 D4 首步并保存完整共享特征。"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import torch

from models.direct_semantic_affinity import _load_complete_lora_state
from models.fused_deployment import (
    FUSED_DEPLOYMENT_FORMAT,
    FusedPhaseAffinityModel,
    build_fused_model_from_bundle,
)
from train_affinity_geometry_g1 import (
    build_system,
    load_geometry_checkpoint_state,
    validate_semantic_geometry_contract,
)
from train_offset_geometry import file_sha256
from utils.config import project_path
from utils.semantic_challenger import build_semantic_challenger


def _freeze(model):
    model.eval()
    model.encoder.trainable_lora = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_backend(config, device):
    """从部署配置构建一致的共享 encoder/语义/affinity；返回模型和来源元数据。"""
    if config.get("sam2", {}).get("input_normalization", "legacy_none") != "legacy_none":
        raise ValueError("S-align/V 后端要求 legacy_none 输入归一化")
    if not config.get("semantic_challenger", {}).get("replace_reference_semantic", False):
        raise ValueError("适配要求 semantic_challenger 同时提供分水岭语义与类别投票")
    cfg = config["affinity_geometry_g1"]
    system, reference_path, _, digest = build_system(config, cfg, device)
    if system.geometry_feature_adapter is not None or system.geometry_highres_refiner is not None:
        raise ValueError("本轮仅适配既有 S-align/V 双头，不包含额外 geometry adapter/refiner")
    geometry_path = Path(project_path(config, config["affinity_deployment"]["checkpoint"]))
    geometry_checkpoint = torch.load(geometry_path, map_location="cpu", weights_only=False)
    load_geometry_checkpoint_state(system, geometry_checkpoint)
    contract = validate_semantic_geometry_contract(
        config, cfg, geometry_checkpoint, reference_path, digest
    )
    semantic, semantic_metadata = build_semantic_challenger(config, system, reference_path, device)
    if semantic is None:
        raise ValueError("适配要求启用 S-align semantic_challenger")
    model = _freeze(FusedPhaseAffinityModel(
        system.reference_model.encoder, semantic, system.geometry_decoder
    ).to(device))
    if not model.parameter_summary()["constraint_passed"]:
        raise ValueError("后端参数量超过赛题限制")

    semantic_path = Path(semantic_metadata["checkpoint"])
    semantic_checkpoint = torch.load(semantic_path, map_location="cpu", weights_only=False)
    reference_checkpoint = torch.load(reference_path, map_location="cpu", weights_only=False)
    for source in (semantic_checkpoint, reference_checkpoint):
        mode = source.get("config", {}).get("sam2", {}).get("input_normalization", "legacy_none")
        if mode != "legacy_none":
            raise ValueError("源权重归一化与 S-align/V 部署不一致")
    loaded_lora = _load_complete_lora_state(model, reference_checkpoint["lora_state_dict"])

    # 与现有 build_fused_deployment_checkpoint 相同的架构记录与载入格式。
    decoder_cfg = copy.deepcopy(semantic_checkpoint.get("config", {}).get("decoder", {}))
    for key in ("fpn_channels", "num_classes", "dropout", "use_bn"):
        decoder_cfg.setdefault(key, config["decoder"][key])
    lora_cfg = semantic_checkpoint.get("config", {}).get("lora", {})
    lora_state = semantic_checkpoint.get("lora_state_dict", {})
    rank = next((int(value.shape[0]) for key, value in lora_state.items() if "lora_A" in key), None)
    if rank is None:
        raise ValueError("源权重缺少 LoRA 架构")
    architecture = {
        "sam2": {
            "config_file": str(config["sam2"]["config_file"]),
            "sam2_repo_path": str(config["sam2"]["sam2_repo_path"]),
            "input_normalization": "legacy_none",
        },
        "lora": {
            "rank": rank, "alpha": float(lora_cfg.get("alpha", 32.0)),
            "target_layers": lora_cfg.get("target_layers"),
        },
        "semantic_decoder": decoder_cfg,
        "affinity_decoder": {
            "affinity_channels": int(model.affinity_decoder.affinity_channels),
            "fpn_channels": int(cfg.get("fpn_channels", 256)),
            "up_channels": int(cfg.get("up_channels", 128)),
            "output_grid": int(model.affinity_decoder.output_grid),
        },
        "geometry_feature_adapter": None,
        "geometry_highres_refiner": None,
    }
    metadata = {
        "architecture": architecture,
        "semantic_geometry_contract": contract,
        "initial_semantic_state_digest": digest,
        "loaded_lora_tensors": loaded_lora,
        "sources": {
            "reference": {
                "path": str(Path(reference_path).resolve()), "sha256": file_sha256(reference_path),
                "epoch": int(reference_checkpoint.get("epoch", -1)),
            },
            "semantic": {**semantic_metadata, "sha256": file_sha256(semantic_path)},
            "affinity": {
                "path": str(geometry_path.resolve()), "sha256": file_sha256(geometry_path),
                "epoch": int(geometry_checkpoint.get("epoch", -1)),
            },
        },
    }
    return model, metadata


def save_backend(model, config, metadata, path, epoch, extra=None):
    """保存标准 fused bundle；训练续跑状态只进入独立 extra 字段。"""
    bundle = {
        "format": FUSED_DEPLOYMENT_FORMAT,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": int(epoch),
        "architecture": copy.deepcopy(metadata["architecture"]),
        "model_state_dict": {
            name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()
        },
        "parameter_summary": model.parameter_summary(),
        "deployment": {
            "fusion": copy.deepcopy(config["affinity_deployment"]),
            "inference": copy.deepcopy(config["inference"]),
        },
        "sources": copy.deepcopy(metadata["sources"]),
        "backend_adaptation": copy.deepcopy(metadata),
        "config": copy.deepcopy(config),
        "extra": extra if extra is not None else {},
    }
    output = Path(project_path(config, path))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    return bundle


def load_backend(path, config, device):
    """严格加载完整 fused 状态；仅 SAM2 代码位置使用当前机器配置。"""
    bundle = torch.load(project_path(config, path), map_location="cpu", weights_only=False)
    if bundle.get("format") != FUSED_DEPLOYMENT_FORMAT:
        raise ValueError(f"不支持的后端格式: {bundle.get('format')!r}")
    architecture = copy.deepcopy(bundle["architecture"])
    if architecture["sam2"].get("input_normalization", "legacy_none") != "legacy_none":
        raise ValueError("当前后端适配仅支持 legacy_none")
    if config.get("sam2", {}).get("input_normalization", "legacy_none") != "legacy_none":
        raise ValueError("当前推理归一化与适配权重不一致")
    architecture["sam2"]["sam2_repo_path"] = config["sam2"]["sam2_repo_path"]
    runtime_bundle = {**bundle, "architecture": architecture}
    model = build_fused_model_from_bundle(runtime_bundle, config, device)
    return _freeze(model), bundle


@torch.no_grad()
def restore_first(restorer, image, seed, strength=1.0):
    """原 T 步 schedule 的首个 x0 估计；独立随机数，不修改默认完整采样。"""
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if strength == 0.0:
        return image
    restorer.eval()
    generator = torch.Generator(device=image.device).manual_seed(int(seed))
    with torch.autocast(device_type=image.device.type, enabled=False):
        condition = image.float()
        step = torch.full((condition.shape[0],), int(restorer.num_steps),
                          dtype=torch.long, device=condition.device)
        state = restorer.q_sample(condition, condition, step, generator=generator)
        clean = restorer.denoise(state, condition, step).float()
        return (condition + strength * (clean - condition)).clamp(0.0, 1.0)
