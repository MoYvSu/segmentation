# -*- coding: utf-8 -*-
"""D5a 后端冷启动：通用 SAM2 + 纯 SSL LoRA + 两个随机任务头。"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch

from models.direct_semantic_affinity import (
    build_direct_semantic_affinity_model,
    configure_direct_training_phase,
)
from utils.config import project_path


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _head_summary(head):
    """保留初始状态摘要，区分冷启动与继续加载旧任务权重。"""
    state = head.state_dict()
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), value.dtype)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        "initialization": "random",
        "loaded_tensors": 0,
        "state_tensors": len(state),
        "parameters": sum(parameter.numel() for parameter in head.parameters()),
        "state_sha256": digest.hexdigest(),
    }


def _validate_config(config):
    initialization = config["backend_adaptation"]["initialization"]
    if initialization.get("mode") != "ssl_random_heads":
        raise ValueError("冷启动必须显式指定 initialization.mode=ssl_random_heads")
    expected_sha = initialization.get("ssl_sha256", "")
    if (not isinstance(expected_sha, str) or len(expected_sha) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha)):
        raise ValueError("冷启动必须指定纯 SSL 权重的 ssl_sha256")
    if config["sam2"].get("input_normalization", "legacy_none") != "legacy_none":
        raise ValueError("冷启动沿用 SSL 的 legacy_none 输入归一化")
    direct = config["direct_semantic_affinity"]
    semantic = direct["semantic_decoder"]
    # 现有 direct 构建器默认打开 highres，必须显式关闭以免继承错误分支。
    if semantic.get("highres") is not False:
        raise ValueError("本轮只允许 simple 语义头：highres 必须显式为 false")
    if semantic.get("semantic_residual", False):
        raise ValueError("simple 冷启动不支持 semantic_residual 分支")
    if int(direct["affinity_decoder"].get("channels", 8)) != 8:
        raise ValueError("既有 affinity 损失与部署要求 8 通道")
    # 避免把旧 direct 训练中的 uncovered 默认值悄悄带入已完成 GT。
    if direct.get("affinity_loss", {}).get("manual_uncovered_as_boundary") is not False:
        raise ValueError("本轮 affinity 损失必须显式忽略未标注区域")


def build_cold_backend(config, device):
    """构建随机双头并返回标准 fused bundle 可直接保存的来源和架构。"""
    _validate_config(config)
    initialization = config["backend_adaptation"]["initialization"]
    ssl_path = Path(project_path(config, config["lora"]["init_from"]))
    sam2_path = Path(project_path(
        config, config["paths"]["weights_dir"], config["paths"]["sam2_ckpt"]
    ))
    ssl_sha = _sha256(ssl_path)
    if ssl_sha != initialization["ssl_sha256"]:
        raise ValueError("纯 SSL 权重 ssl_sha256 不匹配；拒绝旧 joint LoRA 或其他权重")
    sam2_sha = _sha256(sam2_path)
    if initialization.get("sam2_sha256") not in (None, sam2_sha):
        raise ValueError("通用 SAM2 权重 sam2_sha256 不匹配")

    # 此入口只读 SAM2 与完整 SSL LoRA；不会访问旧 semantic/affinity checkpoint。
    model, loaded = build_direct_semantic_affinity_model(config, device)
    configure_direct_training_phase(model, train_lora=False)
    model.eval()
    semantic_cfg = config["direct_semantic_affinity"]["semantic_decoder"]
    affinity_cfg = config["direct_semantic_affinity"]["affinity_decoder"]
    lora_cfg = config["lora"]
    architecture = {
        "sam2": {
            "config_file": str(config["sam2"]["config_file"]),
            "sam2_repo_path": str(config["sam2"]["sam2_repo_path"]),
            "input_normalization": "legacy_none",
        },
        "lora": {
            "rank": int(lora_cfg.get("rank", 16)),
            "alpha": float(lora_cfg.get("alpha", 32.0)),
            "target_layers": copy.deepcopy(lora_cfg.get("target_layers")),
        },
        # fused 读取的是 FPNDecoder 参数名，与 direct 配置字段显式对应。
        "semantic_decoder": {
            "fpn_channels": int(semantic_cfg.get("fpn_channels", 256)),
            "num_classes": 2,
            "dropout": float(semantic_cfg.get("dropout", 0.1)),
            "use_bn": bool(semantic_cfg.get("use_group_norm", True)),
            "semantic_residual": False,
        },
        "affinity_decoder": {
            "affinity_channels": int(model.affinity_decoder.affinity_channels),
            "fpn_channels": int(affinity_cfg.get("fpn_channels", 256)),
            "up_channels": int(affinity_cfg.get("up_channels", 128)),
            "output_grid": int(model.affinity_decoder.output_grid),
        },
        "geometry_feature_adapter": None,
        "geometry_highres_refiner": None,
    }
    heads = {
        "semantic": _head_summary(model.semantic_decoder),
        "affinity": _head_summary(model.affinity_decoder),
    }
    base_tensors = sum(
        "lora_A" not in name and "lora_B" not in name
        for name in model.encoder.trunk.state_dict()
    )
    metadata = {
        "architecture": architecture,
        "initialization": copy.deepcopy(initialization),
        "initial_heads": heads,
        "initial_semantic_state_digest": heads["semantic"]["state_sha256"],
        "initial_affinity_state_digest": heads["affinity"]["state_sha256"],
        "loaded_lora_tensors": int(loaded["loaded_tensors"]),
        "semantic_geometry_contract": "shared_ssl_encoder_random_simple_and_affinity",
        "sources": {
            "sam2": {
                "path": str(sam2_path.resolve()), "sha256": sam2_sha,
                "loaded_tensors": base_tensors,
                "scope": "image_encoder.trunk_only",
            },
            "ssl_lora": {
                "path": str(ssl_path.resolve()), "sha256": ssl_sha,
                "loaded_tensors": int(loaded["loaded_tensors"]),
            },
            "semantic": copy.deepcopy(heads["semantic"]),
            "affinity": copy.deepcopy(heads["affinity"]),
        },
    }
    return model, metadata
