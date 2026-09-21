# -*- coding: utf-8 -*-
"""Stage1初始化及最终两头训练的严格权重交接。"""

import copy
from pathlib import Path

import torch

from models.direct_semantic_affinity import (
    DIRECT_SEMANTIC_AFFINITY_FORMAT,
    _build_direct_architecture,
    _load_complete_lora_state,
)
from models.lora import extract_lora_state_dict
from utils.checkpoint import FORMAT_VERSION, architecture_from_config
from utils.config import project_path


def prefix_state(state, prefix):
    return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}


def initialize_from_stage1(model, checkpoint):
    """只允许完整LoRA、语义coarse及边界FPN；新增highres/affinity输出保持初始值。"""
    state = checkpoint["decoder_state_dict"]
    loaded = _load_complete_lora_state(model, checkpoint["lora_state_dict"])
    model.semantic_decoder.seg_fpn.load_state_dict(prefix_state(state, "seg_fpn."), strict=True)
    model.semantic_decoder.seg_branch.load_state_dict(prefix_state(state, "seg_branch."), strict=True)
    model.affinity_decoder.geometry_fpn.load_state_dict(prefix_state(state, "boundary_fpn."), strict=True)
    return {"source_epoch": checkpoint.get("epoch"), "lora_tensors": loaded,
            "semantic_coarse": "stage1", "affinity_fpn": "stage1_boundary",
            "new_components": ["semantic_residual", "affinity_upsample", "affinity_output"]}


def build_stage1_final_tasks(config, device):
    if config.get("sam2", {}).get("input_normalization", "legacy_none") != "legacy_none":
        raise ValueError("当前Stage1权重要求legacy_none归一化")
    source = project_path(config, config["stage1_final_tasks"]["checkpoint"])
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_mode = checkpoint.get("config", {}).get("sam2", {}).get("input_normalization", "legacy_none")
    if source_mode != "legacy_none":
        raise ValueError("Stage1 checkpoint归一化不匹配")
    model = _build_direct_architecture(config, device)
    return model, initialize_from_stage1(model, checkpoint)


def task_checkpoint(model, config, epoch, phase, selection, split):
    return {
        "format": DIRECT_SEMANTIC_AFFINITY_FORMAT,
        "epoch": epoch + 1, "epoch_index": epoch, "phase": phase,
        "semantic_state_dict": model.semantic_decoder.state_dict(),
        "affinity_state_dict": model.affinity_decoder.state_dict(),
        "lora_state_dict": extract_lora_state_dict(model.encoder),
        "config": config, "selection": selection, "split": split,
    }


def export_final_task_checkpoints(task, stage1, semantic_config, output_dir):
    """导出同一LoRA下的完整语义载体和affinity，供既有最终训练器读取。"""
    if task.get("phase") != "joint_lora":
        raise ValueError("最终训练必须从joint_lora的loss-best开始")
    decoder = dict(stage1["decoder_state_dict"])
    decoder.update(task["semantic_state_dict"])
    # 载体的边界FPN也使用适配后的特征；旧边界输出头不进入最终推理。
    decoder.update({"boundary_fpn." + k: v for k, v in
                    prefix_state(task["affinity_state_dict"], "geometry_fpn.").items()})
    cfg = copy.deepcopy(semantic_config)
    reference = {
        "format_version": FORMAT_VERSION,
        "architecture": architecture_from_config(cfg),
        "decoder_state_dict": decoder,
        "lora_state_dict": task["lora_state_dict"],
        "config": cfg, "epoch": task["epoch_index"], "best_composite_score": 0.0,
        "task_adaptation": {"phase": task["phase"], "selection": task["selection"]},
    }
    geometry = {
        "geometry_state_dict": task["affinity_state_dict"],
        "epoch": task["epoch"], "selection": task["selection"],
        "lora_state_dict": task["lora_state_dict"],
        "config": task["config"], "task_adaptation_export": True,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(reference, output_dir / "reference.pth")
    torch.save(geometry, output_dir / "affinity.pth")
    return reference, geometry
