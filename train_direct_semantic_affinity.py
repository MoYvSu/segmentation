# -*- coding: utf-8 -*-
"""Train semantic and class-agnostic affinity heads directly from SSL LoRA."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.affinity_geometry_augmentation import AffinityGeometryAugmentedDataset
from data.direct_dual_head_dataset import (
    DirectDualHeadDataset,
    load_direct_evaluation_target,
    load_manual_target_archive,
    mask_prediction_to_evaluation_domain,
)
from data.sam2_geometry_dataset import SAM2GeometryDataset
from models.direct_semantic_affinity import (
    DIRECT_SEMANTIC_AFFINITY_FORMAT,
    build_direct_semantic_affinity_model,
    assert_direct_checkpoint_input_contract,
    configure_direct_affinity_recovery,
    configure_direct_semantic_recovery,
    configure_direct_training_phase,
    direct_parameter_groups,
    load_direct_ssl_lora_state,
)
from models.lora import extract_lora_state_dict, load_lora_state_dict
from utils.affinity_deployment import (
    crop_affinity_boundary_output,
    crop_letterbox_output,
    postprocess,
    prepare_image,
    probability_to_logit,
)
from utils.affinity_loss import balanced_affinity_loss, build_affinity_targets_torch
from utils.config import load_config, project_path
from utils.instance_metrics import (
    evaluate_instance_pair,
    summarize_instance_results,
)
from utils.loss import BoundaryLoss
from utils.progressive_aug import ProgressiveAppearanceAug
from utils.run_recorder import RunRecorder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_direct_semantic_affinity")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deterministic_split(names, val_fraction, seed, forced_train_names=()):
    names = sorted({str(name) for name in names})
    forced = {Path(name).stem for name in forced_train_names}
    candidates = [name for name in names if name not in forced]
    generator = np.random.default_rng(int(seed))
    generator.shuffle(candidates)
    val_count = max(1, int(round(len(names) * float(val_fraction))))
    val_names = sorted(candidates[:val_count])
    train_names = sorted(name for name in names if name not in set(val_names))
    if not train_names or not val_names:
        raise ValueError("direct dual-head split requires non-empty train and val")
    return train_names, val_names


def resolve_direct_split(config: dict, names):
    """显式名单优先；检查完整覆盖，避免同 seed 不同划分。"""
    cfg = config["direct_semantic_affinity"]
    split_file = cfg.get("split_file")
    if not split_file:
        return deterministic_split(
            names, float(cfg.get("val_fraction", 0.20)),
            int(cfg.get("seed", 42)),
            cfg.get("forced_train_names", ("train_001", "train_002")),
        )
    payload = json.loads(
        Path(project_path(config, split_file)).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("direct split_file must contain an object")
    groups = []
    for key in ("train", "val"):
        values = payload.get(key)
        if not isinstance(values, list) or not values or any(
            not isinstance(name, str) or not name for name in values
        ):
            raise ValueError(f"direct split_file requires non-empty {key} names")
        if len(values) != len(set(values)):
            raise ValueError(f"direct split_file has duplicate {key} names")
        groups.append(sorted(values))
    train_names, val_names = groups
    if set(train_names) & set(val_names):
        raise ValueError("direct split_file train/val overlap")
    expected = set(names)
    listed = set(train_names) | set(val_names)
    if listed != expected:
        raise ValueError(
            "direct split_file must cover the current labeled dataset exactly: "
            f"missing={sorted(expected - listed)}, extra={sorted(listed - expected)}"
        )
    if "seed" in payload and int(payload["seed"]) != int(cfg.get("seed", 42)):
        raise ValueError("direct split_file seed differs from config")
    forced = {Path(name).stem for name in cfg.get("forced_train_names", ())}
    if forced & set(val_names):
        raise ValueError("direct split_file puts forced training names in validation")
    return train_names, val_names


def validate_direct_run_options(cfg: dict):
    """关闭昂贵部署评估时仍必须按验证损失选优。"""
    deployment_enabled = bool(cfg.get("deployment_validation", {}).get("enabled", True))
    if not deployment_enabled:
        if not bool(cfg.get("loss_selection", {}).get("enabled", False)):
            raise ValueError("disabled deployment_validation requires loss_selection.enabled")
        if any(bool(cfg.get(key, {}).get("enabled", False)) for key in (
            "semantic_recovery", "affinity_recovery"
        )):
            raise ValueError("disabled deployment_validation is supported only by direct dual training")
    fixed_scales = cfg.get("loss_scales")
    if fixed_scales is not None:
        required = {"semantic", "manual_affinity", "pseudo_affinity"}
        if not isinstance(fixed_scales, dict) or set(fixed_scales) != required:
            raise ValueError(f"loss_scales must contain exactly {sorted(required)}")
        try:
            fixed_scales = {key: float(value) for key, value in fixed_scales.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("loss_scales must be positive finite numbers") from exc
        if any(not math.isfinite(value) or value <= 0 for value in fixed_scales.values()):
            raise ValueError("loss_scales must be positive finite numbers")
    return deployment_enabled, fixed_scales


def validate_sam2_geometry_approval(dataset_dir: str | Path):
    dataset_dir = Path(dataset_dir)
    approval_path = dataset_dir / "approval.json"
    manifest_path = dataset_dir / "manifest.jsonl"
    if not approval_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"approved SAM2 geometry requires {approval_path} and {manifest_path}"
        )
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("approved") is not True:
        raise RuntimeError("SAM2 geometry approval.json must contain approved=true")
    for field in ("reviewed_by", "reviewed_at"):
        if not str(approval.get(field, "")).strip():
            raise RuntimeError(f"SAM2 geometry approval.json requires {field}")
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 1 <= len(rows) <= 249:
        raise RuntimeError(f"SAM2 geometry source count must be 1..249, got {len(rows)}")
    hashes = [str(row.get("source_sha256", "")) for row in rows]
    if any(not value for value in hashes) or len(hashes) != len(set(hashes)):
        raise RuntimeError("SAM2 geometry source hashes are missing or duplicated")
    if any(row.get("class_label") is not None for row in rows):
        raise RuntimeError("SAM2 geometry must remain class-agnostic")
    if int(approval.get("source_count", -1)) != len(rows):
        raise RuntimeError("SAM2 geometry approval source_count mismatches manifest")
    masks = list((dataset_dir / "masks").glob("*.npz"))
    if len(masks) != len(rows):
        raise RuntimeError(
            f"SAM2 geometry masks={len(masks)} do not match manifest={len(rows)}"
        )
    return {"source_count": len(rows), "approval": approval}


def validate_manual_target_dataset(dataset_dir: str | Path, raw_dir: str | Path):
    """验证 seeded-watershed manual_target_v2 的文件与来源契约。"""
    dataset_dir = Path(dataset_dir)
    raw_dir = Path(raw_dir)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manual target manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "manual_target_v2":
        raise RuntimeError(
            f"unsupported manual target format: {manifest.get('format')!r}"
        )
    if float(manifest.get("max_fill_distance_native", -1)) != 8.0:
        raise RuntimeError("manual_target_v2 requires reviewed native radius=8")
    if int(manifest.get("version", -1)) != 1:
        raise RuntimeError("manual_target_v2 requires manifest version=1")
    if float(manifest.get("blur_sigma", -1)) != 1.2:
        raise RuntimeError("manual_target_v2 requires reviewed blur_sigma=1.2")
    samples = [str(value) for value in manifest.get("samples", [])]
    if int(manifest.get("sample_count", -1)) != len(samples):
        raise RuntimeError("manual target sample_count mismatches manifest samples")
    if not samples or len(samples) != len(set(samples)):
        raise RuntimeError("manual target samples are empty or duplicated")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    raw_names = {
        path.stem
        for path in raw_dir.iterdir()
        if path.suffix.lower() in extensions and path.with_suffix(".json").is_file()
    }
    if set(samples) != raw_names:
        raise RuntimeError("manual target samples do not match LabelMe image/JSON pairs")
    missing = []
    for stem in samples:
        for path in (
            dataset_dir / f"{stem}_gt.npz",
            dataset_dir / f"{stem}_class.json",
        ):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "manual target files missing: " + ", ".join(missing[:8])
        )
    total_pixels = 0
    valid_pixels = 0
    for stem in samples:
        target_path = dataset_dir / f"{stem}_gt.npz"
        instance_map, _, _, valid, _ = load_manual_target_archive(target_path)
        annotation = json.loads(
            (raw_dir / f"{stem}.json").read_text(encoding="utf-8")
        )
        expected_shape = (
            int(annotation.get("imageHeight") or 0),
            int(annotation.get("imageWidth") or 0),
        )
        if min(expected_shape) <= 0 or instance_map.shape != expected_shape:
            raise RuntimeError(
                f"manual target shape mismatch for {stem}: "
                f"{instance_map.shape} vs {expected_shape}"
            )
        total_pixels += int(instance_map.size)
        valid_pixels += int(valid.sum())
    return {
        "format": manifest["format"],
        "sample_count": len(samples),
        "max_fill_distance_native": float(manifest["max_fill_distance_native"]),
        "coverage": valid_pixels / max(1, total_pixels),
    }


def resolve_direct_monitor_paths(config: dict):
    cfg = config["direct_semantic_affinity"]
    monitor_cfg = cfg.get("monitor", {})
    image_dir = Path(project_path(config, monitor_cfg.get("image_dir", "data/unlabeled")))
    if not image_dir.is_dir():
        raise FileNotFoundError(f"direct monitor image_dir not found: {image_dir}")
    manifest = str(monitor_cfg.get("manifest", "")).strip()
    if manifest:
        manifest_path = Path(project_path(config, manifest))
        if not manifest_path.is_file():
            raise FileNotFoundError(f"direct monitor manifest not found: {manifest_path}")
        names = [
            line.strip()
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        paths = [image_dir / name for name in names]
    else:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        paths = [
            path for path in sorted(image_dir.iterdir())
            if path.suffix.lower() in extensions
        ]
    num_images = int(monitor_cfg.get("num_images", 10))
    selected = paths[:num_images]
    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "direct monitor images not found: " + ", ".join(map(str, missing))
        )
    if len(selected) != num_images:
        raise RuntimeError(
            f"direct monitor requires {num_images} images, found {len(selected)}"
        )
    return selected


def check_sources(config: dict):
    missing = []
    paths_cfg = config["paths"]
    direct_cfg = config["direct_semantic_affinity"]
    validate_direct_run_options(direct_cfg)
    manual_target_cfg = direct_cfg.get("manual_target", {})
    manual_target_enabled = bool(manual_target_cfg.get("enabled", False))
    required_paths = [
        project_path(config, paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"]),
        project_path(config, config["sam2"]["sam2_repo_path"]),
        project_path(config, paths_cfg["raw_data_dir"]),
        project_path(config, config["lora"]["init_from"]),
    ]
    if not manual_target_enabled:
        required_paths.append(project_path(config, config["boundary"]["gt_dir"]))
    if direct_cfg.get("split_file"):
        required_paths.append(project_path(config, direct_cfg["split_file"]))
    for path in required_paths:
        if not Path(path).exists():
            missing.append(str(path))
    manual_target_report = None
    if manual_target_enabled:
        manual_target_dir = project_path(config, manual_target_cfg["dataset_dir"])
        if not Path(manual_target_dir).is_dir():
            missing.append(str(manual_target_dir))
        elif Path(project_path(config, paths_cfg["raw_data_dir"])).is_dir():
            manual_target_report = validate_manual_target_dataset(
                manual_target_dir,
                project_path(config, paths_cfg["raw_data_dir"]),
            )
    pseudo_cfg = direct_cfg.get("sam2_geometry", {})
    approval = None
    if bool(pseudo_cfg.get("enabled", False)):
        dataset_dir = project_path(config, pseudo_cfg["dataset_dir"])
        source_dir = project_path(config, pseudo_cfg["source_dir"])
        if not Path(source_dir).is_dir():
            missing.append(str(source_dir))
        if not Path(dataset_dir).is_dir():
            missing.append(str(dataset_dir))
        else:
            approval = validate_sam2_geometry_approval(dataset_dir)
    monitor_cfg = direct_cfg.get("monitor", {})
    monitor_paths = []
    if bool(monitor_cfg.get("enabled", False)):
        monitor_paths = resolve_direct_monitor_paths(config)
    recovery_modes = [
        direct_cfg.get("semantic_recovery", {}),
        direct_cfg.get("affinity_recovery", {}),
    ]
    enabled_recoveries = [
        recovery_cfg
        for recovery_cfg in recovery_modes
        if bool(recovery_cfg.get("enabled", False))
    ]
    if len(enabled_recoveries) > 1:
        raise ValueError("only one direct recovery mode may be enabled")
    for recovery_cfg in enabled_recoveries:
        recovery_path = Path(
            project_path(config, recovery_cfg["init_from_checkpoint"])
        )
        if not recovery_path.is_file():
            missing.append(str(recovery_path))
    if missing:
        raise FileNotFoundError("missing direct dual-head inputs: " + ", ".join(missing))
    if direct_cfg.get("split_file"):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        resolve_direct_split(config, [
            path.stem
            for path in Path(project_path(config, paths_cfg["raw_data_dir"])).iterdir()
            if path.suffix.lower() in extensions and path.with_suffix(".json").is_file()
        ])
    load_direct_ssl_lora_state(
        project_path(config, config["lora"]["init_from"]), config
    )
    return {
        "sam2_geometry": approval,
        "manual_target": manual_target_report,
        "monitor_images": [path.name for path in monitor_paths],
    }


def build_loaders(config: dict, device):
    cfg = config["direct_semantic_affinity"]
    raw_dir = Path(project_path(config, config["paths"]["raw_data_dir"]))
    manual_target_cfg = cfg.get("manual_target", {})
    manual_target_enabled = bool(manual_target_cfg.get("enabled", False))
    manual_target_dir = (
        Path(project_path(config, manual_target_cfg["dataset_dir"]))
        if manual_target_enabled
        else None
    )
    gt_dir = (
        None
        if manual_target_enabled
        else Path(project_path(config, config["boundary"]["gt_dir"]))
    )
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    names = [
        path.stem
        for path in sorted(raw_dir.iterdir())
        if path.suffix.lower() in extensions and path.with_suffix(".json").is_file()
    ]
    train_names, val_names = resolve_direct_split(config, names)
    common = {
        "data_dir": raw_dir,
        "gt_dir": gt_dir,
        "image_size": int(cfg.get("input_size", 1024)),
        "affinity_grid": int(cfg.get("affinity_grid", 512)),
        "manual_target_dir": manual_target_dir,
        "manual_target_boundary_dilation": int(
            manual_target_cfg.get("boundary_dilation", 2)
        ),
    }
    train_dataset = DirectDualHeadDataset(
        sample_names=train_names,
        augment=True,
        augmentation=cfg.get("augmentation", {}),
        native_crop=cfg.get("native_crop"),
        **common,
    )
    val_dataset = DirectDualHeadDataset(
        sample_names=val_names, augment=False, **common
    )
    epoch_samples = int(cfg.get("manual_samples_per_epoch", len(train_dataset)))
    sampler = WeightedRandomSampler(
        torch.ones(len(train_dataset), dtype=torch.double),
        num_samples=epoch_samples,
        replacement=True,
        generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))),
    )
    workers = int(cfg.get("num_workers", 0))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    manual_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        sampler=sampler,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    pseudo_loader = None
    pseudo_cfg = cfg.get("sam2_geometry", {})
    if bool(pseudo_cfg.get("enabled", False)):
        pseudo_base = SAM2GeometryDataset(
            project_path(config, pseudo_cfg["dataset_dir"]),
            project_path(config, pseudo_cfg["source_dir"]),
            image_size=int(cfg.get("input_size", 1024)),
            output_grid=int(cfg.get("affinity_grid", 512)),
            use_eroded_interiors=bool(pseudo_cfg.get("use_eroded_interiors", False)),
            cache_in_memory=bool(pseudo_cfg.get("cache_in_memory", True)),
        )
        pseudo_dataset = AffinityGeometryAugmentedDataset(
            pseudo_base, cfg.get("augmentation", {})
        )
        pseudo_loader = DataLoader(
            pseudo_dataset,
            batch_size=int(pseudo_cfg.get("batch_size", cfg.get("batch_size", 1))),
            shuffle=True,
            drop_last=False,
            **loader_kwargs,
        )
    return manual_loader, val_loader, pseudo_loader, val_dataset, {
        "train": train_names,
        "val": val_names,
        "seed": int(cfg.get("seed", 42)),
        "sam2_geometry_count": (
            len(pseudo_loader.dataset) if pseudo_loader is not None else 0
        ),
    }


def build_semantic_criterion(config: dict, device):
    loss_cfg = config["direct_semantic_affinity"]["semantic_loss"]
    return BoundaryLoss(
        seg_dice_weight=float(loss_cfg.get("dice_weight", 0.30)),
        semantic_instance_weight=float(loss_cfg.get("instance_weight", 0.75)),
        semantic_core_radius=int(loss_cfg.get("core_radius", 3)),
        semantic_core_min_pixels=int(loss_cfg.get("core_min_pixels", 12)),
        semantic_core_boundary_threshold=float(
            loss_cfg.get("core_boundary_threshold", 0.20)
        ),
        semantic_instance_class_balance=bool(loss_cfg.get("class_balance", True)),
        semantic_instance_ferrite_weight=float(loss_cfg.get("ferrite_weight", 1.0)),
        semantic_instance_hard_gamma=float(loss_cfg.get("hard_gamma", 0.5)),
        semantic_instance_hard_floor=float(loss_cfg.get("hard_floor", 0.50)),
        semantic_instance_pool_weight=float(loss_cfg.get("pool_weight", 0.50)),
        semantic_thin_instance_weight=float(loss_cfg.get("thin_weight", 1.50)),
        semantic_tversky_weight=0.0,
        freeze_boundary=True,
        center_weight=0.0,
    ).to(device)


def compute_semantic_loss(criterion, semantic_logits, batch):
    target = torch.cat(
        [batch["semantic_target"], batch["semantic_boundary"]], dim=1
    )
    prediction = torch.cat(
        [semantic_logits, torch.zeros_like(semantic_logits)], dim=1
    )
    loss, _, _ = criterion(
        prediction,
        target,
        instance_map=batch["semantic_instance_map"],
        seg_weight=batch.get("semantic_loss_weight"),
    )
    return loss


def compute_affinity_loss(affinity_logits, batch, loss_cfg: dict, *, pseudo: bool):
    instance_key = "instance_map" if pseudo else "affinity_instance_map"
    valid_key = "valid_content" if pseudo else "affinity_valid_content"
    instance_map = batch[instance_key]
    valid_content = batch[valid_key]
    manual_source = batch["uncovered_boundary_source"].bool().reshape(-1)
    uncovered_as_boundary = (
        manual_source
        if bool(loss_cfg.get("manual_uncovered_as_boundary", True))
        else torch.zeros_like(manual_source)
    )
    target, edge_valid, uncovered_mask = build_affinity_targets_torch(
        instance_map,
        valid_content,
        uncovered_as_boundary=uncovered_as_boundary,
        return_uncovered_mask=True,
    )
    edge_weight = torch.ones_like(affinity_logits)
    manual_weight = float(loss_cfg.get("manual_uncovered_boundary_weight", 0.20))
    if not 0.0 <= manual_weight <= 1.0:
        raise ValueError("manual_uncovered_boundary_weight must be within [0, 1]")
    edge_weight[uncovered_mask] = manual_weight
    if pseudo:
        pseudo_negative_weight = float(loss_cfg.get("pseudo_negative_weight", 1.0))
        if not 0.0 <= pseudo_negative_weight <= 1.0:
            raise ValueError("pseudo_negative_weight must be within [0, 1]")
        negative = edge_valid & (target <= 0.5)
        edge_weight[negative] = pseudo_negative_weight
    return balanced_affinity_loss(
        affinity_logits,
        target,
        edge_valid,
        negative_weight=float(loss_cfg.get("negative_weight", 1.5)),
        hard_negative_weight=float(loss_cfg.get("hard_negative_weight", 1.0)),
        hard_negative_gamma=float(loss_cfg.get("hard_negative_gamma", 2.0)),
        edge_weight=edge_weight,
        normalize_edge_weights=bool(loss_cfg.get("normalize_edge_weights", True)),
    )


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def next_restarting(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.no_grad()
def calibrate_loss_scales(
    model, manual_loader, pseudo_loader, criterion, loss_cfg, device, batches
):
    model.eval()
    totals = {"semantic": 0.0, "manual_affinity": 0.0, "pseudo_affinity": 0.0}
    counts = {key: 0 for key in totals}
    for index, raw_batch in enumerate(manual_loader):
        if index >= int(batches):
            break
        batch = move_batch(raw_batch, device)
        output = model(batch["image"])
        semantic = compute_semantic_loss(criterion, output["semantic_logits"], batch)
        affinity, _ = compute_affinity_loss(
            output["affinity_logits"], batch, loss_cfg, pseudo=False
        )
        totals["semantic"] += float(semantic)
        totals["manual_affinity"] += float(affinity)
        counts["semantic"] += 1
        counts["manual_affinity"] += 1
    if pseudo_loader is not None:
        for index, raw_batch in enumerate(pseudo_loader):
            if index >= int(batches):
                break
            batch = move_batch(raw_batch, device)
            output = model(batch["image"])
            affinity, _ = compute_affinity_loss(
                output["affinity_logits"], batch, loss_cfg, pseudo=True
            )
            totals["pseudo_affinity"] += float(affinity)
            counts["pseudo_affinity"] += 1
    scales = {
        key: max(1.0e-6, totals[key] / max(1, counts[key])) for key in totals
    }
    if pseudo_loader is None:
        scales["pseudo_affinity"] = 1.0
    return scales


@torch.no_grad()
def validate_semantic(model, loader, device):
    model.eval()
    intersection = [0, 0]
    union = [0, 0]
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = model(batch["image"])["semantic_logits"]
        prediction = (torch.sigmoid(logits) >= 0.5).long()[:, 0]
        target = batch["semantic_target"].long()[:, 0]
        valid = batch["semantic_valid_content"].bool()[:, 0]
        for class_id in (0, 1):
            pred_class = (prediction == class_id) & valid
            target_class = (target == class_id) & valid
            intersection[class_id] += int((pred_class & target_class).sum())
            union[class_id] += int((pred_class | target_class).sum())
    ious = [intersection[index] / max(1, union[index]) for index in (0, 1)]
    return {"semantic_iou_pearlite": ious[0], "semantic_iou_ferrite": ious[1], "semantic_miou": sum(ious) / 2.0}


@torch.no_grad()
def validate_direct_objective(
    model,
    loader,
    criterion,
    loss_cfg,
    scales,
    device,
):
    """固定验证集上的人工目标 loss；SAM2 pseudo 不参与 checkpoint 选优。"""
    model.eval()
    totals = {"semantic": 0.0, "manual_affinity": 0.0, "objective": 0.0}
    intersection = [0, 0]
    union = [0, 0]
    steps = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(batch["image"])
        semantic_loss = compute_semantic_loss(
            criterion, output["semantic_logits"], batch
        )
        affinity_loss, _ = compute_affinity_loss(
            output["affinity_logits"], batch, loss_cfg, pseudo=False
        )
        objective = (
            semantic_loss / float(scales["semantic"])
            + affinity_loss / float(scales["manual_affinity"])
        )
        totals["semantic"] += float(semantic_loss)
        totals["manual_affinity"] += float(affinity_loss)
        totals["objective"] += float(objective)
        steps += 1

        prediction = (torch.sigmoid(output["semantic_logits"]) >= 0.5).long()[:, 0]
        target = batch["semantic_target"].long()[:, 0]
        valid = batch["semantic_valid_content"].bool()[:, 0]
        for class_id in (0, 1):
            pred_class = (prediction == class_id) & valid
            target_class = (target == class_id) & valid
            intersection[class_id] += int((pred_class & target_class).sum())
            union[class_id] += int((pred_class | target_class).sum())

    if steps == 0:
        raise RuntimeError("validation loader produced no batches")
    ious = [intersection[index] / max(1, union[index]) for index in (0, 1)]
    result = {
        "val_objective_loss": totals["objective"] / steps,
        "val_semantic_loss": totals["semantic"] / steps,
        "val_manual_affinity_loss": totals["manual_affinity"] / steps,
        "semantic_iou_pearlite": ious[0],
        "semantic_iou_ferrite": ious[1],
        "semantic_miou": sum(ious) / 2.0,
    }
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise RuntimeError(f"non-finite validation objective: {result}")
    return result


def update_phase_loss_best(selection_state, phase, value, epoch):
    """严格按较小 validation loss 更新本阶段最优，tie 保留较早 epoch。"""
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite validation loss for {phase}: {value}")
    losses = selection_state.setdefault("phase_best_loss", {})
    epochs = selection_state.setdefault("phase_best_epoch", {})
    previous = float(losses.get(str(phase), math.inf))
    if value >= previous:
        return False
    losses[str(phase)] = value
    epochs[str(phase)] = int(epoch) + 1
    return True


@torch.no_grad()
def evaluate_direct_deployment(model, image_paths, config, device):
    cfg = config["direct_semantic_affinity"]
    deploy = cfg["deployment_validation"]
    infer_cfg = config["inference"]
    fusion_kwargs = {
        "distance2_weight": float(deploy.get("distance2_weight", 0.50)),
        "distance4_weight": float(deploy.get("distance4_weight", 0.25)),
        "support_threshold": float(deploy.get("support_threshold", 0.20)),
        "support_temperature": float(deploy.get("support_temperature", 0.05)),
        "short_reduction": str(deploy.get("short_reduction", "mean")),
        "short_softmax_temperature": float(
            deploy.get("short_softmax_temperature", 0.15)
        ),
    }
    manual_target_cfg = cfg.get("manual_target", {})
    manual_target_dir = (
        Path(project_path(config, manual_target_cfg["dataset_dir"]))
        if bool(manual_target_cfg.get("enabled", False))
        else None
    )
    model.eval()
    rows = []
    ground_truth_sources = set()
    ignored_unknown_pixels = 0
    with tempfile.TemporaryDirectory(prefix="direct-dual-deploy-") as temp_dir:
        for image_path in image_paths:
            image, tensor, pad_h, pad_w = prepare_image(
                image_path, int(cfg.get("input_size", 1024)), device
            )
            output = model(tensor)
            semantic_native = crop_letterbox_output(
                output["semantic_logits"],
                int(cfg.get("input_size", 1024)),
                pad_h,
                pad_w,
                image.shape[:2],
            ).cpu()
            boundary_native = crop_affinity_boundary_output(
                {"affinity_logits": output["affinity_logits"]},
                int(cfg.get("input_size", 1024)),
                pad_h,
                pad_w,
                image.shape[:2],
                str(deploy.get("fusion_mode", "gated")),
                fusion_kwargs,
            ).cpu()
            watershed_output = torch.cat(
                [semantic_native, probability_to_logit(boundary_native)], dim=1
            )
            _, pred_map, pred_classes = postprocess(
                watershed_output,
                image.shape[:2],
                temp_dir,
                Path(image_path).stem,
                infer_cfg,
                float(deploy.get("boundary_threshold", 0.65)),
                False,
                image_rgb=image,
            )
            gt_map, gt_classes, gt_valid, gt_audit = (
                load_direct_evaluation_target(
                    image_path,
                    image.shape[:2],
                    manual_target_dir=manual_target_dir,
                )
            )
            pred_eval = mask_prediction_to_evaluation_domain(pred_map, gt_valid)
            rows.append(
                evaluate_instance_pair(gt_map, gt_classes, pred_eval, pred_classes)
            )
            ground_truth_sources.add(str(gt_audit["source"]))
            ignored_unknown_pixels += int(
                gt_audit.get("ignored_unknown_pixels", 0)
            )
    if len(ground_truth_sources) != 1:
        raise RuntimeError(
            f"mixed deployment ground truth sources: {ground_truth_sources}"
        )
    summary = summarize_instance_results(rows)
    summary["ground_truth_source"] = next(iter(ground_truth_sources))
    summary["ignored_unknown_pixels"] = ignored_unknown_pixels
    return summary


@torch.no_grad()
def save_direct_monitor(model, config, epoch, device):
    """保存既有约定的语义 JET 与 affinity 边界 HOT 两种热力图。"""
    cfg = config["direct_semantic_affinity"]
    monitor_cfg = cfg.get("monitor", {})
    output_dir = Path(
        project_path(
            config,
            monitor_cfg.get("output_dir", f"{cfg['output_dir']}/monitor"),
        )
    ) / f"epoch_{epoch + 1:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_max_side = int(monitor_cfg.get("preview_max_side", 0))
    preview_dir = output_dir / "preview"
    if preview_max_side > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)
    deploy = cfg["deployment_validation"]
    fusion_kwargs = {
        "distance2_weight": float(deploy.get("distance2_weight", 0.50)),
        "distance4_weight": float(deploy.get("distance4_weight", 0.25)),
        "support_threshold": float(deploy.get("support_threshold", 0.20)),
        "support_temperature": float(deploy.get("support_temperature", 0.05)),
        "short_reduction": str(deploy.get("short_reduction", "mean")),
        "short_softmax_temperature": float(
            deploy.get("short_softmax_temperature", 0.15)
        ),
    }
    input_size = int(cfg.get("input_size", 1024))
    model.eval()
    for image_path in resolve_direct_monitor_paths(config):
        image, tensor, pad_h, pad_w = prepare_image(image_path, input_size, device)
        output = model(tensor)
        semantic_native = crop_letterbox_output(
            output["semantic_logits"], input_size, pad_h, pad_w, image.shape[:2]
        )
        boundary_native = crop_affinity_boundary_output(
            {"affinity_logits": output["affinity_logits"]},
            input_size,
            pad_h,
            pad_w,
            image.shape[:2],
            str(deploy.get("fusion_mode", "gated")),
            fusion_kwargs,
        )
        semantic = torch.sigmoid(semantic_native)[0, 0].cpu().numpy()
        boundary = boundary_native[0, 0].cpu().numpy()
        semantic_heatmap = cv2.applyColorMap(
            np.clip(semantic * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        boundary_heatmap = cv2.applyColorMap(
            np.clip(boundary * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_HOT,
        )
        stem = Path(image_path).stem
        cv2.imwrite(str(output_dir / f"{stem}_seg.png"), semantic_heatmap)
        cv2.imwrite(str(output_dir / f"{stem}_boundary.png"), boundary_heatmap)
        if preview_max_side > 0:
            height, width = semantic_heatmap.shape[:2]
            scale = min(1.0, preview_max_side / max(height, width))
            preview_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            semantic_preview = cv2.resize(
                semantic_heatmap, preview_size, interpolation=cv2.INTER_AREA
            )
            boundary_preview = cv2.resize(
                boundary_heatmap, preview_size, interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(
                str(preview_dir / f"{stem}_seg.png"), semantic_preview
            )
            cv2.imwrite(
                str(preview_dir / f"{stem}_boundary.png"), boundary_preview
            )
    logger.info(
        "Direct monitor saved: epoch=%d images=%d dir=%s",
        epoch + 1,
        len(resolve_direct_monitor_paths(config)),
        output_dir,
    )


def save_checkpoint(
    path,
    model,
    config,
    epoch,
    phase,
    best_score,
    scales,
    split,
    optimizer,
    scheduler,
    selection_state=None,
    checkpoint_deployment_score=None,
):
    payload = {
        "format": DIRECT_SEMANTIC_AFFINITY_FORMAT,
        "epoch": int(epoch) + 1,
        "epoch_index": int(epoch),
        "phase": str(phase),
        "best_deployment_score": float(best_score) if best_score is not None else None,
        "semantic_state_dict": model.semantic_decoder.state_dict(),
        "affinity_state_dict": model.affinity_decoder.state_dict(),
        "lora_state_dict": extract_lora_state_dict(model.encoder),
        "loss_scales": dict(scales),
        "split": split,
        "config": config,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if selection_state is not None:
        payload["loss_selection"] = dict(selection_state)
    if checkpoint_deployment_score is not None:
        payload["checkpoint_deployment_score"] = float(
            checkpoint_deployment_score
        )
    torch.save(payload, path)


def load_checkpoint(model, path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != DIRECT_SEMANTIC_AFFINITY_FORMAT:
        raise RuntimeError(f"unsupported direct checkpoint: {payload.get('format')!r}")
    assert_direct_checkpoint_input_contract(model, payload)
    model.semantic_decoder.load_state_dict(payload["semantic_state_dict"], strict=True)
    model.affinity_decoder.load_state_dict(payload["affinity_state_dict"], strict=True)
    current_lora = {
        key
        for key in model.encoder.trunk.state_dict()
        if "lora_A" in key or "lora_B" in key
    }
    checkpoint_lora = set(payload["lora_state_dict"])
    if current_lora != checkpoint_lora:
        raise RuntimeError(
            "resume LoRA architecture mismatch: "
            f"missing={sorted(current_lora - checkpoint_lora)[:5]}, "
            f"unexpected={sorted(checkpoint_lora - current_lora)[:5]}"
        )
    loaded = load_lora_state_dict(model.encoder, payload["lora_state_dict"])
    if loaded != len(current_lora):
        raise RuntimeError(f"resume loaded {loaded}/{len(current_lora)} LoRA tensors")
    return payload


def phase_for_epoch(epoch, warmup_epochs):
    return "head_warmup" if int(epoch) < int(warmup_epochs) else "joint_lora"


def learning_rates_for_phase(cfg, phase):
    values = cfg["learning_rates"]
    if phase == "head_warmup":
        return {
            "semantic": values["warmup_semantic"],
            "affinity": values["warmup_affinity"],
            "lora": values["joint_lora"],
        }
    return {
        "semantic": values["joint_semantic"],
        "affinity": values["joint_affinity"],
        "lora": values["joint_lora"],
    }


def forward_affinity_with_frozen_encoder(model, image):
    """只为 affinity 尾训提取冻结特征，跳过 semantic decoder。"""
    with torch.no_grad():
        features = model.encoder(image)
    return model.affinity_decoder(features)["affinity_logits"]


@torch.no_grad()
def calibrate_affinity_recovery_scales(
    model, manual_loader, pseudo_loader, loss_cfg, device, batches
):
    """在 affinity-only 阶段起点重新估计人工与 SAM2 loss 尺度。"""
    model.eval()
    totals = {"manual_affinity": 0.0, "pseudo_affinity": 0.0}
    counts = {key: 0 for key in totals}
    for index, raw_batch in enumerate(manual_loader):
        if index >= int(batches):
            break
        batch = move_batch(raw_batch, device)
        logits = forward_affinity_with_frozen_encoder(model, batch["image"])
        loss, _ = compute_affinity_loss(logits, batch, loss_cfg, pseudo=False)
        totals["manual_affinity"] += float(loss)
        counts["manual_affinity"] += 1
    if pseudo_loader is not None:
        for index, raw_batch in enumerate(pseudo_loader):
            if index >= int(batches):
                break
            batch = move_batch(raw_batch, device)
            logits = forward_affinity_with_frozen_encoder(model, batch["image"])
            loss, _ = compute_affinity_loss(logits, batch, loss_cfg, pseudo=True)
            totals["pseudo_affinity"] += float(loss)
            counts["pseudo_affinity"] += 1
    return {
        "semantic": 1.0,
        "manual_affinity": max(
            1.0e-6,
            totals["manual_affinity"] / max(1, counts["manual_affinity"]),
        ),
        "pseudo_affinity": (
            max(
                1.0e-6,
                totals["pseudo_affinity"] / max(1, counts["pseudo_affinity"]),
            )
            if pseudo_loader is not None
            else 1.0
        ),
    }


def run_affinity_recovery(
    model,
    config,
    manual_loader,
    val_loader,
    pseudo_loader,
    val_dataset,
    split,
    device,
    resume_path=None,
):
    """从语义恢复 checkpoint 出发，只继续训练 affinity decoder。"""
    cfg = config["direct_semantic_affinity"]
    recovery_cfg = cfg["affinity_recovery"]
    output_dir = Path(project_path(config, cfg["output_dir"]))
    source_path = Path(
        project_path(
            config,
            resume_path or recovery_cfg["init_from_checkpoint"],
        )
    )
    source = load_checkpoint(model, source_path, device)
    if source.get("split") != split:
        raise RuntimeError("affinity recovery checkpoint data split differs")
    configure_direct_affinity_recovery(model)

    epochs = int(recovery_cfg.get("epochs", 10))
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.affinity_decoder.parameters()
            if parameter.requires_grad
        ],
        lr=float(recovery_cfg.get("learning_rate", 1.0e-5)),
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
        eps=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(recovery_cfg.get("minimum_learning_rate", 1.0e-6)),
    )
    start_epoch = 0
    if resume_path:
        if source.get("phase") != "affinity_recovery":
            raise RuntimeError("affinity recovery resume checkpoint has wrong phase")
        start_epoch = int(source.get("epoch_index", source["epoch"] - 1)) + 1
        optimizer.load_state_dict(source["optimizer_state_dict"])
        scheduler.load_state_dict(source["scheduler_state_dict"])
        scales = dict(source["loss_scales"])
        best_score = float(source.get("best_deployment_score", -math.inf))
    else:
        scales = calibrate_affinity_recovery_scales(
            model,
            manual_loader,
            pseudo_loader,
            cfg["affinity_loss"],
            device,
            int(
                recovery_cfg.get(
                    "loss_calibration_batches",
                    cfg.get("loss_calibration_batches", 4),
                )
            ),
        )
        best_score = -math.inf

    recorder = RunRecorder(
        config["paths"]["project_root"],
        phase="direct_dual",
        tag="affinity_recovery",
    )
    recorder.save_config(config)
    recorder.manifest["initialization"] = {
        "affinity_recovery_checkpoint": str(source_path.resolve()),
        "source_epoch": int(source["epoch"]),
        "source_phase": str(source.get("phase", "unknown")),
    }
    recorder.manifest["split"] = split
    recorder.manifest["loss_scales"] = scales
    recorder.save_manifest()

    def save_deployment_best(epoch, score):
        best_path = output_dir / "best_direct_dual.pth"
        save_checkpoint(
            best_path,
            model,
            config,
            epoch,
            "affinity_recovery",
            score,
            scales,
            split,
            optimizer,
            scheduler,
        )
        deployment_path = output_dir / "best_deployment_direct_dual.pth"
        shutil.copy2(best_path, deployment_path)
        recorder.copy_checkpoint(best_path, "best_direct_dual.pth")
        recorder.copy_checkpoint(best_path, "best_deployment_direct_dual.pth")

    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    val_paths = [Path(path) for path, _ in val_dataset.samples]
    pseudo_weight = float(cfg.get("pseudo_affinity_weight", 0.50))
    loss_cfg = cfg["affinity_loss"]
    logger.info(
        "Phase=affinity_recovery source_epoch=%d epochs=%d lr=%.3g "
        "pseudo=%s pseudo_weight=%.3f scales=%s",
        int(source["epoch"]),
        epochs,
        optimizer.param_groups[0]["lr"],
        pseudo_loader is not None,
        pseudo_weight,
        scales,
    )

    if start_epoch == 0:
        semantic_metrics = validate_semantic(model, val_loader, device)
        deployment = evaluate_direct_deployment(model, val_paths, config, device)
        best_score = float(deployment["score_total"])
        recorder.append_metrics(
            {
                "epoch": 0,
                "phase": "affinity_recovery_baseline",
                "train_loss": 0.0,
                "train_semantic_loss": 0.0,
                "train_manual_affinity_loss": 0.0,
                "train_pseudo_affinity_loss": 0.0,
                **semantic_metrics,
                "deployment_score_total": deployment["score_total"],
                "deployment_instance_miou": deployment["instance_miou_valid"],
                "deployment_ferrite_area_error": deployment[
                    "ferrite_area_relative_error"
                ],
                "deployment_macro_score": deployment["macro_image_score_total"],
                "deployment_pred_count": deployment["pred_count"],
                "deployment_valid_matches": deployment["valid_matches"],
                "lr_affinity": optimizer.param_groups[0]["lr"],
            }
        )
        save_deployment_best(-1, best_score)
        logger.info(
            "epoch=0 phase=affinity_recovery_baseline sem_miou=%.4f "
            "deploy=%.4f miou=%.4f area_err=%.4f pred=%d matches=%d",
            semantic_metrics["semantic_miou"],
            deployment["score_total"],
            deployment["instance_miou_valid"],
            deployment["ferrite_area_relative_error"],
            deployment["pred_count"],
            deployment["valid_matches"],
        )

    for epoch in range(start_epoch, epochs):
        model.eval()
        model.affinity_decoder.train()
        totals = {
            "loss": 0.0,
            "manual_affinity": 0.0,
            "pseudo_affinity": 0.0,
            "steps": 0,
        }
        pseudo_iterator = iter(pseudo_loader) if pseudo_loader is not None else None
        for raw_manual in manual_loader:
            manual = move_batch(raw_manual, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                manual_logits = forward_affinity_with_frozen_encoder(
                    model, manual["image"]
                )
                manual_loss, _ = compute_affinity_loss(
                    manual_logits, manual, loss_cfg, pseudo=False
                )
                total_loss = manual_loss / scales["manual_affinity"]
                pseudo_loss = torch.zeros((), device=device)
                if pseudo_loader is not None:
                    raw_pseudo, pseudo_iterator = next_restarting(
                        pseudo_loader, pseudo_iterator
                    )
                    pseudo = move_batch(raw_pseudo, device)
                    pseudo_logits = forward_affinity_with_frozen_encoder(
                        model, pseudo["image"]
                    )
                    pseudo_loss, _ = compute_affinity_loss(
                        pseudo_logits, pseudo, loss_cfg, pseudo=True
                    )
                    total_loss = total_loss + pseudo_weight * (
                        pseudo_loss / scales["pseudo_affinity"]
                    )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.affinity_decoder.parameters(),
                float(cfg.get("grad_clip", 1.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(total_loss.detach())
            totals["manual_affinity"] += float(manual_loss.detach())
            totals["pseudo_affinity"] += float(pseudo_loss.detach())
            totals["steps"] += 1
        scheduler.step()

        semantic_metrics = validate_semantic(model, val_loader, device)
        deployment = evaluate_direct_deployment(model, val_paths, config, device)
        row = {
            "epoch": epoch + 1,
            "phase": "affinity_recovery",
            "train_loss": totals["loss"] / max(1, totals["steps"]),
            "train_semantic_loss": 0.0,
            "train_manual_affinity_loss": totals["manual_affinity"]
            / max(1, totals["steps"]),
            "train_pseudo_affinity_loss": totals["pseudo_affinity"]
            / max(1, totals["steps"]),
            **semantic_metrics,
            "deployment_score_total": deployment["score_total"],
            "deployment_instance_miou": deployment["instance_miou_valid"],
            "deployment_ferrite_area_error": deployment[
                "ferrite_area_relative_error"
            ],
            "deployment_macro_score": deployment["macro_image_score_total"],
            "deployment_pred_count": deployment["pred_count"],
            "deployment_valid_matches": deployment["valid_matches"],
            "deployment_ground_truth_source": deployment[
                "ground_truth_source"
            ],
            "deployment_ignored_unknown_pixels": deployment[
                "ignored_unknown_pixels"
            ],
            "lr_affinity": optimizer.param_groups[0]["lr"],
        }
        recorder.append_metrics(row)
        if selection_enabled:
            recorder.manifest["loss_selection"] = selection_state
            recorder.save_manifest()
        latest_path = output_dir / "latest_direct_dual.pth"
        save_checkpoint(
            latest_path,
            model,
            config,
            epoch,
            "affinity_recovery",
            max(best_score, float(deployment["score_total"])),
            scales,
            split,
            optimizer,
            scheduler,
        )
        if float(deployment["score_total"]) > best_score:
            best_score = float(deployment["score_total"])
            save_deployment_best(epoch, best_score)
        checkpoint_interval = int(cfg.get("checkpoint_interval", 10))
        if (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == epochs:
            periodic_path = output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pth"
            save_checkpoint(
                periodic_path,
                model,
                config,
                epoch,
                "affinity_recovery",
                best_score,
                scales,
                split,
                optimizer,
                scheduler,
            )
            logger.info("Periodic checkpoint saved: %s", periodic_path)
        monitor_cfg = cfg.get("monitor", {})
        monitor_interval = int(monitor_cfg.get("interval", 0))
        if bool(monitor_cfg.get("enabled", False)) and monitor_interval > 0 and (
            (epoch + 1) % monitor_interval == 0 or epoch + 1 == epochs
        ):
            save_direct_monitor(model, config, epoch, device)
        logger.info(
            "epoch=%d phase=affinity_recovery train=%.4f manual=%.4f "
            "pseudo=%.4f sem_miou=%.4f deploy=%.4f miou=%.4f "
            "area_err=%.4f pred=%d matches=%d",
            epoch + 1,
            row["train_loss"],
            row["train_manual_affinity_loss"],
            row["train_pseudo_affinity_loss"],
            row["semantic_miou"],
            row["deployment_score_total"],
            row["deployment_instance_miou"],
            row["deployment_ferrite_area_error"],
            row["deployment_pred_count"],
            row["deployment_valid_matches"],
        )
    return 0


def run_semantic_recovery(
    model,
    config,
    manual_loader,
    val_loader,
    val_dataset,
    split,
    criterion,
    device,
    resume_path=None,
):
    """从既有几何 checkpoint 出发，只恢复语义解码头。"""
    cfg = config["direct_semantic_affinity"]
    recovery_cfg = cfg["semantic_recovery"]
    output_dir = Path(project_path(config, cfg["output_dir"]))
    source_path = Path(
        project_path(
            config,
            resume_path or recovery_cfg["init_from_checkpoint"],
        )
    )
    source = load_checkpoint(model, source_path, device)
    if source.get("split") != split:
        raise RuntimeError("semantic recovery checkpoint data split differs")
    configure_direct_semantic_recovery(model)

    epochs = int(recovery_cfg.get("epochs", 20))
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.semantic_decoder.parameters()
            if parameter.requires_grad
        ],
        lr=float(recovery_cfg.get("learning_rate", 1.0e-5)),
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
        eps=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(recovery_cfg.get("minimum_learning_rate", 1.0e-6)),
    )
    start_epoch = 0
    if resume_path:
        if source.get("phase") != "semantic_recovery":
            raise RuntimeError("semantic recovery resume checkpoint has wrong phase")
        start_epoch = int(source.get("epoch_index", source["epoch"] - 1)) + 1
        optimizer.load_state_dict(source["optimizer_state_dict"])
        scheduler.load_state_dict(source["scheduler_state_dict"])

    recorder = RunRecorder(
        config["paths"]["project_root"],
        phase="direct_dual",
        tag="semantic_recovery",
    )
    recorder.save_config(config)
    recorder.manifest["initialization"] = {
        "semantic_recovery_checkpoint": str(source_path.resolve()),
        "source_epoch": int(source["epoch"]),
    }
    recorder.manifest["split"] = split
    recorder.save_manifest()

    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    val_paths = [Path(path) for path, _ in val_dataset.samples]
    best_score = -math.inf
    scales = {"semantic": 1.0, "manual_affinity": 1.0, "pseudo_affinity": 1.0}
    logger.info(
        "Phase=semantic_recovery source_epoch=%d epochs=%d lr=%.3g",
        int(source["epoch"]),
        epochs,
        optimizer.param_groups[0]["lr"],
    )

    for epoch in range(start_epoch, epochs):
        model.eval()
        model.semantic_decoder.train()
        total_loss = 0.0
        steps = 0
        for raw_manual in manual_loader:
            manual = move_batch(raw_manual, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                features = model.encoder(manual["image"])
                semantic_logits = model.semantic_decoder(features, manual["image"])
                loss = compute_semantic_loss(criterion, semantic_logits, manual)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.semantic_decoder.parameters(),
                float(cfg.get("grad_clip", 1.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            steps += 1
        scheduler.step()

        semantic_metrics = validate_semantic(model, val_loader, device)
        deployment = evaluate_direct_deployment(model, val_paths, config, device)
        row = {
            "epoch": epoch + 1,
            "phase": "semantic_recovery",
            "train_loss": total_loss / max(1, steps),
            "train_semantic_loss": total_loss / max(1, steps),
            "train_manual_affinity_loss": 0.0,
            "train_pseudo_affinity_loss": 0.0,
            **semantic_metrics,
            "deployment_score_total": deployment["score_total"],
            "deployment_instance_miou": deployment["instance_miou_valid"],
            "deployment_ferrite_area_error": deployment[
                "ferrite_area_relative_error"
            ],
            "deployment_macro_score": deployment["macro_image_score_total"],
            "deployment_pred_count": deployment["pred_count"],
            "deployment_valid_matches": deployment["valid_matches"],
            "lr_semantic": optimizer.param_groups[0]["lr"],
        }
        recorder.append_metrics(row)
        latest_path = output_dir / "latest_direct_dual.pth"
        save_checkpoint(
            latest_path,
            model,
            config,
            epoch,
            "semantic_recovery",
            max(best_score, float(deployment["score_total"])),
            scales,
            split,
            optimizer,
            scheduler,
        )
        if float(deployment["score_total"]) > best_score:
            best_score = float(deployment["score_total"])
            best_path = output_dir / "best_direct_dual.pth"
            save_checkpoint(
                best_path,
                model,
                config,
                epoch,
                "semantic_recovery",
                best_score,
                scales,
                split,
                optimizer,
                scheduler,
            )
            recorder.copy_checkpoint(best_path, "best_direct_dual.pth")
        checkpoint_interval = int(cfg.get("checkpoint_interval", 10))
        if (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == epochs:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pth",
                model,
                config,
                epoch,
                "semantic_recovery",
                best_score,
                scales,
                split,
                optimizer,
                scheduler,
            )
        monitor_cfg = cfg.get("monitor", {})
        monitor_interval = int(monitor_cfg.get("interval", 0))
        if bool(monitor_cfg.get("enabled", False)) and monitor_interval > 0 and (
            (epoch + 1) % monitor_interval == 0 or epoch + 1 == epochs
        ):
            save_direct_monitor(model, config, epoch, device)
        logger.info(
            "epoch=%d phase=semantic_recovery train=%.4f sem_miou=%.4f "
            "deploy=%.4f miou=%.4f area_err=%.4f pred=%d matches=%d",
            epoch + 1,
            row["train_loss"],
            row["semantic_miou"],
            row["deployment_score_total"],
            row["deployment_instance_miou"],
            row["deployment_ferrite_area_error"],
            row["deployment_pred_count"],
            row["deployment_valid_matches"],
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/train/direct_ssl_semantic_affinity.yaml"
    )
    parser.add_argument("--resume")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        source_report = check_sources(config)
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        if args.check:
            print("PRECHECK FAILED")
            print(f"  - {exc}")
            return 1
        raise
    if args.check:
        print("PRECHECK OK")
        print(json.dumps(source_report, ensure_ascii=False, indent=2))
        return 0

    cfg = config["direct_semantic_affinity"]
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    requested_device = str(config["sam2"].get("device", "cuda"))
    device = torch.device(
        "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        logger.warning("CUDA unavailable; direct dual-head training is CPU-only")
    model, initialization = build_direct_semantic_affinity_model(config, device)
    manual_loader, val_loader, pseudo_loader, val_dataset, split = build_loaders(
        config, device
    )
    output_dir = Path(project_path(config, cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    criterion = build_semantic_criterion(config, device)
    recovery_cfg = cfg.get("semantic_recovery", {})
    if bool(recovery_cfg.get("enabled", False)):
        return run_semantic_recovery(
            model,
            config,
            manual_loader,
            val_loader,
            val_dataset,
            split,
            criterion,
            device,
            resume_path=args.resume,
        )
    affinity_recovery_cfg = cfg.get("affinity_recovery", {})
    if bool(affinity_recovery_cfg.get("enabled", False)):
        return run_affinity_recovery(
            model,
            config,
            manual_loader,
            val_loader,
            pseudo_loader,
            val_dataset,
            split,
            device,
            resume_path=args.resume,
        )
    loss_cfg = cfg["affinity_loss"]
    resume_path = project_path(config, args.resume) if args.resume else None
    resume_payload = (
        load_checkpoint(model, resume_path, device) if resume_path else None
    )
    if resume_payload is not None and resume_payload.get("split") != split:
        raise RuntimeError("resume checkpoint data split differs from current config")
    start_epoch = (
        int(resume_payload.get("epoch_index", resume_payload["epoch"] - 1)) + 1
        if resume_payload
        else 0
    )
    deployment_enabled, fixed_scales = validate_direct_run_options(cfg)
    saved_score = resume_payload.get("best_deployment_score") if resume_payload else None
    best_score = (
        (float(saved_score) if saved_score is not None else -math.inf)
        if deployment_enabled else None
    )
    selection_cfg = cfg.get("loss_selection", {})
    selection_enabled = bool(selection_cfg.get("enabled", False))
    if selection_enabled and resume_payload is not None:
        saved_selection = resume_payload.get("loss_selection")
        if not isinstance(saved_selection, dict):
            raise RuntimeError(
                "loss-selected training cannot resume from a checkpoint without "
                "loss_selection state"
            )
        selection_state = dict(saved_selection)
        selection_state["phase_best_loss"] = dict(
            saved_selection.get("phase_best_loss", {})
        )
        selection_state["phase_best_epoch"] = dict(
            saved_selection.get("phase_best_epoch", {})
        )
    else:
        selection_state = {
            "metric": "val_objective_loss",
            "mode": "min",
            "phase_best_loss": {},
            "phase_best_epoch": {},
        }
    if resume_payload:
        scales = dict(resume_payload["loss_scales"])
        if fixed_scales is not None and scales != fixed_scales:
            raise RuntimeError("resume loss_scales differ from the fixed config")
    elif fixed_scales is not None:
        scales = fixed_scales
    else:
        scales = calibrate_loss_scales(
            model,
            manual_loader,
            pseudo_loader,
            criterion,
            loss_cfg,
            device,
            int(cfg.get("loss_calibration_batches", 4)),
        )
    logger.info("Loss calibration scales: %s", scales)

    warmup_epochs = int(cfg.get("head_warmup_epochs", 5))
    joint_epochs = int(cfg.get("joint_epochs", 20))
    total_epochs = warmup_epochs + joint_epochs
    recorder = RunRecorder(
        config["paths"]["project_root"], phase="direct_dual", tag="ssl_sem_aff"
    )
    recorder.save_config(config)
    recorder.manifest["initialization"] = initialization
    recorder.manifest["split"] = split
    recorder.manifest["loss_scales"] = scales
    recorder.manifest["loss_selection"] = (
        selection_state if selection_enabled else {"enabled": False}
    )
    recorder.save_manifest()

    optimizer = None
    scheduler = None
    active_phase = None
    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    val_paths = [Path(path) for path, _ in val_dataset.samples]
    pseudo_weight = float(cfg.get("pseudo_affinity_weight", 0.50))
    pseudo_start_epoch = int(cfg.get("pseudo_start_epoch", warmup_epochs))
    appearance_cfg = cfg.get("appearance_augmentation", {})
    appearance_augmentor = (
        ProgressiveAppearanceAug(appearance_cfg, device)
        if bool(appearance_cfg.get("enabled", False))
        else None
    )
    if appearance_augmentor is not None:
        logger.info(
            "Direct appearance augmentation enabled: policy=%s max_prob=%.3f",
            appearance_augmentor.policy,
            appearance_augmentor.max_prob,
        )

    for epoch in range(start_epoch, total_epochs):
        if appearance_augmentor is not None:
            appearance_augmentor.set_epoch(epoch)
        phase = phase_for_epoch(epoch, warmup_epochs)
        if phase != active_phase:
            if (
                selection_enabled
                and phase == "joint_lora"
                and epoch == warmup_epochs
                and bool(selection_cfg.get("restore_warmup_best", True))
            ):
                selection_root = (
                    Path(resume_path).parent
                    if resume_path is not None and start_epoch == warmup_epochs
                    else output_dir
                )
                warmup_best_path = (
                    selection_root / "best_head_warmup_by_val_loss.pth"
                )
                if not warmup_best_path.is_file():
                    raise FileNotFoundError(
                        "warmup loss-best checkpoint required before joint phase: "
                        f"{warmup_best_path}"
                    )
                warmup_best = load_checkpoint(model, warmup_best_path, device)
                selection_state["joint_from_warmup_epoch"] = int(
                    warmup_best["epoch"]
                )
                logger.info(
                    "Restored warmup validation-loss best for joint phase: "
                    "epoch=%d path=%s",
                    int(warmup_best["epoch"]),
                    warmup_best_path,
                )
            train_lora = phase == "joint_lora"
            configure_direct_training_phase(model, train_lora=train_lora)
            groups = direct_parameter_groups(
                model, learning_rates_for_phase(cfg, phase)
            )
            optimizer = torch.optim.AdamW(
                groups,
                weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
                eps=1.0e-4,
            )
            phase_epochs = warmup_epochs if phase == "head_warmup" else joint_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, phase_epochs),
                eta_min=float(cfg.get("minimum_learning_rate", 1.0e-6)),
            )
            if (
                resume_payload is not None
                and resume_payload.get("phase") == phase
                and epoch == start_epoch
            ):
                optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            active_phase = phase
            logger.info(
                "Phase=%s train_lora=%s groups=%s",
                phase,
                train_lora,
                [(group["name"], group["lr"]) for group in groups],
            )

        model.train()
        model.encoder.trunk.eval()
        totals = {
            "loss": 0.0,
            "semantic": 0.0,
            "manual_affinity": 0.0,
            "pseudo_affinity": 0.0,
            "steps": 0,
        }
        pseudo_iterator = iter(pseudo_loader) if pseudo_loader is not None else None
        use_pseudo = pseudo_loader is not None and epoch >= pseudo_start_epoch
        for raw_manual in manual_loader:
            manual = move_batch(raw_manual, device)
            if appearance_augmentor is not None:
                manual["image"] = appearance_augmentor(
                    manual["image"], manual.get("semantic_boundary")
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = model(manual["image"])
                semantic_loss = compute_semantic_loss(
                    criterion, output["semantic_logits"], manual
                )
                manual_affinity_loss, _ = compute_affinity_loss(
                    output["affinity_logits"], manual, loss_cfg, pseudo=False
                )
                total_loss = (
                    semantic_loss / scales["semantic"]
                    + manual_affinity_loss / scales["manual_affinity"]
                )
                pseudo_affinity_loss = torch.zeros((), device=device)
                if use_pseudo:
                    raw_pseudo, pseudo_iterator = next_restarting(
                        pseudo_loader, pseudo_iterator
                    )
                    pseudo = move_batch(raw_pseudo, device)
                    if appearance_augmentor is not None:
                        pseudo["image"] = appearance_augmentor(pseudo["image"])
                    pseudo_output = model(pseudo["image"])
                    pseudo_affinity_loss, _ = compute_affinity_loss(
                        pseudo_output["affinity_logits"],
                        pseudo,
                        loss_cfg,
                        pseudo=True,
                    )
                    total_loss = total_loss + pseudo_weight * (
                        pseudo_affinity_loss / scales["pseudo_affinity"]
                    )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(cfg.get("grad_clip", 1.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(total_loss.detach())
            totals["semantic"] += float(semantic_loss.detach())
            totals["manual_affinity"] += float(manual_affinity_loss.detach())
            totals["pseudo_affinity"] += float(pseudo_affinity_loss.detach())
            totals["steps"] += 1
        scheduler.step()

        if selection_enabled:
            validation = validate_direct_objective(
                model,
                val_loader,
                criterion,
                loss_cfg,
                scales,
                device,
            )
            semantic_metrics = {
                key: validation[key]
                for key in (
                    "semantic_iou_pearlite",
                    "semantic_iou_ferrite",
                    "semantic_miou",
                )
            }
        else:
            validation = {}
            semantic_metrics = validate_semantic(model, val_loader, device)
        deployment = (
            evaluate_direct_deployment(model, val_paths, config, device)
            if deployment_enabled else None
        )
        current_deployment_score = float(deployment["score_total"]) if deployment is not None else None
        deployment_improved = deployment is not None and current_deployment_score > best_score
        if deployment_improved:
            best_score = current_deployment_score
        loss_improved = False
        if selection_enabled:
            loss_improved = update_phase_loss_best(
                selection_state,
                phase,
                validation["val_objective_loss"],
                epoch,
            )
        row = {
            "epoch": epoch + 1,
            "phase": phase,
            "train_loss": totals["loss"] / max(1, totals["steps"]),
            "train_semantic_loss": totals["semantic"] / max(1, totals["steps"]),
            "train_manual_affinity_loss": totals["manual_affinity"] / max(1, totals["steps"]),
            "train_pseudo_affinity_loss": totals["pseudo_affinity"] / max(1, totals["steps"]),
            **semantic_metrics,
            **validation,
            "phase_loss_best": int(loss_improved),
            **({
                "deployment_score_total": deployment["score_total"],
                "deployment_instance_miou": deployment["instance_miou_valid"],
                "deployment_ferrite_area_error": deployment["ferrite_area_relative_error"],
                "deployment_macro_score": deployment["macro_image_score_total"],
                "deployment_pred_count": deployment["pred_count"],
                "deployment_valid_matches": deployment["valid_matches"],
            } if deployment is not None else {}),
            "appearance_aug_probability": (
                appearance_augmentor.current_prob
                if appearance_augmentor is not None
                else 0.0
            ),
            **{
                f"lr_{group.get('name', index)}": group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
        }
        recorder.append_metrics(row)
        if loss_improved:
            phase_best_path = output_dir / f"best_{phase}_by_val_loss.pth"
            save_checkpoint(
                phase_best_path,
                model,
                config,
                epoch,
                phase,
                best_score,
                scales,
                split,
                optimizer,
                scheduler,
                selection_state=selection_state,
                checkpoint_deployment_score=current_deployment_score,
            )
            if phase == "joint_lora":
                stable_best_path = output_dir / "best_direct_dual.pth"
                shutil.copy2(phase_best_path, stable_best_path)
                recorder.copy_checkpoint(stable_best_path, "best_direct_dual.pth")
        if deployment_improved:
            deployment_best_name = (
                "best_deployment_direct_dual.pth"
                if selection_enabled
                else "best_direct_dual.pth"
            )
            deployment_best_path = output_dir / deployment_best_name
            save_checkpoint(
                deployment_best_path,
                model,
                config,
                epoch,
                phase,
                best_score,
                scales,
                split,
                optimizer,
                scheduler,
                selection_state=selection_state if selection_enabled else None,
                checkpoint_deployment_score=current_deployment_score,
            )
            recorder.copy_checkpoint(deployment_best_path, deployment_best_name)
        latest_path = output_dir / "latest_direct_dual.pth"
        save_checkpoint(
            latest_path,
            model,
            config,
            epoch,
            phase,
            best_score,
            scales,
            split,
            optimizer,
            scheduler,
            selection_state=selection_state if selection_enabled else None,
            checkpoint_deployment_score=current_deployment_score,
        )
        checkpoint_interval = int(cfg.get("checkpoint_interval", 0))
        if checkpoint_interval > 0 and (
            (epoch + 1) % checkpoint_interval == 0
            or epoch + 1 == warmup_epochs
            or epoch + 1 == total_epochs
        ):
            periodic_path = output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pth"
            save_checkpoint(
                periodic_path,
                model,
                config,
                epoch,
                phase,
                best_score,
                scales,
                split,
                optimizer,
                scheduler,
                selection_state=selection_state if selection_enabled else None,
                checkpoint_deployment_score=current_deployment_score,
            )
            logger.info("Periodic checkpoint saved: %s", periodic_path)
        monitor_cfg = cfg.get("monitor", {})
        monitor_interval = int(monitor_cfg.get("interval", 0))
        if bool(monitor_cfg.get("enabled", False)) and monitor_interval > 0 and (
            (epoch + 1) % monitor_interval == 0 or epoch + 1 == total_epochs
        ):
            save_direct_monitor(model, config, epoch, device)
        if deployment is None:
            logger.info(
                "epoch=%d phase=%s train=%.4f val=%.4f sem_miou=%.4f deployment=disabled",
                epoch + 1, phase, row["train_loss"], row["val_objective_loss"],
                row["semantic_miou"],
            )
            continue
        logger.info(
            "epoch=%d phase=%s train=%.4f val=%s sem_miou=%.4f deploy=%.4f "
            "miou=%.4f area_err=%.4f pred=%d matches=%d",
            epoch + 1,
            phase,
            row["train_loss"],
            (
                f"{row['val_objective_loss']:.4f}"
                if selection_enabled
                else "disabled"
            ),
            row["semantic_miou"],
            row["deployment_score_total"],
            row["deployment_instance_miou"],
            row["deployment_ferrite_area_error"],
            row["deployment_pred_count"],
            row["deployment_valid_matches"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
