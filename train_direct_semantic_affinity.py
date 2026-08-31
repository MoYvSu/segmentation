# -*- coding: utf-8 -*-
"""Train semantic and class-agnostic affinity heads directly from SSL LoRA."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.affinity_geometry_augmentation import AffinityGeometryAugmentedDataset
from data.direct_dual_head_dataset import DirectDualHeadDataset
from data.sam2_geometry_dataset import SAM2GeometryDataset
from models.direct_semantic_affinity import (
    DIRECT_SEMANTIC_AFFINITY_FORMAT,
    build_direct_semantic_affinity_model,
    configure_direct_training_phase,
    direct_parameter_groups,
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
    load_labelme_instances,
    summarize_instance_results,
)
from utils.loss import BoundaryLoss
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


def check_sources(config: dict):
    missing = []
    paths_cfg = config["paths"]
    for path in (
        project_path(config, paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"]),
        project_path(config, config["sam2"]["sam2_repo_path"]),
        project_path(config, paths_cfg["raw_data_dir"]),
        project_path(config, config["boundary"]["gt_dir"]),
        project_path(config, config["lora"]["init_from"]),
    ):
        if not Path(path).exists():
            missing.append(str(path))
    direct_cfg = config["direct_semantic_affinity"]
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
    if missing:
        raise FileNotFoundError("missing direct dual-head inputs: " + ", ".join(missing))
    return {"sam2_geometry": approval}


def build_loaders(config: dict, device):
    cfg = config["direct_semantic_affinity"]
    raw_dir = Path(project_path(config, config["paths"]["raw_data_dir"]))
    gt_dir = Path(project_path(config, config["boundary"]["gt_dir"]))
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    names = [
        path.stem
        for path in sorted(raw_dir.iterdir())
        if path.suffix.lower() in extensions and path.with_suffix(".json").is_file()
    ]
    train_names, val_names = deterministic_split(
        names,
        float(cfg.get("val_fraction", 0.20)),
        int(cfg.get("seed", 42)),
        cfg.get("forced_train_names", ("train_001", "train_002")),
    )
    common = {
        "data_dir": raw_dir,
        "gt_dir": gt_dir,
        "image_size": int(cfg.get("input_size", 1024)),
        "affinity_grid": int(cfg.get("affinity_grid", 512)),
    }
    train_dataset = DirectDualHeadDataset(
        sample_names=train_names,
        augment=True,
        augmentation=cfg.get("augmentation", {}),
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
    model.eval()
    rows = []
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
            gt_map, gt_classes, _ = load_labelme_instances(
                Path(image_path).with_suffix(".json"), image.shape[:2]
            )
            rows.append(evaluate_instance_pair(gt_map, gt_classes, pred_map, pred_classes))
    return summarize_instance_results(rows)


def save_checkpoint(
    path, model, config, epoch, phase, best_score, scales, split, optimizer, scheduler
):
    payload = {
        "format": DIRECT_SEMANTIC_AFFINITY_FORMAT,
        "epoch": int(epoch) + 1,
        "epoch_index": int(epoch),
        "phase": str(phase),
        "best_deployment_score": float(best_score),
        "semantic_state_dict": model.semantic_decoder.state_dict(),
        "affinity_state_dict": model.affinity_decoder.state_dict(),
        "lora_state_dict": extract_lora_state_dict(model.encoder),
        "loss_scales": dict(scales),
        "split": split,
        "config": config,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(model, path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != DIRECT_SEMANTIC_AFFINITY_FORMAT:
        raise RuntimeError(f"unsupported direct checkpoint: {payload.get('format')!r}")
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
    best_score = float(resume_payload.get("best_deployment_score", -math.inf)) if resume_payload else -math.inf
    scales = (
        dict(resume_payload["loss_scales"])
        if resume_payload
        else calibrate_loss_scales(
            model,
            manual_loader,
            pseudo_loader,
            criterion,
            loss_cfg,
            device,
            int(cfg.get("loss_calibration_batches", 4)),
        )
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
    recorder.save_manifest()

    optimizer = None
    scheduler = None
    active_phase = None
    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    val_paths = [Path(path) for path, _ in val_dataset.samples]
    pseudo_weight = float(cfg.get("pseudo_affinity_weight", 0.50))
    pseudo_start_epoch = int(cfg.get("pseudo_start_epoch", warmup_epochs))

    for epoch in range(start_epoch, total_epochs):
        phase = phase_for_epoch(epoch, warmup_epochs)
        if phase != active_phase:
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

        semantic_metrics = validate_semantic(model, val_loader, device)
        deployment = evaluate_direct_deployment(model, val_paths, config, device)
        row = {
            "epoch": epoch + 1,
            "phase": phase,
            "train_loss": totals["loss"] / max(1, totals["steps"]),
            "train_semantic_loss": totals["semantic"] / max(1, totals["steps"]),
            "train_manual_affinity_loss": totals["manual_affinity"] / max(1, totals["steps"]),
            "train_pseudo_affinity_loss": totals["pseudo_affinity"] / max(1, totals["steps"]),
            **semantic_metrics,
            "deployment_score_total": deployment["score_total"],
            "deployment_instance_miou": deployment["instance_miou_valid"],
            "deployment_ferrite_area_error": deployment["ferrite_area_relative_error"],
            "deployment_macro_score": deployment["macro_image_score_total"],
            "deployment_pred_count": deployment["pred_count"],
            "deployment_valid_matches": deployment["valid_matches"],
            **{
                f"lr_{group.get('name', index)}": group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
        }
        recorder.append_metrics(row)
        latest_path = output_dir / "latest_direct_dual.pth"
        save_checkpoint(
            latest_path,
            model,
            config,
            epoch,
            phase,
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
                phase,
                best_score,
                scales,
                split,
                optimizer,
                scheduler,
            )
            recorder.copy_checkpoint(best_path, "best_direct_dual.pth")
        logger.info(
            "epoch=%d phase=%s train=%.4f sem_miou=%.4f deploy=%.4f "
            "miou=%.4f area_err=%.4f pred=%d matches=%d",
            epoch + 1,
            phase,
            row["train_loss"],
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
