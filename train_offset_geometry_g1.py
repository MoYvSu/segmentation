# -*- coding: utf-8 -*-
"""G1: full labeled-data generalization training for center-offset geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import letterbox
from data.offset_geometry_augmentation import OffsetGeometryAugmentedDataset
from data.offset_geometry_dataset import OffsetGeometryDataset
from inference import build_model as build_reference_model
from models.offset_geometry import (
    CenterOffsetGeometryDecoder,
    FrozenSemanticGeometrySystem,
    semantic_state_digest,
)
from tools.audit_offset_geometry_checkpoint import audit_sample
from train_offset_geometry import file_sha256, semantic_contract_audit, set_seed
from utils.center_guided_instances import (
    assign_endpoints_to_centers,
    extract_center_peaks,
)
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair
from utils.offset_geometry_loss import center_offset_loss
from utils.offset_letterbox import geometry_letterbox_metadata


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_offset_geometry_g1")


def deterministic_split(names, val_fraction: float, seed: int, forced_train=()):
    names = sorted(set(names))
    forced = set(forced_train) & set(names)
    candidates = [name for name in names if name not in forced]
    candidates.sort(
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()
    )
    val_count = max(1, int(round(len(names) * float(val_fraction))))
    val_count = min(val_count, max(1, len(candidates) - 1))
    val_names = sorted(candidates[:val_count])
    train_names = sorted(set(names) - set(val_names))
    return train_names, val_names


def build_system(config, geometry_cfg, device):
    reference_path = project_path(config, geometry_cfg["reference_checkpoint"])
    reference_model = build_reference_model(config, device, reference_path)
    geometry_decoder = CenterOffsetGeometryDecoder(
        in_channels=reference_model.encoder.get_stage_channels(),
        fpn_channels=int(geometry_cfg.get("fpn_channels", 256)),
        up_channels=int(geometry_cfg.get("up_channels", 128)),
        output_grid=int(geometry_cfg.get("output_grid", 512)),
    ).to(device)
    init_path = project_path(config, geometry_cfg["geometry_init_checkpoint"])
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    geometry_decoder.load_state_dict(checkpoint["geometry_state_dict"], strict=True)
    system = FrozenSemanticGeometrySystem(reference_model, geometry_decoder).to(device)
    digest = semantic_state_digest(reference_model)
    expected_digest = checkpoint.get("semantic_state_digest")
    if expected_digest and digest != expected_digest:
        raise RuntimeError("G0 geometry checkpoint references a different V6 semantic state")
    return system, reference_path, init_path, digest


@torch.no_grad()
def evaluate_sample(system, sample, grid_size: int):
    device = next(system.geometry_decoder.parameters()).device
    prediction = system.geometry_forward(sample["image"].unsqueeze(0).to(device))
    center_probability = torch.sigmoid(
        prediction["center_logits"]
    )[0, 0].cpu().numpy()
    offsets = prediction["offsets"][0].cpu().numpy() * float(grid_size)
    foreground = sample["foreground"][0].numpy().astype(bool)
    valid = sample["valid_content"][0].numpy().astype(bool)
    gt_instances = sample["instance_map"].numpy().astype(np.int32)
    yy, xx = np.indices(foreground.shape, dtype=np.float32)
    centers, scores, _ = extract_center_peaks(
        center_probability, valid, threshold=0.25, nms_radius=3, max_centers=255
    )
    predicted, assignment = assign_endpoints_to_centers(
        yy + offsets[0], xx + offsets[1], foreground, centers,
        max_assignment_distance=None, min_instance_area=1,
    )
    gt_ids = [int(value) for value in np.unique(gt_instances) if int(value) != 0]
    pred_ids = [int(value) for value in np.unique(predicted) if int(value) != 0]
    metrics = evaluate_instance_pair(
        gt_instances, {value: 0 for value in gt_ids},
        predicted, {value: 0 for value in pred_ids},
    )
    target_offsets = sample["offset_target"].numpy() * float(grid_size)
    offset_mae = float(np.abs(offsets - target_offsets)[:, foreground].mean())
    return {
        "gt_instances": len(gt_ids),
        "pred_instances": len(pred_ids),
        "center_count_abs_error": abs(len(centers) - len(gt_ids)),
        "mean_center_score": float(scores.mean()) if scores.size else 0.0,
        "offset_mae_grid_pixels": offset_mae,
        "instance_miou_valid": float(metrics["instance_miou_valid"]),
        "gt_penalized_miou": float(metrics["gt_penalized_miou"]),
        "unassigned_foreground_pixels": assignment["unassigned_foreground_pixels"],
    }


@torch.no_grad()
def write_unlabeled_monitor(
    system, image_path: Path, output_dir: Path, epoch: int,
    input_size: int, grid_size: int,
):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_lb, _, _, _ = letterbox(image_rgb, input_size)
    tensor = torch.from_numpy(image_lb).permute(2, 0, 1).float().div(255.0)
    device = next(system.geometry_decoder.parameters()).device
    prediction = system.geometry_forward(tensor.unsqueeze(0).to(device))
    probability = torch.sigmoid(prediction["center_logits"])[0, 0].cpu().numpy()
    metadata = geometry_letterbox_metadata(image_rgb.shape[:2], input_size, grid_size)
    valid = np.zeros((grid_size, grid_size), dtype=bool)
    valid[:metadata.content_height, :metadata.content_width] = True
    centers, scores, audit = extract_center_peaks(
        probability, valid, threshold=0.25, nms_radius=3, max_centers=255
    )
    monitor_dir = output_dir / "unlabeled_monitor" / f"epoch_{epoch:03d}" / image_path.stem
    monitor_dir.mkdir(parents=True, exist_ok=True)
    content = probability[:metadata.content_height, :metadata.content_width]
    cv2.imwrite(
        str(monitor_dir / "center_pred_content.png"),
        np.clip(content * 255.0, 0, 255).astype(np.uint8),
    )
    overlay = cv2.resize(
        image_bgr, (metadata.content_width, metadata.content_height),
        interpolation=cv2.INTER_LINEAR,
    )
    cv2.imwrite(
        str(monitor_dir / "input_content.jpg"), overlay,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    for center_y, center_x in centers:
        cv2.circle(overlay, (int(center_x), int(center_y)), 2, (0, 0, 255), -1)
    cv2.imwrite(str(monitor_dir / "center_overlay.png"), overlay)
    (monitor_dir / "metrics.json").write_text(
        json.dumps({
            "image": image_path.name,
            "center_count": int(len(centers)),
            "mean_center_score": float(scores.mean()) if scores.size else 0.0,
            "peak_audit": audit,
            "note": "visual-only unlabeled monitor; never used for checkpoint selection",
        }, indent=2), encoding="utf-8",
    )


def save_checkpoint(
    path: Path, system, config, epoch: int, best_score: float,
    reference_path: str, reference_sha256: str, init_path: str,
    semantic_digest: str, split,
):
    torch.save({
        "format": "offset_geometry_g1_v1",
        "epoch": int(epoch),
        "best_val_gt_penalized_miou": float(best_score),
        "geometry_state_dict": system.geometry_decoder.state_dict(),
        "reference_checkpoint": os.path.abspath(reference_path),
        "reference_checkpoint_sha256": reference_sha256,
        "geometry_init_checkpoint": os.path.abspath(init_path),
        "semantic_state_digest": semantic_digest,
        "split": split,
        "config": config,
    }, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/offset_geometry_g1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["offset_geometry_g1"]
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["sam2"].get("device") == "cuda"
        else "cpu"
    )
    system, reference_path, init_path, semantic_digest = build_system(
        config, cfg, device
    )
    reference_sha256 = file_sha256(reference_path)
    logger.info(
        "Device=%s geometry_params=%.3fM init=%s",
        device, system.geometry_decoder.trainable_param_count() / 1e6, init_path,
    )
    raw_dir = project_path(config, config["paths"]["raw_data_dir"])
    index_dataset = OffsetGeometryDataset(raw_dir)
    all_names = [path.stem for path in index_dataset.samples]
    train_names, val_names = deterministic_split(
        all_names, float(cfg.get("val_fraction", 0.2)), seed,
        cfg.get("forced_train_names", ["train_001", "train_002"]),
    )
    split = {"train": train_names, "val": val_names, "seed": seed}
    logger.info("Split: train=%d val=%d val_names=%s", len(train_names), len(val_names), val_names)
    dataset_kwargs = {
        "image_size": int(cfg.get("input_size", 1024)),
        "output_grid": int(cfg.get("output_grid", 512)),
        "center_sigma_scale": float(cfg.get("center_sigma_scale", 0.12)),
        "center_min_sigma": float(cfg.get("center_min_sigma", 2.0)),
        "center_max_sigma": float(cfg.get("center_max_sigma", 8.0)),
        "cache_in_memory": True,
    }
    train_base = OffsetGeometryDataset(raw_dir, sample_names=train_names, **dataset_kwargs)
    train_dataset = OffsetGeometryAugmentedDataset(
        train_base, cfg.get("augmentation", {})
    )
    val_dataset = OffsetGeometryDataset(raw_dir, sample_names=val_names, **dataset_kwargs)
    loader = DataLoader(
        train_dataset, batch_size=int(cfg.get("batch_size", 2)), shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )
    unlabeled_dir = Path(project_path(config, cfg.get("unlabeled_dir", "data/unlabeled")))
    labeled_set = set(all_names)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    unlabeled_monitors = [
        path for path in sorted(unlabeled_dir.iterdir())
        if path.suffix.lower() in extensions and path.stem not in labeled_set
    ][: int(cfg.get("unlabeled_monitor_count", 2))]

    audit_image = val_dataset[0]["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        semantic_baseline = system.semantic_logits(audit_image).cpu()
    semantic_contract_audit(
        system, audit_image, semantic_baseline, semantic_digest,
        float(cfg.get("semantic_tolerance", 1e-6)),
    )
    output_dir = Path(project_path(config, cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    optimizer = torch.optim.AdamW(
        system.geometry_decoder.parameters(), lr=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(cfg.get("min_learning_rate", 1e-5))
    )
    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    metrics_file = open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics_file, fieldnames=[
        "epoch", "train_loss", "train_center_loss", "train_offset_loss",
        "val_gt_penalized_miou", "val_instance_miou_valid",
        "val_center_count_abs_error", "val_offset_mae_grid_pixels", "learning_rate",
    ])
    writer.writeheader()
    best_score = -1.0
    monitor_interval = int(cfg.get("monitor_interval", 5))
    grid_size = int(cfg.get("output_grid", 512))
    for monitor_path in unlabeled_monitors:
        write_unlabeled_monitor(
            system, monitor_path, output_dir, 0,
            int(cfg.get("input_size", 1024)), grid_size,
        )
    for index in range(min(2, len(val_dataset))):
        audit_sample(system, val_dataset[index], output_dir / "val_monitor" / "epoch_000", grid_size, [None])

    for epoch in range(1, epochs + 1):
        system.train()
        totals = {"loss": 0.0, "center": 0.0, "offset": 0.0, "batches": 0}
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            center_target = batch["center_target"].to(device, non_blocking=True)
            offset_target = batch["offset_target"].to(device, non_blocking=True)
            foreground = batch["foreground"].to(device, non_blocking=True)
            valid_content = batch["valid_content"].to(device, non_blocking=True)
            instance_map = batch["instance_map"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = system.geometry_forward(image)
                loss, loss_metrics = center_offset_loss(
                    prediction, center_target, offset_target, foreground, valid_content,
                    instance_map,
                    center_weight=float(cfg.get("center_weight", 1.0)),
                    offset_weight=float(cfg.get("offset_weight", 5.0)),
                    smooth_l1_beta=float(cfg.get("smooth_l1_beta", 0.005)),
                    offset_reduction="instance_balanced",
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                system.geometry_decoder.parameters(), float(cfg.get("grad_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss_metrics["loss"])
            totals["center"] += float(loss_metrics["center_loss"])
            totals["offset"] += float(loss_metrics["offset_loss"])
            totals["batches"] += 1
        scheduler.step()
        system.eval()
        val_rows = [evaluate_sample(system, val_dataset[index], grid_size) for index in range(len(val_dataset))]
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["batches"],
            "train_center_loss": totals["center"] / totals["batches"],
            "train_offset_loss": totals["offset"] / totals["batches"],
            "val_gt_penalized_miou": float(np.mean([v["gt_penalized_miou"] for v in val_rows])),
            "val_instance_miou_valid": float(np.mean([v["instance_miou_valid"] for v in val_rows])),
            "val_center_count_abs_error": float(np.mean([v["center_count_abs_error"] for v in val_rows])),
            "val_offset_mae_grid_pixels": float(np.mean([v["offset_mae_grid_pixels"] for v in val_rows])),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        writer.writerow(row)
        metrics_file.flush()
        logger.info(
            "epoch=%d train=%.4f center=%.4f offset=%.4f val_pen=%.4f "
            "val_valid=%.4f center_err=%.2f offset_mae=%.2fpx lr=%.2e",
            epoch, row["train_loss"], row["train_center_loss"], row["train_offset_loss"],
            row["val_gt_penalized_miou"], row["val_instance_miou_valid"],
            row["val_center_count_abs_error"], row["val_offset_mae_grid_pixels"],
            row["learning_rate"],
        )
        if row["val_gt_penalized_miou"] > best_score:
            best_score = row["val_gt_penalized_miou"]
            save_checkpoint(
                output_dir / "best_geometry.pth", system, config, epoch, best_score,
                reference_path, reference_sha256, init_path, semantic_digest, split,
            )
        if epoch % monitor_interval == 0 or epoch == epochs:
            semantic_contract_audit(
                system, audit_image, semantic_baseline, semantic_digest,
                float(cfg.get("semantic_tolerance", 1e-6)),
            )
            save_checkpoint(
                output_dir / "latest_geometry.pth", system, config, epoch, best_score,
                reference_path, reference_sha256, init_path, semantic_digest, split,
            )
            for monitor_path in unlabeled_monitors:
                write_unlabeled_monitor(
                    system, monitor_path, output_dir, epoch,
                    int(cfg.get("input_size", 1024)), grid_size,
                )
            for index in range(min(2, len(val_dataset))):
                audit_sample(
                    system, val_dataset[index],
                    output_dir / "val_monitor" / f"epoch_{epoch:03d}",
                    grid_size, [None],
                )
    metrics_file.close()
    logger.info("G1 complete: best_val_gt_penalized_miou=%.6f output=%s", best_score, output_dir)


if __name__ == "__main__":
    main()
