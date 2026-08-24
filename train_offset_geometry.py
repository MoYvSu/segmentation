# -*- coding: utf-8 -*-
"""G0: isolated two-image overfit audit for global center-offset geometry."""

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

from data.offset_geometry_dataset import OffsetGeometryDataset
from inference import build_model as build_reference_model
from models.offset_geometry import (
    CenterOffsetGeometryDecoder,
    FrozenSemanticGeometrySystem,
    semantic_state_digest,
)
from utils.config import load_config, project_path
from utils.flow_instances import cluster_endpoints
from utils.instance_metrics import evaluate_instance_pair
from utils.offset_geometry_loss import center_offset_loss


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_offset_geometry")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def colorize_instances(instance_map: np.ndarray) -> np.ndarray:
    output = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(instance_map):
        value = int(instance_id)
        if value == 0:
            continue
        output[instance_map == value] = (
            (37 * value + 53) % 255,
            (97 * value + 29) % 255,
            (17 * value + 193) % 255,
        )
    return output


@torch.no_grad()
def semantic_contract_audit(
    system: FrozenSemanticGeometrySystem,
    image: torch.Tensor,
    expected_logits: torch.Tensor,
    expected_digest: str,
    tolerance: float,
):
    actual_digest = semantic_state_digest(system.reference_model)
    actual_logits = system.semantic_logits(image).detach().cpu()
    max_abs_diff = float((actual_logits - expected_logits).abs().max())
    if actual_digest != expected_digest:
        raise RuntimeError("V6 semantic/LoRA tensor digest changed during G0")
    if max_abs_diff > float(tolerance):
        raise RuntimeError(
            f"V6 semantic logits changed: max_abs={max_abs_diff:.3e} "
            f"> tolerance={tolerance:.3e}"
        )
    return max_abs_diff


@torch.no_grad()
def write_monitor(system, batch, output_dir: Path, step: int, grid_size: int):
    system.eval()
    image = batch["image"].to(next(system.geometry_decoder.parameters()).device)
    prediction = system.geometry_forward(image)
    center_probability = torch.sigmoid(prediction["center_logits"])[0, 0].cpu().numpy()
    center_target = batch["center_target"][0, 0].numpy()
    offsets = prediction["offsets"][0].cpu().numpy() * float(grid_size)
    foreground = batch["foreground"][0, 0].numpy().astype(bool)
    yy, xx = np.indices(foreground.shape, dtype=np.float32)
    endpoint_y = np.clip(yy + offsets[0], 0, grid_size - 1)
    endpoint_x = np.clip(xx + offsets[1], 0, grid_size - 1)
    pred_instances, reconstruction_audit = cluster_endpoints(
        endpoint_y, endpoint_x, foreground,
        close_radius=1, min_instance_area=1, max_instances=255,
    )
    gt_instances = batch["instance_map"][0].numpy().astype(np.int32)
    gt_ids = [int(value) for value in np.unique(gt_instances) if int(value) != 0]
    pred_ids = [int(value) for value in np.unique(pred_instances) if int(value) != 0]
    geometry_metrics = evaluate_instance_pair(
        gt_instances, {value: 0 for value in gt_ids},
        pred_instances, {value: 0 for value in pred_ids},
    )
    pooled = cv2.dilate(center_probability, np.ones((7, 7), np.uint8))
    peak_count = int(np.sum((center_probability >= pooled - 1e-7) &
                            (center_probability >= 0.25)))
    monitor_dir = output_dir / "monitor" / f"step_{step:04d}"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(monitor_dir / "center_pred.png"),
        np.clip(center_probability * 255.0, 0, 255).astype(np.uint8),
    )
    cv2.imwrite(
        str(monitor_dir / "center_gt.png"),
        np.clip(center_target * 255.0, 0, 255).astype(np.uint8),
    )
    cv2.imwrite(
        str(monitor_dir / "instances_pred.png"), colorize_instances(pred_instances)
    )
    cv2.imwrite(
        str(monitor_dir / "instances_gt.png"), colorize_instances(gt_instances)
    )
    monitor_metrics = {
        "step": int(step),
        "image": batch["image_name"][0],
        "center_peak_count_at_025": peak_count,
        "gt_instance_count": len(gt_ids),
        "pred_instance_count": len(pred_ids),
        "instance_miou_valid": geometry_metrics["instance_miou_valid"],
        "gt_penalized_miou": geometry_metrics["gt_penalized_miou"],
        "reconstruction_audit": reconstruction_audit,
    }
    (monitor_dir / "metrics.json").write_text(
        json.dumps(monitor_metrics, indent=2), encoding="utf-8"
    )
    system.train()
    return monitor_metrics


def save_geometry_checkpoint(
    path: Path,
    system: FrozenSemanticGeometrySystem,
    config,
    step: int,
    reference_path: str,
    reference_sha256: str,
    semantic_digest: str,
    semantic_max_abs_diff: float,
):
    payload = {
        "format": "offset_geometry_g0_v1",
        "step": int(step),
        "geometry_state_dict": system.geometry_decoder.state_dict(),
        "reference_checkpoint": os.path.abspath(reference_path),
        "reference_checkpoint_sha256": reference_sha256,
        "semantic_state_digest": semantic_digest,
        "semantic_max_abs_diff": float(semantic_max_abs_diff),
        "config": config,
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description="G0 offset geometry overfit audit")
    parser.add_argument(
        "--config", default="config/train/offset_geometry_g0.yaml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    geometry_cfg = config["offset_geometry"]
    seed = int(geometry_cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["sam2"].get("device") == "cuda"
        else "cpu"
    )
    logger.info("Device: %s", device)

    reference_path = project_path(config, geometry_cfg["reference_checkpoint"])
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"V6 reference checkpoint missing: {reference_path}")
    reference_sha = file_sha256(reference_path)
    reference_model = build_reference_model(config, device, reference_path)
    geometry_decoder = CenterOffsetGeometryDecoder(
        in_channels=reference_model.encoder.get_stage_channels(),
        fpn_channels=int(geometry_cfg.get("fpn_channels", 256)),
        up_channels=int(geometry_cfg.get("up_channels", 128)),
        output_grid=int(geometry_cfg.get("output_grid", 512)),
    ).to(device)
    if geometry_cfg.get("init_from_v6_boundary_fpn", True):
        geometry_decoder.initialize_fpn_from_boundary(
            reference_model.decoder.boundary_fpn
        )
        logger.info("Geometry FPN initialized from V6 boundary FPN")
    else:
        logger.info("Geometry FPN uses random initialization")
    system = FrozenSemanticGeometrySystem(reference_model, geometry_decoder).to(device)
    logger.info(
        "Trainable geometry parameters: %.3fM",
        geometry_decoder.trainable_param_count() / 1e6,
    )

    dataset = OffsetGeometryDataset(
        project_path(config, config["paths"]["raw_data_dir"]),
        sample_names=geometry_cfg.get("sample_names"),
        image_size=int(geometry_cfg.get("input_size", 1024)),
        output_grid=int(geometry_cfg.get("output_grid", 512)),
        center_sigma_scale=float(geometry_cfg.get("center_sigma_scale", 0.12)),
        center_min_sigma=float(geometry_cfg.get("center_min_sigma", 2.0)),
        center_max_sigma=float(geometry_cfg.get("center_max_sigma", 8.0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(geometry_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(geometry_cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    audit_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    audit_batch = next(iter(audit_loader))
    audit_image = audit_batch["image"].to(device)
    semantic_digest = semantic_state_digest(reference_model)
    with torch.no_grad():
        semantic_baseline = system.semantic_logits(audit_image).detach().cpu()
    semantic_tolerance = float(geometry_cfg.get("semantic_tolerance", 1e-6))
    initial_semantic_diff = semantic_contract_audit(
        system, audit_image, semantic_baseline, semantic_digest, semantic_tolerance
    )
    logger.info(
        "V6 semantic contract passed: digest=%s max_abs=%.3e",
        semantic_digest[:16], initial_semantic_diff,
    )

    output_dir = Path(project_path(config, geometry_cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        geometry_decoder.parameters(),
        lr=float(geometry_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(geometry_cfg.get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(geometry_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    max_steps = int(geometry_cfg.get("max_steps", 400))
    log_interval = int(geometry_cfg.get("log_interval", 20))
    monitor_interval = int(geometry_cfg.get("monitor_interval", 100))
    checkpoint_interval = int(geometry_cfg.get("checkpoint_interval", 100))
    grid_size = int(geometry_cfg.get("output_grid", 512))
    center_weight = float(geometry_cfg.get("center_weight", 1.0))
    offset_weight = float(geometry_cfg.get("offset_weight", 5.0))
    smooth_l1_beta = float(geometry_cfg.get("smooth_l1_beta", 0.02))

    metrics_path = output_dir / "metrics.csv"
    metrics_file = open(metrics_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "step", "loss", "center_loss", "offset_loss",
            "center_peak_probability", "center_background_probability",
            "offset_mae_grid_pixels", "learning_rate",
        ],
    )
    writer.writeheader()
    best_loss = float("inf")
    iterator = iter(loader)
    system.train()
    write_monitor(system, audit_batch, output_dir, 0, grid_size)
    for step in range(1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        image = batch["image"].to(device, non_blocking=True)
        center_target = batch["center_target"].to(device, non_blocking=True)
        offset_target = batch["offset_target"].to(device, non_blocking=True)
        foreground = batch["foreground"].to(device, non_blocking=True)
        valid_content = batch["valid_content"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            prediction = system.geometry_forward(image)
            loss, loss_metrics = center_offset_loss(
                prediction, center_target, offset_target, foreground, valid_content,
                center_weight=center_weight, offset_weight=offset_weight,
                smooth_l1_beta=smooth_l1_beta,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            geometry_decoder.parameters(),
            float(geometry_cfg.get("grad_clip", 1.0)),
        )
        scaler.step(optimizer)
        scaler.update()

        row = {
            "step": step,
            "loss": float(loss_metrics["loss"]),
            "center_loss": float(loss_metrics["center_loss"]),
            "offset_loss": float(loss_metrics["offset_loss"]),
            "center_peak_probability": float(loss_metrics["center_peak_probability"]),
            "center_background_probability": float(
                loss_metrics["center_background_probability"]
            ),
            "offset_mae_grid_pixels": float(
                loss_metrics["offset_mae_normalized"] * grid_size
            ),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        writer.writerow(row)
        if step % log_interval == 0 or step == 1:
            metrics_file.flush()
            logger.info(
                "step=%d loss=%.5f center=%.5f offset=%.5f "
                "peak=%.3f bg=%.3f offset_mae=%.2fpx",
                step, row["loss"], row["center_loss"], row["offset_loss"],
                row["center_peak_probability"],
                row["center_background_probability"],
                row["offset_mae_grid_pixels"],
            )
        if step % monitor_interval == 0 or step == max_steps:
            monitor = write_monitor(system, audit_batch, output_dir, step, grid_size)
            logger.info("monitor step=%d %s", step, monitor)
        if step % checkpoint_interval == 0 or step == max_steps:
            semantic_diff = semantic_contract_audit(
                system, audit_image, semantic_baseline,
                semantic_digest, semantic_tolerance,
            )
            checkpoint_path = output_dir / f"geometry_step_{step:04d}.pth"
            save_geometry_checkpoint(
                checkpoint_path, system, config, step,
                reference_path, reference_sha, semantic_digest, semantic_diff,
            )
        if row["loss"] < best_loss:
            best_loss = row["loss"]
        if step % monitor_interval == 0 or step == max_steps:
            semantic_diff = semantic_contract_audit(
                system, audit_image, semantic_baseline,
                semantic_digest, semantic_tolerance,
            )
            save_geometry_checkpoint(
                output_dir / "latest_geometry.pth", system, config, step,
                reference_path, reference_sha, semantic_digest, semantic_diff,
            )
    metrics_file.close()
    final_diff = semantic_contract_audit(
        system, audit_image, semantic_baseline, semantic_digest, semantic_tolerance
    )
    logger.info(
        "G0 complete: best_loss=%.6f semantic_max_abs=%.3e output=%s",
        best_loss, final_diff, output_dir,
    )


if __name__ == "__main__":
    main()
