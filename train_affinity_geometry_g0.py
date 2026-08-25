# -*- coding: utf-8 -*-
"""G0: two-image overfit audit for local instance affinities."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
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
from models.affinity_geometry import AffinityGeometryDecoder
from models.offset_geometry import FrozenSemanticGeometrySystem, semantic_state_digest
from train_offset_geometry import (
    colorize_instances,
    file_sha256,
    semantic_contract_audit,
    set_seed,
)
from utils.affinity_graph import (
    DEFAULT_AFFINITY_OFFSETS,
    audit_instance_recovery,
    build_affinity_targets,
    reconstruct_affinity_components,
)
from utils.affinity_loss import balanced_affinity_loss, build_affinity_targets_torch
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_affinity_geometry_g0")


def edge_metrics(probability, target, valid):
    predicted = probability >= 0.5
    positive = valid & (target > 0.5)
    negative = valid & ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    tn = int(np.sum(~predicted & negative))
    fn = int(np.sum(~predicted & positive))
    return {
        "edge_precision": tp / max(1, tp + fp),
        "edge_recall": tp / max(1, tp + fn),
        "edge_specificity": tn / max(1, tn + fp),
        "positive_probability": float(probability[positive].mean()) if np.any(positive) else 0.0,
        "negative_probability": float(probability[negative].mean()) if np.any(negative) else 0.0,
    }


@torch.no_grad()
def write_monitor(
    system, dataset, output_dir: Path, step: int, graph_thresholds=0.5,
):
    system.eval()
    device = next(system.geometry_decoder.parameters()).device
    rows = []
    monitor_root = output_dir / "monitor" / f"step_{step:04d}"
    for sample in dataset:
        prediction = system.geometry_forward(sample["image"].unsqueeze(0).to(device))
        probability = torch.sigmoid(prediction["affinity_logits"])[0].cpu().numpy()
        labels = sample["instance_map"].numpy().astype(np.int32)
        target, valid = build_affinity_targets(
            labels, sample["valid_content"][0].numpy().astype(bool)
        )
        instances, graph_audit = reconstruct_affinity_components(
            labels > 0, probability, threshold=graph_thresholds,
            max_instances=255,
        )
        recovery = audit_instance_recovery(labels, instances)
        gt_ids = [int(value) for value in np.unique(labels) if int(value) != 0]
        pred_ids = [int(value) for value in np.unique(instances) if int(value) != 0]
        instance_metrics = evaluate_instance_pair(
            labels, {value: 0 for value in gt_ids},
            instances, {value: 0 for value in pred_ids},
        )
        metrics = {
            "step": int(step),
            "image": sample["image_name"],
            **edge_metrics(probability, target, valid),
            **graph_audit,
            **recovery,
            "instance_miou_valid": float(instance_metrics["instance_miou_valid"]),
            "gt_penalized_miou": float(instance_metrics["gt_penalized_miou"]),
        }
        rows.append(metrics)
        sample_dir = monitor_root / Path(sample["image_name"]).stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / "instances_gt.png"), colorize_instances(labels))
        cv2.imwrite(str(sample_dir / "instances_pred.png"), colorize_instances(instances))
        cv2.imwrite(
            str(sample_dir / "affinity_short_pred.png"),
            np.clip(np.mean(probability[:4], axis=0) * 255.0, 0, 255).astype(np.uint8),
        )
        cv2.imwrite(
            str(sample_dir / "affinity_short_gt.png"),
            np.clip(np.mean(target[:4], axis=0) * 255.0, 0, 255).astype(np.uint8),
        )
        (sample_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = {
        "step": int(step),
        "graph_thresholds": graph_thresholds,
        "mean_edge_precision": float(np.mean([row["edge_precision"] for row in rows])),
        "mean_edge_recall": float(np.mean([row["edge_recall"] for row in rows])),
        "mean_edge_specificity": float(np.mean([row["edge_specificity"] for row in rows])),
        "mean_exact_gt_fraction": float(np.mean([row["exact_gt_instance_fraction"] for row in rows])),
        "mean_gt_penalized_miou": float(np.mean([row["gt_penalized_miou"] for row in rows])),
        "all_exact": bool(all(row["exact_partition"] for row in rows)),
        "samples": rows,
    }
    (monitor_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    system.train()
    return summary


def save_checkpoint(path, system, config, step, reference_path, reference_sha, digest):
    torch.save({
        "format": "affinity_geometry_g0_v1",
        "step": int(step),
        "affinity_offsets": [list(value) for value in DEFAULT_AFFINITY_OFFSETS],
        "geometry_state_dict": system.geometry_decoder.state_dict(),
        "reference_checkpoint": os.path.abspath(reference_path),
        "reference_checkpoint_sha256": reference_sha,
        "semantic_state_digest": digest,
        "config": config,
    }, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/affinity_geometry_g0.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["affinity_geometry"]
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["sam2"].get("device") == "cuda"
        else "cpu"
    )
    reference_path = project_path(config, cfg["reference_checkpoint"])
    reference_sha = file_sha256(reference_path)
    reference_model = build_reference_model(config, device, reference_path)
    decoder = AffinityGeometryDecoder(
        in_channels=reference_model.encoder.get_stage_channels(),
        affinity_channels=len(DEFAULT_AFFINITY_OFFSETS),
        fpn_channels=int(cfg.get("fpn_channels", 256)),
        up_channels=int(cfg.get("up_channels", 128)),
        output_grid=int(cfg.get("output_grid", 512)),
    ).to(device)
    if cfg.get("init_from_v6_boundary_fpn", True):
        decoder.initialize_fpn_from_boundary(reference_model.decoder.boundary_fpn)
        logger.info("Affinity FPN initialized from V6 boundary FPN")
    init_checkpoint = cfg.get("geometry_init_checkpoint")
    if init_checkpoint:
        init_path = project_path(config, init_checkpoint)
        init_payload = torch.load(
            init_path, map_location="cpu", weights_only=False
        )
        decoder.load_state_dict(
            init_payload["geometry_state_dict"], strict=True
        )
        logger.info("Affinity decoder initialized from %s", init_path)
    system = FrozenSemanticGeometrySystem(reference_model, decoder).to(device)
    logger.info(
        "Device=%s trainable_affinity_params=%.3fM",
        device, decoder.trainable_param_count() / 1e6,
    )
    dataset = OffsetGeometryDataset(
        project_path(config, config["paths"]["raw_data_dir"]),
        sample_names=cfg.get("sample_names", ["train_001", "train_002"]),
        image_size=int(cfg.get("input_size", 1024)),
        output_grid=int(cfg.get("output_grid", 512)),
        cache_in_memory=bool(cfg.get("cache_in_memory", True)),
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg.get("batch_size", 1)), shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )
    audit_image = dataset[0]["image"].unsqueeze(0).to(device)
    digest = semantic_state_digest(reference_model)
    with torch.no_grad():
        semantic_baseline = system.semantic_logits(audit_image).cpu()
    semantic_contract_audit(
        system, audit_image, semantic_baseline, digest,
        float(cfg.get("semantic_tolerance", 1e-6)),
    )
    output_dir = Path(project_path(config, cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    metrics_file = open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics_file, fieldnames=[
        "step", "loss", "edge_precision", "edge_recall", "edge_specificity",
        "positive_probability", "negative_probability",
    ])
    writer.writeheader()
    graph_thresholds = cfg.get("graph_thresholds", 0.5)
    write_monitor(system, dataset, output_dir, 0, graph_thresholds)
    max_steps = int(cfg.get("max_steps", 400))
    step = 0
    while step < max_steps:
        for batch in loader:
            step += 1
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["instance_map"].to(device, non_blocking=True)
            valid_content = batch["valid_content"].to(device, non_blocking=True)
            target, edge_valid = build_affinity_targets_torch(
                labels, valid_content
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = system.geometry_forward(image)
                loss, metrics = balanced_affinity_loss(
                    prediction["affinity_logits"], target, edge_valid
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), float(cfg.get("grad_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            row = {
                "step": step,
                "loss": float(metrics["loss"]),
                "edge_precision": float(metrics["precision"]),
                "edge_recall": float(metrics["recall"]),
                "edge_specificity": float(metrics["specificity"]),
                "positive_probability": float(metrics["positive_probability"]),
                "negative_probability": float(metrics["negative_probability"]),
            }
            writer.writerow(row)
            metrics_file.flush()
            if step % int(cfg.get("log_interval", 20)) == 0:
                logger.info(
                    "step=%d loss=%.4f precision=%.4f recall=%.4f specificity=%.4f pos=%.3f neg=%.3f",
                    step, row["loss"], row["edge_precision"], row["edge_recall"],
                    row["edge_specificity"], row["positive_probability"],
                    row["negative_probability"],
                )
            if step % int(cfg.get("monitor_interval", 100)) == 0 or step == max_steps:
                semantic_contract_audit(
                    system, audit_image, semantic_baseline, digest,
                    float(cfg.get("semantic_tolerance", 1e-6)),
                )
                summary = write_monitor(
                    system, dataset, output_dir, step, graph_thresholds
                )
                logger.info(
                    "monitor step=%d exact_fraction=%.4f penalized_miou=%.4f all_exact=%s",
                    step, summary["mean_exact_gt_fraction"],
                    summary["mean_gt_penalized_miou"], summary["all_exact"],
                )
            if step % int(cfg.get("checkpoint_interval", 200)) == 0 or step == max_steps:
                save_checkpoint(
                    output_dir / f"affinity_step_{step:04d}.pth",
                    system, config, step, reference_path, reference_sha, digest,
                )
                save_checkpoint(
                    output_dir / "latest_affinity.pth",
                    system, config, step, reference_path, reference_sha, digest,
                )
            if step >= max_steps:
                break
    metrics_file.close()
    logger.info("Affinity G0 complete: output=%s", output_dir)


if __name__ == "__main__":
    main()
