# -*- coding: utf-8 -*-
"""G1: full labeled-data generalization training for local affinities."""

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
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.affinity_geometry_augmentation import AffinityGeometryAugmentedDataset
from data.dataset import letterbox
from data.offset_geometry_dataset import OffsetGeometryDataset
from data.sam2_geometry_dataset import SAM2GeometryDataset
from inference import build_model as build_reference_model
from models.affinity_geometry import AffinityGeometryDecoder
from models.offset_geometry import FrozenSemanticGeometrySystem, semantic_state_digest
from train_affinity_geometry_g0 import edge_metrics
from train_offset_geometry import (
    colorize_instances,
    file_sha256,
    semantic_contract_audit,
    set_seed,
)
from train_offset_geometry_g1 import deterministic_split
from utils.affinity_graph import (
    DEFAULT_AFFINITY_OFFSETS,
    audit_instance_recovery,
    build_affinity_targets,
    reconstruct_affinity_components,
)
from utils.affinity_loss import balanced_affinity_loss, build_affinity_targets_torch
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair
from utils.offset_letterbox import geometry_letterbox_metadata


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_affinity_geometry_g1")


def build_system(config, cfg, device):
    reference_path = project_path(config, cfg["reference_checkpoint"])
    reference = build_reference_model(config, device, reference_path)
    decoder = AffinityGeometryDecoder(
        in_channels=reference.encoder.get_stage_channels(),
        affinity_channels=len(DEFAULT_AFFINITY_OFFSETS),
        fpn_channels=int(cfg.get("fpn_channels", 256)),
        up_channels=int(cfg.get("up_channels", 128)),
        output_grid=int(cfg.get("output_grid", 512)),
    ).to(device)
    init_path = project_path(config, cfg["geometry_init_checkpoint"])
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    decoder.load_state_dict(checkpoint["geometry_state_dict"], strict=True)
    system = FrozenSemanticGeometrySystem(reference, decoder).to(device)
    digest = semantic_state_digest(reference)
    expected = checkpoint.get("semantic_state_digest")
    if expected and expected != digest:
        raise RuntimeError("Affinity checkpoint references a different V6 semantic state")
    return system, reference_path, init_path, digest


@torch.no_grad()
def evaluate_sample(system, sample, graph_thresholds, return_arrays=False):
    device = next(system.geometry_decoder.parameters()).device
    output = system.geometry_forward(sample["image"].unsqueeze(0).to(device))
    probability = torch.sigmoid(output["affinity_logits"])[0].cpu().numpy()
    labels = sample["instance_map"].numpy().astype(np.int32)
    target, valid = build_affinity_targets(
        labels, sample["valid_content"][0].numpy().astype(bool)
    )
    prediction, graph = reconstruct_affinity_components(
        labels > 0, probability, threshold=graph_thresholds, max_instances=255
    )
    recovery = audit_instance_recovery(labels, prediction)
    gt_ids = [int(value) for value in np.unique(labels) if int(value) != 0]
    pred_ids = [int(value) for value in np.unique(prediction) if int(value) != 0]
    instance_metrics = evaluate_instance_pair(
        labels, {value: 0 for value in gt_ids},
        prediction, {value: 0 for value in pred_ids},
    )
    metrics = {
        **edge_metrics(probability, target, valid),
        **graph,
        **recovery,
        "instance_count_abs_error": abs(len(gt_ids) - len(pred_ids)),
        "instance_miou_valid": float(instance_metrics["instance_miou_valid"]),
        "gt_penalized_miou": float(instance_metrics["gt_penalized_miou"]),
    }
    if return_arrays:
        return metrics, probability, labels, prediction
    return metrics


@torch.no_grad()
def write_val_monitor(system, dataset, output_dir, epoch, graph_thresholds):
    monitor_dir = output_dir / "val_monitor" / f"epoch_{epoch:03d}"
    for index in range(min(2, len(dataset))):
        sample = dataset[index]
        metrics, probability, labels, prediction = evaluate_sample(
            system, sample, graph_thresholds, return_arrays=True
        )
        sample_dir = monitor_dir / Path(sample["image_name"]).stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / "instances_gt.png"), colorize_instances(labels))
        cv2.imwrite(str(sample_dir / "instances_pred.png"), colorize_instances(prediction))
        for name, values in (
            ("short", probability[:4]),
            ("distance2", probability[4:6]),
            ("distance4", probability[6:8]),
        ):
            cv2.imwrite(
                str(sample_dir / f"affinity_{name}.png"),
                np.clip(np.mean(values, axis=0) * 255.0, 0, 255).astype(np.uint8),
            )
        (sample_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )


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
    output = system.geometry_forward(tensor.unsqueeze(0).to(device))
    probability = torch.sigmoid(output["affinity_logits"])[0].cpu().numpy()
    metadata = geometry_letterbox_metadata(image_rgb.shape[:2], input_size, grid_size)
    sample_dir = (
        output_dir / "unlabeled_monitor" / f"epoch_{epoch:03d}" / image_path.stem
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"image": image_path.name, "note": "visual-only; never selects checkpoints"}
    for name, values in (
        ("short", probability[:4]),
        ("distance2", probability[4:6]),
        ("distance4", probability[6:8]),
    ):
        content = np.mean(values, axis=0)[
            : metadata.content_height, : metadata.content_width
        ]
        cv2.imwrite(
            str(sample_dir / f"affinity_{name}.png"),
            np.clip(content * 255.0, 0, 255).astype(np.uint8),
        )
        cv2.imwrite(
            str(sample_dir / f"cut_{name}.png"),
            np.clip((1.0 - content) * 255.0, 0, 255).astype(np.uint8),
        )
        metrics[f"{name}_mean"] = float(content.mean())
    (sample_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def aggregate_validation(rows):
    keys = [
        "gt_penalized_miou", "instance_miou_valid", "instance_count_abs_error",
        "split_gt_instance_count", "merged_pred_instance_count",
        "raw_component_count", "merged_components_for_cap",
        "edge_precision", "edge_recall", "edge_specificity",
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def save_checkpoint(
    path, system, config, epoch, best_score, reference_path, reference_sha,
    init_path, digest, split,
):
    torch.save({
        "format": "affinity_geometry_g1_v1",
        "epoch": int(epoch),
        "best_val_gt_penalized_miou": float(best_score),
        "affinity_offsets": [list(value) for value in DEFAULT_AFFINITY_OFFSETS],
        "geometry_state_dict": system.geometry_decoder.state_dict(),
        "reference_checkpoint": os.path.abspath(reference_path),
        "reference_checkpoint_sha256": reference_sha,
        "geometry_init_checkpoint": os.path.abspath(init_path),
        "semantic_state_digest": digest,
        "split": split,
        "config": config,
    }, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/affinity_geometry_g1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["affinity_geometry_g1"]
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["sam2"].get("device") == "cuda"
        else "cpu"
    )
    system, reference_path, init_path, digest = build_system(config, cfg, device)
    reference_sha = file_sha256(reference_path)
    raw_dir = project_path(config, config["paths"]["raw_data_dir"])
    index_dataset = OffsetGeometryDataset(raw_dir)
    all_names = [path.stem for path in index_dataset.samples]
    train_names, val_names = deterministic_split(
        all_names, float(cfg.get("val_fraction", 0.2)), seed,
        cfg.get("forced_train_names", ["train_001", "train_002"]),
    )
    split = {"train": train_names, "val": val_names, "seed": seed}
    logger.info("Split train=%d val=%d val_names=%s", len(train_names), len(val_names), val_names)
    dataset_kwargs = {
        "image_size": int(cfg.get("input_size", 1024)),
        "output_grid": int(cfg.get("output_grid", 512)),
        "cache_in_memory": True,
    }
    manual_train_base = OffsetGeometryDataset(
        raw_dir, sample_names=train_names, **dataset_kwargs
    )
    augmentation_cfg = cfg.get("augmentation", {})
    dataset_parts = [manual_train_base]
    augmentation_parts = [augmentation_cfg]
    sampling_masses = [1.0]
    sampler = None
    sam2_geometry_cfg = cfg.get("sam2_geometry", {})
    if bool(sam2_geometry_cfg.get("enabled", False)):
        pseudo_dataset_dir = project_path(config, sam2_geometry_cfg["dataset_dir"])
        pseudo_source_dir = project_path(
            config,
            sam2_geometry_cfg.get(
                "source_dir", cfg.get("unlabeled_dir", "data/unlabeled")
            ),
        )
        pseudo_common = {
            "image_size": dataset_kwargs["image_size"],
            "output_grid": dataset_kwargs["output_grid"],
            "use_eroded_interiors": bool(
                sam2_geometry_cfg.get("use_eroded_interiors", False)
            ),
        }
        pseudo_base = SAM2GeometryDataset(
            pseudo_dataset_dir,
            pseudo_source_dir,
            **pseudo_common,
            cache_in_memory=True,
        )
        pseudo_fraction = float(sam2_geometry_cfg.get("pseudo_fraction", 0.5))
        if not 0.0 < pseudo_fraction < 1.0:
            raise ValueError("sam2_geometry.pseudo_fraction must be within (0, 1)")
        dataset_parts.append(pseudo_base)
        augmentation_parts.append(augmentation_cfg)

        native_crop_cfg = sam2_geometry_cfg.get("native_crop", {})
        crop_base = None
        crop_fraction = 0.0
        if bool(native_crop_cfg.get("enabled", False)):
            crop_fraction = float(
                native_crop_cfg.get("fraction_within_pseudo", 0.5)
            )
            if not 0.0 < crop_fraction < 1.0:
                raise ValueError(
                    "sam2_geometry.native_crop.fraction_within_pseudo must be "
                    "within (0, 1)"
                )
            crop_base = SAM2GeometryDataset(
                pseudo_dataset_dir,
                pseudo_source_dir,
                **pseudo_common,
                cache_in_memory=False,
                native_crop_size=int(native_crop_cfg.get("size", 1024)),
                native_crop_min_coverage=float(
                    native_crop_cfg.get("min_coverage", 0.92)
                ),
                native_crop_min_instances=int(
                    native_crop_cfg.get("min_instances", 12)
                ),
                native_crop_min_negative_edge_pixels=int(
                    native_crop_cfg.get("min_negative_edge_pixels", 128)
                ),
                native_crop_stride_fraction=float(
                    native_crop_cfg.get("stride_fraction", 0.5)
                ),
                native_crop_max_candidates=int(
                    native_crop_cfg.get("max_candidates_per_image", 8)
                ),
            )
            crop_augmentation = dict(augmentation_cfg)
            crop_augmentation["crop_probability"] = 0.0
            dataset_parts.append(crop_base)
            augmentation_parts.append(crop_augmentation)

        sampling_masses = [
            1.0 - pseudo_fraction,
            pseudo_fraction * (1.0 - crop_fraction),
        ]
        if crop_base is not None:
            sampling_masses.append(pseudo_fraction * crop_fraction)
        weights = torch.cat(
            [
                torch.full((len(part),), mass / len(part))
                for part, mass in zip(dataset_parts, sampling_masses)
            ]
        ).double()
        epoch_samples = int(
            sam2_geometry_cfg.get("samples_per_epoch", 2 * len(manual_train_base))
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=epoch_samples,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        split["sam2_geometry"] = {
            "count": len(pseudo_base),
            "pseudo_fraction": pseudo_fraction,
            "samples_per_epoch": epoch_samples,
            "dataset_dir": os.path.abspath(
                project_path(config, sam2_geometry_cfg["dataset_dir"])
            ),
        }
        if crop_base is not None:
            split["sam2_geometry"]["native_crop"] = {
                **native_crop_cfg,
                "eligible_source_count": len(crop_base),
            }
        logger.info(
            "SAM2 geometry mix enabled: manual=%d pseudo_full=%d "
            "pseudo_crop=%d masses=%s samples_per_epoch=%d",
            len(manual_train_base),
            len(pseudo_base),
            len(crop_base) if crop_base is not None else 0,
            [round(value, 3) for value in sampling_masses],
            epoch_samples,
        )
    train_parts = [
        AffinityGeometryAugmentedDataset(dataset, augmentation)
        for dataset, augmentation in zip(dataset_parts, augmentation_parts)
    ]
    train_dataset = (
        train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    )
    val_dataset = OffsetGeometryDataset(raw_dir, sample_names=val_names, **dataset_kwargs)
    loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
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
        system, audit_image, semantic_baseline, digest,
        float(cfg.get("semantic_tolerance", 1e-6)),
    )
    output_dir = Path(project_path(config, cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    graph_thresholds = cfg.get("graph_thresholds")
    system.eval()
    baseline_rows = [
        evaluate_sample(system, val_dataset[index], graph_thresholds)
        for index in range(len(val_dataset))
    ]
    baseline = aggregate_validation(baseline_rows)
    (output_dir / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best_score = baseline["gt_penalized_miou"]
    save_checkpoint(
        output_dir / "best_affinity.pth", system, config, 0, best_score,
        reference_path, reference_sha, init_path, digest, split,
    )
    write_val_monitor(system, val_dataset, output_dir, 0, graph_thresholds)
    for path in unlabeled_monitors:
        write_unlabeled_monitor(
            system, path, output_dir, 0,
            int(cfg.get("input_size", 1024)), int(cfg.get("output_grid", 512)),
        )
    optimizer = torch.optim.AdamW(
        system.geometry_decoder.parameters(), lr=float(cfg.get("learning_rate", 5e-5)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(cfg.get("min_learning_rate", 1e-5))
    )
    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    metrics_file = open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    fields = [
        "epoch", "train_loss", "train_edge_precision", "train_edge_recall",
        "val_gt_penalized_miou", "val_instance_miou_valid",
        "val_instance_count_abs_error", "val_split_gt_instance_count",
        "val_merged_pred_instance_count", "val_raw_component_count",
        "val_merged_components_for_cap", "val_edge_precision", "val_edge_recall",
        "val_edge_specificity", "learning_rate",
    ]
    writer = csv.DictWriter(metrics_file, fieldnames=fields)
    writer.writeheader()
    logger.info("Baseline val=%s", baseline)
    for epoch in range(1, epochs + 1):
        system.train()
        totals = {"loss": 0.0, "precision": 0.0, "recall": 0.0, "batches": 0}
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["instance_map"].to(device, non_blocking=True)
            valid = batch["valid_content"].to(device, non_blocking=True)
            target, edge_valid = build_affinity_targets_torch(labels, valid)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = system.geometry_forward(image)
                loss_cfg = cfg.get("loss", {})
                loss, train_metrics = balanced_affinity_loss(
                    output["affinity_logits"],
                    target,
                    edge_valid,
                    negative_weight=float(loss_cfg.get("negative_weight", 1.0)),
                    hard_negative_weight=float(
                        loss_cfg.get("hard_negative_weight", 0.0)
                    ),
                    hard_negative_gamma=float(
                        loss_cfg.get("hard_negative_gamma", 2.0)
                    ),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                system.geometry_decoder.parameters(), float(cfg.get("grad_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(train_metrics["loss"])
            totals["precision"] += float(train_metrics["precision"])
            totals["recall"] += float(train_metrics["recall"])
            totals["batches"] += 1
        scheduler.step()
        system.eval()
        val_rows = [
            evaluate_sample(system, val_dataset[index], graph_thresholds)
            for index in range(len(val_dataset))
        ]
        val = aggregate_validation(val_rows)
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["batches"],
            "train_edge_precision": totals["precision"] / totals["batches"],
            "train_edge_recall": totals["recall"] / totals["batches"],
            **{f"val_{key}": value for key, value in val.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        writer.writerow(row)
        metrics_file.flush()
        logger.info(
            "epoch=%d train=%.4f val_pen=%.4f valid=%.4f count_err=%.1f "
            "split=%.1f merge=%.1f raw=%.1f cap=%.1f edge_p=%.4f edge_r=%.4f",
            epoch, row["train_loss"], row["val_gt_penalized_miou"],
            row["val_instance_miou_valid"], row["val_instance_count_abs_error"],
            row["val_split_gt_instance_count"], row["val_merged_pred_instance_count"],
            row["val_raw_component_count"], row["val_merged_components_for_cap"],
            row["val_edge_precision"], row["val_edge_recall"],
        )
        if row["val_gt_penalized_miou"] > best_score:
            best_score = row["val_gt_penalized_miou"]
            save_checkpoint(
                output_dir / "best_affinity.pth", system, config, epoch, best_score,
                reference_path, reference_sha, init_path, digest, split,
            )
        if epoch % int(cfg.get("monitor_interval", 5)) == 0 or epoch == epochs:
            semantic_contract_audit(
                system, audit_image, semantic_baseline, digest,
                float(cfg.get("semantic_tolerance", 1e-6)),
            )
            save_checkpoint(
                output_dir / "latest_affinity.pth", system, config, epoch, best_score,
                reference_path, reference_sha, init_path, digest, split,
            )
            write_val_monitor(system, val_dataset, output_dir, epoch, graph_thresholds)
            for path in unlabeled_monitors:
                write_unlabeled_monitor(
                    system, path, output_dir, epoch,
                    int(cfg.get("input_size", 1024)), int(cfg.get("output_grid", 512)),
                )
    metrics_file.close()
    logger.info("Affinity G1 complete best_val=%.6f output=%s", best_score, output_dir)


if __name__ == "__main__":
    main()
