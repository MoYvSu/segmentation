# -*- coding: utf-8 -*-
"""Sweep graph thresholds and per-offset errors for an affinity checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.offset_geometry_dataset import OffsetGeometryDataset
from inference import build_model as build_reference_model
from models.affinity_geometry import AffinityGeometryDecoder
from models.offset_geometry import FrozenSemanticGeometrySystem
from utils.affinity_graph import (
    DEFAULT_AFFINITY_OFFSETS,
    audit_instance_recovery,
    build_affinity_targets,
    reconstruct_affinity_components,
)
from utils.config import load_config, project_path


def binary_metrics(probability, target, valid, threshold):
    prediction = probability >= np.asarray(threshold, dtype=np.float32)
    positive = valid & (target > 0.5)
    negative = valid & ~positive
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & negative))
    tn = int(np.sum(~prediction & negative))
    fn = int(np.sum(~prediction & positive))
    return {
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "false_positive_edges": fp,
        "false_negative_edges": fn,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/affinity_geometry_g0.yaml")
    parser.add_argument(
        "--checkpoint", default="outputs/affinity_geometry_g0/latest_affinity.pth"
    )
    parser.add_argument(
        "--output", default="outputs/affinity_geometry_g0/threshold_audit.json"
    )
    parser.add_argument(
        "--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["affinity_geometry"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["sam2"].get("device") == "cuda"
        else "cpu"
    )
    reference = build_reference_model(
        config, device, project_path(config, cfg["reference_checkpoint"])
    )
    decoder = AffinityGeometryDecoder(
        in_channels=reference.encoder.get_stage_channels(),
        affinity_channels=len(DEFAULT_AFFINITY_OFFSETS),
        fpn_channels=int(cfg.get("fpn_channels", 256)),
        up_channels=int(cfg.get("up_channels", 128)),
        output_grid=int(cfg.get("output_grid", 512)),
    ).to(device)
    checkpoint = torch.load(
        project_path(config, args.checkpoint), map_location="cpu", weights_only=False
    )
    decoder.load_state_dict(checkpoint["geometry_state_dict"], strict=True)
    system = FrozenSemanticGeometrySystem(reference, decoder).to(device).eval()
    dataset = OffsetGeometryDataset(
        project_path(config, config["paths"]["raw_data_dir"]),
        sample_names=cfg.get("sample_names"),
        image_size=int(cfg.get("input_size", 1024)),
        output_grid=int(cfg.get("output_grid", 512)),
        cache_in_memory=True,
    )
    predictions = []
    with torch.no_grad():
        for sample in dataset:
            output = system.geometry_forward(sample["image"].unsqueeze(0).to(device))
            predictions.append(
                torch.sigmoid(output["affinity_logits"])[0].cpu().numpy()
            )
    thresholds = [float(value) for value in args.thresholds.split(",")]
    sweep = []
    per_channel = []
    for sample, probability in zip(dataset, predictions):
        labels = sample["instance_map"].numpy().astype(np.int32)
        target, valid = build_affinity_targets(
            labels, sample["valid_content"][0].numpy().astype(bool)
        )
        channels = []
        for channel, offset in enumerate(DEFAULT_AFFINITY_OFFSETS):
            positive = valid[channel] & (target[channel] > 0.5)
            negative = valid[channel] & ~positive
            channels.append({
                "offset": list(offset),
                **binary_metrics(
                    probability[channel], target[channel], valid[channel], 0.5
                ),
                "positive_probability": float(probability[channel][positive].mean()),
                "negative_probability": float(probability[channel][negative].mean()),
            })
        per_channel.append({"image": sample["image_name"], "channels": channels})
        for threshold in thresholds:
            instances, graph = reconstruct_affinity_components(
                labels > 0, probability, threshold=threshold, max_instances=None
            )
            sweep.append({
                "image": sample["image_name"],
                "threshold": threshold,
                **binary_metrics(probability, target, valid, threshold),
                **graph,
                **audit_instance_recovery(labels, instances),
            })
    recipes = {
        "short_t020": [0.20, 0.20, 0.20, 0.20, 1.10, 1.10, 1.10, 1.10],
        "short_t025": [0.25, 0.25, 0.25, 0.25, 1.10, 1.10, 1.10, 1.10],
        "short020_d2_025": [0.20, 0.20, 0.20, 0.20, 0.25, 0.25, 1.10, 1.10],
        "staged_d4_050": [0.20, 0.20, 0.20, 0.20, 0.25, 0.25, 0.50, 0.50],
        "staged_d4_060": [0.20, 0.20, 0.20, 0.20, 0.25, 0.25, 0.60, 0.60],
        "staged_d4_070": [0.20, 0.20, 0.20, 0.20, 0.25, 0.25, 0.70, 0.70],
        "conservative": [0.25, 0.25, 0.25, 0.25, 0.30, 0.30, 0.60, 0.60],
    }
    recipe_rows = []
    for recipe_name, recipe_thresholds in recipes.items():
        for sample, probability in zip(dataset, predictions):
            labels = sample["instance_map"].numpy().astype(np.int32)
            target, valid = build_affinity_targets(
                labels, sample["valid_content"][0].numpy().astype(bool)
            )
            threshold_grid = np.asarray(
                recipe_thresholds, dtype=np.float32
            )[:, None, None]
            instances, graph = reconstruct_affinity_components(
                labels > 0, probability, threshold=recipe_thresholds,
                max_instances=None,
            )
            recipe_rows.append({
                "recipe": recipe_name,
                "image": sample["image_name"],
                **binary_metrics(
                    probability, target, valid, threshold_grid
                ),
                **graph,
                **audit_instance_recovery(labels, instances),
            })
    summary = {
        "checkpoint": os.path.abspath(project_path(config, args.checkpoint)),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "thresholds": thresholds,
        "sweep": sweep,
        "per_channel_at_0_5": per_channel,
        "threshold_recipes": recipes,
        "recipe_sweep": recipe_rows,
    }
    output = Path(project_path(config, args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for threshold in thresholds:
        rows = [row for row in sweep if row["threshold"] == threshold]
        print({
            "threshold": threshold,
            "mean_components": float(np.mean([row["raw_component_count"] for row in rows])),
            "split_gt": int(sum(row["split_gt_instance_count"] for row in rows)),
            "merged_pred": int(sum(row["merged_pred_instance_count"] for row in rows)),
            "precision": float(np.mean([row["precision"] for row in rows])),
            "recall": float(np.mean([row["recall"] for row in rows])),
        })
    for recipe_name in recipes:
        rows = [row for row in recipe_rows if row["recipe"] == recipe_name]
        print({
            "recipe": recipe_name,
            "mean_components": float(np.mean([row["raw_component_count"] for row in rows])),
            "split_gt": int(sum(row["split_gt_instance_count"] for row in rows)),
            "merged_pred": int(sum(row["merged_pred_instance_count"] for row in rows)),
            "precision": float(np.mean([row["precision"] for row in rows])),
            "recall": float(np.mean([row["recall"] for row in rows])),
        })


if __name__ == "__main__":
    main()
