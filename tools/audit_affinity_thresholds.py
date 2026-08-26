# -*- coding: utf-8 -*-
"""Audit raw and fused affinity calibration on the labeled validation split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.offset_geometry_dataset import OffsetGeometryDataset
from train_affinity_geometry_g1 import build_system
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config, project_path


GROUPS = {
    "short": slice(0, 4),
    "distance2": slice(4, 6),
    "distance4": slice(6, 8),
}


def parse_checkpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be ALIAS=PATH")
    alias, path = value.split("=", 1)
    if not alias.strip() or not path.strip():
        raise argparse.ArgumentTypeError("checkpoint must be ALIAS=PATH")
    return alias.strip(), path.strip()


def describe(values: list[np.ndarray]) -> dict:
    if not values:
        return {"count": 0}
    array = np.concatenate(values).astype(np.float32, copy=False)
    if not array.size:
        return {"count": 0}
    quantiles = np.quantile(array, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q10": float(quantiles[2]),
        "q25": float(quantiles[3]),
        "q50": float(quantiles[4]),
        "q75": float(quantiles[5]),
        "q90": float(quantiles[6]),
        "q95": float(quantiles[7]),
        "q99": float(quantiles[8]),
    }


def threshold_metrics(boundary: np.ndarray, interior: np.ndarray, threshold: float) -> dict:
    tp = int(np.count_nonzero(boundary >= threshold))
    fn = int(boundary.size - tp)
    fp = int(np.count_nonzero(interior >= threshold))
    tn = int(interior.size - fp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "boundary_leak_rate": float(1.0 - recall),
        "boundary_pixels": int(boundary.size),
        "interior_pixels": int(interior.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--fusion-mode", choices=("mean", "short", "gated"), default="gated")
    parser.add_argument("--distance2-weight", type=float, default=0.50)
    parser.add_argument("--distance4-weight", type=float, default=0.25)
    parser.add_argument("--support-threshold", type=float, default=0.20)
    parser.add_argument("--support-temperature", type=float, default=0.05)
    parser.add_argument("--short-reduction", choices=("mean", "top2", "softmax"), default="mean")
    parser.add_argument("--short-softmax-temperature", type=float, default=0.15)
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["affinity_geometry_g1"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["sam2"].get("device") == "cuda" else "cpu"
    )
    system, _, _, _ = build_system(config, cfg, device)
    split_path = Path(project_path(config, args.split))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    dataset = OffsetGeometryDataset(
        project_path(config, config["paths"]["raw_data_dir"]),
        sample_names=split["val"],
        image_size=int(cfg.get("input_size", 1024)),
        output_grid=int(cfg.get("output_grid", 512)),
        cache_in_memory=True,
    )
    thresholds = [float(value) for value in args.thresholds.split(",")]
    fusion_kwargs = {
        "distance2_weight": args.distance2_weight,
        "distance4_weight": args.distance4_weight,
        "support_threshold": args.support_threshold,
        "support_temperature": args.support_temperature,
        "short_reduction": args.short_reduction,
        "short_softmax_temperature": args.short_softmax_temperature,
    }
    output_dir = Path(project_path(config, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": str(Path(args.config).resolve()),
        "split": str(split_path.resolve()),
        "validation_images": split["val"],
        "fusion": {"mode": args.fusion_mode, **fusion_kwargs},
        "checkpoints": {},
    }
    csv_rows = []

    for alias, checkpoint_name in args.checkpoint:
        checkpoint_path = Path(project_path(config, checkpoint_name))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        system.geometry_decoder.load_state_dict(checkpoint["geometry_state_dict"], strict=True)
        system.eval()
        raw = {
            group: {"positive": [], "negative": [], "uncovered_negative": []}
            for group in GROUPS
        }
        fused_boundary_values: list[np.ndarray] = []
        fused_interior_values: list[np.ndarray] = []

        with torch.no_grad():
            for sample in dataset:
                image = sample["image"].unsqueeze(0).to(device)
                labels = sample["instance_map"].unsqueeze(0).to(device)
                valid_content = sample["valid_content"].unsqueeze(0).to(device)
                uncovered_source = sample["uncovered_boundary_source"].unsqueeze(0).to(device)
                target, edge_valid, uncovered_edge = build_affinity_targets_torch(
                    labels,
                    valid_content,
                    uncovered_as_boundary=uncovered_source,
                    return_uncovered_mask=True,
                )
                logits = system.geometry_forward(image)["affinity_logits"]
                probability = torch.sigmoid(logits)
                fused = affinity_boundary_probability(
                    logits, mode=args.fusion_mode, **fusion_kwargs
                )[0, 0]

                target_np = target[0].cpu().numpy()
                valid_np = edge_valid[0].cpu().numpy().astype(bool)
                uncovered_np = uncovered_edge[0].cpu().numpy().astype(bool)
                probability_np = probability[0].cpu().numpy()
                for group, channel_slice in GROUPS.items():
                    group_valid = valid_np[channel_slice]
                    group_target = target_np[channel_slice]
                    group_probability = probability_np[channel_slice]
                    group_uncovered = uncovered_np[channel_slice]
                    raw[group]["positive"].append(
                        group_probability[group_valid & (group_target > 0.5)]
                    )
                    raw[group]["negative"].append(
                        group_probability[group_valid & (group_target < 0.5) & ~group_uncovered]
                    )
                    raw[group]["uncovered_negative"].append(
                        group_probability[group_uncovered]
                    )

                short_valid = valid_np[:4]
                short_negative = short_valid & (target_np[:4] < 0.5)
                short_positive = short_valid & (target_np[:4] > 0.5)
                boundary_pixels = np.any(short_negative, axis=0)
                interior_pixels = (~boundary_pixels) & (np.sum(short_positive, axis=0) >= 2)
                fused_np = fused.cpu().numpy()
                fused_boundary_values.append(fused_np[boundary_pixels])
                fused_interior_values.append(fused_np[interior_pixels])

        boundary = np.concatenate(fused_boundary_values)
        interior = np.concatenate(fused_interior_values)
        sweep = [threshold_metrics(boundary, interior, value) for value in thresholds]
        best = max(sweep, key=lambda row: row["f1"])
        checkpoint_report = {
            "path": str(checkpoint_path.resolve()),
            "epoch": int(checkpoint.get("epoch", checkpoint.get("step", -1))),
            "raw_affinity": {
                group: {kind: describe(values) for kind, values in kinds.items()}
                for group, kinds in raw.items()
            },
            "fused_boundary_probability": {
                "true_boundary": describe(fused_boundary_values),
                "instance_interior": describe(fused_interior_values),
            },
            "best_f1_threshold": best,
            "threshold_sweep": sweep,
        }
        report["checkpoints"][alias] = checkpoint_report
        for row in sweep:
            csv_rows.append({"checkpoint": alias, **row})

    (output_dir / "affinity_threshold_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "affinity_threshold_sweep.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
