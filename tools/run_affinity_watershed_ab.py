# -*- coding: utf-8 -*-
"""A/B V6 boundary watershed against inverted mean affinity watershed."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.mim_dataset import list_images
from train_affinity_geometry_g1 import (
    build_system,
    load_geometry_checkpoint_state,
)
from utils.affinity_deployment import (
    affinity_mean_boundary_probability,
    crop_letterbox_output,
    postprocess,
    predict_maps,
    probability_to_logit,
)
from utils.config import load_config, project_path
from utils.instance_metrics import (
    evaluate_instance_pair,
    load_labelme_instances,
    summarize_instance_results,
)


def flatten_metrics(image_name, metrics, seconds):
    row = {
        "image": image_name,
        "seconds": float(seconds),
        "gt_count": metrics["gt_count"],
        "pred_count": metrics["pred_count"],
        "valid_matches": metrics["valid_matches"],
        "instance_miou_valid": metrics["instance_miou_valid"],
        "gt_penalized_miou": metrics["gt_penalized_miou"],
        "symmetric_penalized_miou": metrics["symmetric_penalized_miou"],
        "ferrite_mean_area_gt": metrics["ferrite_mean_area_gt"],
        "ferrite_mean_area_pred": metrics["ferrite_mean_area_pred"],
        "ferrite_area_relative_error": metrics["ferrite_area_relative_error"],
        "score_total": metrics["score_total"],
    }
    for class_name in ("pearlite", "ferrite"):
        values = metrics["classes"][class_name]
        for key in (
            "gt_count", "pred_count", "valid_matches", "instance_miou_valid",
            "gt_penalized_miou", "split_gt_count", "merged_pred_count",
        ):
            row[f"{class_name}_{key}"] = values[key]
    return row


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/affinity_geometry_g1.yaml")
    parser.add_argument(
        "--affinity-checkpoint",
        default="outputs/affinity_geometry_g1/best_affinity.pth",
    )
    parser.add_argument(
        "--split", default="outputs/affinity_geometry_g1/split.json"
    )
    parser.add_argument(
        "--output-dir", default="outputs/experiments/affinity_mean_watershed_ab"
    )
    parser.add_argument("--thresholds", default="0.25,0.35,0.45")
    parser.add_argument(
        "--fusion-mode", choices=("mean", "short", "gated"), default="mean"
    )
    parser.add_argument("--distance2-weight", type=float, default=0.50)
    parser.add_argument("--distance4-weight", type=float, default=0.25)
    parser.add_argument("--support-threshold", type=float, default=0.20)
    parser.add_argument("--support-temperature", type=float, default=0.05)
    parser.add_argument(
        "--short-reduction", choices=("mean", "top2", "softmax"), default="mean"
    )
    parser.add_argument("--short-softmax-temperature", type=float, default=0.15)
    parser.add_argument("--monitor-dir", default="data/test")
    parser.add_argument("--monitor-count", type=int, default=10)
    parser.add_argument(
        "--reference-boundary-threshold", type=float, default=0.35,
        help="Stable V6 reference threshold; affinity-only marker settings are disabled.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["affinity_geometry_g1"]
    device_name = config["sam2"].get("device", "cuda")
    device = torch.device(
        "cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu"
    )
    system, _, _, _ = build_system(config, cfg, device)
    checkpoint_path = project_path(config, args.affinity_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_geometry_checkpoint_state(system, checkpoint)
    system.eval()

    image_size = int(cfg.get("input_size", config["data"]["image_size"]))
    infer_cfg = config["inference"]
    raw_dir = Path(project_path(config, config["paths"]["raw_data_dir"]))
    split = json.loads(
        Path(project_path(config, args.split)).read_text(encoding="utf-8")
    )
    val_names = split["val"]
    val_images = [raw_dir / f"{name}.jpg" for name in val_names]
    missing = [str(path) for path in val_images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"validation images missing: {missing}")

    thresholds = [float(value) for value in args.thresholds.split(",")]
    fusion_kwargs = {
        "distance2_weight": args.distance2_weight,
        "distance4_weight": args.distance4_weight,
        "support_threshold": args.support_threshold,
        "support_temperature": args.support_temperature,
        "short_reduction": args.short_reduction,
        "short_softmax_temperature": args.short_softmax_temperature,
    }
    baseline_threshold = float(args.reference_boundary_threshold)
    reference_infer_cfg = dict(infer_cfg)
    reference_infer_cfg.update(
        {
            "marker_boundary_low_threshold": None,
            "marker_boundary_reconstruction_steps": 0,
            "semantic_vote_mode": "hard_majority",
            "semantic_vote_erode_width": 0,
        }
    )
    arms = [("v6_boundary", baseline_threshold, "reference")]
    arms.extend(
        (f"affinity_{args.fusion_mode}_bt{int(round(value * 100)):03d}", value, "affinity")
        for value in thresholds
    )
    output_root = Path(project_path(config, args.output_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_by_arm = {name: [] for name, _, _ in arms}
    rows_by_arm = {name: [] for name, _, _ in arms}

    for image_path in val_images:
        basename = image_path.stem
        started = time.time()
        image, reference_output, affinity_output, boundary_prob = predict_maps(
            system, image_path, image_size, device, args.fusion_mode, fusion_kwargs
        )
        prediction_seconds = time.time() - started
        gt_map, gt_class_map, _ = load_labelme_instances(
            image_path.with_suffix(".json"), image.shape[:2]
        )
        for arm_name, threshold, source in arms:
            arm_dir = output_root / "val" / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)
            selected_output = reference_output if source == "reference" else affinity_output
            arm_infer_cfg = reference_infer_cfg if source == "reference" else infer_cfg
            _, pred_map, pred_class_map = postprocess(
                selected_output,
                image.shape[:2],
                arm_dir,
                basename,
                arm_infer_cfg,
                threshold,
                True,
                image_rgb=image,
            )
            if source == "affinity":
                cv2.imwrite(
                    str(arm_dir / f"{basename}_affinity_{args.fusion_mode}_boundary.png"),
                    (boundary_prob[0, 0].numpy() * 255).astype(np.uint8),
                )
            metrics = evaluate_instance_pair(
                gt_map, gt_class_map, pred_map, pred_class_map
            )
            metrics_by_arm[arm_name].append(metrics)
            rows_by_arm[arm_name].append(
                flatten_metrics(basename, metrics, prediction_seconds)
            )
            print(
                f"[{arm_name}] {basename} pred={metrics['pred_count']} "
                f"valid={metrics['valid_matches']} score={metrics['score_total']:.3f}",
                flush=True,
            )

    summaries = {}
    for arm_name, _, _ in arms:
        summaries[arm_name] = summarize_instance_results(metrics_by_arm[arm_name])
        write_csv(
            output_root / "val" / arm_name / "metrics_per_image.csv",
            rows_by_arm[arm_name],
        )

    monitor_dir = Path(project_path(config, args.monitor_dir))
    monitor_images = [Path(path) for path in list_images(str(monitor_dir))][
        : max(0, int(args.monitor_count))
    ]
    for image_path in monitor_images:
        basename = image_path.stem
        image, _, affinity_output, boundary_prob = predict_maps(
            system, image_path, image_size, device, args.fusion_mode, fusion_kwargs
        )
        for arm_name, threshold, source in arms:
            if source != "affinity":
                continue
            arm_dir = output_root / "monitor" / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)
            postprocess(
                affinity_output,
                image.shape[:2],
                arm_dir,
                basename,
                infer_cfg,
                threshold,
                True,
                image_rgb=image,
            )
            cv2.imwrite(
                str(arm_dir / f"{basename}_affinity_{args.fusion_mode}_boundary.png"),
                (boundary_prob[0, 0].numpy() * 255).astype(np.uint8),
            )

    report = {
        "config": os.path.abspath(args.config),
        "affinity_checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_epoch": int(
            checkpoint.get("epoch", checkpoint.get("step", -1))
        ),
        "split": os.path.abspath(project_path(config, args.split)),
        "validation_images": val_names,
        "definition": "short-range affinity boundary with optional gated long-range reinforcement",
        "ground_truth_usage": (
            "scoring only after submission-style prediction; never used as "
            "foreground, marker, or postprocess input"
        ),
        "fusion": {"mode": args.fusion_mode, **fusion_kwargs},
        "reference_boundary_threshold": baseline_threshold,
        "fixed_postprocess": {
            "min_instance_area": int(infer_cfg.get("min_instance_area", 50)),
            "bridge_width": int(infer_cfg.get("bridge_width", 1)),
            "watershed_dilate_width": int(infer_cfg.get("watershed_dilate_width", 1)),
            "marker_border_seal_width": int(
                infer_cfg.get("marker_border_seal_width", 0)
            ),
            "marker_boundary_low_threshold": infer_cfg.get(
                "marker_boundary_low_threshold"
            ),
            "marker_boundary_reconstruction_steps": int(
                infer_cfg.get("marker_boundary_reconstruction_steps", 0)
            ),
            "semantic_vote_mode": str(
                infer_cfg.get("semantic_vote_mode", "hard_majority")
            ),
            "semantic_vote_erode_width": int(
                infer_cfg.get("semantic_vote_erode_width", 0)
            ),
            "semantic_vote_threshold": float(
                infer_cfg.get("semantic_vote_threshold", 0.5)
            ),
            "center_seeds": False,
            "max_instance_id": int(infer_cfg.get("max_instance_id", 255)),
        },
        "summaries": summaries,
    }
    report_path = output_root / "ab_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"A/B report: {report_path}")


if __name__ == "__main__":
    main()
