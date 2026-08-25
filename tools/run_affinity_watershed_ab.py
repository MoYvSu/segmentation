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
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import letterbox
from data.mim_dataset import list_images
from train_affinity_geometry_g1 import build_system
from utils.affinity_fusion import affinity_boundary_probability
from utils.config import load_config, project_path
from utils.instance_metrics import (
    evaluate_instance_pair,
    load_labelme_instances,
    summarize_instance_results,
)
from utils.post_process import post_process_prediction_boundary


def affinity_mean_boundary_probability(affinity_logits: torch.Tensor) -> torch.Tensor:
    """Return ``1 - mean(sigmoid(affinity))`` as a one-channel boundary map."""
    return affinity_boundary_probability(affinity_logits, mode="mean")


def probability_to_logit(probability: torch.Tensor, eps: float = 1.0e-5):
    value = probability.clamp(float(eps), 1.0 - float(eps))
    return torch.log(value) - torch.log1p(-value)


def crop_letterbox_output(
    output: torch.Tensor,
    image_size: int,
    pad_h: int,
    pad_w: int,
    original_size,
):
    out_h, out_w = output.shape[-2:]
    content_h = max(1, int(round((image_size - pad_h) * out_h / image_size)))
    content_w = max(1, int(round((image_size - pad_w) * out_w / image_size)))
    output = output[:, :, :content_h, :content_w]
    return F.interpolate(output, size=original_size, mode="bilinear", align_corners=True)


def prepare_image(image_path: str | Path, image_size: int, device):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_lb, _, pad_h, pad_w = letterbox(rgb, image_size)
    tensor = (
        torch.from_numpy(image_lb).permute(2, 0, 1).float().unsqueeze(0).to(device)
        / 255.0
    )
    return rgb, tensor, int(pad_h), int(pad_w)


@torch.no_grad()
def predict_maps(
    system,
    image_path,
    image_size,
    device,
    fusion_mode="mean",
    fusion_kwargs=None,
):
    image, tensor, pad_h, pad_w = prepare_image(image_path, image_size, device)
    original_size = image.shape[:2]
    reference_output = system.reference_model(tensor)
    affinity_output = system.geometry_forward(tensor)["affinity_logits"]
    reference_native = crop_letterbox_output(
        reference_output, image_size, pad_h, pad_w, original_size
    ).cpu()
    affinity_boundary = affinity_boundary_probability(
        affinity_output,
        mode=fusion_mode,
        **(fusion_kwargs or {}),
    )
    affinity_boundary_native = crop_letterbox_output(
        affinity_boundary, image_size, pad_h, pad_w, original_size
    ).cpu()
    affinity_logits_native = probability_to_logit(affinity_boundary_native)
    affinity_watershed_output = torch.cat(
        [reference_native[:, :1], affinity_logits_native], dim=1
    )
    return image, reference_native, affinity_watershed_output, affinity_boundary_native


def postprocess(
    output,
    original_size,
    output_dir,
    basename,
    infer_cfg,
    boundary_threshold,
    save_visualization,
):
    return post_process_prediction_boundary(
        output=output,
        original_size=original_size,
        output_dir=str(output_dir),
        image_basename=basename,
        min_instance_area=int(infer_cfg.get("min_instance_area", 50)),
        max_instance_id=int(infer_cfg.get("max_instance_id", 255)),
        threshold=float(infer_cfg.get("threshold", 0.5)),
        boundary_threshold=float(boundary_threshold),
        boundary_logit_scale=1.0,
        sem_edge_boost_alpha=0.0,
        sem_edge_merge_weight=0.0,
        watershed_dilate_width=int(infer_cfg.get("watershed_dilate_width", 1)),
        bridge_width=int(infer_cfg.get("bridge_width", 1)),
        use_center_seeds=False,
        save_visualization=save_visualization,
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
    parser.add_argument("--monitor-dir", default="data/test")
    parser.add_argument("--monitor-count", type=int, default=10)
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
    system.geometry_decoder.load_state_dict(
        checkpoint["geometry_state_dict"], strict=True
    )
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
    }
    baseline_threshold = float(infer_cfg.get("boundary_threshold", 0.35))
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
            _, pred_map, pred_class_map = postprocess(
                selected_output,
                image.shape[:2],
                arm_dir,
                basename,
                infer_cfg,
                threshold,
                True,
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
            )
            cv2.imwrite(
                str(arm_dir / f"{basename}_affinity_{args.fusion_mode}_boundary.png"),
                (boundary_prob[0, 0].numpy() * 255).astype(np.uint8),
            )

    report = {
        "config": os.path.abspath(args.config),
        "affinity_checkpoint": os.path.abspath(checkpoint_path),
        "split": os.path.abspath(project_path(config, args.split)),
        "validation_images": val_names,
        "definition": "short-range affinity boundary with optional gated long-range reinforcement",
        "fusion": {"mode": args.fusion_mode, **fusion_kwargs},
        "fixed_postprocess": {
            "min_instance_area": int(infer_cfg.get("min_instance_area", 50)),
            "bridge_width": int(infer_cfg.get("bridge_width", 1)),
            "watershed_dilate_width": int(infer_cfg.get("watershed_dilate_width", 1)),
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
