# -*- coding: utf-8 -*-
"""Run a fixed-label A/B between 1024 letterbox and native tiled logits.

This is intentionally an experiment driver rather than a second production
inference pipeline.  Both arms use the same checkpoint and the exact same
post-processing parameters; only logit generation differs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import letterbox, split_train_val_indices
from inference import _predict_with_tta, build_model
from utils.config import load_config, project_path
from utils.instance_metrics import (
    evaluate_instance_pair,
    load_labelme_instances,
    summarize_instance_results,
)
from utils.post_process import post_process_prediction_boundary
from utils.scale_policy import resolution_scaled_min_area


@torch.no_grad()
def predict_letterbox_logits(model, image_rgb, device, image_size, use_tta=False):
    """Reproduce the current production letterbox path and return native logits."""
    height, width = image_rgb.shape[:2]
    image_lb, _, pad_h, pad_w = letterbox(image_rgb, image_size)
    tensor = (
        torch.from_numpy(image_lb).float().permute(2, 0, 1).unsqueeze(0).to(device)
        / 255.0
    )
    output = _predict_with_tta(model, tensor, use_tta=use_tta)
    out_h, out_w = output.shape[-2:]
    content_h = int(round((image_size - pad_h) * out_h / image_size))
    content_w = int(round((image_size - pad_w) * out_w / image_size))
    output = output[:, :, :max(1, content_h), :max(1, content_w)]
    return F.interpolate(
        output, size=(height, width), mode="bilinear", align_corners=True
    ).cpu()


def tile_starts(length, tile_size, overlap):
    """Cover an axis completely while anchoring the final tile at the far edge."""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def blend_window(tile_size, floor=0.05):
    """Positive raised-cosine window; nonzero floor preserves outer image edges."""
    one_dimensional = np.hanning(tile_size).astype(np.float32)
    one_dimensional = np.maximum(one_dimensional, float(floor))
    return np.outer(one_dimensional, one_dimensional).astype(np.float32)


@torch.no_grad()
def predict_native_tiled_logits(
    model,
    image_rgb,
    device,
    tile_size=1024,
    overlap=256,
    blend_floor=0.05,
    use_tta=False,
):
    """Predict overlapping native-resolution tiles and blend logits globally."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile overlap must satisfy 0 <= overlap < tile_size")

    height, width = image_rgb.shape[:2]
    y_starts = tile_starts(height, tile_size, overlap)
    x_starts = tile_starts(width, tile_size, overlap)
    window = blend_window(tile_size, blend_floor)
    accumulation = None
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for top in y_starts:
        for left in x_starts:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = image_rgb[top:bottom, left:right]
            valid_h, valid_w = tile.shape[:2]
            pad_h = tile_size - valid_h
            pad_w = tile_size - valid_w
            if pad_h or pad_w:
                tile = cv2.copyMakeBorder(
                    tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
                )

            tensor = (
                torch.from_numpy(tile).float().permute(2, 0, 1).unsqueeze(0).to(device)
                / 255.0
            )
            output = _predict_with_tta(model, tensor, use_tta=use_tta)
            output = F.interpolate(
                output, size=(tile_size, tile_size),
                mode="bilinear", align_corners=True,
            )[0, :, :valid_h, :valid_w].float().cpu().numpy()

            if accumulation is None:
                accumulation = np.zeros(
                    (output.shape[0], height, width), dtype=np.float32
                )
            local_weight = window[:valid_h, :valid_w]
            accumulation[:, top:bottom, left:right] += output * local_weight[None]
            weight_sum[top:bottom, left:right] += local_weight

    if accumulation is None or np.any(weight_sum <= 0):
        raise RuntimeError("Native tile coverage failed")
    blended = accumulation / weight_sum[None]
    return torch.from_numpy(blended).unsqueeze(0)


def discover_labeled_images(data_dir):
    paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        paths.extend(glob.glob(os.path.join(data_dir, pattern)))
    return sorted(
        path for path in paths
        if os.path.exists(os.path.join(data_dir, Path(path).stem + ".json"))
    )


def select_subset(image_paths, subset, train_ratio, seed):
    if subset == "all":
        return image_paths
    indices = split_train_val_indices(
        len(image_paths), train_ratio=train_ratio, seed=seed, split=subset
    )
    return [image_paths[int(index)] for index in indices]


def flatten_metrics(image_name, metrics, elapsed):
    row = {
        "image": image_name,
        "seconds": elapsed,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/inference/v6_reference.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/experiments/native_tiles_v6")
    parser.add_argument("--subset", choices=("train", "val", "all"), default="val")
    parser.add_argument(
        "--modes", nargs="+", choices=("letterbox", "native_tiles"),
        default=("letterbox", "native_tiles"),
    )
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-overlap", type=int, default=256)
    parser.add_argument("--tile-blend-floor", type=float, default=0.05)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--save-visualization", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = project_path(config, args.data_dir)
    output_root = project_path(config, args.output_dir)
    os.makedirs(output_root, exist_ok=True)
    image_size = int(config["data"]["image_size"])
    tile_size = int(args.tile_size or image_size)
    if tile_size != image_size:
        raise ValueError(
            "This controlled A/B keeps the SAM2 input fixed; tile_size must equal image_size"
        )

    infer_cfg = config["inference"]
    checkpoint = project_path(
        config,
        args.checkpoint or infer_cfg.get(
            "stage2_checkpoint", "outputs/stage2_v6/best_model_stage2.pth"
        ),
    )
    device = config["sam2"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = build_model(config, device, checkpoint)

    all_images = discover_labeled_images(data_dir)
    images = select_subset(
        all_images,
        args.subset,
        float(config["data"].get("train_ratio", 0.8)),
        int(config["data"].get("seed", 42)),
    )
    if not images:
        raise SystemExit(f"No labeled images selected from {data_dir}")

    summaries = {}
    rows_by_mode = {}
    for mode in args.modes:
        mode_dir = os.path.join(output_root, mode)
        os.makedirs(mode_dir, exist_ok=True)
        metrics_list = []
        rows = []
        for image_path in images:
            basename = Path(image_path).stem
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            started = time.time()
            if mode == "letterbox":
                logits = predict_letterbox_logits(
                    model, image_rgb, device, image_size,
                    use_tta=bool(args.tta or infer_cfg.get("tta", False)),
                )
            else:
                logits = predict_native_tiled_logits(
                    model, image_rgb, device,
                    tile_size=tile_size,
                    overlap=args.tile_overlap,
                    blend_floor=args.tile_blend_floor,
                    use_tta=bool(args.tta or infer_cfg.get("tta", False)),
                )

            effective_min_area = resolution_scaled_min_area(
                int(infer_cfg.get("min_instance_area", 50)),
                image_rgb.shape[:2],
                infer_cfg.get("resolution_aware_min_area", {}),
            )
            _, pred_map, pred_class_map = post_process_prediction_boundary(
                output=logits,
                original_size=image_rgb.shape[:2],
                output_dir=mode_dir,
                image_basename=basename,
                min_instance_area=effective_min_area,
                max_instance_id=int(infer_cfg.get("max_instance_id", 255)),
                threshold=float(infer_cfg.get("threshold", 0.5)),
                boundary_threshold=float(infer_cfg.get("boundary_threshold", 0.5)),
                boundary_logit_scale=float(infer_cfg.get("boundary_logit_scale", 1.0)),
                sem_edge_boost_alpha=float(infer_cfg.get("sem_edge_boost_alpha", 0.0)),
                sem_edge_merge_weight=float(infer_cfg.get("sem_edge_merge_weight", 0.0)),
                sem_edge_smooth=float(infer_cfg.get("sem_edge_smooth", 1.0)),
                watershed_dilate_width=int(infer_cfg.get("watershed_dilate_width", 2)),
                bridge_width=int(infer_cfg.get("bridge_width", 1)),
                use_center_seeds=bool(infer_cfg.get("center_seeds", False)),
                center_threshold=float(infer_cfg.get("center_threshold", 0.25)),
                center_nms_kernel=int(infer_cfg.get("center_nms_kernel", 9)),
                save_visualization=args.save_visualization,
            )
            gt_map, gt_class_map, _ = load_labelme_instances(
                os.path.join(data_dir, basename + ".json"), pred_map.shape
            )
            metrics = evaluate_instance_pair(
                gt_map, gt_class_map, pred_map, pred_class_map
            )
            elapsed = time.time() - started
            metrics_list.append(metrics)
            rows.append(flatten_metrics(basename, metrics, elapsed))
            print(
                f"[{mode}] {basename}: pred={metrics['pred_count']} "
                f"valid={metrics['valid_matches']} score={metrics['score_total']:.3f} "
                f"time={elapsed:.1f}s",
                flush=True,
            )

        summary = summarize_instance_results(metrics_list)
        summaries[mode] = summary
        rows_by_mode[mode] = rows
        with open(os.path.join(mode_dir, "metrics_per_image.csv"), "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    comparison = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(checkpoint),
        "data_dir": os.path.abspath(data_dir),
        "subset": args.subset,
        "images": [Path(path).name for path in images],
        "tile_size": tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_blend_floor": args.tile_blend_floor,
        "tta": bool(args.tta or infer_cfg.get("tta", False)),
        "summaries": summaries,
    }
    if "letterbox" in summaries and "native_tiles" in summaries:
        comparison["native_minus_letterbox"] = {
            key: float(summaries["native_tiles"][key]) - float(summaries["letterbox"][key])
            for key in (
                "instance_miou_valid", "gt_penalized_miou",
                "symmetric_penalized_miou", "ferrite_area_relative_error",
                "score_miou", "score_area", "score_total",
            )
        }
    report_path = os.path.join(output_root, "ab_summary.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"A/B report: {report_path}")


if __name__ == "__main__":
    main()
