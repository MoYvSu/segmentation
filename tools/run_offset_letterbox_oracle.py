# -*- coding: utf-8 -*-
"""O1 oracle for a global 1024-letterbox center/offset geometry pass."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import split_train_val_indices
from utils.flow_instances import (
    build_center_offset_target,
    cluster_endpoints,
    endpoints_from_center_offsets,
    majority_class_map,
)
from utils.instance_metrics import (
    evaluate_instance_pair,
    load_labelme_instances,
    summarize_instance_results,
)
from utils.offset_letterbox import (
    inverse_letterbox_instances,
    letterbox_instance_geometry,
)


def collect_samples(data_dir: Path, subset: str, train_ratio: float, seed: int):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    samples = [
        path for path in sorted(data_dir.iterdir())
        if path.suffix.lower() in extensions and path.with_suffix(".json").is_file()
    ]
    if subset == "all":
        return samples
    indices = split_train_val_indices(len(samples), train_ratio, seed, subset)
    return [samples[int(index)] for index in indices]


def stable_rng(seed: int, condition: str, image_name: str):
    token = f"{seed}:{condition}:{image_name}".encode("utf-8")
    return np.random.default_rng(zlib.crc32(token) & 0xFFFFFFFF)


def condition_name(grid: int, noise: float) -> str:
    noise_text = f"{noise:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"global_lb1024_g{grid}_n{noise_text}"


def compact_metrics(metrics):
    keys = (
        "gt_count", "pred_count", "valid_matches", "instance_miou_valid",
        "gt_penalized_miou", "symmetric_penalized_miou",
        "ferrite_gt_count", "ferrite_pred_count",
        "ferrite_area_relative_error", "score_total", "classes",
    )
    return {key: metrics[key] for key in keys}


def main():
    parser = argparse.ArgumentParser(
        description="O1 global letterbox offset-grid oracle"
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--output-dir", default="outputs/experiments/offset_letterbox_o1"
    )
    parser.add_argument("--subset", choices=("train", "val", "all"), default="val")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--output-grids", nargs="+", type=int,
                        default=(256, 512, 1024))
    parser.add_argument("--endpoint-noise-px", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--close-radius", type=int, default=1)
    parser.add_argument("--min-instance-area", type=int, default=50)
    parser.add_argument("--max-instances", type=int, default=255)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(data_dir, args.subset, args.train_ratio, args.seed)
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No labeled samples selected")

    results = defaultdict(list)
    details = defaultdict(list)
    for image_index, image_path in enumerate(samples, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        gt_map, gt_class_map, gt_audit = load_labelme_instances(
            image_path.with_suffix(".json"), image.shape[:2]
        )
        print(
            f"[{image_index}/{len(samples)}] {image_path.name}: "
            f"shape={gt_map.shape} gt={len(gt_class_map)}"
        )
        for grid_size in args.output_grids:
            geometry_map, valid_content, metadata = letterbox_instance_geometry(
                gt_map, input_size=args.input_size, output_grid=grid_size
            )
            foreground = (geometry_map > 0) & valid_content
            offset_target = build_center_offset_target(geometry_map)
            area_scale = (
                metadata.content_height * metadata.content_width
                / float(gt_map.shape[0] * gt_map.shape[1])
            )
            min_area_grid = max(1, int(math.ceil(args.min_instance_area * area_scale)))
            for noise in args.endpoint_noise_px:
                condition = condition_name(grid_size, noise)
                rng = stable_rng(args.seed, condition, image_path.name)
                endpoint_y, endpoint_x = endpoints_from_center_offsets(
                    offset_target, foreground, noise, rng
                )
                pred_grid, reconstruction_audit = cluster_endpoints(
                    endpoint_y, endpoint_x, foreground,
                    close_radius=args.close_radius,
                    min_instance_area=min_area_grid,
                    max_instances=args.max_instances,
                )
                pred_full = inverse_letterbox_instances(pred_grid, metadata)
                pred_class_map = majority_class_map(
                    pred_full, gt_map, gt_class_map
                )
                declared = set(pred_class_map)
                if declared:
                    keep = np.isin(
                        pred_full, np.fromiter(declared, dtype=np.int32)
                    )
                    pred_full = np.where(keep, pred_full, 0).astype(np.int32)
                else:
                    pred_full.fill(0)
                metrics = evaluate_instance_pair(
                    gt_map, gt_class_map, pred_full, pred_class_map
                )
                results[condition].append(metrics)
                details[condition].append({
                    "image": image_path.name,
                    "metadata": metadata.to_dict(),
                    "gt_audit": gt_audit,
                    "reconstruction_audit": reconstruction_audit,
                    "min_instance_area_grid": min_area_grid,
                    "metrics": compact_metrics(metrics),
                })
                print(
                    f"  {condition}: pred={metrics['pred_count']} "
                    f"valid={metrics['valid_matches']} "
                    f"miou={metrics['instance_miou_valid']:.4f} "
                    f"area_err={metrics['ferrite_area_relative_error']:.4f} "
                    f"score={metrics['score_total']:.2f}"
                )

    summaries = {
        condition: summarize_instance_results(rows)
        for condition, rows in results.items()
    }
    report = {
        "experiment": "O1 global letterbox center-offset oracle",
        "data_dir": str(data_dir),
        "subset": args.subset,
        "images": [path.name for path in samples],
        "settings": vars(args),
        "summaries": summaries,
        "details": details,
    }
    report_path = output_dir / "offset_letterbox_oracle_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== summaries ===")
    for condition, summary in summaries.items():
        print(
            f"{condition}: pred={summary['pred_count']} "
            f"valid={summary['valid_matches']} "
            f"miou={summary['instance_miou_valid']:.5f} "
            f"gt_pen={summary['gt_penalized_miou']:.5f} "
            f"area_err={summary['ferrite_area_relative_error']:.5f} "
            f"score={summary['score_total']:.3f}"
        )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
