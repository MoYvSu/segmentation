# -*- coding: utf-8 -*-
"""Evaluate submission-style instance outputs against LabelMe polygons."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.instance_metrics import (
    evaluate_instance_pair,
    load_class_map,
    load_labelme_instances,
    summarize_instance_results,
)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _flatten_row(image_name, metrics, audit):
    row = {
        "image": image_name,
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
        **{f"gt_audit_{key}": value for key, value in audit.items()},
    }
    for class_name, values in metrics["classes"].items():
        for key in (
            "gt_count", "pred_count", "valid_matches", "instance_miou_valid",
            "gt_penalized_miou", "valid_match_recall", "valid_match_precision",
            "split_gt_count", "merged_pred_count",
        ):
            row[f"{class_name}_{key}"] = values[key]
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Competition-proxy instance mIoU + ferrite mean-area evaluation"
    )
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", default="data/raw")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    pred_dir = os.path.abspath(args.pred_dir)
    gt_dir = os.path.abspath(args.gt_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(pred_dir, "evaluation"))
    os.makedirs(output_dir, exist_ok=True)

    mask_paths = sorted(glob.glob(os.path.join(pred_dir, "*_inst.png")))
    if not mask_paths:
        raise SystemExit(f"No *_inst.png predictions found in {pred_dir}")

    details = []
    csv_rows = []
    missing = []
    for mask_path in mask_paths:
        filename = os.path.basename(mask_path)
        basename = filename[:-len("_inst.png")]
        class_path = os.path.join(pred_dir, f"{basename}_class.json")
        gt_path = os.path.join(gt_dir, f"{basename}.json")
        if not os.path.exists(class_path) or not os.path.exists(gt_path):
            missing.append({"image": basename, "class_json": class_path, "gt_json": gt_path})
            continue

        pred_map = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if pred_map is None:
            raise FileNotFoundError(mask_path)
        if pred_map.ndim == 3:
            raise ValueError(f"Prediction must be single-channel: {mask_path}")
        pred_class_map = load_class_map(class_path)
        gt_map, gt_class_map, audit = load_labelme_instances(gt_path, pred_map.shape)
        metrics = evaluate_instance_pair(
            gt_map, gt_class_map, pred_map, pred_class_map,
            iou_threshold=args.iou_threshold,
        )
        details.append({"image": basename, "metrics": metrics, "gt_audit": audit})
        csv_rows.append(_flatten_row(basename, metrics, audit))

    if not details:
        raise SystemExit(
            f"No prediction had both class JSON and LabelMe GT; missing={len(missing)}"
        )

    summary = summarize_instance_results(item["metrics"] for item in details)
    report = {
        "protocol": {
            "same_class_only": True,
            "assignment": "Hungarian maximum total IoU, independently per class",
            "valid_iou_threshold": args.iou_threshold,
            "score_miou": "50 * mean IoU over valid matches",
            "score_area": "50 * max(0, 1 - ferrite mean-area relative error)",
            "aggregation": "dataset-level instance and area totals",
            "note": "GT/matching edge cases should be confirmed against organizer implementation",
        },
        "summary": summary,
        "images": details,
        "missing": missing,
    }

    json_path = os.path.join(output_dir, "instance_metrics.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, ensure_ascii=False, indent=2)
    csv_path = os.path.join(output_dir, "instance_metrics_per_image.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    print(f"Detailed JSON: {json_path}")
    print(f"Per-image CSV: {csv_path}")
    if missing:
        print(f"Skipped {len(missing)} predictions without matching GT/class JSON")


if __name__ == "__main__":
    main()
