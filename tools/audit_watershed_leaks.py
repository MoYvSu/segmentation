# -*- coding: utf-8 -*-
"""Audit marker leakage and topology-preserving boundary repairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.instance_metrics import (
    evaluate_instance_pair,
    load_labelme_instances,
    summarize_instance_results,
)
from utils.post_process import boundary_watershed_separation


def hysteresis_boundary(probability, low_threshold, high_threshold):
    """Keep weak boundary components only when connected to a strong pixel."""
    probability = np.asarray(probability, dtype=np.float32)
    weak = (probability >= float(low_threshold)).astype(np.uint8)
    strong = probability >= float(high_threshold)
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    output = np.zeros_like(weak)
    for label_id in range(1, count):
        component = labels == label_id
        if np.any(strong & component):
            output[component] = 1
    return output


def close_boundary(boundary, radius):
    binary = (np.asarray(boundary) > 0).astype(np.uint8)
    if int(radius) <= 0:
        return binary
    # A rectangular kernel closes one-pixel gaps in already-thinned axial and
    # diagonal contours. OpenCV's 3x3 ellipse is a cross and can reopen such
    # gaps during erosion.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * int(radius) + 1, 2 * int(radius) + 1)
    )
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def skeleton_belt(boundary, bridge_width, dilate_width):
    binary = (np.asarray(boundary) > 0).astype(np.uint8) * 255
    if int(bridge_width) > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(bridge_width) + 1, 2 * int(bridge_width) + 1),
        )
        binary = cv2.dilate(binary, kernel)
    if np.any(binary):
        try:
            skeleton = cv2.ximgproc.thinning(
                binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
        except (AttributeError, cv2.error):
            from skimage.morphology import skeletonize

            skeleton = (skeletonize(binary > 0) * 255).astype(np.uint8)
    else:
        skeleton = binary
    if int(dilate_width) > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(dilate_width) + 1, 2 * int(dilate_width) + 1),
        )
        belt = cv2.dilate(skeleton, kernel)
    else:
        belt = skeleton
    return skeleton, belt


def marker_leak_metrics(
    gt_map, boundary, bridge_width, dilate_width, min_area, border_seal_width=0
):
    _, belt = skeleton_belt(boundary, bridge_width, dilate_width)
    border_width = max(0, int(border_seal_width))
    if border_width > 0:
        belt[:border_width, :] = 255
        belt[-border_width:, :] = 255
        belt[:, :border_width] = 255
        belt[:, -border_width:] = 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(belt), connectivity=8
    )
    marker_ids = [
        label_id
        for label_id in range(1, count)
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= int(min_area)
    ]
    merged_markers = 0
    gt_ids_in_merged_markers = set()
    for marker_id in marker_ids:
        gt_ids = {
            int(value) for value in np.unique(gt_map[labels == marker_id])
            if int(value) > 0
        }
        if len(gt_ids) > 1:
            merged_markers += 1
            gt_ids_in_merged_markers.update(gt_ids)
    return {
        "marker_count": len(marker_ids),
        "merged_marker_count": int(merged_markers),
        "gt_instances_in_merged_markers": int(len(gt_ids_in_merged_markers)),
        "boundary_rate": float((np.asarray(boundary) > 0).mean()),
        "skeleton_belt_rate": float((belt > 0).mean()),
    }


def semantic_from_saved_mask(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return (bgr[:, :, 1] > bgr[:, :, 0]).astype(np.uint8)


def variants(probability):
    primary = (probability >= 0.35).astype(np.uint8)
    base = {
        "t035_bridge1": (primary, 1, 1, None, None, None),
        "t035_bridge2": (primary, 2, 1, None, None, None),
        "t035_bridge3": (primary, 3, 1, None, None, None),
        "t030_bridge1": (
            (probability >= 0.30).astype(np.uint8), 1, 1, None, None, None
        ),
        "t030_bridge2": (
            (probability >= 0.30).astype(np.uint8), 2, 1, None, None, None
        ),
        "t035_close1": (
            close_boundary(probability >= 0.35, 1), 1, 1, None, None, None
        ),
        "t035_close2": (
            close_boundary(probability >= 0.35, 2), 1, 1, None, None, None
        ),
        "hyst025_045": (
            hysteresis_boundary(probability, 0.25, 0.45),
            1, 1, None, None, None,
        ),
        "hyst025_045_close1": (
            close_boundary(hysteresis_boundary(probability, 0.25, 0.45), 1),
            1, 1, None, None, None,
        ),
    }
    for threshold in (0.15, 0.20, 0.25, 0.30):
        marker_boundary = (probability >= threshold).astype(np.uint8)
        for marker_bridge_width in (1, 2):
            name = f"dual_marker_t{int(threshold * 100):03d}_b{marker_bridge_width}"
            base[name] = (
                primary,
                1,
                1,
                marker_boundary,
                marker_bridge_width,
                1,
            )
    for border_width in (1, 2, 3, 4, 6, 8):
        base[f"border_seal_w{border_width}"] = (
            primary, 1, 1, None, None, None, border_width
        )
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        default="downloads/affinity_mean_watershed_ab_20260825/val/affinity_mean_bt035",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--output", default="downloads/affinity_mean_watershed_ab_20260825/leak_audit.json"
    )
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument(
        "--only-prefix", default="", help="Run only variants with this prefix"
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    raw_dir = Path(args.raw_dir)
    probability_paths = sorted(
        experiment_dir.glob("*_affinity_mean_boundary.png")
    )
    if not probability_paths:
        raise ValueError(f"no affinity boundary maps in {experiment_dir}")

    rows = []
    metrics_by_variant = {}
    for probability_path in probability_paths:
        name = probability_path.name.replace("_affinity_mean_boundary.png", "")
        probability = cv2.imread(str(probability_path), cv2.IMREAD_GRAYSCALE)
        if probability is None:
            raise FileNotFoundError(probability_path)
        probability = probability.astype(np.float32) / 255.0
        semantic = semantic_from_saved_mask(experiment_dir / f"{name}_mask.png")
        gt_map, gt_class_map, _ = load_labelme_instances(
            raw_dir / f"{name}.json", probability.shape
        )

        for variant_name, variant in variants(probability).items():
            if args.only_prefix and not variant_name.startswith(args.only_prefix):
                continue
            if len(variant) == 6:
                variant = (*variant, 0)
            (
                boundary,
                bridge_width,
                dilate_width,
                marker_boundary,
                marker_bridge_width,
                marker_dilate_width,
                marker_border_seal_width,
            ) = variant
            pred_map, pred_class_map = boundary_watershed_separation(
                semantic,
                boundary,
                dilate_width=dilate_width,
                min_area=args.min_area,
                max_instance_id=255,
                bridge_width=bridge_width,
                center_prob=None,
                marker_boundary_mask=marker_boundary,
                marker_bridge_width=marker_bridge_width,
                marker_dilate_width=marker_dilate_width,
                marker_border_seal_width=marker_border_seal_width,
            )
            metrics = evaluate_instance_pair(
                gt_map, gt_class_map, pred_map, pred_class_map
            )
            leak = marker_leak_metrics(
                gt_map,
                boundary if marker_boundary is None else marker_boundary,
                bridge_width if marker_bridge_width is None else marker_bridge_width,
                dilate_width if marker_dilate_width is None else marker_dilate_width,
                args.min_area,
                marker_border_seal_width,
            )
            metrics_by_variant.setdefault(variant_name, []).append(metrics)
            rows.append({
                "image": name,
                "variant": variant_name,
                "pred_count": int(metrics["pred_count"]),
                "valid_matches": int(metrics["valid_matches"]),
                "gt_penalized_miou": float(metrics["gt_penalized_miou"]),
                "ferrite_area_relative_error": float(
                    metrics["ferrite_area_relative_error"]
                ),
                "score_total": float(metrics["score_total"]),
                **leak,
            })

    summaries = {
        name: {
            **summarize_instance_results(metric_list),
            "mean_marker_count": float(np.mean([
                row["marker_count"] for row in rows if row["variant"] == name
            ])),
            "mean_merged_marker_count": float(np.mean([
                row["merged_marker_count"] for row in rows if row["variant"] == name
            ])),
            "mean_gt_instances_in_merged_markers": float(np.mean([
                row["gt_instances_in_merged_markers"]
                for row in rows if row["variant"] == name
            ])),
        }
        for name, metric_list in metrics_by_variant.items()
    }
    report = {
        "experiment_dir": os.path.abspath(experiment_dir),
        "raw_dir": os.path.abspath(raw_dir),
        "rows": rows,
        "summaries": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, summary in summaries.items():
        print({
            "variant": name,
            "pred_count": summary["pred_count"],
            "gt_penalized_miou": round(summary["gt_penalized_miou"], 6),
            "ferrite_area_relative_error": round(
                summary["ferrite_area_relative_error"], 6
            ),
            "score_total": round(summary["score_total"], 6),
            "mean_marker_count": round(summary["mean_marker_count"], 2),
            "mean_gt_in_leaky_markers": round(
                summary["mean_gt_instances_in_merged_markers"], 2
            ),
        })
    print(f"Leak audit: {output}")


if __name__ == "__main__":
    main()
