# -*- coding: utf-8 -*-
"""Audit G1 center calibration and single-seed suitability on non-convex GT."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import letterbox
from data.offset_geometry_dataset import OffsetGeometryDataset
from train_offset_geometry_g1 import build_system
from utils.center_guided_instances import (
    assign_endpoints_to_centers,
    extract_center_peaks,
)
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair
from utils.offset_letterbox import geometry_letterbox_metadata


def parse_values(text, cast=float):
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def evaluate_prediction(sample, center_probability, offsets, threshold, nms_radius):
    grid_size = int(sample["instance_map"].shape[-1])
    foreground = sample["foreground"][0].numpy().astype(bool)
    valid = sample["valid_content"][0].numpy().astype(bool)
    gt = sample["instance_map"].numpy().astype(np.int32)
    centers, scores, peak_audit = extract_center_peaks(
        center_probability, valid, threshold=float(threshold),
        nms_radius=int(nms_radius), max_centers=255,
    )
    yy, xx = np.indices(foreground.shape, dtype=np.float32)
    predicted, assignment = assign_endpoints_to_centers(
        yy + offsets[0], xx + offsets[1], foreground, centers,
        max_assignment_distance=None, min_instance_area=1,
    )
    gt_ids = [int(value) for value in np.unique(gt) if int(value) != 0]
    pred_ids = [int(value) for value in np.unique(predicted) if int(value) != 0]
    metrics = evaluate_instance_pair(
        gt, {value: 0 for value in gt_ids},
        predicted, {value: 0 for value in pred_ids},
    )
    return {
        "gt_count": len(gt_ids),
        "center_count": len(centers),
        "center_count_abs_error": abs(len(centers) - len(gt_ids)),
        "mean_center_score": float(scores.mean()) if scores.size else 0.0,
        "instance_miou_valid": float(metrics["instance_miou_valid"]),
        "gt_penalized_miou": float(metrics["gt_penalized_miou"]),
        "cap_drops": int(peak_audit["dropped_for_cap"]),
        "unassigned": int(assignment["unassigned_foreground_pixels"]),
    }


def line_visible_fraction(mask, center_y, center_x, max_samples=96):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0
    indices = np.linspace(
        0, ys.size - 1, min(int(max_samples), ys.size), dtype=np.int64
    )
    visible = 0
    for index in indices:
        target_y, target_x = int(ys[index]), int(xs[index])
        steps = max(abs(target_y - center_y), abs(target_x - center_x)) + 1
        line_y = np.rint(np.linspace(center_y, target_y, steps)).astype(np.int32)
        line_x = np.rint(np.linspace(center_x, target_x, steps)).astype(np.int32)
        visible += int(bool(np.all(mask[line_y, line_x])))
    return visible / max(1, len(indices))


def instance_topology(instance_map, image_name):
    rows = []
    for instance_id in np.unique(instance_map):
        instance_id = int(instance_id)
        if instance_id == 0:
            continue
        mask = instance_map == instance_id
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        local = mask[y0:y1, x0:x1].astype(np.uint8)
        distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
        local_y, local_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        center_y, center_x = y0 + int(local_y), x0 + int(local_x)
        centroid_y, centroid_x = int(round(float(ys.mean()))), int(round(float(xs.mean())))
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_points = np.concatenate(contours, axis=0) if contours else np.empty((0, 1, 2))
        hull_area = (
            float(cv2.contourArea(cv2.convexHull(contour_points)))
            if contour_points.shape[0] >= 3 else float(mask.sum())
        )
        component_count = cv2.connectedComponents(local, connectivity=8)[0] - 1
        max_distance = float(
            np.sqrt((ys - center_y) ** 2 + (xs - center_x) ** 2).max()
        )
        rows.append({
            "image": image_name,
            "instance_id": instance_id,
            "area": int(mask.sum()),
            "solidity": float(mask.sum() / max(hull_area, 1.0)),
            "component_count": int(component_count),
            "centroid_inside": bool(mask[centroid_y, centroid_x]),
            "edt_center_inside": bool(mask[center_y, center_x]),
            "star_visible_fraction": float(
                line_visible_fraction(mask, center_y, center_x)
            ),
            "max_center_distance": max_distance,
            "max_center_distance_over_sqrt_area": float(
                max_distance / max(np.sqrt(mask.sum()), 1.0)
            ),
        })
    return rows


@torch.no_grad()
def predict_labeled(system, samples):
    device = next(system.geometry_decoder.parameters()).device
    predictions = {}
    for sample in samples:
        output = system.geometry_forward(sample["image"].unsqueeze(0).to(device))
        predictions[sample["image_name"]] = {
            "center": torch.sigmoid(output["center_logits"])[0, 0].cpu().numpy(),
            "offsets": output["offsets"][0].cpu().numpy()
            * float(sample["instance_map"].shape[-1]),
        }
    return predictions


@torch.no_grad()
def predict_unlabeled(system, paths, input_size, grid_size):
    device = next(system.geometry_decoder.parameters()).device
    results = {}
    for path in paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_lb, _, _, _ = letterbox(image_rgb, input_size)
        tensor = torch.from_numpy(image_lb).permute(2, 0, 1).float().div(255.0)
        output = system.geometry_forward(tensor.unsqueeze(0).to(device))
        probability = torch.sigmoid(output["center_logits"])[0, 0].cpu().numpy()
        metadata = geometry_letterbox_metadata(image_rgb.shape[:2], input_size, grid_size)
        valid = np.zeros((grid_size, grid_size), dtype=bool)
        valid[:metadata.content_height, :metadata.content_width] = True
        results[path.name] = {"center": probability, "valid": valid}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/offset_geometry_g1.yaml")
    parser.add_argument("--checkpoint", default="outputs/offset_geometry_g1/best_geometry.pth")
    parser.add_argument("--output-dir", default="outputs/offset_geometry_g1/audit_center_generalization")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40")
    parser.add_argument("--nms-radii", default="2,3,4,5")
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["offset_geometry_g1"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["sam2"].get("device") == "cuda"
        else "cpu"
    )
    system, _, _, semantic_digest = build_system(config, cfg, device)
    checkpoint_path = project_path(config, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    system.geometry_decoder.load_state_dict(checkpoint["geometry_state_dict"], strict=True)
    if checkpoint.get("semantic_state_digest") != semantic_digest:
        raise RuntimeError("G1 checkpoint semantic digest mismatch")
    system.eval()
    output_dir = Path(project_path(config, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(project_path(config, cfg["output_dir"])) / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    raw_dir = project_path(config, config["paths"]["raw_data_dir"])
    all_dataset = OffsetGeometryDataset(
        raw_dir, image_size=int(cfg.get("input_size", 1024)),
        output_grid=int(cfg.get("output_grid", 512)), cache_in_memory=True,
    )
    by_name = {
        Path(sample["image_name"]).stem: sample
        for sample in (all_dataset[index] for index in range(len(all_dataset)))
    }
    val_samples = [by_name[name] for name in split["val"]]
    predictions = predict_labeled(system, val_samples)
    thresholds = parse_values(args.thresholds, float)
    nms_radii = parse_values(args.nms_radii, int)
    sweep_rows = []
    for threshold in thresholds:
        sample_rows = [
            evaluate_prediction(
                sample, predictions[sample["image_name"]]["center"],
                predictions[sample["image_name"]]["offsets"], threshold, 3,
            )
            for sample in val_samples
        ]
        sweep_rows.append({
            "threshold": threshold,
            "nms_radius": 3,
            "val_gt_penalized_miou": float(np.mean([row["gt_penalized_miou"] for row in sample_rows])),
            "val_instance_miou_valid": float(np.mean([row["instance_miou_valid"] for row in sample_rows])),
            "center_count_abs_error": float(np.mean([row["center_count_abs_error"] for row in sample_rows])),
            "mean_pred_center_count": float(np.mean([row["center_count"] for row in sample_rows])),
            "mean_gt_count": float(np.mean([row["gt_count"] for row in sample_rows])),
            "mean_center_score": float(np.mean([row["mean_center_score"] for row in sample_rows])),
            "cap_drops": int(sum(row["cap_drops"] for row in sample_rows)),
        })
    best = max(sweep_rows, key=lambda row: row["val_gt_penalized_miou"])

    labeled_names = set(by_name)
    unlabeled_dir = Path(project_path(config, cfg.get("unlabeled_dir", "data/unlabeled")))
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    unlabeled_paths = [
        path for path in sorted(unlabeled_dir.iterdir())
        if path.suffix.lower() in extensions and path.stem not in labeled_names
    ][: int(cfg.get("unlabeled_monitor_count", 2))]
    unlabeled_predictions = predict_unlabeled(
        system, unlabeled_paths, int(cfg.get("input_size", 1024)),
        int(cfg.get("output_grid", 512)),
    )
    center_count_rows = []
    for threshold in thresholds:
        for nms_radius in nms_radii:
            row = {"threshold": threshold, "nms_radius": nms_radius}
            val_counts = []
            for sample in val_samples:
                probability = predictions[sample["image_name"]]["center"]
                valid = sample["valid_content"][0].numpy().astype(bool)
                centers, _, audit = extract_center_peaks(
                    probability, valid, threshold, nms_radius, 255
                )
                val_counts.append(len(centers))
                row["val_cap_drops"] = row.get("val_cap_drops", 0) + audit["dropped_for_cap"]
            row["val_mean_center_count"] = float(np.mean(val_counts))
            for image_name, prediction in unlabeled_predictions.items():
                centers, _, audit = extract_center_peaks(
                    prediction["center"], prediction["valid"], threshold, nms_radius, 255
                )
                row[f"{Path(image_name).stem}_center_count"] = len(centers)
                row[f"{Path(image_name).stem}_cap_drops"] = audit["dropped_for_cap"]
            center_count_rows.append(row)

    topology_rows = []
    for sample in by_name.values():
        topology_rows.extend(
            instance_topology(
                sample["instance_map"].numpy().astype(np.int32), sample["image_name"]
            )
        )
    topology = {
        "instance_count": len(topology_rows),
        "edt_center_outside_count": int(sum(not row["edt_center_inside"] for row in topology_rows)),
        "centroid_outside_fraction": float(np.mean([not row["centroid_inside"] for row in topology_rows])),
        "multi_component_fraction": float(np.mean([row["component_count"] > 1 for row in topology_rows])),
        "median_solidity": float(np.median([row["solidity"] for row in topology_rows])),
        "p10_solidity": float(np.percentile([row["solidity"] for row in topology_rows], 10)),
        "median_star_visible_fraction": float(np.median([row["star_visible_fraction"] for row in topology_rows])),
        "fraction_star_visibility_below_0_8": float(np.mean([row["star_visible_fraction"] < 0.8 for row in topology_rows])),
        "p90_max_center_distance_over_sqrt_area": float(np.percentile([row["max_center_distance_over_sqrt_area"] for row in topology_rows], 90)),
    }
    with open(output_dir / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
        writer.writeheader(); writer.writerows(sweep_rows)
    with open(output_dir / "center_count_grid.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(center_count_rows[0]))
        writer.writeheader(); writer.writerows(center_count_rows)
    (output_dir / "summary.json").write_text(json.dumps({
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_threshold": best,
        "threshold_sweep": sweep_rows,
        "topology": topology,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "topology_instances.json").write_text(
        json.dumps(topology_rows, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"best_threshold": best, "topology": topology}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
