# -*- coding: utf-8 -*-
"""GT-free audit of instance splits introduced by boundary reconstruction.

New seams are valid where the baseline treats neighboring pixels as one
foreground instance while reconstruction assigns different non-zero IDs.
Their image evidence is compared with ordinary interior neighbor pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def _read_instance(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    if value.ndim == 3:
        value = value[..., 0]
    return value.astype(np.int32)


def _read_image(path: Path, shape):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[:2] != tuple(shape):
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _label_boundary(labels):
    result = np.zeros(labels.shape, dtype=np.uint8)
    for axis in (0, 1):
        if axis == 0:
            left = (slice(None, -1), slice(None))
            right = (slice(1, None), slice(None))
        else:
            left = (slice(None), slice(None, -1))
            right = (slice(None), slice(1, None))
        different = labels[left] != labels[right]
        left_mask = result[left]
        right_mask = result[right]
        left_mask[different] = 255
        right_mask[different] = 255
    return result


def _pixel_values(baseline, reconstructed, gray, boundary):
    baseline_boundary = _label_boundary(baseline)
    reconstructed_boundary = _label_boundary(reconstructed)
    exclusion = cv2.dilate(
        baseline_boundary, np.ones((5, 5), np.uint8), iterations=1
    )
    seam_mask = (
        (reconstructed_boundary > 0)
        & (exclusion == 0)
        & (baseline > 0)
    ).astype(np.uint8) * 255
    all_boundaries = cv2.dilate(
        cv2.bitwise_or(baseline_boundary, reconstructed_boundary),
        np.ones((7, 7), np.uint8),
        iterations=1,
    )
    interior_mask = (baseline > 0) & (all_boundaries == 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    local_contrast = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    seam = seam_mask > 0
    return {
        "seam_contrast": local_contrast[seam],
        "seam_gradient": gradient[seam],
        "seam_affinity": boundary[seam],
        "interior_contrast": local_contrast[interior_mask],
        "interior_gradient": gradient[interior_mask],
        "interior_affinity": boundary[interior_mask],
        "seam_mask": seam_mask,
    }


def _safe_mean(value):
    return float(value.mean()) if value.size else 0.0


def _area_statistics(labels, prefix):
    ids, counts = np.unique(labels, return_counts=True)
    counts = counts[ids > 0]
    return {
        f"{prefix}_min_area": int(counts.min()) if counts.size else 0,
        f"{prefix}_median_area": float(np.median(counts)) if counts.size else 0.0,
        f"{prefix}_instances_le_200px": int(np.count_nonzero(counts <= 200)),
        f"{prefix}_instances_le_500px": int(np.count_nonzero(counts <= 500)),
    }


def _overlay(image, seam_mask):
    output = image.copy()
    wide = cv2.dilate(seam_mask, np.ones((3, 3), np.uint8), iterations=1) > 0
    output[wide] = (0, 0, 255)
    return output


def _fit_panel(image, width=480, height=360):
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (
        max(1, round(image.shape[1] * scale)),
        max(1, round(image.shape[0] * scale)),
    )
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 245, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--reconstruct-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    reconstruct_dir = Path(args.reconstruct_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    visuals = []
    for baseline_path in sorted(baseline_dir.glob("*_inst.png")):
        name = baseline_path.name.removesuffix("_inst.png")
        reconstructed_path = reconstruct_dir / f"{name}_inst.png"
        boundary_path = reconstruct_dir / f"{name}_affinity_gated_boundary.png"
        image_path = next(
            (image_dir / f"{name}{suffix}" for suffix in (".jpg", ".png", ".jpeg")
             if (image_dir / f"{name}{suffix}").exists()),
            None,
        )
        if not reconstructed_path.exists() or image_path is None:
            continue
        baseline = _read_instance(baseline_path)
        reconstructed = _read_instance(reconstructed_path)
        image, gray = _read_image(image_path, baseline.shape)
        boundary = cv2.imread(str(boundary_path), cv2.IMREAD_GRAYSCALE)
        if boundary is None:
            boundary = np.zeros(baseline.shape, np.uint8)
        elif boundary.shape != baseline.shape:
            boundary = cv2.resize(
                boundary, (baseline.shape[1], baseline.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        values = _pixel_values(
            baseline, reconstructed, gray, boundary.astype(np.float32)
        )
        row = {
            "image": name,
            "baseline_instances": int(np.count_nonzero(np.unique(baseline))),
            "reconstruct_instances": int(np.count_nonzero(np.unique(reconstructed))),
            "new_seam_pixels": int(values["seam_contrast"].size),
            "seam_local_contrast": _safe_mean(values["seam_contrast"]),
            "interior_local_contrast": _safe_mean(values["interior_contrast"]),
            "seam_gradient": _safe_mean(values["seam_gradient"]),
            "interior_gradient": _safe_mean(values["interior_gradient"]),
            "seam_affinity_boundary": _safe_mean(values["seam_affinity"]),
            "interior_affinity_boundary": _safe_mean(values["interior_affinity"]),
            **_area_statistics(baseline, "baseline"),
            **_area_statistics(reconstructed, "reconstruct"),
        }
        row["local_contrast_ratio"] = row["seam_local_contrast"] / max(
            row["interior_local_contrast"], 1e-6
        )
        row["gradient_ratio"] = row["seam_gradient"] / max(
            row["interior_gradient"], 1e-6
        )
        rows.append(row)
        visuals.append((row["new_seam_pixels"], name, image, values["seam_mask"]))
        cv2.imwrite(
            str(output_dir / f"{name}_new_seams_overlay.png"),
            _overlay(image, values["seam_mask"]),
        )

    if not rows:
        raise RuntimeError("no matching baseline/reconstruction instance maps")
    with open(output_dir / "metrics_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "images": len(rows),
        "baseline_instances": int(sum(row["baseline_instances"] for row in rows)),
        "reconstruct_instances": int(sum(row["reconstruct_instances"] for row in rows)),
        "added_instances": int(sum(
            row["reconstruct_instances"] - row["baseline_instances"] for row in rows
        )),
        "new_seam_pixels": int(sum(row["new_seam_pixels"] for row in rows)),
        "mean_local_contrast_ratio": float(np.mean(
            [row["local_contrast_ratio"] for row in rows]
        )),
        "mean_gradient_ratio": float(np.mean(
            [row["gradient_ratio"] for row in rows]
        )),
        "baseline_instances_le_200px": int(sum(
            row["baseline_instances_le_200px"] for row in rows
        )),
        "reconstruct_instances_le_200px": int(sum(
            row["reconstruct_instances_le_200px"] for row in rows
        )),
        "baseline_min_area": int(min(row["baseline_min_area"] for row in rows)),
        "reconstruct_min_area": int(min(row["reconstruct_min_area"] for row in rows)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected = sorted(visuals, reverse=True)[:args.top_k]
    header = np.full((70, 960, 3), 255, np.uint8)
    cv2.putText(
        header, "Reconstruction-only seams (red) on original test images",
        (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA,
    )
    panels = [header]
    for _, name, image, seam_mask in selected:
        left = _fit_panel(image)
        right = _fit_panel(_overlay(image, seam_mask))
        cv2.putText(left, f"{name} original", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(right, f"{name} new seams", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
        panels.append(np.concatenate([left, right], axis=1))
    cv2.imwrite(
        str(output_dir / "new_seams_report.png"),
        np.concatenate(panels, axis=0),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
