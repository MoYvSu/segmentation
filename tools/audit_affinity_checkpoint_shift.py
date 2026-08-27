# -*- coding: utf-8 -*-
"""GT-free affinity shift audit on G4b reconstruction-supported seams."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_gray(path):
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return value


def label_boundary(labels):
    result = np.zeros(labels.shape, dtype=np.uint8)
    for axis in (0, 1):
        if axis == 0:
            left, right = (slice(None, -1), slice(None)), (slice(1, None), slice(None))
        else:
            left, right = (slice(None), slice(None, -1)), (slice(None), slice(1, None))
        different = labels[left] != labels[right]
        result[left][different] = 255
        result[right][different] = 255
    return result


def new_seam_mask(baseline, reconstructed):
    baseline_boundary = label_boundary(baseline)
    reconstructed_boundary = label_boundary(reconstructed)
    exclusion = cv2.dilate(
        baseline_boundary, np.ones((5, 5), np.uint8), iterations=1
    )
    return (
        (reconstructed_boundary > 0)
        & (exclusion == 0)
        & (baseline > 0)
    )


def q(value, quantile):
    return float(np.quantile(value, quantile)) if value.size else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--g4b-dir", required=True)
    parser.add_argument("--g5-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--low-threshold", type=float, default=0.45)
    parser.add_argument("--high-threshold", type=float, default=0.72)
    args = parser.parse_args()
    baseline_dir = Path(args.baseline_dir)
    g4b_dir = Path(args.g4b_dir)
    g5_dir = Path(args.g5_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for g4b_inst_path in sorted(g4b_dir.glob("*_inst.png")):
        stem = g4b_inst_path.name.removesuffix("_inst.png")
        baseline_inst_path = baseline_dir / f"{stem}_inst.png"
        g5_inst_path = g5_dir / f"{stem}_inst.png"
        g4b_prob_path = g4b_dir / f"{stem}_affinity_gated_boundary.png"
        g5_prob_path = g5_dir / f"{stem}_affinity_gated_boundary.png"
        if not all(path.exists() for path in (
            baseline_inst_path, g5_inst_path, g4b_prob_path, g5_prob_path
        )):
            continue
        baseline = read_gray(baseline_inst_path)
        g4b_inst = read_gray(g4b_inst_path)
        g5_inst = read_gray(g5_inst_path)
        g4b_prob = read_gray(g4b_prob_path).astype(np.float32) / 255.0
        g5_prob = read_gray(g5_prob_path).astype(np.float32) / 255.0
        seam = new_seam_mask(baseline, g4b_inst)
        interior = (baseline > 0) & (
            cv2.dilate(label_boundary(g4b_inst), np.ones((7, 7), np.uint8)) == 0
        )
        g4b_seam = g4b_prob[seam]
        g5_seam = g5_prob[seam]
        g4b_interior = g4b_prob[interior]
        g5_interior = g5_prob[interior]
        bridge = (
            (g4b_seam >= args.low_threshold)
            & (g4b_seam < args.high_threshold)
        )
        supported = g4b_seam >= args.low_threshold
        dropout = supported & (g5_seam < args.low_threshold)
        strong = g4b_seam >= args.high_threshold
        row = {
            "image": stem,
            "g4b_instances": int(np.count_nonzero(np.unique(g4b_inst))),
            "g5_instances": int(np.count_nonzero(np.unique(g5_inst))),
            "new_seam_pixels": int(seam.sum()),
            "g4b_seam_mean": float(g4b_seam.mean()),
            "g5_seam_mean": float(g5_seam.mean()),
            "seam_delta_mean": float((g5_seam - g4b_seam).mean()),
            "g4b_seam_q10": q(g4b_seam, 0.10),
            "g5_seam_q10": q(g5_seam, 0.10),
            "g4b_bridge_pixels": int(bridge.sum()),
            "g4b_supported_pixels": int(supported.sum()),
            "g5_retained_supported_fraction": float(
                (g5_seam[supported] >= args.low_threshold).mean()
                if supported.any() else 1.0
            ),
            "support_dropout_pixels": int(dropout.sum()),
            "support_dropout_fraction": float(
                dropout.sum() / max(1, supported.sum())
            ),
            "g4b_strong_fraction": float(strong.mean()),
            "g5_strong_fraction": float(
                (g5_seam >= args.high_threshold).mean()
            ),
            "interior_delta_mean": float(
                (g5_interior - g4b_interior).mean()
            ),
        }
        rows.append(row)
        dropout_map = np.zeros(seam.shape, np.uint8)
        dropout_map[seam] = dropout.astype(np.uint8) * 255
        cv2.imwrite(str(output_dir / f"{stem}_support_dropout.png"), dropout_map)
    if not rows:
        raise RuntimeError("no matching comparison outputs")
    with (output_dir / "metrics_per_image.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    supported_total = sum(row["g4b_supported_pixels"] for row in rows)
    dropout_total = sum(row["support_dropout_pixels"] for row in rows)
    seam_total = sum(row["new_seam_pixels"] for row in rows)
    summary = {
        "images": len(rows),
        "g4b_instances": sum(row["g4b_instances"] for row in rows),
        "g5_instances": sum(row["g5_instances"] for row in rows),
        "new_seam_pixels": seam_total,
        "g4b_seam_mean": float(np.average(
            [row["g4b_seam_mean"] for row in rows],
            weights=[row["new_seam_pixels"] for row in rows],
        )),
        "g5_seam_mean": float(np.average(
            [row["g5_seam_mean"] for row in rows],
            weights=[row["new_seam_pixels"] for row in rows],
        )),
        "supported_pixels": supported_total,
        "support_dropout_pixels": dropout_total,
        "support_dropout_fraction": dropout_total / max(1, supported_total),
        "mean_retained_supported_fraction_per_image": float(np.mean(
            [row["g5_retained_supported_fraction"] for row in rows]
        )),
        "mean_interior_delta": float(np.mean(
            [row["interior_delta_mean"] for row in rows]
        )),
        "low_threshold": args.low_threshold,
        "high_threshold": args.high_threshold,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
