# -*- coding: utf-8 -*-
"""Audit class-only changes between two fixed-geometry submission outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def quantiles(values):
    if not values:
        return {"min": 0, "q25": 0, "median": 0, "q75": 0, "max": 0}
    array = np.asarray(values, dtype=np.float64)
    result = np.quantile(array, [0.0, 0.25, 0.50, 0.75, 1.0])
    return {
        key: float(value)
        for key, value in zip(("min", "q25", "median", "q75", "max"), result)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    flips = []
    geometry_mismatches = []
    baseline_counts = {"ferrite": 0, "pearlite": 0}
    candidate_counts = {"ferrite": 0, "pearlite": 0}
    color_used = 0
    compared = 0

    for baseline_class_path in sorted(baseline_dir.glob("*_class.json")):
        stem = baseline_class_path.name.removesuffix("_class.json")
        candidate_class_path = candidate_dir / baseline_class_path.name
        if not candidate_class_path.is_file():
            raise FileNotFoundError(candidate_class_path)
        baseline_class = load_json(baseline_class_path)
        candidate_class = load_json(candidate_class_path)
        if set(baseline_class) != set(candidate_class):
            geometry_mismatches.append({"image": stem, "reason": "instance_ids"})
            continue
        baseline_map = cv2.imread(
            str(baseline_dir / f"{stem}_inst.png"), cv2.IMREAD_UNCHANGED
        )
        candidate_map = cv2.imread(
            str(candidate_dir / f"{stem}_inst.png"), cv2.IMREAD_UNCHANGED
        )
        if baseline_map is None or candidate_map is None or not np.array_equal(
            baseline_map, candidate_map
        ):
            geometry_mismatches.append({"image": stem, "reason": "instance_map"})
            continue

        baseline_audit = load_json(
            baseline_dir / f"{stem}_class_confidence.json"
        )["instances"]
        candidate_audit = load_json(
            candidate_dir / f"{stem}_class_confidence.json"
        )["instances"]
        for instance_id in sorted(baseline_class, key=int):
            before = int(baseline_class[instance_id])
            after = int(candidate_class[instance_id])
            baseline_counts["ferrite" if before == 1 else "pearlite"] += 1
            candidate_counts["ferrite" if after == 1 else "pearlite"] += 1
            details = candidate_audit[instance_id]
            color_used += int(bool(details.get("color_used", False)))
            if before != after:
                flips.append({
                    "image": stem,
                    "instance_id": int(instance_id),
                    "area": int(details["area"]),
                    "from": before,
                    "to": after,
                    "baseline_score": float(
                        baseline_audit[instance_id]["ferrite_score"]
                    ),
                    "candidate_score": float(details["ferrite_score"]),
                    "semantic_score": float(
                        details.get("semantic_score", details["ferrite_score"])
                    ),
                    "color_score": details.get("color_score"),
                    "color_used": bool(details.get("color_used", False)),
                })
        compared += 1

    areas = [row["area"] for row in flips]
    report = {
        "baseline_dir": str(baseline_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "images_compared": compared,
        "geometry_mismatches": geometry_mismatches,
        "baseline_counts": baseline_counts,
        "candidate_counts": candidate_counts,
        "flip_count": len(flips),
        "pearlite_to_ferrite": sum(
            row["from"] == 0 and row["to"] == 1 for row in flips
        ),
        "ferrite_to_pearlite": sum(
            row["from"] == 1 and row["to"] == 0 for row in flips
        ),
        "flip_area": quantiles(areas),
        "color_used_count": color_used,
        "flips": flips,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "flips"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
