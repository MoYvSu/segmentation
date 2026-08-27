# -*- coding: utf-8 -*-
"""Sweep cached V6/E7b instance-class gates without rerunning inference.

The prediction directory must contain the fixed instance maps and the
``*_class_confidence.json`` files produced by ``conservative_dual`` inference.
Only the instance-to-class mapping is recomputed; geometry is never changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.instance_metrics import (  # noqa: E402
    evaluate_instance_pair,
    load_labelme_instances,
    summarize_instance_results,
)


CLASS_PEARLITE = 0
CLASS_FERRITE = 1


@dataclass(frozen=True)
class GateArm:
    name: str
    p2f_base_min: float
    p2f_candidate_min: float
    p2f_min_core_gain: float
    f2p_base_max: float
    f2p_candidate_max: float


DEFAULT_ARMS = (
    GateArm("strict", 0.35, 0.85, 0.08, 0.65, 0.15),
    GateArm("medium", 0.30, 0.80, 0.05, 0.70, 0.20),
    GateArm("relaxed", 0.25, 0.75, 0.03, 0.75, 0.25),
)


def parse_arm(value: str) -> GateArm:
    """Parse NAME:P2F_BASE:P2F_CAND:P2F_GAIN:F2P_BASE:F2P_CAND."""
    fields = [field.strip() for field in value.split(":")]
    if len(fields) != 6 or not fields[0]:
        raise argparse.ArgumentTypeError(
            "arm must be NAME:P2F_BASE:P2F_CAND:P2F_GAIN:F2P_BASE:F2P_CAND"
        )
    try:
        thresholds = [float(field) for field in fields[1:]]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid arm threshold: {value}") from exc
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise argparse.ArgumentTypeError("all arm thresholds must be within [0, 1]")
    return GateArm(fields[0], *thresholds)


def classify_cached_instance(details: Mapping, arm: GateArm) -> tuple[int, str]:
    """Apply the production conservative-dual gate to one cached instance."""
    hard_ratio = float(details["hard_ratio"])
    candidate_core = float(details["candidate_core_score"])
    core_gain = float(details["dual_core_gain"])
    base_class = CLASS_FERRITE if hard_ratio > 0.5 else CLASS_PEARLITE
    candidate_class = (
        CLASS_FERRITE if candidate_core > 0.5 else CLASS_PEARLITE
    )
    if base_class == candidate_class:
        return base_class, "agreement"
    if base_class == CLASS_PEARLITE:
        if (
            hard_ratio >= arm.p2f_base_min
            and candidate_core >= arm.p2f_candidate_min
            and core_gain >= arm.p2f_min_core_gain
        ):
            return candidate_class, "pearlite_to_ferrite_black_rim"
    elif (
        hard_ratio <= arm.f2p_base_max
        and candidate_core <= arm.f2p_candidate_max
    ):
        return candidate_class, "ferrite_to_pearlite_high_confidence"
    return base_class, "gate_rejected"


def quantiles(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {key: 0.0 for key in ("min", "q25", "median", "q75", "max")}
    result = np.quantile(np.asarray(values, dtype=np.float64), [0, .25, .5, .75, 1])
    return {
        key: float(value)
        for key, value in zip(("min", "q25", "median", "q75", "max"), result)
    }


def build_gt_index(gt_dir: Optional[Path]) -> Dict[str, Path]:
    if gt_dir is None:
        return {}
    candidates: Dict[str, list[Path]] = {}
    for path in gt_dir.rglob("*.json"):
        candidates.setdefault(path.stem, []).append(path)
    duplicate = {stem: paths for stem, paths in candidates.items() if len(paths) > 1}
    if duplicate:
        examples = ", ".join(
            f"{stem} ({len(paths)})" for stem, paths in list(duplicate.items())[:5]
        )
        raise ValueError(f"duplicate GT JSON stems under {gt_dir}: {examples}")
    return {stem: paths[0] for stem, paths in candidates.items()}


def load_prediction_cache(prediction_dir: Path) -> list[dict]:
    rows = []
    confidence_paths = sorted(prediction_dir.glob("*_class_confidence.json"))
    if not confidence_paths:
        raise FileNotFoundError(
            f"no *_class_confidence.json files found in {prediction_dir}"
        )
    for confidence_path in confidence_paths:
        stem = confidence_path.name.removesuffix("_class_confidence.json")
        instance_path = prediction_dir / f"{stem}_inst.png"
        instance_map = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
        if instance_map is None:
            raise FileNotFoundError(instance_path)
        payload = json.loads(confidence_path.read_text(encoding="utf-8"))
        instances = payload.get("instances", {})
        present_ids = {int(value) for value in np.unique(instance_map) if int(value)}
        cached_ids = {int(value) for value in instances}
        if present_ids != cached_ids:
            raise ValueError(
                f"cached ids differ from instance map for {stem}: "
                f"map_only={sorted(present_ids - cached_ids)}, "
                f"cache_only={sorted(cached_ids - present_ids)}"
            )
        required = {"hard_ratio", "candidate_core_score", "dual_core_gain", "area"}
        for instance_id, details in instances.items():
            missing = required - set(details)
            if missing:
                raise ValueError(f"{stem} instance {instance_id} missing {sorted(missing)}")
        rows.append({"stem": stem, "instance_map": instance_map, "instances": instances})
    return rows


def evaluate_arm(cache: list[dict], arm: GateArm, gt_index: Mapping[str, Path]) -> dict:
    flips = []
    image_rows = []
    metric_rows = []
    class_counts = Counter()
    for image in cache:
        stem = image["stem"]
        instance_map = image["instance_map"]
        class_map = {}
        image_flips = []
        for instance_id, details in image["instances"].items():
            numeric_id = int(instance_id)
            base_class = (
                CLASS_FERRITE if float(details["hard_ratio"]) > 0.5 else CLASS_PEARLITE
            )
            predicted_class, reason = classify_cached_instance(details, arm)
            class_map[numeric_id] = predicted_class
            class_counts["ferrite" if predicted_class else "pearlite"] += 1
            if predicted_class != base_class:
                area = int(details["area"])
                row = {
                    "image": stem,
                    "instance_id": numeric_id,
                    "area": area,
                    "area_fraction": float(area / instance_map.size),
                    "from": base_class,
                    "to": predicted_class,
                    "hard_ratio": float(details["hard_ratio"]),
                    "candidate_core_score": float(details["candidate_core_score"]),
                    "candidate_full_score": float(details["candidate_full_score"]),
                    "core_gain": float(details["dual_core_gain"]),
                    "reason": reason,
                }
                flips.append(row)
                image_flips.append(row)

        image_report = {
            "image": stem,
            "instance_count": len(class_map),
            "flip_count": len(image_flips),
            "pearlite_to_ferrite": sum(row["from"] == 0 for row in image_flips),
            "ferrite_to_pearlite": sum(row["from"] == 1 for row in image_flips),
        }
        gt_path = gt_index.get(stem)
        if gt_index and gt_path is None:
            raise FileNotFoundError(f"no GT JSON for cached prediction {stem}")
        if gt_path is not None:
            gt_map, gt_class_map, gt_audit = load_labelme_instances(
                gt_path, instance_map.shape
            )
            metrics = evaluate_instance_pair(
                gt_map, gt_class_map, instance_map, class_map
            )
            metric_rows.append(metrics)
            image_report["gt_json"] = str(gt_path)
            image_report["gt_audit"] = gt_audit
            image_report["metrics"] = metrics
        image_rows.append(image_report)

    report = {
        "gate": asdict(arm),
        "images": len(cache),
        "instances": sum(row["instance_count"] for row in image_rows),
        "class_counts": dict(class_counts),
        "flip_count": len(flips),
        "pearlite_to_ferrite": sum(row["from"] == 0 for row in flips),
        "ferrite_to_pearlite": sum(row["from"] == 1 for row in flips),
        "flip_area": quantiles(row["area"] for row in flips),
        "flip_area_fraction": quantiles(row["area_fraction"] for row in flips),
        "per_image_flip_count": quantiles(row["flip_count"] for row in image_rows),
        "image_results": image_rows,
        "flips": flips,
    }
    if metric_rows:
        report["metrics"] = summarize_instance_results(metric_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--gt-dir",
        help="Optional LabelMe root; enables class-aware competition-proxy scoring.",
    )
    parser.add_argument(
        "--arm",
        action="append",
        type=parse_arm,
        help=(
            "Repeatable NAME:P2F_BASE:P2F_CAND:P2F_GAIN:F2P_BASE:F2P_CAND. "
            "Defaults to strict, medium, and relaxed."
        ),
    )
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir).resolve()
    arms = tuple(args.arm or DEFAULT_ARMS)
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("arm names must be unique")
    gt_dir = Path(args.gt_dir).resolve() if args.gt_dir else None
    cache = load_prediction_cache(prediction_dir)
    gt_index = build_gt_index(gt_dir)
    reports = {arm.name: evaluate_arm(cache, arm, gt_index) for arm in arms}
    summary = {
        name: {
            key: value
            for key, value in report.items()
            if key not in {"image_results", "flips"}
        }
        for name, report in reports.items()
    }
    payload = {
        "prediction_dir": str(prediction_dir),
        "gt_dir": None if gt_dir is None else str(gt_dir),
        "geometry_policy": "fixed cached instance maps; class mapping only",
        "area_gate": "none",
        "arms": reports,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Gate sweep: {output}")


if __name__ == "__main__":
    main()
