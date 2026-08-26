# -*- coding: utf-8 -*-
"""Run the canonical deployment evaluator over multiple affinity checkpoints.

Each child run uses the exact deployment path implemented by
``run_affinity_watershed_ab.py``: predicted V6 semantics, affinity fusion,
boundary watershed, instance class voting, and the competition-proxy scorer.
Ground truth is loaded only after a submission-style prediction is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "tools" / "run_affinity_watershed_ab.py"
ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_checkpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be ALIAS=PATH")
    alias, path = (item.strip() for item in value.split("=", 1))
    if not alias or not path:
        raise argparse.ArgumentTypeError("checkpoint must be ALIAS=PATH")
    if not ALIAS_PATTERN.fullmatch(alias):
        raise argparse.ArgumentTypeError(
            "checkpoint alias may contain only letters, digits, dot, dash, underscore"
        )
    return alias, path


def _finite(value, fallback: float) -> float:
    number = float(value)
    return number if math.isfinite(number) else float(fallback)


def selection_key(row: dict) -> tuple[float, float, float]:
    """Rank by the official proxy, then stable diagnostic tie-breakers."""
    return (
        _finite(row["score_total"], -math.inf),
        _finite(row["instance_miou_valid"], -math.inf),
        -_finite(row["ferrite_area_relative_error"], math.inf),
    )


def flatten_candidate(alias: str, arm: str, summary: dict, checkpoint: str) -> dict:
    classes = summary["classes"]
    return {
        "candidate": "v6_boundary" if arm == "v6_boundary" else f"{alias}/{arm}",
        "checkpoint_alias": "v6" if arm == "v6_boundary" else alias,
        "checkpoint": "V6 reference boundary" if arm == "v6_boundary" else checkpoint,
        "arm": arm,
        "score_total": float(summary["score_total"]),
        "score_miou": float(summary["score_miou"]),
        "score_area": float(summary["score_area"]),
        "instance_miou_valid": float(summary["instance_miou_valid"]),
        "gt_penalized_miou": float(summary["gt_penalized_miou"]),
        "ferrite_area_relative_error": float(
            summary["ferrite_area_relative_error"]
        ),
        "ferrite_mean_area_gt": float(summary["ferrite_mean_area_gt"]),
        "ferrite_mean_area_pred": float(summary["ferrite_mean_area_pred"]),
        "gt_count": int(summary["gt_count"]),
        "pred_count": int(summary["pred_count"]),
        "valid_matches": int(summary["valid_matches"]),
        "ferrite_valid_matches": int(classes["ferrite"]["valid_matches"]),
        "pearlite_valid_matches": int(classes["pearlite"]["valid_matches"]),
        "ferrite_merged_pred_count": int(classes["ferrite"]["merged_pred_count"]),
        "pearlite_merged_pred_count": int(classes["pearlite"]["merged_pred_count"]),
    }


def write_ranking(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="0.50,0.55,0.60")
    parser.add_argument(
        "--fusion-mode", choices=("mean", "short", "gated"), default="gated"
    )
    parser.add_argument("--distance2-weight", type=float, default=0.50)
    parser.add_argument("--distance4-weight", type=float, default=0.25)
    parser.add_argument("--support-threshold", type=float, default=0.20)
    parser.add_argument("--support-temperature", type=float, default=0.05)
    parser.add_argument(
        "--short-reduction", choices=("mean", "top2", "softmax"), default="mean"
    )
    parser.add_argument("--short-softmax-temperature", type=float, default=0.15)
    parser.add_argument("--monitor-dir", default="data/test")
    parser.add_argument("--monitor-count", type=int, default=0)
    args = parser.parse_args()

    aliases = [alias for alias, _ in args.checkpoint]
    if len(aliases) != len(set(aliases)):
        raise ValueError("checkpoint aliases must be unique")

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    child_reports = {}
    rows = []
    for index, (alias, checkpoint) in enumerate(args.checkpoint):
        child_output = output_root / "runs" / alias
        command = [
            sys.executable,
            str(RUNNER),
            "--config", args.config,
            "--affinity-checkpoint", checkpoint,
            "--split", args.split,
            "--output-dir", str(child_output),
            "--thresholds", args.thresholds,
            "--fusion-mode", args.fusion_mode,
            "--distance2-weight", str(args.distance2_weight),
            "--distance4-weight", str(args.distance4_weight),
            "--support-threshold", str(args.support_threshold),
            "--support-temperature", str(args.support_temperature),
            "--short-reduction", args.short_reduction,
            "--short-softmax-temperature", str(args.short_softmax_temperature),
            "--monitor-dir", args.monitor_dir,
            "--monitor-count", str(args.monitor_count),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        report_path = child_output / "ab_summary.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        child_reports[alias] = {
            "checkpoint": checkpoint,
            "checkpoint_epoch": report.get("checkpoint_epoch"),
            "report": str(report_path),
        }
        for arm, summary in report["summaries"].items():
            if arm == "v6_boundary" and index > 0:
                continue
            rows.append(flatten_candidate(alias, arm, summary, checkpoint))

    deployment_rows = [row for row in rows if row["checkpoint_alias"] != "v6"]
    if not deployment_rows:
        raise RuntimeError("no affinity deployment candidates were evaluated")
    ranked = sorted(deployment_rows, key=selection_key, reverse=True)
    baseline_rows = [row for row in rows if row["checkpoint_alias"] == "v6"]
    ranking_rows = ranked + baseline_rows
    write_ranking(output_root / "deployment_ranking.csv", ranking_rows)

    report = {
        "protocol": {
            "prediction": (
                "predicted V6 semantics -> affinity fusion -> boundary watershed "
                "-> class voting"
            ),
            "ground_truth_usage": (
                "scoring only; never used as foreground, marker, or postprocess input"
            ),
            "selection_metric": "competition-proxy score_total",
            "score_total": (
                "50 * valid matched-instance mIoU + 50 * max(0, 1 - ferrite "
                "mean-area relative error)"
            ),
            "tie_breakers": [
                "instance_miou_valid descending",
                "ferrite_area_relative_error ascending",
            ],
            "oracle_geometry_metrics": (
                "diagnostic only and forbidden for checkpoint promotion"
            ),
        },
        "config": args.config,
        "split": args.split,
        "thresholds": [float(value) for value in args.thresholds.split(",")],
        "fusion": {
            "mode": args.fusion_mode,
            "short_reduction": args.short_reduction,
            "distance2_weight": args.distance2_weight,
            "distance4_weight": args.distance4_weight,
            "support_threshold": args.support_threshold,
            "support_temperature": args.support_temperature,
            "short_softmax_temperature": args.short_softmax_temperature,
        },
        "checkpoints": child_reports,
        "winner": ranked[0],
        "ranking": ranking_rows,
    }
    report_path = output_root / "deployment_selection.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["winner"], ensure_ascii=False, indent=2))
    print(f"Deployment selection: {report_path}")
    print(f"Deployment ranking: {output_root / 'deployment_ranking.csv'}")


if __name__ == "__main__":
    main()