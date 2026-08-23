# -*- coding: utf-8 -*-
"""O0 oracle experiment for offset/flow-based instance reconstruction.

This script converts LabelMe GT polygons to a dense vector representation,
throws away the GT ids, reconstructs instances only from vector endpoints, and
then evaluates with the competition-oriented instance proxy. It is intended to
answer whether a representation is viable before training a neural head.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import split_train_val_indices
from utils.flow_instances import (
    build_center_offset_target,
    build_edt_flow_target,
    build_tile_local_center_offset_target,
    cluster_endpoints,
    endpoints_from_center_offsets,
    integrate_flow,
    majority_class_map,
    perturb_unit_flow,
    resize_label_map,
    stride_shape,
)
from utils.instance_metrics import (
    evaluate_instance_pair,
    load_labelme_instances,
    summarize_instance_results,
)


def _parse_float_list(values: Iterable[str]) -> List[float]:
    return [float(value) for value in values]


def _collect_samples(data_dir: Path, subset: str, train_ratio: float, seed: int):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    samples = [
        path for path in sorted(data_dir.iterdir())
        if path.suffix.lower() in image_extensions
        and path.with_suffix(".json").is_file()
    ]
    if subset == "all":
        return samples
    indices = split_train_val_indices(
        len(samples), train_ratio=float(train_ratio), seed=int(seed), split=subset
    )
    return [samples[int(index)] for index in indices]


def _stable_rng(seed: int, condition: str, image_name: str):
    token = f"{seed}:{condition}:{image_name}".encode("utf-8")
    return np.random.default_rng(zlib.crc32(token) & 0xFFFFFFFF)


def _condition_name(method: str, stride: int, noise: float) -> str:
    noise_text = f"{noise:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{method}_s{stride}_n{noise_text}"


def _colorize(instance_map: np.ndarray) -> np.ndarray:
    output = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(instance_map):
        if int(instance_id) == 0:
            continue
        value = int(instance_id)
        color = (
            (37 * value + 53) % 255,
            (97 * value + 29) % 255,
            (17 * value + 193) % 255,
        )
        output[instance_map == value] = color
    return output


def _metric_row(metrics: Dict) -> Dict:
    return {
        "gt_count": metrics["gt_count"],
        "pred_count": metrics["pred_count"],
        "valid_matches": metrics["valid_matches"],
        "instance_miou_valid": metrics["instance_miou_valid"],
        "gt_penalized_miou": metrics["gt_penalized_miou"],
        "symmetric_penalized_miou": metrics["symmetric_penalized_miou"],
        "ferrite_gt_count": metrics["ferrite_gt_count"],
        "ferrite_pred_count": metrics["ferrite_pred_count"],
        "ferrite_area_relative_error": metrics["ferrite_area_relative_error"],
        "score_total": metrics["score_total"],
        "classes": metrics["classes"],
    }


def _reconstruct_condition(
    method: str,
    low_instances: np.ndarray,
    noise: float,
    rng: np.random.Generator,
    *,
    integration_steps: int,
    integration_step_size: float,
    edt_smooth_sigma: float,
    close_radius: int,
    min_instance_area_low: int,
    max_instances: int,
    tile_size_low: int,
    tile_overlap_low: int,
):
    foreground = low_instances > 0
    if method in ("center", "center_tile_local"):
        if method == "center":
            target = build_center_offset_target(low_instances)
        else:
            target = build_tile_local_center_offset_target(
                low_instances,
                tile_size=tile_size_low,
                overlap=tile_overlap_low,
            )
        endpoint_y, endpoint_x = endpoints_from_center_offsets(
            target, foreground, endpoint_noise_px=noise, rng=rng
        )
    elif method == "edt":
        target = build_edt_flow_target(
            low_instances, smooth_sigma=float(edt_smooth_sigma)
        )
        target = perturb_unit_flow(target, foreground, noise_std=noise, rng=rng)
        endpoint_y, endpoint_x = integrate_flow(
            target, foreground, steps=int(integration_steps),
            step_size=float(integration_step_size),
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    return cluster_endpoints(
        endpoint_y, endpoint_x, foreground,
        close_radius=int(close_radius),
        min_instance_area=int(min_instance_area_low),
        max_instances=int(max_instances),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Oracle center-offset / EDT-flow instance reconstruction"
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/experiments/flow_oracle_o0")
    parser.add_argument("--subset", choices=("train", "val", "all"), default="val")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods", nargs="+",
        choices=("center", "center_tile_local", "edt"),
        default=("center", "edt"),
    )
    parser.add_argument("--strides", nargs="+", type=int, default=(4,))
    parser.add_argument("--center-noise-px", nargs="+", default=("0", "1", "2"))
    parser.add_argument("--edt-noise-std", nargs="+", default=("0", "0.05", "0.10"))
    parser.add_argument("--integration-steps", type=int, default=160)
    parser.add_argument("--integration-step-size", type=float, default=1.0)
    parser.add_argument("--edt-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--close-radius", type=int, default=1)
    parser.add_argument("--min-instance-area", type=int, default=50)
    parser.add_argument("--max-instances", type=int, default=255)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-visualizations", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _collect_samples(
        data_dir, args.subset, args.train_ratio, args.seed
    )
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No labeled samples selected")

    method_noise = {
        "center": _parse_float_list(args.center_noise_px),
        "center_tile_local": _parse_float_list(args.center_noise_px),
        "edt": _parse_float_list(args.edt_noise_std),
    }
    conditions = [
        (method, int(stride), float(noise))
        for method in args.methods
        for stride in args.strides
        for noise in method_noise[method]
    ]
    results = defaultdict(list)
    details = defaultdict(list)

    for sample_index, image_path in enumerate(samples, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        gt_instances, gt_class_map, gt_audit = load_labelme_instances(
            image_path.with_suffix(".json"), image.shape[:2]
        )
        print(
            f"[{sample_index}/{len(samples)}] {image_path.name}: "
            f"shape={image.shape[:2]} gt={len(gt_class_map)} "
            f"uncovered={gt_audit['uncovered_pixels']}"
        )

        for method, stride, noise in conditions:
            condition = _condition_name(method, stride, noise)
            low_shape = stride_shape(gt_instances.shape, stride)
            low_instances = resize_label_map(gt_instances, low_shape)
            min_area_low = max(
                1, int(math.ceil(float(args.min_instance_area) / (stride * stride)))
            )
            rng = _stable_rng(args.seed, condition, image_path.name)
            pred_low, reconstruction_audit = _reconstruct_condition(
                method, low_instances, noise, rng,
                integration_steps=args.integration_steps,
                integration_step_size=args.integration_step_size,
                edt_smooth_sigma=args.edt_smooth_sigma,
                close_radius=args.close_radius,
                min_instance_area_low=min_area_low,
                max_instances=args.max_instances,
                tile_size_low=max(1, int(math.ceil(args.tile_size / stride))),
                tile_overlap_low=max(
                    0, int(math.ceil(args.tile_overlap / stride))
                ),
            )
            pred_full = resize_label_map(pred_low, gt_instances.shape)
            pred_class_map = majority_class_map(
                pred_full, gt_instances, gt_class_map
            )
            # An endpoint basin with no labeled semantic support is invalid for
            # the competition output. Remove it instead of inventing a class.
            declared_ids = set(pred_class_map)
            if declared_ids:
                valid = np.isin(pred_full, np.fromiter(declared_ids, dtype=np.int32))
                pred_full = np.where(valid, pred_full, 0).astype(np.int32)
            else:
                pred_full.fill(0)
            metrics = evaluate_instance_pair(
                gt_instances, gt_class_map, pred_full, pred_class_map
            )
            results[condition].append(metrics)
            details[condition].append({
                "image": image_path.name,
                "stride": stride,
                "noise": noise,
                "noise_unit": (
                    "decoder-grid endpoint pixels" if method.startswith("center")
                    else "unit-flow component standard deviation"
                ),
                "low_shape": list(low_shape),
                "gt_audit": gt_audit,
                "reconstruction_audit": reconstruction_audit,
                "metrics": _metric_row(metrics),
            })
            print(
                f"  {condition}: pred={metrics['pred_count']} "
                f"match={metrics['valid_matches']} "
                f"miou={metrics['instance_miou_valid']:.4f} "
                f"area_err={metrics['ferrite_area_relative_error']:.4f} "
                f"score={metrics['score_total']:.2f}"
            )

            if args.save_visualizations:
                condition_dir = output_dir / condition
                condition_dir.mkdir(parents=True, exist_ok=True)
                stem = image_path.stem
                cv2.imwrite(str(condition_dir / f"{stem}_pred.png"),
                            pred_full.astype(np.uint8))
                cv2.imwrite(str(condition_dir / f"{stem}_pred_color.png"),
                            _colorize(pred_full))
                if not (condition_dir / f"{stem}_gt_color.png").exists():
                    cv2.imwrite(str(condition_dir / f"{stem}_gt_color.png"),
                                _colorize(gt_instances))

    summaries = {
        condition: summarize_instance_results(rows)
        for condition, rows in results.items()
    }
    report = {
        "experiment": "O0 oracle flow representation",
        "data_dir": str(data_dir),
        "subset": args.subset,
        "images": [path.name for path in samples],
        "settings": vars(args),
        "summaries": summaries,
        "details": details,
    }
    report_path = output_dir / "flow_oracle_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
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
