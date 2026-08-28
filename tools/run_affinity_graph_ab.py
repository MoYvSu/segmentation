# -*- coding: utf-8 -*-
"""Compare E10a-only watershed with directional G4b affinity partitions.

This is a GT-free geometry screening experiment.  E10a is the sole semantic
source for both watershed terrain/class voting; G4b supplies geometry only.
The graph arms preserve all eight directional affinity channels instead of
collapsing them into one scalar boundary probability.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.mim_dataset import list_images
from train_affinity_geometry_g1 import build_system, load_geometry_checkpoint_state
from utils.affinity_deployment import (
    postprocess,
    predict_directional_maps_with_challenger,
    probability_to_logit,
)
from utils.affinity_graph import (
    reconstruct_affinity_components,
    regularize_affinity_components,
)
from utils.config import load_config, project_path
from utils.post_process import classify_instance_partition
from utils.semantic_challenger import build_semantic_challenger
from visualize_instances import visualize_instance_map


DEFAULT_IMAGES = (
    "test_003,test_009,test_019,test_024,test_026,"
    "test_044,test_045,test_048,test_052,test_055"
)


def _vote_options(infer_cfg):
    return {
        "core_fraction": float(infer_cfg.get("semantic_vote_core_fraction", 0.40)),
        "core_min_pixels": int(infer_cfg.get("semantic_vote_core_min_pixels", 8)),
        "core_distance_power": float(
            infer_cfg.get("semantic_vote_core_distance_power", 2.0)
        ),
        "color_uncertain_low": float(
            infer_cfg.get("semantic_vote_color_uncertain_low", 0.35)
        ),
        "color_uncertain_high": float(
            infer_cfg.get("semantic_vote_color_uncertain_high", 0.65)
        ),
        "color_weight": float(infer_cfg.get("semantic_vote_color_weight", 0.25)),
        "color_min_separation": float(
            infer_cfg.get("semantic_vote_color_min_separation", 1.0)
        ),
    }


def _save_partition(output_dir, stem, instance_map, class_map, audit):
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{stem}_inst.png"), instance_map)
    class_json = {str(key): int(value) for key, value in class_map.items()}
    (output_dir / f"{stem}_class.json").write_text(
        json.dumps(class_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact_audit = {key: value for key, value in audit.items() if key != "votes"}
    (output_dir / f"{stem}_graph_audit.json").write_text(
        json.dumps(compact_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    color = visualize_instance_map(instance_map, class_json)
    cv2.imwrite(str(output_dir / f"{stem}_inst_color.png"), color)


def _thresholds(short_threshold, distance2_threshold, distance4_threshold):
    return [
        float(short_threshold),
        float(short_threshold),
        float(short_threshold),
        float(short_threshold),
        float(distance2_threshold),
        float(distance2_threshold),
        float(distance4_threshold),
        float(distance4_threshold),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml",
    )
    parser.add_argument("--output-dir", default="outputs/experiments/e10a_graph_ab")
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--short-thresholds", default="0.30,0.35,0.40")
    parser.add_argument("--distance2-threshold", type=float, default=0.55)
    parser.add_argument("--distance4-threshold", type=float, default=0.65)
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["affinity_geometry_g1"]
    infer_cfg = config["inference"]
    deployment = config["affinity_deployment"]
    device = torch.device(
        "cuda"
        if config["sam2"].get("device") == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    system, reference_path, _, _ = build_system(config, cfg, device)
    checkpoint_path = Path(project_path(config, deployment["checkpoint"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_geometry_checkpoint_state(system, checkpoint)
    challenger, challenger_metadata = build_semantic_challenger(
        config, system, reference_path, device
    )
    if challenger is None:
        raise RuntimeError("This experiment requires the E10a semantic challenger")
    system.eval()

    test_dir = Path(project_path(config, config["inference"]["test_dir"]))
    all_images = str(args.images).strip().lower() == "all"
    requested = (
        None
        if all_images
        else {item.strip() for item in args.images.split(",") if item.strip()}
    )
    image_paths = [
        Path(path) for path in list_images(str(test_dir))
        if requested is None or Path(path).stem in requested
    ]
    if requested is not None:
        missing = sorted(requested - {path.stem for path in image_paths})
        if missing:
            raise FileNotFoundError(f"requested images missing: {missing}")

    output_root = Path(project_path(config, args.output_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    fusion_kwargs = {
        "distance2_weight": float(deployment.get("distance2_weight", 0.50)),
        "distance4_weight": float(deployment.get("distance4_weight", 0.25)),
        "support_threshold": float(deployment.get("support_threshold", 0.20)),
        "support_temperature": float(deployment.get("support_temperature", 0.05)),
        "short_reduction": str(deployment.get("short_reduction", "mean")),
        "short_softmax_temperature": float(
            deployment.get("short_softmax_temperature", 0.15)
        ),
    }
    graph_arms = {
        f"graph_short{value:.2f}": _thresholds(
            value, args.distance2_threshold, args.distance4_threshold
        )
        for value in [float(item) for item in args.short_thresholds.split(",")]
    }
    summary = {"e10a_watershed": [], **{name: [] for name in graph_arms}}

    for image_path in image_paths:
        started = time.time()
        image, semantic_logits, affinity_grid, boundary_native = (
            predict_directional_maps_with_challenger(
                system,
                challenger,
                image_path,
                int(cfg.get("input_size", 1024)),
                device,
                str(deployment.get("fusion_mode", "gated")),
                fusion_kwargs,
            )
        )
        stem = image_path.stem
        semantic_probability = torch.sigmoid(semantic_logits)[0, 0].numpy()

        # Strict single-semantic baseline: E10a drives both watershed terrain
        # and class voting.  No V6 semantic logits are passed to postprocess.
        watershed_output = torch.cat(
            [semantic_logits, probability_to_logit(boundary_native)], dim=1
        )
        watershed_dir = output_root / "e10a_watershed"
        _, watershed_map, watershed_classes = postprocess(
            watershed_output,
            image.shape[:2],
            watershed_dir,
            stem,
            infer_cfg,
            float(config["inference"]["boundary_threshold"]),
            True,
            image_rgb=image,
            semantic_challenger_logits=None,
        )
        cv2.imwrite(
            str(watershed_dir / f"{stem}_inst_color.png"),
            visualize_instance_map(
                watershed_map,
                {str(key): value for key, value in watershed_classes.items()},
            ),
        )
        summary["e10a_watershed"].append({
            "image": stem,
            "instances": len(watershed_classes),
        })

        affinity = affinity_grid[0].numpy()
        graph_shape = affinity.shape[1:]
        foreground = np.ones(graph_shape, dtype=bool)
        for arm_name, thresholds in graph_arms.items():
            raw_grid, graph_audit = reconstruct_affinity_components(
                foreground,
                affinity,
                threshold=thresholds,
                max_instances=None,
            )
            # Translate the native output area floor to the decoder grid.
            # Low-affinity boundary-belt pixels become an unassigned seam and
            # are then uniquely completed from retained affinity cores.
            grid_min_area = max(
                1,
                int(np.ceil(
                    float(infer_cfg.get("min_instance_area", 50))
                    * float(np.prod(graph_shape))
                    / float(image.shape[0] * image.shape[1])
                )),
            )
            core_grid, regularize_audit = regularize_affinity_components(
                raw_grid,
                min_component_area=grid_min_area,
            )
            native_regions = cv2.resize(
                core_grid.astype(np.int32),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            instance_map, class_map, finish_audit = classify_instance_partition(
                native_regions,
                semantic_probability,
                min_area=int(infer_cfg.get("min_instance_area", 50)),
                max_instance_id=int(infer_cfg.get("max_instance_id", 255)),
                semantic_vote_mode=str(
                    infer_cfg.get("semantic_vote_mode", "probability_mean")
                ),
                semantic_vote_threshold=float(
                    infer_cfg.get("semantic_vote_threshold", 0.5)
                ),
                semantic_vote_erode_width=int(
                    infer_cfg.get("semantic_vote_erode_width", 0)
                ),
                semantic_vote_options=_vote_options(infer_cfg),
            )
            arm_dir = output_root / arm_name
            audit = {
                "relation_thresholds": thresholds,
                "graph_grid": list(graph_shape),
                "graph_min_component_area": grid_min_area,
                **graph_audit,
                **{f"regularized_{key}": value for key, value in regularize_audit.items()},
                **finish_audit,
            }
            _save_partition(arm_dir, stem, instance_map, class_map, audit)
            summary[arm_name].append({
                "image": stem,
                "instances": len(class_map),
                "raw_components": int(graph_audit["raw_component_count"]),
                "unassigned_pixels": int(finish_audit["unassigned_pixels"]),
            })
        elapsed = time.time() - started
        print(f"{stem}: {elapsed:.1f}s", flush=True)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "definition": "GT-free E10a-only semantic geometry A/B",
        "important_threshold_note": (
            "graph thresholds are same-instance affinity thresholds; boundary "
            "threshold 0.65 corresponds approximately to affinity 0.35"
        ),
        "semantic_source": challenger_metadata,
        "affinity_checkpoint": str(checkpoint_path),
        "graph_arms": graph_arms,
        "distance2_threshold": args.distance2_threshold,
        "distance4_threshold": args.distance4_threshold,
        "summary": summary,
    }
    (output_root / "graph_ab_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
