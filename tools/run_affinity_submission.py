# -*- coding: utf-8 -*-
"""Single-command, GT-free affinity submission inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.mim_dataset import list_images
from train_affinity_geometry_g1 import (
    build_system,
    load_geometry_checkpoint_state,
    validate_semantic_geometry_contract,
)
from train_offset_geometry import file_sha256
from utils.affinity_deployment import (
    postprocess,
    predict_maps,
    predict_maps_with_challenger,
)
from utils.config import load_config, project_path
from utils.semantic_challenger import build_semantic_challenger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/inference/final_affinity_g4b.yaml",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["affinity_geometry_g1"]
    deployment = config["affinity_deployment"]
    checkpoint_path = Path(project_path(
        config, args.checkpoint or deployment["checkpoint"]
    ))
    test_dir = Path(project_path(
        config, args.test_dir or config["inference"]["test_dir"]
    ))
    output_dir = Path(project_path(
        config, args.output_dir or config["inference"]["output_dir"]
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if config["sam2"].get("device") == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    system, reference_path, _, digest = build_system(config, cfg, device)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    load_geometry_checkpoint_state(system, checkpoint)
    semantic_contract = validate_semantic_geometry_contract(
        config, cfg, checkpoint, reference_path, digest
    )
    semantic_challenger, challenger_metadata = build_semantic_challenger(
        config, system, reference_path, device
    )
    replace_reference_semantic = bool(
        config.get("semantic_challenger", {}).get(
            "replace_reference_semantic", False
        )
    )
    system.eval()
    fusion_mode = str(deployment.get("fusion_mode", "gated"))
    fusion_kwargs = {
        "distance2_weight": float(deployment.get("distance2_weight", 0.50)),
        "distance4_weight": float(deployment.get("distance4_weight", 0.25)),
        "support_threshold": float(deployment.get("support_threshold", 0.20)),
        "support_temperature": float(
            deployment.get("support_temperature", 0.05)
        ),
        "short_reduction": str(deployment.get("short_reduction", "mean")),
        "short_softmax_temperature": float(
            deployment.get("short_softmax_temperature", 0.15)
        ),
    }
    threshold = float(config["inference"]["boundary_threshold"])
    save_visualization = bool(
        config.get("post_process", {}).get("save_visualization", True)
    )
    results = []
    for image_name in list_images(str(test_dir)):
        image_path = Path(image_name)
        started = time.time()
        challenger_logits = None
        if semantic_challenger is None:
            image, _, affinity_output, _ = predict_maps(
                system,
                image_path,
                int(cfg.get("input_size", 1024)),
                device,
                fusion_mode,
                fusion_kwargs,
            )
        else:
            image, _, affinity_output, _, challenger_logits = (
                predict_maps_with_challenger(
                    system,
                    semantic_challenger,
                    image_path,
                    int(cfg.get("input_size", 1024)),
                    device,
                    fusion_mode,
                    fusion_kwargs,
                    replace_reference_semantic=replace_reference_semantic,
                )
            )
        _, instance_map, class_map = postprocess(
            affinity_output,
            image.shape[:2],
            output_dir,
            image_path.stem,
            config["inference"],
            threshold,
            save_visualization,
            image_rgb=image,
            semantic_challenger_logits=challenger_logits,
        )
        results.append({
            "image": image_path.name,
            "instances": int(len(class_map)),
            "ferrite": int(sum(int(value) == 1 for value in class_map.values())),
            "pearlite": int(sum(int(value) == 0 for value in class_map.values())),
            "max_instance_id": int(instance_map.max()),
            "seconds": time.time() - started,
        })
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "format": "affinity_submission_v1",
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "reference_checkpoint": os.path.abspath(reference_path),
        "reference_checkpoint_sha256": file_sha256(reference_path),
        "semantic_state_digest": digest,
        "semantic_geometry_contract": semantic_contract,
        "semantic_challenger": challenger_metadata,
        "semantic_source": (
            "challenger_only" if replace_reference_semantic else "reference_with_challenger_vote"
        ),
        "fusion": {"mode": fusion_mode, **fusion_kwargs},
        "inference": {
            "boundary_threshold": threshold,
            "marker_border_seal_width": int(
                config["inference"].get("marker_border_seal_width", 0)
            ),
            "marker_boundary_low_threshold": config["inference"].get(
                "marker_boundary_low_threshold"
            ),
            "marker_boundary_reconstruction_steps": int(
                config["inference"].get(
                    "marker_boundary_reconstruction_steps", 0
                )
            ),
            "semantic_vote_mode": config["inference"].get(
                "semantic_vote_mode", "hard_majority"
            ),
            "semantic_vote": {
                key: config["inference"].get(key, default)
                for key, default in {
                    "semantic_vote_threshold": 0.5,
                    "semantic_vote_core_fraction": 0.40,
                    "semantic_vote_core_min_pixels": 8,
                    "semantic_vote_core_distance_power": 2.0,
                    "semantic_vote_color_uncertain_low": 0.35,
                    "semantic_vote_color_uncertain_high": 0.65,
                    "semantic_vote_color_weight": 0.25,
                    "semantic_vote_color_min_separation": 1.0,
                    "semantic_vote_dual_p2f_base_min": 0.35,
                    "semantic_vote_dual_p2f_candidate_min": 0.85,
                    "semantic_vote_dual_f2p_base_max": 0.65,
                    "semantic_vote_dual_f2p_candidate_max": 0.15,
                    "semantic_vote_dual_p2f_min_core_gain": 0.08,
                }.items()
            },
            "max_instance_id": int(
                config["inference"].get("max_instance_id", 255)
            ),
        },
        "images": results,
    }
    (output_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "images": len(results),
        "instances": sum(row["instances"] for row in results),
        "checkpoint_epoch": manifest["checkpoint_epoch"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
