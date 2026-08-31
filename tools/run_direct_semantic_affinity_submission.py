# -*- coding: utf-8 -*-
"""Submission inference for the direct SSL semantic-affinity challenger."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.mim_dataset import list_images
from models.direct_semantic_affinity import load_direct_semantic_affinity_model
from train_offset_geometry import file_sha256
from utils.affinity_deployment import (
    crop_affinity_boundary_output,
    crop_letterbox_output,
    postprocess,
    prepare_image,
    probability_to_logit,
)
from utils.config import load_config, project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/train/direct_ssl_semantic_affinity.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/direct_ssl_semantic_affinity/best_direct_dual.pth",
    )
    parser.add_argument("--test-dir")
    parser.add_argument(
        "--output-dir", default="outputs/submission_direct_ssl_semantic_affinity"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["direct_semantic_affinity"]
    infer_cfg = config["inference"]
    deployment = cfg["deployment_validation"]
    device = torch.device(
        "cuda"
        if config["sam2"].get("device") == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    checkpoint_path = Path(project_path(config, args.checkpoint))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model, checkpoint = load_direct_semantic_affinity_model(
        checkpoint_path, config, device
    )
    test_dir = Path(
        project_path(config, args.test_dir or infer_cfg.get("test_dir", "data/test"))
    )
    output_dir = Path(project_path(config, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    input_size = int(cfg.get("input_size", 1024))
    fusion_mode = str(deployment.get("fusion_mode", "gated"))
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

    results = []
    for image_name in list_images(str(test_dir)):
        image_path = Path(image_name)
        started = time.time()
        image, tensor, pad_h, pad_w = prepare_image(
            image_path, input_size, device
        )
        with torch.no_grad():
            outputs = model(tensor)
        semantic_native = crop_letterbox_output(
            outputs["semantic_logits"],
            input_size,
            pad_h,
            pad_w,
            image.shape[:2],
        ).cpu()
        boundary_native = crop_affinity_boundary_output(
            outputs,
            input_size,
            pad_h,
            pad_w,
            image.shape[:2],
            fusion_mode,
            fusion_kwargs,
        ).cpu()
        postprocess_output = torch.cat(
            [semantic_native, probability_to_logit(boundary_native)], dim=1
        )
        _, instance_map, class_map = postprocess(
            postprocess_output,
            image.shape[:2],
            output_dir,
            image_path.stem,
            infer_cfg,
            float(deployment.get("boundary_threshold", 0.65)),
            bool(config.get("post_process", {}).get("save_visualization", False)),
            image_rgb=image,
        )
        max_instance_id = int(instance_map.max())
        if max_instance_id > int(infer_cfg.get("max_instance_id", 255)):
            raise RuntimeError(
                f"{image_path.name} instance id {max_instance_id} exceeds limit"
            )
        results.append(
            {
                "image": image_path.name,
                "instances": len(class_map),
                "ferrite": sum(int(value) == 1 for value in class_map.values()),
                "pearlite": sum(int(value) == 0 for value in class_map.values()),
                "max_instance_id": max_instance_id,
                "seconds": time.time() - started,
            }
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "format": "direct_semantic_affinity_submission_v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_format": checkpoint["format"],
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_phase": checkpoint.get("phase"),
        "best_deployment_score": checkpoint.get("best_deployment_score"),
        "parameter_summary": model.parameter_summary(),
        "semantic_source": "direct_semantic_only",
        "affinity_source": "direct_affinity_only",
        "fusion": {"mode": fusion_mode, **fusion_kwargs},
        "boundary_threshold": float(deployment.get("boundary_threshold", 0.65)),
        "images": results,
    }
    (output_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "images": len(results),
                "instances": sum(item["instances"] for item in results),
                "checkpoint": str(checkpoint_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
