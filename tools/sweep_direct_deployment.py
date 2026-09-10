# -*- coding: utf-8 -*-
"""Small deployment-parameter sweep for a trained direct dual-head model."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.direct_dual_head_dataset import (
    DirectDualHeadDataset,
    load_direct_evaluation_target,
    mask_prediction_to_evaluation_domain,
    semantic_from_instance_classes,
)
from models.direct_semantic_affinity import load_direct_semantic_affinity_model
from utils.affinity_deployment import (
    crop_affinity_boundary_output,
    crop_letterbox_output,
    postprocess,
    prepare_image,
    probability_to_logit,
)
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair, summarize_instance_results


def parse_values(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("parameter sweep requires at least one value")
    return values


@torch.no_grad()
def cache_validation_outputs(model, image_paths, config, device):
    direct_cfg = config["direct_semantic_affinity"]
    deploy = direct_cfg["deployment_validation"]
    fusion_kwargs = {
        "distance2_weight": float(deploy.get("distance2_weight", 0.50)),
        "distance4_weight": float(deploy.get("distance4_weight", 0.25)),
        "support_threshold": float(deploy.get("support_threshold", 0.20)),
        "support_temperature": float(deploy.get("support_temperature", 0.05)),
        "short_reduction": str(deploy.get("short_reduction", "mean")),
        "short_softmax_temperature": float(
            deploy.get("short_softmax_temperature", 0.15)
        ),
    }
    image_size = int(direct_cfg.get("input_size", 1024))
    manual_target_cfg = direct_cfg.get("manual_target", {})
    manual_target_dir = (
        Path(project_path(config, manual_target_cfg["dataset_dir"]))
        if bool(manual_target_cfg.get("enabled", False))
        else None
    )
    model.eval()
    cached = []
    for image_path in image_paths:
        image, tensor, pad_h, pad_w = prepare_image(
            image_path, image_size, device
        )
        output = model(tensor)
        semantic_native = crop_letterbox_output(
            output["semantic_logits"],
            image_size,
            pad_h,
            pad_w,
            image.shape[:2],
        ).cpu()
        boundary_native = crop_affinity_boundary_output(
            {"affinity_logits": output["affinity_logits"]},
            image_size,
            pad_h,
            pad_w,
            image.shape[:2],
            str(deploy.get("fusion_mode", "gated")),
            fusion_kwargs,
        ).cpu()
        gt_map, gt_classes, gt_valid, gt_audit = load_direct_evaluation_target(
            image_path,
            image.shape[:2],
            manual_target_dir=manual_target_dir,
        )
        cached.append(
            {
                "image_path": Path(image_path),
                "image": image,
                "watershed_output": torch.cat(
                    [semantic_native, probability_to_logit(boundary_native)], dim=1
                ),
                "gt_map": gt_map,
                "gt_classes": gt_classes,
                "gt_valid": gt_valid,
                "gt_audit": gt_audit,
                "semantic_gt": semantic_from_instance_classes(
                    gt_map, gt_classes
                )[0],
            }
        )
    return cached


def evaluate_cached_semantic(cached, threshold=0.5):
    intersection = [0, 0]
    union = [0, 0]
    for item in cached:
        probability = torch.sigmoid(item["watershed_output"][:, :1])[0, 0].numpy()
        prediction = probability >= float(threshold)
        target = item["semantic_gt"]
        valid = item["gt_valid"]
        for class_id in (0, 1):
            pred_class = (prediction == bool(class_id)) & valid
            target_class = (target == class_id) & valid
            intersection[class_id] += int((pred_class & target_class).sum())
            union[class_id] += int((pred_class | target_class).sum())
    ious = [intersection[index] / max(1, union[index]) for index in (0, 1)]
    return {
        "pixel_threshold": float(threshold),
        "iou_pearlite": ious[0],
        "iou_ferrite": ious[1],
        "miou": sum(ious) / 2.0,
    }


def evaluate_cached(cached, config, boundary_threshold, semantic_threshold):
    infer_cfg = dict(config["inference"])
    infer_cfg["semantic_vote_threshold"] = float(semantic_threshold)
    rows = []
    with tempfile.TemporaryDirectory(prefix="direct-dual-sweep-") as temp_dir:
        for item in cached:
            _, pred_map, pred_classes = postprocess(
                item["watershed_output"],
                item["image"].shape[:2],
                temp_dir,
                item["image_path"].stem,
                infer_cfg,
                float(boundary_threshold),
                False,
                image_rgb=item["image"],
            )
            rows.append(
                evaluate_instance_pair(
                    item["gt_map"],
                    item["gt_classes"],
                    mask_prediction_to_evaluation_domain(
                        pred_map, item["gt_valid"]
                    ),
                    pred_classes,
                )
            )
    return summarize_instance_results(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/train/direct_ssl_semantic_affinity.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/direct_ssl_semantic_affinity/best_direct_dual.pth",
    )
    parser.add_argument("--boundary-thresholds", default="0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--semantic-thresholds", default="0.50")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(project_path(config, args.checkpoint))
    model, checkpoint = load_direct_semantic_affinity_model(
        checkpoint_path, config, device
    )
    direct_cfg = config["direct_semantic_affinity"]
    manual_target_cfg = direct_cfg.get("manual_target", {})
    manual_target_dir = (
        Path(project_path(config, manual_target_cfg["dataset_dir"]))
        if bool(manual_target_cfg.get("enabled", False))
        else None
    )
    dataset = DirectDualHeadDataset(
        project_path(config, config["paths"]["raw_data_dir"]),
        (
            None
            if manual_target_dir is not None
            else project_path(config, config["boundary"]["gt_dir"])
        ),
        sample_names=checkpoint["split"]["val"],
        image_size=int(direct_cfg.get("input_size", 1024)),
        affinity_grid=int(direct_cfg.get("affinity_grid", 512)),
        augment=False,
        manual_target_dir=manual_target_dir,
        manual_target_boundary_dilation=int(
            manual_target_cfg.get("boundary_dilation", 2)
        ),
    )
    image_paths = [path for path, _ in dataset.samples]
    cached = cache_validation_outputs(model, image_paths, config, device)
    rows = []
    for boundary_threshold in parse_values(args.boundary_thresholds):
        for semantic_threshold in parse_values(args.semantic_thresholds):
            metrics = evaluate_cached(
                cached,
                config,
                boundary_threshold,
                semantic_threshold,
            )
            row = {
                "boundary_threshold": boundary_threshold,
                "semantic_vote_threshold": semantic_threshold,
                "score_total": float(metrics["score_total"]),
                "instance_miou": float(metrics["instance_miou_valid"]),
                "gt_penalized_miou": float(metrics["gt_penalized_miou"]),
                "symmetric_penalized_miou": float(
                    metrics["symmetric_penalized_miou"]
                ),
                "ferrite_area_error": float(
                    metrics["ferrite_area_relative_error"]
                ),
                "ferrite_mean_area_gt": float(metrics["ferrite_mean_area_gt"]),
                "ferrite_mean_area_pred": float(metrics["ferrite_mean_area_pred"]),
                "gt_count": int(metrics["gt_count"]),
                "pred_count": int(metrics["pred_count"]),
                "valid_matches": int(metrics["valid_matches"]),
                "ferrite_pred_count": int(metrics["classes"]["ferrite"]["pred_count"]),
                "pearlite_pred_count": int(metrics["classes"]["pearlite"]["pred_count"]),
                "ferrite_valid_matches": int(
                    metrics["classes"]["ferrite"]["valid_matches"]
                ),
                "pearlite_valid_matches": int(
                    metrics["classes"]["pearlite"]["valid_matches"]
                ),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: row["score_total"], reverse=True)
    semantic_metrics = evaluate_cached_semantic(cached)
    output_path = Path(project_path(config, args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "validation_images": [path.name for path in image_paths],
                "ground_truth_source": cached[0]["gt_audit"]["source"],
                "ignored_unknown_pixels": sum(
                    int(item["gt_audit"].get("ignored_unknown_pixels", 0))
                    for item in cached
                ),
                "semantic_metrics": semantic_metrics,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("BEST " + json.dumps(rows[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
