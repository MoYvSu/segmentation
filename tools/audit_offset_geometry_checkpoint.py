# -*- coding: utf-8 -*-
"""Audit a trained center-offset geometry checkpoint without retraining."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.offset_geometry_dataset import OffsetGeometryDataset
from inference import build_model as build_reference_model
from models.offset_geometry import (
    CenterOffsetGeometryDecoder,
    FrozenSemanticGeometrySystem,
    semantic_state_digest,
)
from utils.center_guided_instances import (
    assign_endpoints_to_centers,
    extract_center_peaks,
)
from utils.config import load_config, project_path
from utils.flow_instances import cluster_endpoints
from utils.instance_metrics import evaluate_instance_pair


def colorize_instances(instance_map: np.ndarray) -> np.ndarray:
    output = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(instance_map):
        value = int(instance_id)
        if value == 0:
            continue
        output[instance_map == value] = (
            (37 * value + 53) % 255,
            (97 * value + 29) % 255,
            (17 * value + 193) % 255,
        )
    return output


def instance_metrics(gt_instances: np.ndarray, predicted: np.ndarray):
    gt_ids = [int(value) for value in np.unique(gt_instances) if int(value) != 0]
    pred_ids = [int(value) for value in np.unique(predicted) if int(value) != 0]
    metrics = evaluate_instance_pair(
        gt_instances, {value: 0 for value in gt_ids},
        predicted, {value: 0 for value in pred_ids},
    )
    return {
        "pred_instance_count": len(pred_ids),
        "instance_miou_valid": float(metrics["instance_miou_valid"]),
        "gt_penalized_miou": float(metrics["gt_penalized_miou"]),
    }


def parse_radii(value: str):
    result = []
    for item in value.split(","):
        item = item.strip().lower()
        result.append(None if item in {"none", "unlimited"} else float(item))
    return result


@torch.no_grad()
def audit_sample(system, sample, output_dir: Path, grid_size: int, radii):
    device = next(system.geometry_decoder.parameters()).device
    image = sample["image"].unsqueeze(0).to(device)
    prediction = system.geometry_forward(image)
    center_probability = torch.sigmoid(
        prediction["center_logits"]
    )[0, 0].cpu().numpy()
    offsets = prediction["offsets"][0].cpu().numpy() * float(grid_size)
    foreground = sample["foreground"][0].numpy().astype(bool)
    valid_content = sample["valid_content"][0].numpy().astype(bool)
    gt_instances = sample["instance_map"].numpy().astype(np.int32)
    center_target = sample["center_target"][0].numpy()
    content_height, content_width = map(int, sample["content_shape"].tolist())
    yy, xx = np.indices(foreground.shape, dtype=np.float32)
    endpoint_y = np.clip(yy + offsets[0], 0, grid_size - 1)
    endpoint_x = np.clip(xx + offsets[1], 0, grid_size - 1)
    centers, center_scores, peak_audit = extract_center_peaks(
        center_probability, valid_content,
        threshold=0.25, nms_radius=3, max_centers=255,
    )
    gt_centers = np.argwhere(center_target >= 1.0 - 1e-6)
    exact_matches = len(
        {tuple(value) for value in centers.tolist()}
        & {tuple(value) for value in gt_centers.tolist()}
    )

    sample_dir = output_dir / Path(sample["image_name"]).stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_rgb = np.clip(
        sample["image"].permute(1, 2, 0).numpy() * 255.0, 0, 255
    ).astype(np.uint8)
    input_content_height = int(round(content_height * image_rgb.shape[0] / grid_size))
    input_content_width = int(round(content_width * image_rgb.shape[1] / grid_size))
    input_boundary = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if input_content_height < input_boundary.shape[0]:
        cv2.line(
            input_boundary, (0, input_content_height - 1),
            (input_content_width - 1, input_content_height - 1), (0, 0, 255), 2,
        )
    cv2.imwrite(str(sample_dir / "input_letterbox_boundary.png"), input_boundary)
    cv2.imwrite(
        str(sample_dir / "center_pred_full.png"),
        np.clip(center_probability * 255.0, 0, 255).astype(np.uint8),
    )
    cv2.imwrite(
        str(sample_dir / "center_pred_content.png"),
        np.clip(
            center_probability[:content_height, :content_width] * 255.0, 0, 255
        ).astype(np.uint8),
    )
    cv2.imwrite(
        str(sample_dir / "center_gt_content.png"),
        np.clip(
            center_target[:content_height, :content_width] * 255.0, 0, 255
        ).astype(np.uint8),
    )
    cv2.imwrite(
        str(sample_dir / "instances_gt_content.png"),
        colorize_instances(gt_instances[:content_height, :content_width]),
    )

    baseline, baseline_audit = cluster_endpoints(
        endpoint_y, endpoint_x, foreground,
        close_radius=1, min_instance_area=1, max_instances=255,
    )
    cv2.imwrite(
        str(sample_dir / "instances_endpoint_components_content.png"),
        colorize_instances(baseline[:content_height, :content_width]),
    )
    spatial_voronoi, spatial_voronoi_audit = assign_endpoints_to_centers(
        yy, xx, foreground, centers,
        max_assignment_distance=None, min_instance_area=1,
    )
    target_offsets = sample["offset_target"].numpy() * float(grid_size)
    oracle_offset, oracle_offset_audit = assign_endpoints_to_centers(
        yy + target_offsets[0], xx + target_offsets[1], foreground, centers,
        max_assignment_distance=None, min_instance_area=1,
    )
    cv2.imwrite(
        str(sample_dir / "instances_spatial_voronoi_content.png"),
        colorize_instances(spatial_voronoi[:content_height, :content_width]),
    )
    cv2.imwrite(
        str(sample_dir / "instances_oracle_offset_content.png"),
        colorize_instances(oracle_offset[:content_height, :content_width]),
    )
    result = {
        "image": sample["image_name"],
        "foreground_source": "ground_truth_conditional_geometry_audit",
        "gt_instance_count": int(sample["instance_count"]),
        "predicted_center_count": int(len(centers)),
        "exact_center_matches": exact_matches,
        "mean_center_score": float(center_scores.mean()) if center_scores.size else 0.0,
        "offset_mae_grid_pixels": float(
            np.abs(offsets - sample["offset_target"].numpy() * grid_size)
            [:, foreground].mean()
        ),
        "peak_audit": peak_audit,
        "endpoint_components": {
            **instance_metrics(gt_instances, baseline),
            "audit": baseline_audit,
        },
        "spatial_voronoi_without_offset": {
            **instance_metrics(gt_instances, spatial_voronoi),
            "audit": spatial_voronoi_audit,
        },
        "oracle_offset_upper_bound": {
            **instance_metrics(gt_instances, oracle_offset),
            "audit": oracle_offset_audit,
        },
        "center_guided": {},
    }
    for radius in radii:
        label = "unlimited" if radius is None else f"r{radius:g}"
        guided, guided_audit = assign_endpoints_to_centers(
            endpoint_y, endpoint_x, foreground, centers,
            max_assignment_distance=radius, min_instance_area=1,
        )
        cv2.imwrite(
            str(sample_dir / f"instances_center_guided_{label}_content.png"),
            colorize_instances(guided[:content_height, :content_width]),
        )
        result["center_guided"][label] = {
            **instance_metrics(gt_instances, guided),
            "audit": guided_audit,
        }
    (sample_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/offset_geometry_g0.yaml")
    parser.add_argument(
        "--checkpoint", default="outputs/offset_geometry_g0/geometry_step_0400.pth"
    )
    parser.add_argument(
        "--output-dir", default="outputs/offset_geometry_g0/audit_g05_step_0400"
    )
    parser.add_argument("--radii", default="4,6,8,12,none")
    args = parser.parse_args()
    config = load_config(args.config)
    geometry_cfg = config["offset_geometry"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["sam2"].get("device") == "cuda"
        else "cpu"
    )
    reference_path = project_path(config, geometry_cfg["reference_checkpoint"])
    reference_model = build_reference_model(config, device, reference_path)
    geometry_decoder = CenterOffsetGeometryDecoder(
        in_channels=reference_model.encoder.get_stage_channels(),
        fpn_channels=int(geometry_cfg.get("fpn_channels", 256)),
        up_channels=int(geometry_cfg.get("up_channels", 128)),
        output_grid=int(geometry_cfg.get("output_grid", 512)),
    ).to(device)
    checkpoint_path = project_path(config, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    geometry_decoder.load_state_dict(checkpoint["geometry_state_dict"], strict=True)
    system = FrozenSemanticGeometrySystem(reference_model, geometry_decoder).to(device)
    current_digest = semantic_state_digest(reference_model)
    expected_digest = checkpoint.get("semantic_state_digest")
    if expected_digest and current_digest != expected_digest:
        raise RuntimeError("V6 semantic digest differs from geometry checkpoint")
    dataset = OffsetGeometryDataset(
        project_path(config, config["paths"]["raw_data_dir"]),
        sample_names=geometry_cfg.get("sample_names"),
        image_size=int(geometry_cfg.get("input_size", 1024)),
        output_grid=int(geometry_cfg.get("output_grid", 512)),
        center_sigma_scale=float(geometry_cfg.get("center_sigma_scale", 0.12)),
        center_min_sigma=float(geometry_cfg.get("center_min_sigma", 2.0)),
        center_max_sigma=float(geometry_cfg.get("center_max_sigma", 8.0)),
        cache_in_memory=True,
    )
    output_dir = Path(project_path(config, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    radii = parse_radii(args.radii)
    results = [
        audit_sample(
            system, dataset[index], output_dir,
            int(geometry_cfg.get("output_grid", 512)), radii,
        )
        for index in range(len(dataset))
    ]
    summary = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "semantic_digest": current_digest,
        "samples": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    print(f"audit written to {output_dir}")


if __name__ == "__main__":
    main()
