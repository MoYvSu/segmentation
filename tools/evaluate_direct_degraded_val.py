# -*- coding: utf-8 -*-
"""在固定、标签保持的退化视图上评估 direct 双头 checkpoint。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from data.dataset import letterbox
from data.direct_dual_head_dataset import (
    load_direct_evaluation_target,
    mask_prediction_to_evaluation_domain,
    semantic_from_instance_classes,
)
from models.direct_semantic_affinity import load_direct_semantic_affinity_model
from utils.affinity_deployment import (
    crop_affinity_boundary_output,
    crop_letterbox_output,
    postprocess,
    probability_to_logit,
)
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair, summarize_instance_results
from utils.quality_aware import assess_image_quality
from visualize_instances import visualize_instance_map


VARIANTS = ("clean", "defocus_v1", "low_light_v1", "defocus_low_light_v1")
_SAFE_ALIAS = re.compile(r"^[A-Za-z0-9_.-]+$")


def apply_fixed_degradation(image_rgb: np.ndarray, variant: str) -> np.ndarray:
    """生成确定性 uint8 RGB 压力测试视图，不修改输入数组。"""
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("degraded-val expects uint8 RGB image with three channels")
    if variant not in VARIANTS:
        raise ValueError(f"unsupported degraded-val variant: {variant}")
    output = image_rgb.copy()
    if variant in {"defocus_v1", "defocus_low_light_v1"}:
        output = cv2.GaussianBlur(
            output,
            (9, 9),
            sigmaX=1.6,
            sigmaY=1.6,
            borderType=cv2.BORDER_REFLECT_101,
        )
    if variant in {"low_light_v1", "defocus_low_light_v1"}:
        values = output.astype(np.float32) / 255.0
        values = np.clip(0.70 * np.power(values, 1.20), 0.0, 1.0)
        output = np.rint(values * 255.0).astype(np.uint8)
    return output


def _prepare_rgb(image_rgb: np.ndarray, image_size: int, device):
    image_lb, _, pad_h, pad_w = letterbox(image_rgb, image_size)
    tensor = (
        torch.from_numpy(image_lb)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device)
        / 255.0
    )
    return tensor, int(pad_h), int(pad_w)


def _parse_checkpoints(values, config):
    checkpoints = []
    aliases = set()
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint must use ALIAS=PATH")
        alias, raw_path = value.split("=", 1)
        alias = alias.strip()
        if not _SAFE_ALIAS.fullmatch(alias) or alias in aliases:
            raise ValueError(f"invalid or duplicated checkpoint alias: {alias!r}")
        path = Path(project_path(config, raw_path.strip()))
        if not path.is_file():
            raise FileNotFoundError(path)
        aliases.add(alias)
        checkpoints.append((alias, path))
    return checkpoints


def _resolve_output_dir(config, override=None):
    root = Path(config["paths"]["project_root"]).resolve()
    allowed = (root / "outputs" / "analysis").resolve()
    raw = override or config.get("degraded_validation", {}).get(
        "output_dir", "outputs/analysis/direct_degraded_val_v1"
    )
    target = Path(project_path(config, raw)).resolve()
    try:
        target.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(
            f"degraded-val output must stay inside {allowed}, got {target}"
        ) from exc
    return target


def _checkpoint_val_names(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    split = payload.get("split")
    if not isinstance(split, dict) or not split.get("val"):
        raise RuntimeError(f"checkpoint has no fixed validation split: {path}")
    return tuple(str(name) for name in split["val"])


def _resolve_val_paths(config, names):
    raw_dir = Path(project_path(config, config["paths"]["raw_data_dir"]))
    paths_by_stem = {
        path.stem: path
        for path in raw_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    }
    missing = [name for name in names if Path(name).stem not in paths_by_stem]
    if missing:
        raise FileNotFoundError(f"validation images not found: {missing}")
    return [paths_by_stem[Path(name).stem] for name in names]


def _semantic_counts(
    probability: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
):
    prediction = probability >= 0.5
    valid = (
        np.ones(target.shape, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if valid.shape != target.shape:
        raise ValueError(f"semantic valid {valid.shape} != target {target.shape}")
    counts = []
    for class_id in (0, 1):
        pred_class = (prediction == bool(class_id)) & valid
        target_class = (target == class_id) & valid
        counts.append(
            (
                int(np.logical_and(pred_class, target_class).sum()),
                int(np.logical_or(pred_class, target_class).sum()),
            )
        )
    return counts


def _save_panel(path, image_rgb, semantic, boundary, instance_map, class_map):
    max_width = 640
    scale = min(1.0, max_width / float(image_rgb.shape[1]))
    size = (
        max(1, int(round(image_rgb.shape[1] * scale))),
        max(1, int(round(image_rgb.shape[0] * scale))),
    )
    input_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    semantic_bgr = cv2.applyColorMap(
        np.clip(semantic * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    boundary_bgr = cv2.applyColorMap(
        np.clip(boundary * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT
    )
    instance_bgr = visualize_instance_map(instance_map, class_map)
    cells = [
        cv2.resize(cell, size, interpolation=cv2.INTER_AREA)
        for cell in (input_bgr, semantic_bgr, boundary_bgr, instance_bgr)
    ]
    top = np.concatenate(cells[:2], axis=1)
    bottom = np.concatenate(cells[2:], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.concatenate([top, bottom], axis=0)):
        raise RuntimeError(f"failed to write degraded-val panel: {path}")


@torch.no_grad()
def _evaluate_checkpoint(model, image_paths, config, variants, output_dir, save_panels):
    direct_cfg = config["direct_semantic_affinity"]
    deploy = direct_cfg["deployment_validation"]
    infer_cfg = config["inference"]
    input_size = int(direct_cfg.get("input_size", 1024))
    gt_dir = Path(project_path(config, config["boundary"]["gt_dir"]))
    manual_target_cfg = direct_cfg.get("manual_target", {})
    manual_target_dir = (
        Path(project_path(config, manual_target_cfg["dataset_dir"]))
        if bool(manual_target_cfg.get("enabled", False))
        else None
    )
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
    summaries = {}
    model.eval()
    for variant in variants:
        rows = []
        per_image = []
        semantic_intersection = [0, 0]
        semantic_union = [0, 0]
        quality_rows = []
        with tempfile.TemporaryDirectory(prefix="direct-degraded-val-") as temp_dir:
            for image_path in image_paths:
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(image_path)
                original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image_rgb = apply_fixed_degradation(original_rgb, variant)
                tensor, pad_h, pad_w = _prepare_rgb(image_rgb, input_size, model.encoder.device)
                output = model(tensor)
                semantic_native = crop_letterbox_output(
                    output["semantic_logits"],
                    input_size,
                    pad_h,
                    pad_w,
                    image_rgb.shape[:2],
                ).cpu()
                boundary_native = crop_affinity_boundary_output(
                    {"affinity_logits": output["affinity_logits"]},
                    input_size,
                    pad_h,
                    pad_w,
                    image_rgb.shape[:2],
                    str(deploy.get("fusion_mode", "gated")),
                    fusion_kwargs,
                ).cpu()
                watershed_output = torch.cat(
                    [semantic_native, probability_to_logit(boundary_native)], dim=1
                )
                _, pred_map, pred_classes = postprocess(
                    watershed_output,
                    image_rgb.shape[:2],
                    temp_dir,
                    image_path.stem,
                    infer_cfg,
                    float(deploy.get("boundary_threshold", 0.59)),
                    False,
                    image_rgb=image_rgb,
                )
                gt_map, gt_classes, gt_valid, gt_audit = (
                    load_direct_evaluation_target(
                        image_path,
                        image_rgb.shape[:2],
                        manual_target_dir=manual_target_dir,
                    )
                )
                instance_row = evaluate_instance_pair(
                    gt_map,
                    gt_classes,
                    mask_prediction_to_evaluation_domain(pred_map, gt_valid),
                    pred_classes,
                )
                rows.append(instance_row)

                semantic = torch.sigmoid(semantic_native)[0, 0].numpy()
                if manual_target_dir is not None:
                    semantic_gt, _ = semantic_from_instance_classes(
                        gt_map, gt_classes
                    )
                else:
                    gt_path = gt_dir / f"{image_path.stem}_gt.npz"
                    with np.load(gt_path) as gt_payload:
                        semantic_gt = gt_payload["semantic"]
                counts = _semantic_counts(semantic, semantic_gt, gt_valid)
                for class_id, (intersection, union) in enumerate(counts):
                    semantic_intersection[class_id] += intersection
                    semantic_union[class_id] += union
                quality = assess_image_quality(image_rgb)
                quality_rows.append(quality)
                per_image.append(
                    {
                        "image": image_path.name,
                        "score_total": instance_row["score_total"],
                        "instance_miou_valid": instance_row["instance_miou_valid"],
                        "pred_count": instance_row["pred_count"],
                        "valid_matches": instance_row["valid_matches"],
                        "quality": quality,
                        "ground_truth_source": gt_audit["source"],
                    }
                )
                if save_panels:
                    _save_panel(
                        output_dir / variant / f"{image_path.stem}_panel.jpg",
                        image_rgb,
                        semantic,
                        boundary_native[0, 0].numpy(),
                        pred_map,
                        pred_classes,
                    )
        summary = summarize_instance_results(rows)
        semantic_ious = [
            semantic_intersection[index] / max(1, semantic_union[index])
            for index in (0, 1)
        ]
        summary.update(
            {
                "semantic_iou_pearlite": semantic_ious[0],
                "semantic_iou_ferrite": semantic_ious[1],
                "semantic_miou": sum(semantic_ious) / 2.0,
                "input_brightness_mean": float(
                    np.mean([row["brightness"] for row in quality_rows])
                ),
                "input_contrast_mean": float(
                    np.mean([row["contrast"] for row in quality_rows])
                ),
                "input_sharpness_mean": float(
                    np.mean([row["sharpness"] for row in quality_rows])
                ),
                "per_image": per_image,
                "ground_truth_source": per_image[0]["ground_truth_source"],
            }
        )
        summaries[variant] = summary
    clean = summaries["clean"]
    delta_keys = (
        "score_total",
        "instance_miou_valid",
        "ferrite_area_relative_error",
        "semantic_miou",
        "pred_count",
        "valid_matches",
    )
    for variant, summary in summaries.items():
        summary["delta_from_clean"] = {
            key: float(summary[key]) - float(clean[key]) for key in delta_keys
        }
    return summaries


def _summary_rows(alias, summaries):
    for variant, summary in summaries.items():
        yield {
            "checkpoint": alias,
            "variant": variant,
            "score_total": summary["score_total"],
            "instance_miou_valid": summary["instance_miou_valid"],
            "ferrite_area_relative_error": summary["ferrite_area_relative_error"],
            "semantic_miou": summary["semantic_miou"],
            "pred_count": summary["pred_count"],
            "valid_matches": summary["valid_matches"],
            "input_brightness_mean": summary["input_brightness_mean"],
            "input_contrast_mean": summary["input_contrast_mean"],
            "input_sharpness_mean": summary["input_sharpness_mean"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, help="ALIAS=PATH，可重复"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--save-panels", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = _resolve_output_dir(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = _parse_checkpoints(args.checkpoint, config)
    variants = tuple(
        config.get("degraded_validation", {}).get("variants", VARIANTS)
    )
    if not variants or variants[0] != "clean" or any(v not in VARIANTS for v in variants):
        raise ValueError(f"degraded_validation.variants must start with clean: {variants}")
    splits = {alias: _checkpoint_val_names(path) for alias, path in checkpoints}
    expected_split = next(iter(splits.values()))
    if any(split != expected_split for split in splits.values()):
        raise RuntimeError(f"checkpoint validation splits differ: {splits}")
    image_paths = _resolve_val_paths(config, expected_split)
    requested = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    report = {
        "config": str(Path(args.config).resolve()),
        "variants": list(variants),
        "validation_images": [path.name for path in image_paths],
        "checkpoints": {},
    }
    csv_rows = []
    for alias, checkpoint_path in checkpoints:
        model, payload = load_direct_semantic_affinity_model(
            checkpoint_path, config, requested
        )
        summaries = _evaluate_checkpoint(
            model,
            image_paths,
            config,
            variants,
            output_dir / alias,
            args.save_panels,
        )
        report["checkpoints"][alias] = {
            "path": str(checkpoint_path.resolve()),
            "epoch": payload.get("epoch"),
            "input_normalization": model.encoder.input_normalization,
            "summaries": summaries,
        }
        csv_rows.extend(_summary_rows(alias, summaries))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"output_dir": str(output_dir), "rows": csv_rows}, indent=2))


if __name__ == "__main__":
    main()
