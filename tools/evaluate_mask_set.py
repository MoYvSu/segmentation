# -*- coding: utf-8 -*-
"""Evaluate mask-set predictions on the checkpoint's fixed manual-GT split."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.direct_dual_head_dataset import load_manual_target_archive, mask_prediction_to_evaluation_domain
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair, summarize_instance_results, validate_instance_prediction
from utils.mask_set_inference import infer_mask_set_image


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def resolve_evaluation_paths(config, payload, image_dir=None, names=None):
    """Keep checkpoint split membership authoritative unless an override is explicit."""
    split = payload.get("split")
    if not isinstance(split, dict) or not split.get("val"):
        raise ValueError("checkpoint must carry its fixed train/val split")
    split_file = config.get("mask_set", {}).get("split_file")
    if split_file:
        configured = json.loads(Path(project_path(config, split_file)).read_text(encoding="utf-8"))
        for part in ("train", "val"):
            current = {Path(name).stem for name in configured.get(part, [])}
            saved = {Path(name).stem for name in split.get(part, [])}
            if current != saved:
                raise ValueError(f"config {part} split differs from the checkpoint split")
    directory = Path(project_path(config, image_dir or config["paths"]["raw_data_dir"]))
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    by_stem = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in extensions:
            if path.stem in by_stem:
                raise ValueError(f"ambiguous image stem {path.stem} in {directory}")
            by_stem[path.stem] = path
    selected = [Path(name).stem for name in names] if names else (
        list(by_stem) if image_dir is not None else [Path(name).stem for name in split["val"]]
    )
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate evaluation image names")
    missing = [stem for stem in selected if stem not in by_stem]
    if missing:
        raise FileNotFoundError(f"evaluation images missing: {missing}")
    if not selected:
        raise ValueError("no evaluation images selected")
    return [by_stem[stem] for stem in selected], bool(image_dir is not None or names), split


def _phase_preview(instances, classes, valid=None):
    # 固定类别配色使 GT/预测可直接对照，所有实例的分区线仍单独保留。
    lut = np.full((int(instances.max()) + 1, 3), 70, dtype=np.uint8)
    for instance_id, class_id in classes.items():
        if int(instance_id) < len(lut):
            lut[int(instance_id)] = (172, 125, 218) if int(class_id) == 0 else (109, 211, 137)
    image = lut[instances]
    known = np.ones(instances.shape, dtype=bool) if valid is None else valid.astype(bool)
    edge = np.zeros(instances.shape, dtype=bool)
    edge[:-1] |= (instances[:-1] != instances[1:]) & known[:-1] & known[1:]
    edge[:, :-1] |= (instances[:, :-1] != instances[:, 1:]) & known[:, :-1] & known[:, 1:]
    image[edge] = 0
    image[~known] = 70
    return image


def save_preview(path, rgb, prediction, classes, gt=None, gt_classes=None, valid=None):
    width = min(640, rgb.shape[1])
    shape = (width, max(1, round(rgb.shape[0] * width / rgb.shape[1])))
    panels = [(rgb, "RGB"),
              (_phase_preview(gt, gt_classes, valid) if gt is not None else rgb,
               "GT; unknown=gray" if gt is not None else "No GT: visual only"),
              (_phase_preview(prediction, classes), "Mask-set prediction")]
    rendered = []
    for values, label in panels:
        canvas = np.full((shape[1] + 48, width, 3), 255, dtype=np.uint8)
        canvas[48:] = cv2.resize(values, shape, interpolation=cv2.INTER_NEAREST)
        cv2.putText(canvas, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "purple=pearlite green=ferrite; black=edges", (8, 39),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        rendered.append(canvas)
    image = cv2.cvtColor(np.concatenate(rendered, axis=1), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to save preview: {path}")


def evaluate_model(model, payload, config, output_dir, device, image_dir=None, names=None):
    """Save complete native outputs; mask unknown only inside the scoring domain."""
    paths, override, split = resolve_evaluation_paths(config, payload, image_dir, names)
    target_dir = Path(project_path(config, config["mask_set"]["manual_target_dir"]))
    output_dir = Path(output_dir)
    prediction_dir, visual_dir = output_dir / "predictions", output_dir / "visual"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    rows, scores = [], []
    preferred = {"train_172", "train_873", "test_026"}
    previews = {path.stem for path in paths if path.stem in preferred}
    if not previews:
        previews = {path.stem for path in paths[:2]}
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances, classes, diagnostics = infer_mask_set_image(model, rgb, config, device)
        validate_instance_prediction(instances, classes)
        if instances.dtype != np.uint16 or instances.shape != rgb.shape[:2]:
            raise ValueError("mask-set output must be a native-size uint16 instance map")
        instance_path = prediction_dir / f"{path.stem}_inst.png"
        if not cv2.imwrite(str(instance_path), instances):
            raise OSError(f"failed to save instances: {instance_path}")
        class_path = prediction_dir / f"{path.stem}_class.json"
        class_path.write_text(json.dumps(classes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target_path = target_dir / f"{path.stem}_gt.npz"
        row = {"stem": path.stem, "image_path": str(path), "has_gt": target_path.is_file(),
               "instance_png": str(instance_path.relative_to(output_dir)),
               "class_json": str(class_path.relative_to(output_dir)), "diagnostics": diagnostics}
        gt = gt_classes = valid = None
        if target_path.is_file():
            gt, gt_classes, _, valid, provenance = load_manual_target_archive(target_path)
            if gt.shape != instances.shape:
                raise ValueError(f"GT shape mismatch for {path.name}")
            scoring_prediction = mask_prediction_to_evaluation_domain(instances, valid)
            metrics = evaluate_instance_pair(gt, gt_classes, scoring_prediction, classes)
            scores.append(metrics)
            row["metrics"] = metrics
            row["ignored_unknown_pixels"] = int(provenance["residual_unknown"].sum())
        elif not override:
            raise FileNotFoundError(f"fixed validation GT missing: {target_path}")
        if path.stem in previews:
            preview_path = visual_dir / f"{path.stem}_preview.png"
            save_preview(preview_path, rgb, instances, classes, gt, gt_classes, valid)
            row["preview"] = str(preview_path.relative_to(output_dir))
        rows.append(row)
        print(f"{path.stem}: {len(classes)} instances, has_gt={row['has_gt']}", flush=True)
    report = {
        "format": "mask_set_evaluation_v1", "scope": "explicit_analysis_override" if override else "checkpoint_fixed_validation",
        "protocol": "Repository competition proxy; same-class instance matching and ferrite mean area. Saved PNGs retain every predicted pixel; remaining GT unknown is masked only for scoring. Both phases are independent instance classes.",
        "checkpoint_epoch": payload.get("epoch"), "checkpoint_phase": payload.get("phase"),
        "checkpoint_split": split, "evaluated_names": [path.stem for path in paths],
        "inference": config["mask_set"].get("inference", {}),
        "aggregate": summarize_instance_results(scores) if scores else None, "images": rows,
    }
    return _json_safe(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-dir", help="Explicit alternate image directory; without --names evaluate all its images.")
    parser.add_argument("--names", nargs="+", help="Explicit analysis subset; bypasses default checkpoint val selection.")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = project_path(config, args.checkpoint)
    output_dir = Path(project_path(config, args.output_dir))
    from models.mask_set import load_mask_set_model
    model, payload = load_mask_set_model(checkpoint, config, torch.device(args.device))
    report = evaluate_model(model, payload, config, output_dir, torch.device(args.device), args.image_dir, args.names)
    report.update(config_path=str(Path(args.config).resolve()), checkpoint=str(checkpoint))
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
