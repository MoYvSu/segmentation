# -*- coding: utf-8 -*-
"""Export SAM2 automatic-mask candidates as editable LabelMe polygons.

The exported label is deliberately ``sam2_candidate``: generic SAM2 proposes
geometry but cannot determine ferrite/pearlite phase.  A human must inspect and
rename accepted shapes to ``0``/``1`` before these files become training labels.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def select_non_overlapping_masks(
    records: Sequence[Dict],
    image_shape: Sequence[int],
    *,
    min_area: int,
    max_area_fraction: float,
    max_overlap_fraction: float,
    max_masks: int,
) -> List[Dict]:
    image_area = int(image_shape[0] * image_shape[1])
    candidates = []
    for record in records:
        area = int(record.get("area", 0))
        if area < min_area or area > image_area * max_area_fraction:
            continue
        score = float(record.get("predicted_iou", 0.0)) * float(
            record.get("stability_score", 0.0)
        )
        candidates.append((score, area, record))
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    occupied = np.zeros(tuple(image_shape[:2]), dtype=bool)
    accepted: List[Dict] = []
    for _, area, record in candidates:
        mask = np.asarray(record["segmentation"], dtype=bool)
        overlap = int(np.logical_and(mask, occupied).sum()) / max(area, 1)
        if overlap > max_overlap_fraction:
            continue
        accepted.append(record)
        occupied |= mask
        if len(accepted) >= max_masks:
            break
    return accepted


def mask_to_polygons(mask: np.ndarray, min_contour_area: float, epsilon: float):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        tolerance = max(0.5, epsilon * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, tolerance, True).reshape(-1, 2)
        if len(approx) >= 3:
            polygons.append([[float(x), float(y)] for x, y in approx])
    return polygons


def records_to_labelme(
    records: Sequence[Dict],
    image_path: str,
    image_shape: Sequence[int],
    *,
    label: str = "sam2_candidate",
    min_contour_area: float = 100.0,
    epsilon: float = 0.002,
) -> Dict:
    shapes = []
    for group_id, record in enumerate(records):
        polygons = mask_to_polygons(
            np.asarray(record["segmentation"]), min_contour_area, epsilon
        )
        for polygon in polygons:
            shapes.append(
                {
                    "label": label,
                    "points": polygon,
                    "group_id": group_id,
                    "description": (
                        f"SAM2 candidate; predicted_iou={float(record.get('predicted_iou', 0.0)):.4f}; "
                        f"stability={float(record.get('stability_score', 0.0)):.4f}; "
                        f"area={int(record.get('area', 0))}"
                    ),
                    "shape_type": "polygon",
                    "flags": {"sam2_generated": True, "human_verified": False},
                    "mask": None,
                }
            )
    return {
        "version": "6.3.1",
        "flags": {
            "sam2_generated": True,
            "human_verified": False,
            "training_use_forbidden_until_verified": True,
        },
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": int(image_shape[0]),
        "imageWidth": int(image_shape[1]),
    }


def save_overlay(image: np.ndarray, records: Sequence[Dict], path: Path):
    overlay = image.copy()
    rng = np.random.default_rng(42)
    for record in records:
        mask = np.asarray(record["segmentation"], dtype=bool)
        color = rng.integers(40, 256, size=3, dtype=np.uint8)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def parse_indices(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="SAM2 automatic masks -> LabelMe")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam2-repo", default="segment-anything-2")
    parser.add_argument("--config-file", default="configs/sam2/sam2_hiera_b+.yaml")
    parser.add_argument("--checkpoint", default="weights/sam2_hiera_base_plus.pt")
    parser.add_argument("--indices", default="0,250,500,750")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.86)
    parser.add_argument("--stability-thresh", type=float, default=0.92)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--max-area-fraction", type=float, default=0.80)
    parser.add_argument("--max-overlap-fraction", type=float, default=0.30)
    parser.add_argument("--max-masks", type=int, default=255)
    args = parser.parse_args()

    repo = str(Path(args.sam2_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    from data.mim_dataset import list_images

    images = list_images(args.input_dir)
    indices = parse_indices(args.indices)
    selected = [images[index] for index in indices]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(
        config_file=args.config_file,
        ckpt_path=str(Path(args.checkpoint).resolve()),
        device=device,
        mode="eval",
        apply_postprocessing=True,
    )
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        min_mask_region_area=args.min_area,
        output_mode="binary_mask",
        use_m2m=True,
    )

    summary = []
    for path in selected:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        amp_enabled = device == "cuda" and torch.cuda.is_bf16_supported()
        with torch.inference_mode(), torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            records = generator.generate(image)
        accepted = select_non_overlapping_masks(
            records,
            image.shape,
            min_area=args.min_area,
            max_area_fraction=args.max_area_fraction,
            max_overlap_fraction=args.max_overlap_fraction,
            max_masks=args.max_masks,
        )
        destination_image = output_dir / Path(path).name
        shutil.copy2(path, destination_image)
        document = records_to_labelme(
            accepted, destination_image.name, image.shape
        )
        json_path = output_dir / f"{Path(path).stem}.json"
        json_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        overlay_path = output_dir / f"{Path(path).stem}_sam2_overlay.jpg"
        save_overlay(image, accepted, overlay_path)
        summary.append(
            {
                "image": destination_image.name,
                "raw_masks": len(records),
                "accepted_masks": len(accepted),
                "labelme_shapes": len(document["shapes"]),
            }
        )
        print(summary[-1])
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
