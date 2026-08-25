# -*- coding: utf-8 -*-
"""Build a class-agnostic SAM2 pseudo-instance geometry dataset.

The output is deliberately separate from LabelMe phase labels.  It may only
supervise geometry tasks; ferrite/pearlite class targets remain ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enforce_source_limit(
    source_hashes: Iterable[str],
    existing_hashes: Iterable[str] = (),
    *,
    limit: int = 249,
) -> int:
    combined = set(existing_hashes) | set(source_hashes)
    if len(combined) > int(limit):
        raise ValueError(
            f"pseudo-instance sources={len(combined)} exceeds hard limit={limit}"
        )
    return len(combined)


def partition_mask_records(
    records: Sequence[Dict],
    image_shape: Sequence[int],
    *,
    min_area: int,
    max_area_fraction: float,
    max_overlap_fraction: float,
    max_instances: int = 255,
):
    """Convert overlapping SAM2 proposals into one unambiguous instance map."""
    height, width = int(image_shape[0]), int(image_shape[1])
    image_area = height * width
    candidates = []
    rejection_counts = {
        "area": 0,
        "overlap": 0,
        "empty_after_partition": 0,
    }
    for record in records:
        mask = np.asarray(record["segmentation"], dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(f"mask shape {mask.shape} != image {(height, width)}")
        area = int(mask.sum())
        if area < min_area or area > image_area * max_area_fraction:
            rejection_counts["area"] += 1
            continue
        score = float(record.get("predicted_iou", 0.0)) * float(
            record.get("stability_score", 0.0)
        )
        candidates.append((score, area, record, mask))
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    occupied = np.zeros((height, width), dtype=bool)
    accepted = []
    for score, original_area, record, mask in candidates:
        overlap = int(np.logical_and(mask, occupied).sum()) / max(original_area, 1)
        if overlap > max_overlap_fraction:
            rejection_counts["overlap"] += 1
            continue
        unique_mask = mask & ~occupied
        unique_area = int(unique_mask.sum())
        if unique_area < min_area:
            rejection_counts["empty_after_partition"] += 1
            continue
        accepted.append(
            {
                "mask": unique_mask,
                "score": score,
                "predicted_iou": float(record.get("predicted_iou", 0.0)),
                "stability_score": float(record.get("stability_score", 0.0)),
                "original_area": original_area,
                "partition_area": unique_area,
            }
        )
        occupied |= unique_mask

    if len(accepted) > int(max_instances):
        raise ValueError(
            f"accepted SAM2 instances={len(accepted)} exceeds per-image cap="
            f"{max_instances}; refusing silent truncation"
        )

    instance_map = np.zeros((height, width), dtype=np.uint16)
    scores = np.zeros(len(accepted) + 1, dtype=np.float32)
    stability = np.zeros(len(accepted) + 1, dtype=np.float32)
    for instance_id, item in enumerate(accepted, start=1):
        instance_map[item["mask"]] = instance_id
        scores[instance_id] = item["predicted_iou"]
        stability[instance_id] = item["stability_score"]
    return instance_map, scores, stability, rejection_counts


def read_manifest(path: Path) -> List[Dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_indices(value: str) -> List[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate image indices are not allowed")
    return indices


def save_overlay(image: np.ndarray, instance_map: np.ndarray, path: Path):
    overlay = image.copy()
    rng = np.random.default_rng(42)
    for instance_id in range(1, int(instance_map.max()) + 1):
        mask = instance_map == instance_id
        color = rng.integers(40, 256, size=3, dtype=np.uint8)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--sam2-repo", default="segment-anything-2")
    parser.add_argument("--config-file", default="configs/sam2/sam2_hiera_b+.yaml")
    parser.add_argument("--checkpoint", default="weights/sam2_hiera_base_plus.pt")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.86)
    parser.add_argument("--stability-thresh", type=float, default=0.92)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--max-area-fraction", type=float, default=0.80)
    parser.add_argument("--max-overlap-fraction", type=float, default=0.30)
    parser.add_argument("--max-instances", type=int, default=255)
    parser.add_argument("--max-source-images", type=int, default=249)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if "test" in {part.lower() for part in input_dir.parts}:
        raise ValueError("test images are forbidden for pseudo-instance generation")
    if args.max_source_images > 249:
        raise ValueError("max-source-images cannot exceed the compliance cap 249")

    from data.mim_dataset import list_images

    images = [Path(path).resolve() for path in list_images(str(input_dir))]
    indices = parse_indices(args.indices)
    if any(index < 0 or index >= len(images) for index in indices):
        raise IndexError(f"indices must be within [0, {len(images) - 1}]")
    selected = [images[index] for index in indices]
    source_hashes = [sha256_file(path) for path in selected]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("selected files contain duplicate image content")

    output_dir = Path(args.output_dir).resolve()
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    existing = read_manifest(manifest_path)
    existing_hashes = {row["source_sha256"] for row in existing}
    enforce_source_limit(
        source_hashes,
        existing_hashes,
        limit=args.max_source_images,
    )
    duplicates = existing_hashes & set(source_hashes)
    if duplicates:
        raise ValueError("one or more selected sources already exist in the manifest")

    repo = str(Path(args.sam2_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

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
    checkpoint_sha256 = sha256_file(Path(args.checkpoint).resolve())
    generation = {
        "config_file": args.config_file,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "points_per_side": args.points_per_side,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_thresh": args.stability_thresh,
        "min_area": args.min_area,
        "max_area_fraction": args.max_area_fraction,
        "max_overlap_fraction": args.max_overlap_fraction,
    }

    for index, image_path, source_hash in zip(indices, selected, source_hashes):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        amp_enabled = device == "cuda" and torch.cuda.is_bf16_supported()
        with torch.inference_mode(), torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            records = generator.generate(image)
        instance_map, scores, stability, rejected = partition_mask_records(
            records,
            image.shape,
            min_area=args.min_area,
            max_area_fraction=args.max_area_fraction,
            max_overlap_fraction=args.max_overlap_fraction,
            max_instances=args.max_instances,
        )
        mask_path = masks_dir / f"{image_path.stem}.npz"
        np.savez_compressed(
            mask_path,
            instance_map=instance_map,
            geometry_valid=(instance_map > 0),
            instance_predicted_iou=scores,
            instance_stability=stability,
        )
        overlay_path = overlays_dir / f"{image_path.stem}.jpg"
        save_overlay(image, instance_map, overlay_path)
        row = {
            "source_index": index,
            "source_file": str(image_path),
            "source_sha256": source_hash,
            "source_role": "competition_unlabeled_train",
            "class_label": None,
            "geometry_supervision": "positive_affinity_default; negative_edges_opt_in",
            "raw_mask_count": len(records),
            "accepted_instance_count": int(instance_map.max()),
            "covered_fraction": float((instance_map > 0).mean()),
            "rejection_counts": rejected,
            "mask_file": str(mask_path.relative_to(output_dir)),
            "overlay_file": str(overlay_path.relative_to(output_dir)),
            "generation": generation,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
