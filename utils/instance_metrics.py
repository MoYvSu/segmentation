# -*- coding: utf-8 -*-
"""Competition-oriented class-aware instance metrics.

The training metrics in :mod:`utils.metrics` are pixel semantic metrics.  This
module deliberately stays independent from them and evaluates the submitted
``uint8`` instance map plus its instance-to-class mapping.

The public rule text averages IoU only over one-to-one, same-class matches with
IoU >= 0.5.  That number is reported as ``instance_miou_valid``.  Diagnostic
variants that assign zero to unmatched instances are reported as well, because
the valid-match-only average can otherwise hide severe over/under-segmentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


CLASS_PEARLITE = 0
CLASS_FERRITE = 1
CLASS_NAMES = {CLASS_PEARLITE: "pearlite", CLASS_FERRITE: "ferrite"}
VALID_LABELS = {
    "0": CLASS_PEARLITE,
    "pearlite": CLASS_PEARLITE,
    "珠光体": CLASS_PEARLITE,
    "1": CLASS_FERRITE,
    "ferrite": CLASS_FERRITE,
    "ferrite_core": CLASS_FERRITE,
    "铁素体": CLASS_FERRITE,
}


def normalize_class_id(value) -> int:
    """Normalize JSON/LabelMe class values to 0=pearlite, 1=ferrite."""
    if isinstance(value, (int, np.integer)) and int(value) in CLASS_NAMES:
        return int(value)
    key = str(value).strip().lower()
    if key not in VALID_LABELS:
        raise ValueError(f"Unsupported instance class: {value!r}")
    return VALID_LABELS[key]


def load_class_map(path: str | Path) -> Dict[int, int]:
    """Load a submission-style ``instance id -> class`` JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    result: Dict[int, int] = {}
    for key, value in raw.items():
        instance_id = int(key)
        if not 1 <= instance_id <= 255:
            raise ValueError(f"Instance id outside [1, 255]: {instance_id}")
        result[instance_id] = normalize_class_id(value)
    return result


def load_labelme_instances(
    json_path: str | Path,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Dict[int, int], Dict[str, int]]:
    """Rasterize each labeled polygon as one ground-truth instance.

    LabelMe polygons can share their one-pixel rasterized outline.  To keep the
    result deterministic, earlier polygons retain already assigned pixels and
    later polygons receive only previously unassigned pixels.  The number of
    overlap pixels is returned as an audit field.
    """
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if image_shape is None:
        height = int(data.get("imageHeight") or 0)
        width = int(data.get("imageWidth") or 0)
        if height <= 0 or width <= 0:
            raise ValueError(
                f"LabelMe dimensions missing in {json_path}; pass image_shape"
            )
    else:
        height, width = map(int, image_shape)

    instance_map = np.zeros((height, width), dtype=np.int32)
    class_map: Dict[int, int] = {}
    overlap_pixels = 0
    skipped_shapes = 0

    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        if label not in VALID_LABELS:
            continue
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            skipped_shapes += 1
            continue

        polygon = np.round(points).astype(np.int32).reshape((-1, 1, 2))
        shape_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(shape_mask, [polygon], 1)
        occupied = (shape_mask > 0) & (instance_map > 0)
        overlap_pixels += int(occupied.sum())
        paint = (shape_mask > 0) & (instance_map == 0)
        if not np.any(paint):
            skipped_shapes += 1
            continue

        instance_id = len(class_map) + 1
        instance_map[paint] = instance_id
        class_map[instance_id] = VALID_LABELS[label]

    audit = {
        "polygon_count": len(class_map),
        "overlap_pixels": overlap_pixels,
        "skipped_shapes": skipped_shapes,
        "uncovered_pixels": int((instance_map == 0).sum()),
    }
    return instance_map, class_map, audit


def validate_instance_prediction(
    instance_map: np.ndarray,
    class_map: Mapping[int, int],
) -> None:
    """Validate the competition's 8-bit instance-map contract."""
    if instance_map.ndim != 2:
        raise ValueError(f"Instance map must be 2-D, got {instance_map.shape}")
    if not np.issubdtype(instance_map.dtype, np.integer):
        raise ValueError(f"Instance map must be integer, got {instance_map.dtype}")
    if int(instance_map.min()) < 0 or int(instance_map.max()) > 255:
        raise ValueError("Instance map values must stay within [0, 255]")
    present = {int(value) for value in np.unique(instance_map) if int(value) != 0}
    declared = {int(key) for key in class_map}
    missing = sorted(present - declared)
    if missing:
        raise ValueError(f"Prediction has instance ids missing from class JSON: {missing}")
    for value in class_map.values():
        normalize_class_id(value)


def _class_iou_matrix(
    gt_map: np.ndarray,
    gt_ids: Sequence[int],
    pred_map: np.ndarray,
    pred_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return IoU/intersection matrices and per-instance areas for one class."""
    n_gt, n_pred = len(gt_ids), len(pred_ids)
    gt_areas = np.zeros(n_gt, dtype=np.int64)
    pred_areas = np.zeros(n_pred, dtype=np.int64)
    intersections = np.zeros((n_gt, n_pred), dtype=np.int64)
    if n_gt == 0 and n_pred == 0:
        return intersections.astype(np.float64), intersections, gt_areas, pred_areas

    gt_lut = np.full(int(gt_map.max()) + 1, -1, dtype=np.int32)
    pred_lut = np.full(int(pred_map.max()) + 1, -1, dtype=np.int32)
    for index, instance_id in enumerate(gt_ids):
        gt_lut[int(instance_id)] = index
    for index, instance_id in enumerate(pred_ids):
        pred_lut[int(instance_id)] = index

    gt_index = gt_lut[gt_map.astype(np.int64, copy=False)]
    pred_index = pred_lut[pred_map.astype(np.int64, copy=False)]
    gt_valid = gt_index >= 0
    pred_valid = pred_index >= 0
    if n_gt:
        gt_areas = np.bincount(gt_index[gt_valid], minlength=n_gt).astype(np.int64)
    if n_pred:
        pred_areas = np.bincount(pred_index[pred_valid], minlength=n_pred).astype(np.int64)
    both = gt_valid & pred_valid
    if n_gt and n_pred and np.any(both):
        flat_pairs = gt_index[both] * n_pred + pred_index[both]
        intersections = np.bincount(
            flat_pairs, minlength=n_gt * n_pred
        ).reshape(n_gt, n_pred).astype(np.int64)

    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    return iou, intersections, gt_areas, pred_areas


def _safe_ratio(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def evaluate_instance_pair(
    gt_map: np.ndarray,
    gt_class_map: Mapping[int, int],
    pred_map: np.ndarray,
    pred_class_map: Mapping[int, int],
    iou_threshold: float = 0.5,
    topology_overlap_threshold: float = 0.10,
) -> Dict:
    """Evaluate one prediction against one class-aware instance ground truth."""
    if gt_map.shape != pred_map.shape:
        raise ValueError(f"Shape mismatch: GT {gt_map.shape}, pred {pred_map.shape}")
    validate_instance_prediction(pred_map, pred_class_map)

    class_results: Dict[str, Dict] = {}
    all_match_ious: List[float] = []
    total_gt = 0
    total_pred = 0
    ferrite_gt_area_sum = 0
    ferrite_pred_area_sum = 0
    ferrite_gt_count = 0
    ferrite_pred_count = 0

    for class_id, class_name in CLASS_NAMES.items():
        gt_ids = sorted(
            int(instance_id) for instance_id, value in gt_class_map.items()
            if normalize_class_id(value) == class_id and np.any(gt_map == int(instance_id))
        )
        pred_ids = sorted(
            int(instance_id) for instance_id, value in pred_class_map.items()
            if normalize_class_id(value) == class_id and np.any(pred_map == int(instance_id))
        )
        iou, intersections, gt_areas, pred_areas = _class_iou_matrix(
            gt_map, gt_ids, pred_map, pred_ids
        )

        matches: List[Tuple[int, int, float]] = []
        if iou.size:
            gt_assignment, pred_assignment = linear_sum_assignment(-iou)
            for gt_index, pred_index in zip(gt_assignment, pred_assignment):
                value = float(iou[gt_index, pred_index])
                if value + 1e-12 >= float(iou_threshold):
                    matches.append((gt_ids[gt_index], pred_ids[pred_index], value))

        match_ious = [item[2] for item in matches]
        all_match_ious.extend(match_ious)
        gt_count, pred_count = len(gt_ids), len(pred_ids)
        total_gt += gt_count
        total_pred += pred_count

        if class_id == CLASS_FERRITE:
            ferrite_gt_area_sum += int(gt_areas.sum())
            ferrite_pred_area_sum += int(pred_areas.sum())
            ferrite_gt_count += gt_count
            ferrite_pred_count += pred_count

        gt_overlap_fraction = np.divide(
            intersections,
            gt_areas[:, None],
            out=np.zeros_like(intersections, dtype=np.float64),
            where=gt_areas[:, None] > 0,
        ) if gt_count and pred_count else np.zeros((gt_count, pred_count))
        pred_overlap_fraction = np.divide(
            intersections,
            pred_areas[None, :],
            out=np.zeros_like(intersections, dtype=np.float64),
            where=pred_areas[None, :] > 0,
        ) if gt_count and pred_count else np.zeros((gt_count, pred_count))
        split_gt_count = int(
            np.sum((gt_overlap_fraction >= topology_overlap_threshold).sum(axis=1) >= 2)
        ) if gt_count else 0
        merged_pred_count = int(
            np.sum((pred_overlap_fraction >= topology_overlap_threshold).sum(axis=0) >= 2)
        ) if pred_count else 0

        iou_sum = float(sum(match_ious))
        valid_count = len(matches)
        class_results[class_name] = {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "valid_matches": valid_count,
            "unmatched_gt": gt_count - valid_count,
            "unmatched_pred": pred_count - valid_count,
            "instance_miou_valid": _safe_ratio(iou_sum, valid_count),
            "gt_penalized_miou": _safe_ratio(iou_sum, gt_count),
            "symmetric_penalized_miou": _safe_ratio(iou_sum, max(gt_count, pred_count)),
            "valid_match_recall": _safe_ratio(valid_count, gt_count),
            "valid_match_precision": _safe_ratio(valid_count, pred_count),
            "split_gt_count": split_gt_count,
            "merged_pred_count": merged_pred_count,
            "gt_area_sum": int(gt_areas.sum()),
            "pred_area_sum": int(pred_areas.sum()),
            "matches": [
                {"gt_id": gt_id, "pred_id": pred_id, "iou": value}
                for gt_id, pred_id, value in matches
            ],
        }

    valid_matches = len(all_match_ious)
    iou_sum = float(sum(all_match_ious))
    valid_miou = _safe_ratio(iou_sum, valid_matches)
    mean_area_gt = _safe_ratio(ferrite_gt_area_sum, ferrite_gt_count)
    mean_area_pred = _safe_ratio(ferrite_pred_area_sum, ferrite_pred_count)
    if mean_area_gt > 0:
        area_relative_error = abs(mean_area_pred - mean_area_gt) / mean_area_gt
    elif mean_area_pred == 0:
        area_relative_error = 0.0
    else:
        area_relative_error = float("inf")

    return {
        "gt_count": total_gt,
        "pred_count": total_pred,
        "valid_matches": valid_matches,
        "instance_miou_valid": valid_miou,
        "gt_penalized_miou": _safe_ratio(iou_sum, total_gt),
        "symmetric_penalized_miou": _safe_ratio(iou_sum, max(total_gt, total_pred)),
        "score_miou": 50.0 * valid_miou,
        "ferrite_gt_count": ferrite_gt_count,
        "ferrite_pred_count": ferrite_pred_count,
        "ferrite_gt_area_sum": ferrite_gt_area_sum,
        "ferrite_pred_area_sum": ferrite_pred_area_sum,
        "ferrite_mean_area_gt": mean_area_gt,
        "ferrite_mean_area_pred": mean_area_pred,
        "ferrite_area_relative_error": area_relative_error,
        "score_area": 50.0 * max(0.0, 1.0 - area_relative_error),
        "score_total": 50.0 * valid_miou + 50.0 * max(0.0, 1.0 - area_relative_error),
        "classes": class_results,
    }


def summarize_instance_results(results: Iterable[Mapping]) -> Dict:
    """Aggregate per-image results using dataset-level instance/area totals."""
    rows = list(results)
    if not rows:
        raise ValueError("No instance-evaluation results to summarize")

    total_gt = sum(int(row["gt_count"]) for row in rows)
    total_pred = sum(int(row["pred_count"]) for row in rows)
    valid_matches = sum(int(row["valid_matches"]) for row in rows)
    iou_sum = sum(
        float(class_result["instance_miou_valid"])
        * int(class_result["valid_matches"])
        for row in rows
        for class_result in row["classes"].values()
    )
    valid_miou = _safe_ratio(iou_sum, valid_matches)

    ferrite_gt_count = sum(int(row["ferrite_gt_count"]) for row in rows)
    ferrite_pred_count = sum(int(row["ferrite_pred_count"]) for row in rows)
    ferrite_gt_area_sum = sum(int(row["ferrite_gt_area_sum"]) for row in rows)
    ferrite_pred_area_sum = sum(int(row["ferrite_pred_area_sum"]) for row in rows)
    mean_area_gt = _safe_ratio(ferrite_gt_area_sum, ferrite_gt_count)
    mean_area_pred = _safe_ratio(ferrite_pred_area_sum, ferrite_pred_count)
    if mean_area_gt > 0:
        area_relative_error = abs(mean_area_pred - mean_area_gt) / mean_area_gt
    elif mean_area_pred == 0:
        area_relative_error = 0.0
    else:
        area_relative_error = float("inf")

    class_summary = {}
    for class_name in CLASS_NAMES.values():
        gt_count = sum(int(row["classes"][class_name]["gt_count"]) for row in rows)
        pred_count = sum(int(row["classes"][class_name]["pred_count"]) for row in rows)
        matches = sum(int(row["classes"][class_name]["valid_matches"]) for row in rows)
        class_iou_sum = sum(
            float(row["classes"][class_name]["instance_miou_valid"])
            * int(row["classes"][class_name]["valid_matches"])
            for row in rows
        )
        class_summary[class_name] = {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "valid_matches": matches,
            "unmatched_gt": gt_count - matches,
            "unmatched_pred": pred_count - matches,
            "instance_miou_valid": _safe_ratio(class_iou_sum, matches),
            "gt_penalized_miou": _safe_ratio(class_iou_sum, gt_count),
            "symmetric_penalized_miou": _safe_ratio(
                class_iou_sum, max(gt_count, pred_count)
            ),
            "valid_match_recall": _safe_ratio(matches, gt_count),
            "valid_match_precision": _safe_ratio(matches, pred_count),
            "split_gt_count": sum(
                int(row["classes"][class_name]["split_gt_count"]) for row in rows
            ),
            "merged_pred_count": sum(
                int(row["classes"][class_name]["merged_pred_count"]) for row in rows
            ),
        }

    macro_score = float(np.mean([float(row["score_total"]) for row in rows]))
    return {
        "num_images": len(rows),
        "gt_count": total_gt,
        "pred_count": total_pred,
        "valid_matches": valid_matches,
        "instance_miou_valid": valid_miou,
        "gt_penalized_miou": _safe_ratio(iou_sum, total_gt),
        "symmetric_penalized_miou": _safe_ratio(iou_sum, max(total_gt, total_pred)),
        "score_miou": 50.0 * valid_miou,
        "ferrite_gt_count": ferrite_gt_count,
        "ferrite_pred_count": ferrite_pred_count,
        "ferrite_mean_area_gt": mean_area_gt,
        "ferrite_mean_area_pred": mean_area_pred,
        "ferrite_area_relative_error": area_relative_error,
        "score_area": 50.0 * max(0.0, 1.0 - area_relative_error),
        "score_total": 50.0 * valid_miou + 50.0 * max(0.0, 1.0 - area_relative_error),
        "macro_image_score_total": macro_score,
        "classes": class_summary,
    }
