# -*- coding: utf-8 -*-
"""Instance-level semantic voting with adaptive cores and optional Lab prior."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


CLASS_PEARLITE = 0
CLASS_FERRITE = 1


def adaptive_instance_core(
    instance_mask: np.ndarray,
    fraction: float = 0.40,
    min_pixels: int = 8,
    distance_power: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a shape-adaptive interior core and distance weights.

    Distance transform is restricted to the instance bounding box.  This is
    mathematically equivalent to a full-image transform with a zero exterior,
    while avoiding one megapixel-scale CPU transform per predicted instance.
    """
    mask = np.asarray(instance_mask, dtype=bool)
    area = int(mask.sum())
    core = np.zeros(mask.shape, dtype=bool)
    weights = np.zeros(mask.shape, dtype=np.float32)
    if area <= 0:
        return core, weights

    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    local_mask = mask[y0:y1, x0:x1]
    padded = np.pad(
        local_mask.astype(np.uint8), 1, mode="constant", constant_values=0
    )
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    values = distance[local_mask]
    fraction = float(np.clip(fraction, 0.05, 1.0))
    target = min(area, max(min(int(min_pixels), area), int(round(area * fraction)), 1))
    index = values.size - target
    cutoff = float(np.partition(values, index)[index])
    local_core = local_mask & (distance >= cutoff)
    if int(local_core.sum()) < target:
        flat_indices = np.flatnonzero(local_mask)
        selected = flat_indices[np.argsort(distance.flat[flat_indices])[-target:]]
        local_core = np.zeros(local_mask.shape, dtype=bool)
        local_core.flat[selected] = True

    local_weights = np.zeros(local_mask.shape, dtype=np.float32)
    maximum = float(distance[local_core].max()) if np.any(local_core) else 0.0
    if maximum <= 1.0e-6:
        local_weights[local_core] = 1.0
    else:
        normalized = np.clip(distance / maximum, 0.0, 1.0)
        local_weights[local_core] = np.maximum(
            np.power(normalized[local_core], float(distance_power)), 0.05
        )
    core[y0:y1, x0:x1] = local_core
    weights[y0:y1, x0:x1] = local_weights
    return core, weights


def robust_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    if values.size >= 8:
        low, high = np.quantile(values, [0.10, 0.90])
        values = np.clip(values, low, high)
    total = float(weights.sum())
    if total <= 1.0e-8:
        return float(values.mean())
    return float(np.sum(values * weights) / total)


def build_adaptive_lab_prior(image_rgb: np.ndarray) -> Dict:
    """Build a GT-free, per-image ferrite brightness prior."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must be HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        maximum = float(np.nanmax(image)) if image.size else 0.0
        image = np.clip(image * (255.0 if maximum <= 1.5 else 1.0), 0.0, 255.0)
        image = image.astype(np.uint8)
    lightness = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0]
    smooth = cv2.GaussianBlur(lightness, (0, 0), 1.0)
    threshold, _ = cv2.threshold(
        smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    low = smooth <= threshold
    high = ~low
    low_fraction = float(low.mean())
    high_fraction = float(high.mean())
    available = min(low_fraction, high_fraction) >= 0.05
    if available:
        low_values = smooth[low].astype(np.float32) / 255.0
        high_values = smooth[high].astype(np.float32) / 255.0
        low_mean = float(low_values.mean())
        high_mean = float(high_values.mean())
        pooled_std = float(np.sqrt(0.5 * (low_values.var() + high_values.var())))
        separation = (high_mean - low_mean) / max(pooled_std, 1.0e-3)
        transition = max(0.03, 0.25 * (high_mean - low_mean))
    else:
        low_mean = high_mean = float(smooth.mean() / 255.0)
        separation = 0.0
        transition = 0.05
    threshold_unit = float(threshold / 255.0)
    normalized = smooth.astype(np.float32) / 255.0
    ferrite_probability = 1.0 / (
        1.0 + np.exp(-np.clip((normalized - threshold_unit) / transition, -20.0, 20.0))
    )
    return {
        "available": bool(available),
        "threshold": threshold_unit,
        "separation": float(separation),
        "low_fraction": low_fraction,
        "high_fraction": high_fraction,
        "low_mean": low_mean,
        "high_mean": high_mean,
        "ferrite_probability": ferrite_probability.astype(np.float32),
    }


def instance_semantic_vote(
    instance_mask: np.ndarray,
    semantic_mask: np.ndarray,
    semantic_probability: Optional[np.ndarray] = None,
    candidate_semantic_probability: Optional[np.ndarray] = None,
    mode: str = "hard_majority",
    erode_width: int = 0,
    threshold: float = 0.5,
    lab_prior: Optional[Dict] = None,
    core_fraction: float = 0.40,
    core_min_pixels: int = 8,
    core_distance_power: float = 2.0,
    color_uncertain_low: float = 0.35,
    color_uncertain_high: float = 0.65,
    color_weight: float = 0.25,
    color_min_separation: float = 1.0,
    dual_p2f_base_min: float = 0.35,
    dual_p2f_candidate_min: float = 0.85,
    dual_f2p_base_max: float = 0.65,
    dual_f2p_candidate_max: float = 0.15,
    dual_p2f_min_core_gain: float = 0.08,
    return_details: bool = False,
):
    """Classify one instance and optionally expose the vote components."""
    instance_mask = np.asarray(instance_mask, dtype=bool)
    vote_mask = instance_mask.copy()
    area = int(instance_mask.sum())
    if area <= 0:
        details = {
            "semantic_score": 0.0,
            "hard_ratio": 0.0,
            "core_pixels": 0,
            "color_score": None,
            "color_used": False,
            "lab_separation": None,
            "lab_threshold": None,
        }
        return (CLASS_PEARLITE, 0.0, details) if return_details else (CLASS_PEARLITE, 0.0)

    erode_width = max(0, int(erode_width))
    if erode_width > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_width + 1, 2 * erode_width + 1)
        )
        eroded = cv2.erode(vote_mask.astype(np.uint8), kernel) > 0
        minimum_core = min(64, max(8, area // 10))
        if int(eroded.sum()) >= minimum_core:
            vote_mask = eroded

    semantic_mask = np.asarray(semantic_mask)
    hard_ratio = float(np.mean(semantic_mask[vote_mask] > 0))
    normalized_mode = str(mode).strip().lower()
    core_mask = vote_mask
    core_weights = np.ones(instance_mask.shape, dtype=np.float32)
    dual_details = {}
    if normalized_mode == "hard_majority" or semantic_probability is None:
        score = hard_ratio
    else:
        probability = np.asarray(semantic_probability, dtype=np.float32)
        probability_score = float(np.mean(probability[vote_mask]))
        if normalized_mode == "probability_mean":
            score = probability_score
        elif normalized_mode == "hybrid":
            score = 0.5 * (hard_ratio + probability_score)
        elif normalized_mode in {"adaptive_core", "adaptive_core_lab"}:
            core_mask, core_weights = adaptive_instance_core(
                instance_mask,
                fraction=core_fraction,
                min_pixels=core_min_pixels,
                distance_power=core_distance_power,
            )
            score = robust_weighted_mean(
                probability[core_mask], core_weights[core_mask]
            )
        elif normalized_mode == "conservative_dual":
            if candidate_semantic_probability is None:
                raise ValueError(
                    "conservative_dual requires candidate_semantic_probability"
                )
            core_mask, core_weights = adaptive_instance_core(
                instance_mask,
                fraction=core_fraction,
                min_pixels=core_min_pixels,
                distance_power=core_distance_power,
            )
            candidate_probability = np.asarray(
                candidate_semantic_probability, dtype=np.float32
            )
            if candidate_probability.shape != probability.shape:
                raise ValueError(
                    "candidate semantic probability shape mismatch: "
                    f"{candidate_probability.shape} vs {probability.shape}"
                )
            base_core_score = robust_weighted_mean(
                probability[core_mask], core_weights[core_mask]
            )
            candidate_core_score = robust_weighted_mean(
                candidate_probability[core_mask], core_weights[core_mask]
            )
            candidate_full_score = float(
                np.mean(candidate_probability[instance_mask])
            )
            base_class = CLASS_FERRITE if hard_ratio > float(threshold) else CLASS_PEARLITE
            candidate_class = (
                CLASS_FERRITE
                if candidate_core_score > float(threshold)
                else CLASS_PEARLITE
            )
            core_gain = max(
                float(candidate_core_score - candidate_full_score),
                float(base_core_score - hard_ratio),
            )
            override = False
            reason = "agreement"
            if base_class != candidate_class:
                reason = "gate_rejected"
                if base_class == CLASS_PEARLITE:
                    override = (
                        hard_ratio >= float(dual_p2f_base_min)
                        and candidate_core_score >= float(dual_p2f_candidate_min)
                        and core_gain >= float(dual_p2f_min_core_gain)
                    )
                    if override:
                        reason = "pearlite_to_ferrite_black_rim"
                else:
                    override = (
                        hard_ratio <= float(dual_f2p_base_max)
                        and candidate_core_score <= float(dual_f2p_candidate_max)
                    )
                    if override:
                        reason = "ferrite_to_pearlite_high_confidence"
            score = candidate_core_score if override else hard_ratio
            dual_details = {
                "base_core_score": float(base_core_score),
                "candidate_core_score": float(candidate_core_score),
                "candidate_full_score": float(candidate_full_score),
                "candidate_class": int(candidate_class),
                "dual_core_gain": float(core_gain),
                "dual_override": bool(override),
                "dual_reason": reason,
            }
        else:
            raise ValueError(
                "semantic_vote_mode must be hard_majority, probability_mean, "
                "hybrid, adaptive_core, adaptive_core_lab, or conservative_dual; "
                f"got {mode!r}"
            )
    semantic_score = float(score)

    color_score = None
    color_used = False
    if (
        normalized_mode == "adaptive_core_lab"
        and lab_prior is not None
        and bool(lab_prior.get("available", False))
        and float(lab_prior.get("separation", 0.0)) >= float(color_min_separation)
        and float(color_uncertain_low) <= semantic_score <= float(color_uncertain_high)
    ):
        color_probability = np.asarray(
            lab_prior["ferrite_probability"], dtype=np.float32
        )
        if color_probability.shape == instance_mask.shape:
            color_score = robust_weighted_mean(
                color_probability[core_mask], core_weights[core_mask]
            )
            blend = float(np.clip(color_weight, 0.0, 1.0))
            score = (1.0 - blend) * semantic_score + blend * color_score
            color_used = True

    score = float(np.clip(score, 0.0, 1.0))
    cls = CLASS_FERRITE if score > float(threshold) else CLASS_PEARLITE
    if return_details:
        details = {
            "semantic_score": semantic_score,
            "hard_ratio": hard_ratio,
            "core_pixels": int(core_mask.sum()),
            "color_score": None if color_score is None else float(color_score),
            "color_used": bool(color_used),
            "lab_separation": (
                None if lab_prior is None
                else float(lab_prior.get("separation", 0.0))
            ),
            "lab_threshold": (
                None if lab_prior is None
                else float(lab_prior.get("threshold", 0.0))
            ),
            **dual_details,
        }
        return cls, score, details
    return cls, score
