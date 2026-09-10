# -*- coding: utf-8 -*-
"""Deterministic mask-set decoding without watershed or same-class merging."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from data.dataset import letterbox


def _numpy_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _probability(value, name):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def postprocess_mask_set(
    mask_logits,
    class_logits,
    original_shape,
    padding_hw,
    *,
    input_size: int = 1024,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_area: int = 1,
    min_overlap_ratio: float = 0.0,
):
    """Return native uint16 IDs, ID-to-class JSON data and query diagnostics.

    Each query chooses one of classes 0, 1, or empty=2. Eligible masks compete
    by class probability times mask probability; exact ties retain the lower
    query index. Area/overlap rejection happens after competition and leaves
    rejected pixels unassigned. Both material classes keep independent IDs.

    ``padding_hw`` is the exact bottom/right padding on the square input.
    Each query is bilinearly restored to that input grid, unpadded exactly,
    and resized to the native image before sigmoid. At most one native mask,
    one winning-score map and one owner map are held, never Q native masks.
    """
    score_threshold = _probability(score_threshold, "score_threshold")
    mask_threshold = _probability(mask_threshold, "mask_threshold")
    min_overlap_ratio = _probability(min_overlap_ratio, "min_overlap_ratio")
    min_area_value = float(min_area)
    if not np.isfinite(min_area_value) or min_area_value < 1 or not min_area_value.is_integer():
        raise ValueError("min_area must be a finite positive integer")
    min_area = int(min_area_value)
    input_size_value = float(input_size)
    if not np.isfinite(input_size_value) or input_size_value < 1 or not input_size_value.is_integer():
        raise ValueError("input_size must be a finite positive integer")
    input_size = int(input_size_value)
    height, width = map(int, original_shape)
    pad_h, pad_w = map(int, padding_hw)
    if height < 1 or width < 1 or not (0 <= pad_h < input_size and 0 <= pad_w < input_size):
        raise ValueError("invalid original_shape or input-grid padding_hw")
    if len(mask_logits.shape) != 3 or min(mask_logits.shape[1:]) < 1:
        raise ValueError("mask_logits must have shape [Q,H,W]")
    logits = _numpy_float(class_logits)
    query_count = int(mask_logits.shape[0])
    if logits.shape != (query_count, 3) or not np.isfinite(logits).all():
        raise ValueError("class_logits must be finite [Q,3] for classes 0,1,empty=2")
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    class_probability = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    categories = class_probability.argmax(axis=1)
    scores = class_probability[np.arange(query_count), categories]
    candidates = np.flatnonzero((categories != 2) & (scores >= score_threshold))
    owner = np.full((height, width), -1, dtype=np.int32)
    winning_score = np.full((height, width), -1, dtype=np.float32)
    original_areas = np.zeros(query_count, dtype=np.int64)

    for query in candidates:
        mask = _numpy_float(mask_logits[int(query)])
        if not np.isfinite(mask).all():
            raise ValueError(f"non-finite mask logits for query {query}")
        # 先回到输入网格再去精确填充，避免奇数 padding 在低分辨率网格上半像素错位。
        mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        mask = mask[:input_size - pad_h, :input_size - pad_w]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        np.clip(mask, -80.0, 80.0, out=mask)
        np.negative(mask, out=mask)
        np.exp(mask, out=mask)
        mask += 1.0
        np.reciprocal(mask, out=mask)
        eligible = mask >= mask_threshold
        original_areas[query] = int(eligible.sum())
        mask *= float(scores[query])
        replace = eligible & (mask > winning_score)
        owner[replace] = int(query)
        winning_score[replace] = mask[replace]

    assigned_counts = np.bincount(owner[owner >= 0], minlength=query_count)
    output = np.zeros((height, width), dtype=np.uint16)
    classes, query_rows = {}, []
    for query in candidates:
        area = int(assigned_counts[query])
        original_area = int(original_areas[query])
        overlap_ratio = area / original_area if original_area else 0.0
        accepted = area >= min_area and overlap_ratio >= min_overlap_ratio
        instance_id = 0
        if accepted:
            instance_id = len(classes) + 1
            if instance_id > np.iinfo(np.uint16).max:
                raise ValueError("mask-set output exceeds the 65535 instance-ID limit")
            output[owner == query] = instance_id
            classes[instance_id] = int(categories[query])
        query_rows.append({
            "query": int(query), "class": int(categories[query]), "class_score": float(scores[query]),
            "thresholded_mask_area": original_area, "assigned_area": area,
            "assigned_to_mask_ratio": float(overlap_ratio), "accepted": bool(accepted),
            "instance_id": instance_id,
        })
    diagnostics = {
        "queries": query_count, "eligible_queries": int(len(candidates)), "instances": len(classes),
        "unassigned_pixels": int((output == 0).sum()), "native_shape": [height, width],
        "input_size": input_size, "padding_hw": [pad_h, pad_w],
        "settings": {"score_threshold": score_threshold, "mask_threshold": mask_threshold,
                     "min_area": min_area, "min_overlap_ratio": min_overlap_ratio},
        "assignment": "argmax class_score * mask_probability over thresholded candidate masks; ties use lower query index",
        "query_results": query_rows,
    }
    return output, classes, diagnostics


@torch.no_grad()
def infer_mask_set_image(model, rgb: np.ndarray, config: dict, device):
    """Run one whole image; the model owns encoder input normalization."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("infer_mask_set_image expects uint8 RGB [H,W,3]")
    cfg = config.get("mask_set", {})
    input_size = int(cfg.get("input_size", 1024))
    inference = cfg.get("inference", {})
    image_lb, _, pad_h, pad_w = letterbox(rgb, input_size)
    tensor = torch.from_numpy(image_lb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    was_training = bool(model.training)
    model.eval()
    try:
        prediction = model(tensor)
    finally:
        model.train(was_training)
    if prediction["pred_masks"].shape[0] != 1 or prediction["pred_logits"].shape[0] != 1:
        raise ValueError("mask-set inference expects one output batch item")
    return postprocess_mask_set(
        prediction["pred_masks"][0], prediction["pred_logits"][0], rgb.shape[:2], (pad_h, pad_w),
        input_size=input_size, score_threshold=inference.get("score_threshold", 0.5),
        mask_threshold=inference.get("mask_threshold", 0.5), min_area=inference.get("min_area", 1),
        min_overlap_ratio=inference.get("min_overlap_ratio", 0.0),
    )
