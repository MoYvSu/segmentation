# -*- coding: utf-8 -*-
"""用人工实例作为 marker，对 LabelMe 窄接缝做保守补全。

原始多边形像素始终保持不变。只允许 watershed 在距已标实例不超过
``max_fill_distance`` 的未覆盖像素中生长；更宽的未知区域继续保留为 0。
该模块生成的是带来源记录的派生候选 GT，不会改写 LabelMe JSON。
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.measure import label
from skimage.segmentation import find_boundaries, watershed


def _validate_inputs(
    image_bgr: np.ndarray, instance_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image_bgr)
    labels = np.asarray(instance_map)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_bgr must have shape [H,W,3], got {image.shape}")
    if labels.ndim != 2 or labels.shape != image.shape[:2]:
        raise ValueError(
            f"instance_map shape {labels.shape} does not match image {image.shape[:2]}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"instance_map must be integer, got {labels.dtype}")
    if int(labels.min()) < 0 or int(labels.max()) > 65535:
        raise ValueError("instance ids must stay within [0, 65535]")
    if not np.any(labels > 0):
        raise ValueError("instance_map has no positive marker")
    return image.astype(np.uint8, copy=False), labels.astype(np.int32, copy=False)


def lab_gradient_elevation(
    image_bgr: np.ndarray,
    *,
    blur_sigma: float = 1.2,
) -> np.ndarray:
    """返回轻度平滑 Lab 图像的多通道 Scharr 梯度幅值。"""
    image = np.asarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_bgr must have shape [H,W,3], got {image.shape}")
    if float(blur_sigma) < 0:
        raise ValueError("blur_sigma must be non-negative")
    if float(blur_sigma) > 0:
        image = cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=float(blur_sigma),
            sigmaY=float(blur_sigma),
            borderType=cv2.BORDER_REFLECT_101,
        )
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    magnitude_sq = np.zeros(lab.shape[:2], dtype=np.float32)
    for channel in range(3):
        gx = cv2.Scharr(
            lab[:, :, channel], cv2.CV_32F, 1, 0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        gy = cv2.Scharr(
            lab[:, :, channel], cv2.CV_32F, 0, 1,
            borderType=cv2.BORDER_REFLECT_101,
        )
        magnitude_sq += gx * gx + gy * gy
    return np.sqrt(magnitude_sq, out=magnitude_sq).astype(np.float32, copy=False)


def _component_audit(instance_map: np.ndarray) -> Tuple[int, int]:
    instance_count = int(np.sum(np.unique(instance_map) > 0))
    component_count = int(label(instance_map, background=0, connectivity=None).max())
    return instance_count, component_count


def complete_narrow_instance_gaps(
    image_bgr: np.ndarray,
    instance_map: np.ndarray,
    *,
    max_fill_distance: float = 8.0,
    blur_sigma: float = 1.2,
    elevation: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float | int]]:
    """在窄未覆盖带中执行 image-gradient seeded watershed。

    Returns:
        completed: 保留原 ID 的 ``int32[H,W]`` 派生实例图。
        fill_mask: 本次补入像素的 ``bool[H,W]`` 来源掩码。
        distance_to_seed: 未覆盖像素到最近人工标注的欧氏距离。
        audit: 覆盖率、拓扑和补区图像梯度诊断。
    """
    image, labels = _validate_inputs(image_bgr, instance_map)
    if float(max_fill_distance) <= 0:
        raise ValueError("max_fill_distance must be positive")

    uncovered = labels == 0
    distance_to_seed = cv2.distanceTransform(
        uncovered.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ).astype(np.float32, copy=False)
    fill_mask = uncovered & (distance_to_seed <= float(max_fill_distance))
    allowed = (labels > 0) | fill_mask

    if elevation is None:
        height_map = lab_gradient_elevation(image, blur_sigma=blur_sigma)
    else:
        height_map = np.asarray(elevation, dtype=np.float32)
        if height_map.shape != labels.shape:
            raise ValueError(
                f"elevation shape {height_map.shape} != labels {labels.shape}"
            )

    grown = watershed(
        height_map,
        markers=labels,
        mask=allowed,
        connectivity=1,
        compactness=0.0,
        watershed_line=False,
    ).astype(np.int32, copy=False)
    completed = labels.copy()
    completed[fill_mask] = grown[fill_mask]
    if np.any(completed[fill_mask] == 0):
        raise RuntimeError("watershed left eligible gap pixels unassigned")
    changed_original = int(np.sum(completed[labels > 0] != labels[labels > 0]))
    if changed_original:
        raise RuntimeError(f"watershed changed {changed_original} original pixels")

    original_instances, original_components = _component_audit(labels)
    completed_instances, completed_components = _component_audit(completed)
    new_boundary = find_boundaries(completed, mode="inner", connectivity=1)
    boundary_touching_fill = new_boundary & cv2.dilate(
        fill_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ).astype(bool)
    fill_gradient = float(height_map[fill_mask].mean()) if fill_mask.any() else 0.0
    boundary_gradient = (
        float(height_map[boundary_touching_fill].mean())
        if boundary_touching_fill.any()
        else 0.0
    )
    total_pixels = int(labels.size)
    original_covered = int(np.sum(labels > 0))
    filled_pixels = int(fill_mask.sum())
    residual_unknown = int(np.sum(completed == 0))
    audit: Dict[str, float | int] = {
        "max_fill_distance": float(max_fill_distance),
        "instance_count": original_instances,
        "completed_instance_count": completed_instances,
        "original_component_count": original_components,
        "completed_component_count": completed_components,
        "original_component_excess": original_components - original_instances,
        "completed_component_excess": completed_components - completed_instances,
        "total_pixels": total_pixels,
        "original_covered_pixels": original_covered,
        "original_uncovered_pixels": int(uncovered.sum()),
        "filled_pixels": filled_pixels,
        "residual_unknown_pixels": residual_unknown,
        "original_coverage": original_covered / total_pixels,
        "completed_coverage": (original_covered + filled_pixels) / total_pixels,
        "fraction_of_gap_filled": filled_pixels / max(1, int(uncovered.sum())),
        "changed_original_pixels": changed_original,
        "fill_mean_gradient": fill_gradient,
        "new_boundary_mean_gradient": boundary_gradient,
        "boundary_to_fill_gradient_ratio": boundary_gradient / max(fill_gradient, 1.0e-6),
        "new_boundary_pixels_touching_fill": int(boundary_touching_fill.sum()),
    }
    return completed, fill_mask, distance_to_seed, audit
