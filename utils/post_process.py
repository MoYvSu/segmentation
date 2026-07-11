# -*- coding: utf-8 -*-
"""
Adaptive size restoration and topological separation post-processing
=====================================================================
1. Dynamic upsampling: F.interpolate to original image size (H, W)
2. Binary thresholding: Sigmoid + 0.5 threshold on classification logits
3. Distance field compensation: inverse transform + spatial scale + re-normalize
4. Watershed separation: use distance field to split touching grains
5. Topological instance ID assignment: connected components + watershed

Class definition: 0=pearlite, 1=ferrite
"""

import json
import os
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


CLASS_PEARLITE = 0
CLASS_FERRITE = 1


# ---------------------------------------------------------------------------
# Distance field compensation
# ---------------------------------------------------------------------------

def compensate_distance_field(
    dist_norm: np.ndarray,
    spatial_scale: float,
    scale_factor: float = 10.0,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Compensate normalized distance field for spatial scale differences.

    Uses inverse transform: de-normalize -> scale -> re-normalize.
    Linear multiplication cannot correctly invert the non-linear normalization.
    """
    dist_clipped = np.clip(dist_norm, 0.0, 1.0 - eps)
    dist_raw = dist_clipped * scale_factor / (1.0 - dist_clipped + eps)
    dist_raw_corrected = dist_raw * spatial_scale
    dist_compensated = dist_raw_corrected / (dist_raw_corrected + scale_factor)
    return dist_compensated.astype(np.float32)


# ---------------------------------------------------------------------------
# Basic output conversion
# ---------------------------------------------------------------------------

def restore_to_original_size(
    output: torch.Tensor,
    original_size: Tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """Upsample decoder output to original image size."""
    h, w = original_size
    return F.interpolate(output, size=(h, w), mode=mode, align_corners=align_corners)


def output_to_binary_mask(
    output: torch.Tensor,
    threshold: float = 0.5,
    original_size: Optional[Tuple[int, int]] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> np.ndarray:
    """Convert dual-channel model output to binary mask via Sigmoid + threshold."""
    if output.ndim == 3:
        output = output.unsqueeze(0)
    elif output.ndim != 4:
        raise ValueError("output ndim should be 3 or 4")

    if original_size is not None:
        output = restore_to_original_size(output, original_size, mode=mode, align_corners=align_corners)

    seg_logits = output[:, 0]
    seg_prob = torch.sigmoid(seg_logits)
    pred = (seg_prob > threshold).cpu().numpy().astype(np.uint8)

    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    return pred


def output_to_distance_field(
    output: torch.Tensor,
    original_size: Optional[Tuple[int, int]] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> np.ndarray:
    """Extract distance field prediction from dual-channel model output."""
    if output.ndim == 3:
        output = output.unsqueeze(0)
    elif output.ndim != 4:
        raise ValueError("output ndim should be 3 or 4")

    if original_size is not None:
        output = restore_to_original_size(output, original_size, mode=mode, align_corners=align_corners)

    dist_field = output[:, 1].cpu().numpy().astype(np.float32)

    if dist_field.ndim == 3 and dist_field.shape[0] == 1:
        dist_field = dist_field[0]
    return dist_field


# ---------------------------------------------------------------------------
# Gaussian weight blending
# ---------------------------------------------------------------------------

def gaussian_weight_map(size: int, sigma_scale: float = 0.25) -> np.ndarray:
    """Generate 2D Gaussian weight map for tiled inference blending."""
    sigma = size * sigma_scale
    center = (size - 1) / 2.0
    ax = np.arange(size, dtype=np.float32) - center
    xx, yy = np.meshgrid(ax, ax)
    weight = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    weight /= weight.max()
    return weight


# ---------------------------------------------------------------------------
# Watershed instance separation
# ---------------------------------------------------------------------------

def _get_dynamic_kernel_size(
    image_area: int,
    base_area: int = 1024 * 1024,
    base_kernel: int = 3,
    max_kernel: int = 15,
) -> int:
    """Dynamically compute morphological kernel size based on image area."""
    scale = image_area / base_area
    kernel = int(base_kernel * (scale ** 0.5))
    kernel = max(base_kernel, min(kernel, max_kernel))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def watershed_separation(
    ferrite_mask: np.ndarray,
    dist_field: np.ndarray,
    min_distance: int = 5,
    kernel_size: Optional[int] = None,
) -> np.ndarray:
    """
    Watershed separation using distance field to split touching grains.

    Algorithm:
    1. Dilate + erode distance field to find local maxima
    2. Extract seed points (local maxima that are strictly greater than eroded values)
    3. Fallback: use 75th percentile threshold if no seeds found
    4. Run cv2.watershed with seed markers
    """
    h, w = ferrite_mask.shape[:2]

    if kernel_size is None:
        kernel_size = _get_dynamic_kernel_size(h * w)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Step 1: dilate + erode to find local maxima
    dist_dilated = cv2.dilate(dist_field, kernel)
    dist_eroded = cv2.erode(dist_field, kernel)

    # Step 2: local maxima = value >= dilated AND value > eroded (exclude plateaus)
    local_max = (dist_field >= dist_dilated - 1e-6) & (dist_field > dist_eroded + 1e-6) & (ferrite_mask > 0)

    # Fallback: use percentile threshold if too strict
    if local_max.sum() == 0:
        valid_dist = dist_field[ferrite_mask > 0]
        if valid_dist.size > 0:
            threshold = np.percentile(valid_dist, 75)
            local_max = (dist_field > threshold) & (ferrite_mask > 0)

    local_max = local_max.astype(np.uint8)

    # Step 3: filter small seed regions
    if local_max.sum() > 0:
        num_seeds, seed_labels = cv2.connectedComponents(local_max, connectivity=8)
        if num_seeds > 1:
            for sid in range(1, num_seeds):
                if (seed_labels == sid).sum() < min_distance:
                    seed_labels[seed_labels == sid] = 0
    else:
        seed_labels = np.zeros((h, w), dtype=np.int32)

    # Step 4: fallback to connected components if no seeds
    if seed_labels.max() == 0:
        num_labels, labels = cv2.connectedComponents(ferrite_mask, connectivity=8)
        return labels.astype(np.int32)

    # Step 5: watershed transform
    markers = seed_labels.copy().astype(np.int32)
    markers[ferrite_mask == 0] = 1
    markers[markers > 0] = markers[markers > 0] + 1

    vis_image = cv2.cvtColor(ferrite_mask * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(vis_image, markers)

    labels = np.where(markers <= 1, 0, markers - 1).astype(np.int32)
    return labels


# ---------------------------------------------------------------------------
# Topological instance separation
# ---------------------------------------------------------------------------

def topo_instance_separation(
    mask: np.ndarray,
    dist_field: Optional[np.ndarray] = None,
    min_instance_area: int = 50,
    max_instance_id: int = 255,
    connectivity: int = 8,
    use_watershed: bool = True,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Topological instance separation and global ID assignment.

    For ferrite (class 1): watershed separation if dist_field provided, else connected components.
    For pearlite (class 0): connected components analysis.
    Filter instances smaller than min_instance_area.
    Assign IDs 1~255 by descending area.
    """
    h, w = mask.shape[:2]
    inst_map = np.zeros((h, w), dtype=np.uint8)
    class_map = {}

    instances = []

    # 1. Ferrite watershed separation
    ferrite_binary = (mask == CLASS_FERRITE).astype(np.uint8)
    if ferrite_binary.sum() > 0:
        if use_watershed and dist_field is not None:
            ferrite_labels = watershed_separation(
                ferrite_binary,
                dist_field,
                kernel_size=_get_dynamic_kernel_size(h * w),
            )
        else:
            _, ferrite_labels = cv2.connectedComponents(ferrite_binary, connectivity=connectivity)

        unique_labels = np.unique(ferrite_labels)
        unique_labels = unique_labels[unique_labels > 0]

        for label_id in unique_labels:
            inst_mask = (ferrite_labels == label_id)
            area = int(inst_mask.sum())
            if area < min_instance_area:
                continue
            instances.append((area, CLASS_FERRITE, inst_mask))

    # 2. Pearlite connected components
    pearlite_binary = (mask == CLASS_PEARLITE).astype(np.uint8)
    if pearlite_binary.sum() > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            pearlite_binary, connectivity=connectivity
        )
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_instance_area:
                continue
            inst_mask = (labels == label_id)
            instances.append((area, CLASS_PEARLITE, inst_mask))

    # 3. Sort by area descending
    instances.sort(key=lambda x: x[0], reverse=True)

    # 4. Assign IDs 1~255
    current_id = 1
    for area, class_label, inst_mask in instances:
        if current_id > max_instance_id:
            current_id = max_instance_id
        inst_map[inst_mask] = current_id
        class_map[current_id] = int(class_label)
        if current_id < max_instance_id:
            current_id += 1

    return inst_map, class_map


# ---------------------------------------------------------------------------
# Full post-processing pipeline
# ---------------------------------------------------------------------------

def post_process_prediction(
    output: torch.Tensor,
    original_size: Tuple[int, int],
    output_dir: str,
    image_basename: str,
    min_instance_area: int = 50,
    max_instance_id: int = 255,
    connectivity: int = 8,
    interpolate_mode: str = "bilinear",
    align_corners: bool = True,
    threshold: float = 0.5,
    save_visualization: bool = True,
    dist_scale_factor: float = 10.0,
    spatial_scale: float = 1.0,
    use_watershed: bool = True,
) -> Dict[str, str]:
    """
    Full post-processing pipeline:
    1. Upsample to original size
    2. Sigmoid + threshold -> binary mask
    3. Distance field spatial compensation
    4. Topological separation (watershed + connected components)
    5. Save _inst.png and _class.json
    """
    os.makedirs(output_dir, exist_ok=True)

    mask = output_to_binary_mask(
        output, threshold=threshold, original_size=original_size,
        mode=interpolate_mode, align_corners=align_corners,
    )

    dist_field = output_to_distance_field(
        output, original_size=original_size,
        mode=interpolate_mode, align_corners=align_corners,
    )

    if spatial_scale != 1.0:
        dist_field = compensate_distance_field(
            dist_field, spatial_scale=spatial_scale, scale_factor=dist_scale_factor,
        )

    inst_map, class_map = topo_instance_separation(
        mask, dist_field=dist_field,
        min_instance_area=min_instance_area, max_instance_id=max_instance_id,
        connectivity=connectivity, use_watershed=use_watershed,
    )

    inst_path = os.path.join(output_dir, f"{image_basename}_inst.png")
    cv2.imwrite(inst_path, inst_map)

    class_json_path = os.path.join(output_dir, f"{image_basename}_class.json")
    class_json = {str(k): v for k, v in class_map.items()}
    with open(class_json_path, "w", encoding="utf-8") as f:
        json.dump(class_json, f, ensure_ascii=False, indent=2)

    output_paths = {"inst_path": inst_path, "class_json_path": class_json_path}

    if save_visualization:
        mask_path = os.path.join(output_dir, f"{image_basename}_mask.png")
        color_map = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_map[mask == CLASS_PEARLITE] = [0, 0, 128]
        color_map[mask == CLASS_FERRITE] = [0, 128, 0]
        cv2.imwrite(mask_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
        output_paths["mask_path"] = mask_path

        dist_path = os.path.join(output_dir, f"{image_basename}_dist.png")
        dist_vis = (dist_field * 255).astype(np.uint8)
        cv2.imwrite(dist_path, dist_vis)
        output_paths["dist_path"] = dist_path

    return output_paths