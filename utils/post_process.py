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


def _get_adaptive_max_filter_size(
    image_area: int,
    min_size: int = 15,
    max_size: int = 31,
    base_area: int = 1024 * 1024,
) -> int:
    """Compute adaptive maximum filter size based on image area.

    Returns an odd integer in [min_size, max_size], scaling with sqrt(area).
    For 1024x1024 -> 15, for 2448x2048 -> ~31.
    """
    scale = image_area / base_area
    size = int(min_size * (scale ** 0.5))
    size = max(min_size, min(size, max_size))
    if size % 2 == 0:
        size += 1
    return size


def watershed_separation(
    ferrite_mask: np.ndarray,
    dist_field: np.ndarray,
    alpha: float = 0.75,
    beta: float = 0.05,
    max_filter_size: Optional[int] = None,
    area_ratio_threshold: float = 0.2,
    min_island_area: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Watershed separation using local maximum filter + adaptive threshold.

    Replaces the previous Sigmoid-stretch approach. Instead of a global
    non-linear transform, this method computes a per-pixel adaptive
    threshold from the local maximum of the distance field, making it
    robust to varying grain sizes without manual tuning.

    Algorithm:
    1. Purify: zero out non-ferrite regions
    2. Local maximum filter via cv2.dilate (rectangular kernel)
    3. Adaptive threshold: Threshold = alpha * local_max + beta
    4. Highland binary extraction: dist_purified > Threshold & ferrite_mask
    5. Connected components + relative area topology filter
       (remove islands < area_ratio_threshold * max_island_area)
    6. cv2.moments centroid extraction -> single-pixel markers
    7. 3x3 ellipse dilation of markers
    8. skimage watershed on -dist_purified with mask=ferrite_mask

    Args:
        ferrite_mask: [H, W] binary mask (1=ferrite, 0=background)
        dist_field: [H, W] normalized distance field [0,1] from model
        alpha: Adaptive threshold coefficient (default 0.75).
               Threshold = alpha * local_max + beta.
        beta: Global bias constant (default 0.05).
        max_filter_size: Local max filter window size (odd int).
                         None = auto-adaptive based on image area (15~31).
        area_ratio_threshold: Islands with area < max_island_area * this ratio
                              are filtered out as noise (default 0.2 = 20%).
        min_island_area: Absolute minimum island area to keep (default 5).
    """
    from skimage.segmentation import watershed as sk_watershed

    h, w = ferrite_mask.shape[:2]

    # Step 1: Purify - zero out non-ferrite regions to prevent energy leakage
    dist_purified = dist_field * (ferrite_mask > 0).astype(np.float32)

    # Step 2: Local maximum filter via cv2.dilate (rectangular kernel)
    if max_filter_size is None:
        max_filter_size = _get_adaptive_max_filter_size(h * w)
    if max_filter_size % 2 == 0:
        max_filter_size += 1
    rect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max_filter_size, max_filter_size)
    )
    local_max_map = cv2.dilate(dist_purified, rect_kernel)

    # Step 3: Adaptive threshold matrix
    threshold_map = alpha * local_max_map + beta

    # Step 4: Highland binary extraction
    highland_binary = (dist_purified > threshold_map) & (ferrite_mask > 0)
    highland_mask = highland_binary.astype(np.uint8)

    # Step 5: Connected components + relative area topology filter
    num_islands, island_labels, stats, _ = cv2.connectedComponentsWithStats(
        highland_mask, connectivity=8
    )

    if num_islands <= 1:
        # No highlands found - fallback to connected components on ferrite mask
        num_labels, labels = cv2.connectedComponents(ferrite_mask, connectivity=8)
        return labels.astype(np.int32), dist_purified

    # Find max island area (excluding background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_island_area = int(np.max(areas)) if len(areas) > 0 else 0

    # Filter: keep islands that pass BOTH relative and absolute area filters
    valid_islands = []
    for island_id in range(1, num_islands):
        area = int(stats[island_id, cv2.CC_STAT_AREA])
        if area < min_island_area:
            continue
        if max_island_area > 0 and area < max_island_area * area_ratio_threshold:
            continue
        valid_islands.append(island_id)

    if len(valid_islands) == 0:
        num_labels, labels = cv2.connectedComponents(ferrite_mask, connectivity=8)
        return labels.astype(np.int32), dist_purified

    # Step 6: cv2.moments centroid extraction -> single-pixel markers
    seed_labels = np.zeros((h, w), dtype=np.int32)
    for new_id, island_id in enumerate(valid_islands, start=1):
        island_pixel_mask = (island_labels == island_id).astype(np.uint8)
        M = cv2.moments(island_pixel_mask)
        if M["m00"] == 0:
            continue
        cx = int(round(M["m10"] / M["m00"]))
        cy = int(round(M["m01"] / M["m00"]))
        # Clamp to image bounds
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        seed_labels[cy, cx] = new_id

    # Check if any seeds survived
    if seed_labels.max() == 0:
        num_labels, labels = cv2.connectedComponents(ferrite_mask, connectivity=8)
        return labels.astype(np.int32), dist_purified

    # Step 7: 3x3 ellipse dilation of markers
    seed_mask = (seed_labels > 0).astype(np.uint8)
    seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed_mask_dilated = cv2.dilate(seed_mask, seed_kernel)

    # Re-assign IDs to dilated seeds via connected components
    _, seed_labels_final = cv2.connectedComponents(seed_mask_dilated, connectivity=8)

    if seed_labels_final.max() == 0:
        num_labels, labels = cv2.connectedComponents(ferrite_mask, connectivity=8)
        return labels.astype(np.int32), dist_purified

    # Step 8: Watershed on inverted purified distance field
    markers = seed_labels_final.copy()
    markers[ferrite_mask == 0] = -1  # mark background explicitly

    labels = sk_watershed(
        -dist_purified,  # negate: centers become basins
        markers=markers,
        mask=ferrite_mask > 0,  # restrict to ferrite region
    )

    # Ensure background is 0
    labels = np.where(ferrite_mask > 0, labels, 0).astype(np.int32)
    return labels, dist_purified


# ---------------------------------------------------------------------------
# Active Contour edge smoothing
# ---------------------------------------------------------------------------

def smooth_instance_edges(
    label_map: np.ndarray,
    dist_purified: np.ndarray,
    snake_alpha: float = 0.02,
    snake_beta: float = 0.2,
    snake_max_iter: int = 80,
    area_shrink_threshold: float = 0.15,
    max_contour_points: int = 500,
) -> np.ndarray:
    """
    Smooth instance edges using Active Contour Model (Snake).

    Processes each instance independently:
    A. Extract binary ROI for current instance
    B. cv2.findContours -> initial contour (downsampled)
    C. skimage.segmentation.active_contour evolution
    D. cv2.fillPoly to rebuild smoothed mask
    + Area shrink defense: if smoothed area < (1 - threshold) * original,
      fall back to original mask.

    Args:
        label_map: [H, W] int32 label map from watershed (0=background)
        dist_purified: [H, W] float32 distance field (guides Snake)
        snake_alpha: Elasticity coefficient (default 0.02)
        snake_beta: Rigidity coefficient (default 0.2)
        snake_max_iter: Max Snake iterations (default 80)
        area_shrink_threshold: Max allowed area shrink ratio (default 0.15)
        max_contour_points: Max contour points for Snake (default 500)

    Returns:
        Smoothed label_map (same dtype as input)
    """
    from skimage.segmentation import active_contour

    h, w = label_map.shape[:2]
    smooth_labels = np.zeros_like(label_map)

    unique_ids = np.unique(label_map)
    unique_ids = unique_ids[unique_ids > 0]

    # Sort by area descending (largest first)
    id_areas = []
    for uid in unique_ids:
        id_areas.append((int((label_map == uid).sum()), uid))
    id_areas.sort(reverse=True)

    for _, inst_id in id_areas:
        # Step A: Binary ROI
        binary_roi = (label_map == inst_id).astype(np.uint8)
        original_area = int(binary_roi.sum())

        if original_area < 10:
            smooth_labels[binary_roi > 0] = inst_id
            continue

        # Step B: Extract contour
        contours, _ = cv2.findContours(
            binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 0:
            smooth_labels[binary_roi > 0] = inst_id
            continue

        # Take largest contour
        contour = max(contours, key=cv2.contourArea)
        contour = contour.squeeze(1)  # (N, 2) in (x, y) = (col, row)

        if len(contour) < 5:
            smooth_labels[binary_roi > 0] = inst_id
            continue

        # Downsample to max_contour_points (equidistant)
        if len(contour) > max_contour_points:
            indices = np.linspace(0, len(contour) - 1, max_contour_points, dtype=int)
            contour = contour[indices]

        # Convert to (row, col) for skimage
        init_contour = contour[:, [1, 0]].astype(np.float64)

        # Step C: Active Contour evolution
        try:
            snake = active_contour(
                dist_purified,
                init_contour,
                alpha=snake_alpha,
                beta=snake_beta,
                max_num_iter=snake_max_iter,
            )
        except Exception:
            # Fallback on any Snake error
            smooth_labels[binary_roi > 0] = inst_id
            continue

        # Convert back to (col, row) -> int32 for cv2
        snake_pts = np.round(snake[:, [1, 0]]).astype(np.int32)
        # Clamp to image bounds
        snake_pts[:, 0] = np.clip(snake_pts[:, 0], 0, w - 1)
        snake_pts[:, 1] = np.clip(snake_pts[:, 1], 0, h - 1)

        # Step D: Area shrink defense
        snake_area = int(cv2.contourArea(snake_pts))
        if snake_area <= 0:
            smooth_labels[binary_roi > 0] = inst_id
            continue
        if snake_area < original_area * (1.0 - area_shrink_threshold):
            smooth_labels[binary_roi > 0] = inst_id
            continue

        # Fill smoothed contour
        cv2.fillPoly(smooth_labels, [snake_pts], int(inst_id))

    return smooth_labels


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
    alpha: float = 0.75,
    beta: float = 0.05,
    max_filter_size: Optional[int] = None,
    area_ratio_threshold: float = 0.2,
    min_island_area: int = 5,
    enable_snake_smoothing: bool = False,
    snake_alpha: float = 0.02,
    snake_beta: float = 0.2,
    snake_max_iter: int = 80,
    snake_area_shrink_threshold: float = 0.15,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Topological instance separation and global ID assignment.

    For ferrite (class 1): watershed separation if dist_field provided, else connected components.
    For pearlite (class 0): connected components analysis.
    Filter instances smaller than min_instance_area.
    Assign IDs 1~255 by descending area.

    Args:
        alpha: Adaptive threshold coefficient for watershed (default 0.75).
        beta: Global bias for watershed adaptive threshold (default 0.05).
        max_filter_size: Local max filter window size. None=adaptive.
        area_ratio_threshold: Relative area filter ratio (default 0.2).
        min_island_area: Absolute min island area for seed extraction (default 5).
        enable_snake_smoothing: If True, apply Active Contour smoothing to
                                ferrite watershed labels (default False).
        snake_alpha: Snake elasticity coefficient (default 0.02).
        snake_beta: Snake rigidity coefficient (default 0.2).
        snake_max_iter: Snake max iterations (default 80).
        snake_area_shrink_threshold: Max area shrink ratio before fallback (default 0.15).
    """
    h, w = mask.shape[:2]
    inst_map = np.zeros((h, w), dtype=np.uint8)
    class_map = {}

    instances = []

    # 1. Ferrite watershed separation
    ferrite_binary = (mask == CLASS_FERRITE).astype(np.uint8)
    if ferrite_binary.sum() > 0:
        if use_watershed and dist_field is not None:
            ferrite_labels, dist_purified = watershed_separation(
                ferrite_binary,
                dist_field,
                alpha=alpha,
                beta=beta,
                max_filter_size=max_filter_size,
                area_ratio_threshold=area_ratio_threshold,
                min_island_area=min_island_area,
            )
            # Optional: Active Contour edge smoothing (ferrite only)
            if enable_snake_smoothing:
                ferrite_labels = smooth_instance_edges(
                    ferrite_labels,
                    dist_purified,
                    snake_alpha=snake_alpha,
                    snake_beta=snake_beta,
                    snake_max_iter=snake_max_iter,
                    area_shrink_threshold=snake_area_shrink_threshold,
                )
        else:
            _, ferrite_labels = cv2.connectedComponents(ferrite_binary, connectivity=connectivity)

        unique_labels = np.unique(ferrite_labels)
        unique_labels = unique_labels[unique_labels > 0]

        for label_id in unique_labels:
            # FIX: filter out non-ferrite pixels to prevent area inflation
            # caused by background marker leakage.
            inst_mask = (ferrite_labels == label_id) & (ferrite_binary > 0)
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
    alpha: float = 0.75,
    beta: float = 0.05,
    max_filter_size: Optional[int] = None,
    area_ratio_threshold: float = 0.2,
    min_island_area: int = 5,
    enable_snake_smoothing: bool = False,
    snake_alpha: float = 0.02,
    snake_beta: float = 0.2,
    snake_max_iter: int = 80,
    snake_area_shrink_threshold: float = 0.15,
) -> Tuple[Dict[str, str], np.ndarray, Dict[int, int]]:
    """
    Full post-processing pipeline:
    1. Upsample to original size
    2. Sigmoid + threshold -> binary mask
    3. Distance field spatial compensation
    4. Topological separation (watershed + connected components)
    5. Save _inst.png and _class.json

    Args:
        alpha: Adaptive threshold coefficient for watershed (default 0.75).
        beta: Global bias for watershed adaptive threshold (default 0.05).
        max_filter_size: Local max filter window size. None=adaptive.
        area_ratio_threshold: Relative area filter ratio (default 0.2).
        min_island_area: Absolute min island area for seed extraction (default 5).
        enable_snake_smoothing: Active Contour edge smoothing toggle (default False).
        snake_alpha: Snake elasticity coefficient (default 0.02).
        snake_beta: Snake rigidity coefficient (default 0.2).
        snake_max_iter: Snake max iterations (default 80).
        snake_area_shrink_threshold: Max area shrink ratio before fallback (default 0.15).
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
        alpha=alpha, beta=beta, max_filter_size=max_filter_size,
        area_ratio_threshold=area_ratio_threshold, min_island_area=min_island_area,
        enable_snake_smoothing=enable_snake_smoothing,
        snake_alpha=snake_alpha, snake_beta=snake_beta,
        snake_max_iter=snake_max_iter,
        snake_area_shrink_threshold=snake_area_shrink_threshold,
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

    # Return inst_map and class_map so callers can reuse without re-computing
    return output_paths, inst_map, class_map


# ---------------------------------------------------------------------------
# Vector Field Centroid Collapse + DBSCAN Clustering (Phase 2)
# ---------------------------------------------------------------------------

def centroid_collapse_clustering(
    ferrite_mask: np.ndarray,
    vx_field: np.ndarray,
    vy_field: np.ndarray,
    image_size: int = 1024,
    dbscan_eps: float = 5.0,
    dbscan_min_samples: int = 3,
    downsample_grid: int = 4,
    min_instance_area: int = 50,
    max_instance_id: int = 255,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Vector field centroid collapse + DBSCAN clustering for instance separation.

    Algorithm:
    1. Extract foreground pixels (ferrite_mask > 0)
    2. Denormalize: offset = vx * max(h,w)
    3. Centroid collapse: coords_collapsed = coords_orig + offsets
    4. Grid downsampling: bucket aggregation to reduce point count
    5. DBSCAN clustering in collapsed space
    6. Map cluster labels back to original pixel positions
    7. Area filter + ID assignment (1-255 by descending area)

    Args:
        ferrite_mask: [H, W] binary mask (1=ferrite, 0=background)
        vx_field: [H, W] normalized X offset [-1, 1]
        vy_field: [H, W] normalized Y offset [-1, 1]
        image_size: letterbox target size (default 1024). Actual denormalization uses max(h,w).
        dbscan_eps: DBSCAN neighborhood radius in collapsed space (default 5.0)
        dbscan_min_samples: DBSCAN minimum samples per cluster (default 3)
        downsample_grid: grid size for downsampling (default 4 = 4x4 buckets)
        min_instance_area: minimum instance area in pixels (default 50)
        max_instance_id: maximum instance ID (default 255, uint8)

    Returns:
        inst_map: [H, W] uint8 instance map (0=background, 1-255 by descending area)
        class_map: dict {instance_id: class_label} (all 1=ferrite for this function)
    """
    from sklearn.cluster import DBSCAN

    h, w = ferrite_mask.shape[:2]

    # Step 1: Extract foreground pixels
    ys, xs = np.where(ferrite_mask > 0)
    if len(ys) == 0:
        return np.zeros((h, w), dtype=np.uint8), {}

    # Step 2: Denormalize offsets
    # 训练时向量场在 1024×1024 letterbox 空间中生成: offset = (centroid - pixel) / image_size
    # 推理时坐标 xs/ys 在原始分辨率空间（如 2584×1936），需要将偏移量也缩放到原始空间
    # denorm_factor = image_size / scale = image_size / (image_size / max(h,w)) = max(h, w)
    denorm_factor = float(max(h, w))
    vx_vals = vx_field[ys, xs] * denorm_factor
    vy_vals = vy_field[ys, xs] * denorm_factor

    # Step 3: Centroid collapse
    collapsed_x = xs.astype(np.float64) + vx_vals
    collapsed_y = ys.astype(np.float64) + vy_vals
    coords_collapsed = np.stack([collapsed_x, collapsed_y], axis=1)  # [N, 2]

    # Step 4: Grid downsampling
    if downsample_grid > 1:
        # Assign each point to a grid bucket
        grid_x = (collapsed_x / downsample_grid).astype(np.int32)
        grid_y = (collapsed_y / downsample_grid).astype(np.int32)

        # Aggregate: mean position per bucket, collect original indices
        bucket_map = {}  # (gx, gy) -> list of original indices
        for i in range(len(ys)):
            key = (grid_x[i], grid_y[i])
            if key not in bucket_map:
                bucket_map[key] = []
            bucket_map[key].append(i)

        # Create downsampled coordinates and keep mapping to original indices
        ds_coords = []
        ds_to_orig = []
        for key, indices in bucket_map.items():
            mean_x = collapsed_x[indices].mean()
            mean_y = collapsed_y[indices].mean()
            ds_coords.append([mean_x, mean_y])
            ds_to_orig.append(indices)
        ds_coords = np.array(ds_coords, dtype=np.float64)
    else:
        ds_coords = coords_collapsed
        ds_to_orig = [[i] for i in range(len(ys))]

    # Step 5: DBSCAN clustering
    clustering = DBSCAN(
        eps=dbscan_eps,
        min_samples=dbscan_min_samples,
        algorithm="auto",
        n_jobs=-1,
    ).fit(ds_coords)

    cluster_labels_ds = clustering.labels_  # -1 = noise

    # Step 6: Map cluster labels back to original pixel positions
    # Assign each original pixel the cluster label of its downsampled bucket
    pixel_labels = np.full(len(ys), -1, dtype=np.int32)
    for ds_idx, orig_indices in enumerate(ds_to_orig):
        label = cluster_labels_ds[ds_idx]
        for orig_idx in orig_indices:
            pixel_labels[orig_idx] = label

    # Handle noise points: assign to nearest non-noise cluster
    noise_mask = pixel_labels == -1
    if noise_mask.any() and not noise_mask.all():
        noise_indices = np.where(noise_mask)[0]
        non_noise_indices = np.where(~noise_mask)[0]
        if len(non_noise_indices) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(coords_collapsed[non_noise_indices])
            _, nearest = tree.query(coords_collapsed[noise_indices])
            pixel_labels[noise_indices] = pixel_labels[non_noise_indices[nearest]]

    # If all noise (DBSCAN failed), fallback to connected components
    if pixel_labels.max() < 0:
        _, cc_labels = cv2.connectedComponents(ferrite_mask, connectivity=8)
        pixel_labels = cc_labels[ys, xs]

    # Step 7: Build instance map + area filter + ID assignment
    # Relabel to consecutive IDs
    unique_labels = np.unique(pixel_labels)
    unique_labels = unique_labels[unique_labels >= 0]  # remove remaining -1

    instances = []
    for label_id in unique_labels:
        inst_mask = pixel_labels == label_id
        area = int(inst_mask.sum())
        if area < min_instance_area:
            continue
        instances.append((area, inst_mask))

    # Sort by area descending
    instances.sort(key=lambda x: x[0], reverse=True)

    # Build instance map
    inst_map = np.zeros((h, w), dtype=np.uint8)
    class_map = {}
    current_id = 1
    for area, inst_mask in instances:
        if current_id > max_instance_id:
            current_id = max_instance_id
        pixel_indices = np.where(inst_mask)[0]
        inst_map[ys[pixel_indices], xs[pixel_indices]] = current_id
        class_map[current_id] = CLASS_FERRITE
        if current_id < max_instance_id:
            current_id += 1

    return inst_map, class_map


def post_process_prediction_vector(
    output: torch.Tensor,
    original_size: Tuple[int, int],
    output_dir: str,
    image_basename: str,
    image_size: int = 1024,
    min_instance_area: int = 50,
    max_instance_id: int = 255,
    threshold: float = 0.5,
    dbscan_eps: float = 5.0,
    dbscan_min_samples: int = 3,
    downsample_grid: int = 4,
    save_visualization: bool = True,
) -> Tuple[Dict[str, str], np.ndarray, Dict[int, int]]:
    """
    Full post-processing pipeline using vector field centroid collapse.

    Steps:
    1. Upsample to original size
    2. Sigmoid + threshold -> binary mask
    3. Extract Vx/Vy vector field channels
    4. Centroid collapse + DBSCAN clustering (ferrite)
    5. Connected components (pearlite)
    6. Save _inst.png and _class.json

    Args:
        output: [B, 3, H, W] or [3, H, W] model output
        original_size: (H, W) original image size
        output_dir: directory for output files
        image_basename: basename for output files
        image_size: normalization factor used in training (default 1024)
        min_instance_area: minimum instance area filter
        max_instance_id: maximum instance ID (uint8)
        threshold: binary classification threshold
        dbscan_eps: DBSCAN neighborhood radius
        dbscan_min_samples: DBSCAN minimum samples
        downsample_grid: grid size for downsampling
        save_visualization: save mask/vec visualization
    """
    os.makedirs(output_dir, exist_ok=True)

    if output.ndim == 3:
        output = output.unsqueeze(0)
    elif output.ndim != 4:
        raise ValueError("output ndim should be 3 or 4")

    # Upsample to original size
    h, w = original_size
    output = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=True)

    # Extract channels
    seg_logits = output[0, 0].cpu()  # [H, W]
    vx_field = output[0, 1].cpu().numpy().astype(np.float32)  # [H, W]
    vy_field = output[0, 2].cpu().numpy().astype(np.float32)  # [H, W]

    # Binary mask
    seg_prob = torch.sigmoid(seg_logits)
    mask = (seg_prob > threshold).numpy().astype(np.uint8)  # [H, W]

    # Ferrite instances via centroid collapse
    ferrite_binary = (mask == CLASS_FERRITE).astype(np.uint8)
    if ferrite_binary.sum() > 0:
        ferrite_inst_map, ferrite_class_map = centroid_collapse_clustering(
            ferrite_binary,
            vx_field,
            vy_field,
            image_size=image_size,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            downsample_grid=downsample_grid,
            min_instance_area=min_instance_area,
            max_instance_id=max_instance_id,
        )
    else:
        ferrite_inst_map = np.zeros((h, w), dtype=np.uint8)
        ferrite_class_map = {}

    # Pearlite instances via connected components
    pearlite_binary = (mask == CLASS_PEARLITE).astype(np.uint8)
    pearlite_instances = []
    if pearlite_binary.sum() > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            pearlite_binary, connectivity=8
        )
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_instance_area:
                continue
            pearlite_instances.append((area, labels == label_id))

    # Merge: ferrite instances already have IDs 1..N
    # Append pearlite instances with continuing IDs
    inst_map = ferrite_inst_map.copy()
    class_map = dict(ferrite_class_map)
    current_id = max(class_map.keys()) + 1 if class_map else 1

    for area, inst_mask in sorted(pearlite_instances, key=lambda x: x[0], reverse=True):
        if current_id > max_instance_id:
            current_id = max_instance_id
        inst_map[inst_mask] = current_id
        class_map[current_id] = CLASS_PEARLITE
        if current_id < max_instance_id:
            current_id += 1

    # Save outputs
    inst_path = os.path.join(output_dir, f"{image_basename}_inst.png")
    cv2.imwrite(inst_path, inst_map)

    class_json_path = os.path.join(output_dir, f"{image_basename}_class.json")
    class_json = {str(k): v for k, v in class_map.items()}
    with open(class_json_path, "w", encoding="utf-8") as f:
        json.dump(class_json, f, ensure_ascii=False, indent=2)

    output_paths = {"inst_path": inst_path, "class_json_path": class_json_path}

    if save_visualization:
        mask_path = os.path.join(output_dir, f"{image_basename}_mask.png")
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        color_map[mask == CLASS_PEARLITE] = [0, 0, 128]
        color_map[mask == CLASS_FERRITE] = [0, 128, 0]
        cv2.imwrite(mask_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
        output_paths["mask_path"] = mask_path

        # Vector field visualization: color-coded by direction
        vec_vis = np.zeros((h, w, 3), dtype=np.uint8)
        vec_vis[..., 0] = np.clip(((vx_field + 1) * 127.5), 0, 255).astype(np.uint8)
        vec_vis[..., 1] = np.clip(((vy_field + 1) * 127.5), 0, 255).astype(np.uint8)
        vec_vis[..., 2] = 128
        vec_path = os.path.join(output_dir, f"{image_basename}_vec.png")
        cv2.imwrite(vec_path, cv2.cvtColor(vec_vis, cv2.COLOR_RGB2BGR))
        output_paths["vec_path"] = vec_path

    return output_paths, inst_map, class_map
