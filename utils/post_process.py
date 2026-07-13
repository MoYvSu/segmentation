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
    sigmoid_x0: float = 0.8,
    sigmoid_k: float = 0.3,
    highland_threshold: float = 0.8,
) -> np.ndarray:
    """
    Watershed separation using Sigmoid-stretched distance field.

    Since adjacent ferrite grains have no semantic gap (they are connected
    in the binary mask), seed isolation relies entirely on the topological
    properties of the continuous distance field.

    Algorithm:
    1. Gaussian blur distance field to suppress noise
    2. Purify: zero out non-ferrite regions
    3. Inverse-normalize to raw pixel distance
    4. Adaptive Sigmoid stretch: center = dist_max * sigmoid_x0
       (deepens valleys, flattens plateaus, scales to any grain size)
    5. Extract highland islands via hard threshold
    6. Connected components on highlands -> centroid per island = unique seed
    7. Dilate seeds with fixed (3,3) ellipse kernel
    8. Watershed on inverted smoothed distance field

    Args:
        ferrite_mask: [H, W] binary mask (1=ferrite, 0=background)
        dist_field: [H, W] normalized distance field [0,1] from model
        min_distance: minimum island area to keep as seed
        kernel_size: Gaussian blur kernel size
        sigmoid_x0: Sigmoid center as a FRACTION of global max raw distance.
                    0.8 means transition at 80% of peak distance.
                    Lower = more aggressive (smaller highlands). Default 0.8.
        sigmoid_k: Sigmoid gain in raw pixel distance space.
                   Higher = sharper transition. Default 0.3.
        highland_threshold: Hard threshold on Sigmoid output [0,1] to
                            extract plateau islands. Default 0.7.
    """
    from skimage.segmentation import watershed as sk_watershed

    h, w = ferrite_mask.shape[:2]

    if kernel_size is None:
        kernel_size = _get_dynamic_kernel_size(h * w)

    # Step 1: Gaussian blur to suppress noise
    blur_ksize = max(3, kernel_size)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    dist_smooth = cv2.GaussianBlur(dist_field, (blur_ksize, blur_ksize), 0)

    # Step 2: Purify - zero out non-ferrite regions to prevent energy leakage
    dist_purified = dist_smooth * (ferrite_mask > 0).astype(np.float32)

    # Step 3: Inverse-normalize to raw pixel distance
    eps = 1e-7
    dist_clipped = np.clip(dist_purified, 0.0, 1.0 - eps)
    dist_raw = dist_clipped * 10.0 / (1.0 - dist_clipped + eps)

    # Step 4: Adaptive Sigmoid stretch
    # Center = fraction of global max distance. This ensures the transition
    # zone automatically scales to any grain size.
    # For two touching circles radius=50: max~50px, saddle~40px
    #   x0 = 0.8 * 50 = 40 -> saddle at 40 -> sigma(0)=0.5 -> threshold 0.7
    #   filters out saddle region, separating the two plateaus.
    valid_dist = dist_raw[ferrite_mask > 0]
    if valid_dist.size > 0:
        dist_max = float(np.max(valid_dist))
    else:
        dist_max = 0.0
    x0_effective = dist_max * sigmoid_x0

    dist_stretched = 1.0 / (1.0 + np.exp(-sigmoid_k * (dist_raw - x0_effective)))
    # Re-zero background after Sigmoid
    dist_stretched[ferrite_mask == 0] = 0.0

    # Step 5: Extract highland islands via hard threshold
    highland_mask = (dist_stretched > highland_threshold) & (ferrite_mask > 0)
    highland_mask = highland_mask.astype(np.uint8)

    # Step 6: Connected components on highlands -> centroid per island
    num_islands, island_labels, stats, centroids = cv2.connectedComponentsWithStats(
        highland_mask, connectivity=8
    )

    if num_islands <= 1:
        # No highlands found - fallback to connected components on ferrite mask
        num_labels, labels = cv2.connectedComponents(ferrite_mask, connectivity=8)
        return labels.astype(np.int32)

    # Build seed markers from island centroids
    seed_labels = np.zeros((h, w), dtype=np.int32)
    seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for island_id in range(1, num_islands):
        area = int(stats[island_id, cv2.CC_STAT_AREA])
        if area < min_distance:
            continue
        cx = int(round(centroids[island_id, 0]))
        cy = int(round(centroids[island_id, 1]))
        # Clamp to image bounds
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        # Mark centroid pixel with island ID
        seed_labels[cy, cx] = island_id

    # Check if any seeds survived area filter
    if seed_labels.max() == 0:
        num_labels, labels = cv2.connectedComponents(ferrite_mask, connectivity=8)
        return labels.astype(np.int32)

    # Step 7: Dilate seeds with fixed (3,3) ellipse kernel for stability
    seed_mask = (seed_labels > 0).astype(np.uint8)
    seed_mask_dilated = cv2.dilate(seed_mask, seed_kernel)

    # Re-assign IDs to dilated seeds via connected components
    _, seed_labels_final = cv2.connectedComponents(seed_mask_dilated, connectivity=8)

    if seed_labels_final.max() == 0:
        num_labels, labels = cv2.connectedComponents(ferrite_mask, connectivity=8)
        return labels.astype(np.int32)

    # Step 8: Watershed on inverted smoothed distance field
    markers = seed_labels_final.copy()
    markers[ferrite_mask == 0] = -1  # mark background explicitly

    labels = sk_watershed(
        -dist_smooth,  # negate: centers become basins
        markers=markers,
        mask=ferrite_mask > 0,  # restrict to ferrite region
    )

    # Ensure background is 0
    labels = np.where(ferrite_mask > 0, labels, 0).astype(np.int32)
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
    sigmoid_x0: float = 0.8,
    sigmoid_k: float = 0.3,
    highland_threshold: float = 0.8,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Topological instance separation and global ID assignment.

    For ferrite (class 1): watershed separation if dist_field provided, else connected components.
    For pearlite (class 0): connected components analysis.
    Filter instances smaller than min_instance_area.
    Assign IDs 1~255 by descending area.

    Args:
        sigmoid_x0: Sigmoid center as fraction of max raw distance (default 0.8).
        sigmoid_k: Sigmoid gain in raw pixel distance space (default 0.3).
        highland_threshold: Hard threshold on Sigmoid output for seed extraction (default 0.8).
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
                sigmoid_x0=sigmoid_x0,
                sigmoid_k=sigmoid_k,
                highland_threshold=highland_threshold,
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
    sigmoid_x0: float = 0.8,
    sigmoid_k: float = 0.3,
    highland_threshold: float = 0.8,
) -> Tuple[Dict[str, str], np.ndarray, Dict[int, int]]:
    """
    Full post-processing pipeline:
    1. Upsample to original size
    2. Sigmoid + threshold -> binary mask
    3. Distance field spatial compensation
    4. Topological separation (watershed + connected components)
    5. Save _inst.png and _class.json

    Args:
        sigmoid_x0: Sigmoid center as fraction of max raw distance (default 0.8).
        sigmoid_k: Sigmoid gain in raw pixel distance space (default 0.3).
        highland_threshold: Hard threshold on Sigmoid output for seed extraction (default 0.8).
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
        sigmoid_x0=sigmoid_x0, sigmoid_k=sigmoid_k,
        highland_threshold=highland_threshold,
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