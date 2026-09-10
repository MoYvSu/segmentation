# -*- coding: utf-8 -*-
"""μSAM 风格的实例距离目标与 seeded-watershed 重建。

这里只实现 μSAM Automatic Instance Segmentation (AIS) 的三通道表示和
后处理，不依赖 μSAM/torch-em 的额外运行时依赖：

1. foreground；
2. 每实例归一化的中心距离（中心为 0）；
3. 每实例归一化的反转边界距离（深部为 0、边界为 1）。

中心距离只参与 marker 生成；watershed 的 height map 仅使用反转边界距离。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries, relabel_sequential, watershed


_EPS = 1.0e-7


def _validate_instance_map(instance_map: np.ndarray) -> np.ndarray:
    if instance_map.ndim != 2:
        raise ValueError(f"instance_map must be 2-D, got {instance_map.shape}")
    if not np.issubdtype(instance_map.dtype, np.integer):
        raise ValueError(f"instance_map must be integer, got {instance_map.dtype}")
    if int(instance_map.min()) < 0:
        raise ValueError("instance_map must not contain negative ids")
    return instance_map.astype(np.int32, copy=False)


def build_musam_targets(
    instance_map: np.ndarray,
    valid_content: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """由实例图生成 μSAM AIS 三通道真值。

    距离定义与 ``torch_em.transform.label.PerObjectDistanceTransform`` 一致。
    ``instance_map == 0`` 是本项目未覆盖区域，不作为 distance/foreground
    监督；返回的 ``supervision_valid`` 明确保留这一 ignore 契约。为复刻
    μSAM 的数值约定，两个距离通道在实例外仍填 1。

    Returns:
        targets: ``float32[3,H,W]``，依次为 foreground、center、boundary。
        supervision_valid: ``bool[H,W]``，仅已覆盖实例像素为真。
        audit: 实例数和凹实例中心修正计数。
    """
    source_labels = _validate_instance_map(instance_map).copy()
    if valid_content is None:
        content = np.ones(source_labels.shape, dtype=bool)
    else:
        content = np.asarray(valid_content, dtype=bool)
        if content.shape != source_labels.shape:
            raise ValueError(
                f"valid_content shape mismatch: {content.shape} vs {source_labels.shape}"
            )
        source_labels[~content] = 0

    original_instance_count = int(np.sum(np.unique(source_labels) > 0))
    # μSAM/torch-em 的 PerObjectDistanceTransform 默认 apply_label=True：
    # 对相同数值的每个连通分量重新编号。skimage 默认全连通与官方实现一致。
    labels = label(source_labels, background=0, connectivity=None).astype(
        np.int32, copy=False
    )

    foreground = (labels > 0).astype(np.float32)
    center_distances = np.ones(labels.shape, dtype=np.float32)
    boundary_distances = np.ones(labels.shape, dtype=np.float32)
    inner_boundaries = find_boundaries(labels, mode="inner")

    corrected_centers = 0
    objects = regionprops(labels)
    for prop in objects:
        instance_id = int(prop.label)
        min_row, min_col, max_row, max_col = map(int, prop.bbox)
        box = (slice(min_row, max_row), slice(min_col, max_col))
        object_mask = labels[box] == instance_id
        object_boundary = inner_boundaries[box]

        # μSAM/torch-em 先计算物体内部到 inner boundary 的距离；物体外清零。
        raw_to_boundary = distance_transform_edt(~object_boundary)
        raw_to_boundary[~object_mask] = 0.0
        deepest = np.unravel_index(
            int(np.argmax(raw_to_boundary)), raw_to_boundary.shape
        )

        center_global = np.round(prop.centroid).astype(np.int64)
        center = (
            int(center_global[0] - min_row),
            int(center_global[1] - min_col),
        )
        center_in_bounds = (
            0 <= center[0] < object_mask.shape[0]
            and 0 <= center[1] < object_mask.shape[1]
        )
        if not center_in_bounds or not bool(object_mask[center]):
            center = (int(deepest[0]), int(deepest[1]))
            corrected_centers += 1

        yy, xx = np.indices(object_mask.shape, dtype=np.float32)
        raw_center = np.sqrt(
            (yy - float(center[0])) ** 2 + (xx - float(center[1])) ** 2
        )
        raw_boundary = float(raw_to_boundary[deepest]) - raw_to_boundary
        raw_center[~object_mask] = 0.0
        raw_boundary[~object_mask] = 0.0

        center_scale = float(np.max(np.abs(raw_center))) + _EPS
        boundary_scale = float(np.max(np.abs(raw_boundary))) + _EPS
        normalized_center = raw_center / center_scale
        normalized_boundary = raw_boundary / boundary_scale

        center_crop = center_distances[box]
        boundary_crop = boundary_distances[box]
        center_crop[object_mask] = normalized_center[object_mask]
        boundary_crop[object_mask] = normalized_boundary[object_mask]

    supervision_valid = (labels > 0) & content
    targets = np.stack(
        (foreground, center_distances, boundary_distances), axis=0
    ).astype(np.float32, copy=False)
    audit = {
        "instance_count": len(objects),
        "original_instance_count": original_instance_count,
        "component_split_increase": len(objects) - original_instance_count,
        "corrected_centers": int(corrected_centers),
        "covered_pixels": int(supervision_valid.sum()),
        "ignored_pixels": int(content.sum() - supervision_valid.sum()),
        "padding_pixels": int((~content).sum()),
    }
    return targets, supervision_valid, audit


def perturb_musam_predictions(
    targets: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """以固定高斯噪声模拟轻度回归误差，结果截断到 ``[0, 1]``。"""
    predictions = np.asarray(targets, dtype=np.float32)
    if predictions.ndim != 3 or predictions.shape[0] != 3:
        raise ValueError(f"targets must have shape [3,H,W], got {predictions.shape}")
    if float(noise_std) < 0:
        raise ValueError("noise_std must be non-negative")
    if float(noise_std) == 0:
        return predictions.copy()
    noise = rng.normal(0.0, float(noise_std), size=predictions.shape)
    return np.clip(predictions + noise, 0.0, 1.0).astype(np.float32)


def musam_watershed(
    predictions: np.ndarray,
    *,
    center_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    foreground_threshold: float = 0.5,
    foreground_smoothing: float = 1.0,
    distance_smoothing: float = 1.6,
    min_size: int = 0,
    return_debug: bool = False,
):
    """严格按 μSAM AIS 约定从三通道预测重建实例。

    marker 是低中心距离、低反转边界距离与前景 mask 的交集；watershed
    的 height map 仅为反转边界距离。未读取 GT 实例 ID 或实例数量。
    """
    values = np.asarray(predictions, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError(f"predictions must have shape [3,H,W], got {values.shape}")
    if min_size < 0:
        raise ValueError("min_size must be non-negative")

    foreground, center_distances, boundary_distances = values
    if foreground_smoothing > 0:
        foreground = gaussian_filter(
            foreground, sigma=float(foreground_smoothing), mode="reflect"
        )
    if distance_smoothing > 0:
        center_distances = gaussian_filter(
            center_distances, sigma=float(distance_smoothing), mode="reflect"
        )
        boundary_distances = gaussian_filter(
            boundary_distances, sigma=float(distance_smoothing), mode="reflect"
        )

    foreground_mask = foreground > float(foreground_threshold)
    marker_mask = (
        foreground_mask
        & (center_distances < float(center_threshold))
        & (boundary_distances < float(boundary_threshold))
    )
    markers = label(marker_mask, connectivity=None).astype(np.int32, copy=False)
    instances = watershed(
        boundary_distances,
        markers=markers,
        mask=foreground_mask,
    ).astype(np.int32, copy=False)

    if min_size > 0 and instances.max() > 0:
        ids, sizes = np.unique(instances, return_counts=True)
        small = ids[(ids != 0) & (sizes < int(min_size))]
        if small.size:
            instances[np.isin(instances, small)] = 0
            instances = relabel_sequential(instances)[0].astype(
                np.int32, copy=False
            )

    if not return_debug:
        return instances
    return instances, {
        "foreground": foreground.astype(np.float32, copy=False),
        "center_distances": center_distances.astype(np.float32, copy=False),
        "boundary_distances": boundary_distances.astype(np.float32, copy=False),
        "foreground_mask": foreground_mask,
        "marker_mask": marker_mask,
        "markers": markers,
        "marker_count": int(markers.max()),
    }
