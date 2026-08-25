# -*- coding: utf-8 -*-
"""
边界预测版后处理（当前推理唯一路径）
====================================
1. Letterbox 输出上采样回原图尺寸
2. Sigmoid + 阈值 -> 语义掩码 / 边界掩码 / 中心热图
3. 中心热图峰值提供实例种子，边界骨架带作为分水岭障碍 -> 实例 ID + 类别映射

说明：旧向量场 / 距离场 / Snake 等废弃管线已删除，本文件仅保留
`boundary_watershed_separation` 与 `post_process_prediction_boundary`。

Class definition: 0=pearlite, 1=ferrite
"""

import json
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


CLASS_PEARLITE = 0
CLASS_FERRITE = 1


def center_heatmap_markers(
    center_prob: np.ndarray,
    threshold: float = 0.25,
    nms_kernel: int = 9,
) -> Tuple[np.ndarray, int]:
    """Convert a center heatmap into one watershed marker per local peak.

    The heatmap is trained with one Gaussian peak per LabelMe polygon.  NMS is
    deliberately performed before connected components so a broad Gaussian
    contributes one seed rather than an area-sized seed.  ``nms_kernel`` is
    the minimum approximate separation between two seeds in output pixels.
    """
    prob = np.asarray(center_prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"center_prob must be [H, W], got {prob.shape}")
    kernel = max(3, int(nms_kernel))
    if kernel % 2 == 0:
        kernel += 1
    local_max = cv2.dilate(prob, np.ones((kernel, kernel), np.uint8))
    peak_mask = ((prob >= float(threshold)) & (prob >= local_max - 1e-6)).astype(np.uint8)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(peak_mask, 8)
    markers = np.zeros(prob.shape, dtype=np.int32)
    peaks = []
    for label_id in range(1, num):
        x, y = np.round(centroids[label_id]).astype(int)
        x = int(np.clip(x, 0, prob.shape[1] - 1))
        y = int(np.clip(y, 0, prob.shape[0] - 1))
        peaks.append((float(prob[y, x]), x, y))
    # Stable order makes output IDs reproducible across equivalent peaks.
    peaks.sort(key=lambda item: (-item[0], item[2], item[1]))
    for marker_id, (_, x, y) in enumerate(peaks, start=1):
        markers[y, x] = marker_id
    return markers, len(peaks)


# ---------------------------------------------------------------------------
# Boundary-based Watershed Instance Separation
# ---------------------------------------------------------------------------

def _boundary_skeleton_belt(
    boundary_mask: np.ndarray,
    bridge_width: int,
    dilate_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a binary boundary map into its skeleton and barrier belt."""
    boundary_binary = (boundary_mask > 0).astype(np.uint8) * 255
    if bridge_width > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * bridge_width + 1, 2 * bridge_width + 1)
        )
        boundary_binary = cv2.dilate(boundary_binary, k)
    if not np.any(boundary_binary):
        skeleton = boundary_binary
    else:
        try:
            skeleton = cv2.ximgproc.thinning(
                boundary_binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
        except (AttributeError, cv2.error):
            from skimage.morphology import skeletonize as sk_skeletonize

            skeleton = (sk_skeletonize(boundary_binary > 0) * 255).astype(np.uint8)

    if dilate_width > 0:
        kernel_size = 2 * dilate_width + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        skeleton_belt = cv2.dilate(skeleton, kernel)
    else:
        skeleton_belt = skeleton
    return skeleton, skeleton_belt


def boundary_watershed_separation(
    semantic_mask: np.ndarray,
    boundary_mask: np.ndarray,
    dilate_width: int = 2,
    min_area: int = 50,
    max_instance_id: int = 255,
    bridge_width: int = 1,
    center_prob: Optional[np.ndarray] = None,
    center_threshold: float = 0.25,
    center_nms_kernel: int = 9,
    marker_boundary_mask: Optional[np.ndarray] = None,
    marker_bridge_width: Optional[int] = None,
    marker_dilate_width: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    基于边界预测的受阻分水岭实例分割。

    流程：
    1. 骨架化边界掩码 -> 1px 线条
    2. 膨胀 dilate_width 像素 -> 骨架带（防线）
    3. 全图减去骨架带 -> 独立晶核
    4. 中心热图局部峰值 -> 种子；没有中心热图时回退到空间核心连通域
    5. 受阻分水岭缝合 -> 实例图
    6. 每个实例区域语义投票 -> 晶粒类别
    7. 面积过滤 + ID 分配
    """
    max_instance_id = int(max_instance_id)
    if not 1 <= max_instance_id <= 255:
        raise ValueError(
            f"max_instance_id must be within [1, 255], got {max_instance_id}"
        )
    h, w = semantic_mask.shape[:2]

    # Step 1: 骨架化边界（可选先膨胀桥接"双峰/双线"输出：
    # 边界概率剖面呈两条脊时，阈值化后是两条分离的细带，骨架化得到粗糙双线；
    # 先膨胀 bridge_width 像素把两条脊合成一条带，骨架取中轴即为平滑单线）
    skeleton, skeleton_belt = _boundary_skeleton_belt(
        boundary_mask, bridge_width, dilate_width
    )
    marker_skeleton_belt = skeleton_belt
    if marker_boundary_mask is not None:
        _, marker_skeleton_belt = _boundary_skeleton_belt(
            marker_boundary_mask,
            bridge_width if marker_bridge_width is None else marker_bridge_width,
            dilate_width if marker_dilate_width is None else marker_dilate_width,
        )
    # Step 3/4: 生成种子。中心峰避免“一个连通核心=一个实例”的欠分割，
    # 仍保留旧核心种子作为旧 checkpoint 和低置信中心热图的安全回退。
    markers = np.zeros((h, w), dtype=np.int32)
    center_count = 0
    if center_prob is not None:
        markers, center_count = center_heatmap_markers(
            center_prob,
            threshold=center_threshold,
            nms_kernel=center_nms_kernel,
        )
    valid_id = center_count

    if center_count == 0:
        cores = cv2.bitwise_not(marker_skeleton_belt)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cores, connectivity=8
        )
        valid_id = 0
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            valid_id += 1
            markers[labels == label_id] = valid_id

    if valid_id == 0:
        return np.zeros((h, w), dtype=np.uint8), {}

    # Step 5: 受阻分水岭
    img_for_ws = np.full((h, w, 3), 128, dtype=np.uint8)
    img_for_ws[semantic_mask > 0] = [200, 200, 200]
    overlay = img_for_ws.copy()
    overlay[skeleton_belt > 0] = [255, 255, 255]
    img_for_ws = cv2.addWeighted(img_for_ws, 0.7, overlay, 0.3, 0)

    ws_result = cv2.watershed(img_for_ws, markers.copy())
    ws_result[ws_result < 0] = 0

    # Step 6: 语义投票 -> 晶粒类别
    instances = []
    unique_labels = np.unique(ws_result)
    unique_labels = unique_labels[unique_labels > 0]

    for label_id in unique_labels:
        inst_mask = ws_result == label_id
        area = int(inst_mask.sum())
        if area < min_area:
            continue
        ferrite_ratio = float((semantic_mask[inst_mask] > 0).sum()) / area
        cls = CLASS_FERRITE if ferrite_ratio > 0.5 else CLASS_PEARLITE
        instances.append((area, cls, inst_mask))

    # Step 7: 按面积降序分配 ID。若候选超过 8-bit 提交格式上限，
    # 保留最大的 max_instance_id-1 个，剩余区域统一写入最后一个 ID。
    # 最后一个 ID 的类别由全部溢出像素重新投票，避免历史逻辑中类别被
    # 最后一个小碎片覆盖。这里不以目标实例数或平均面积反向调节分割参数。
    instances.sort(key=lambda x: x[0], reverse=True)
    inst_map = np.zeros((h, w), dtype=np.uint8)
    class_map = {}
    if len(instances) <= max_instance_id:
        for current_id, (_, cls, inst_mask) in enumerate(instances, start=1):
            inst_map[inst_mask] = current_id
            class_map[current_id] = int(cls)
    else:
        keep_count = max_instance_id - 1
        for current_id, (_, cls, inst_mask) in enumerate(
            instances[:keep_count], start=1
        ):
            inst_map[inst_mask] = current_id
            class_map[current_id] = int(cls)

        overflow_mask = np.zeros((h, w), dtype=bool)
        for _, _, inst_mask in instances[keep_count:]:
            overflow_mask |= inst_mask
        overflow_area = int(overflow_mask.sum())
        if overflow_area > 0:
            overflow_ferrite = int((semantic_mask[overflow_mask] > 0).sum())
            inst_map[overflow_mask] = max_instance_id
            class_map[max_instance_id] = int(
                CLASS_FERRITE
                if overflow_ferrite / overflow_area > 0.5
                else CLASS_PEARLITE
            )

    if int(inst_map.max()) > max_instance_id or len(class_map) > max_instance_id:
        raise RuntimeError(
            "Internal error: watershed output exceeded the configured instance cap"
        )
    return inst_map, class_map


def _norm_p95(resp):
    """P95 分位数归一化到 [0,1]（稳健，抗离群）。"""
    p95 = float(np.percentile(resp, 95))
    if p95 <= 1e-8:
        return np.zeros_like(resp, dtype=np.float32)
    return (resp / p95).astype(np.float32)


def semantic_edge_map(seg_prob, smooth=1.0):
    """语义梯度图 |∇seg|（P95 归一化），用于补缺式边界融合。"""
    src = seg_prob
    if smooth > 0:
        src = cv2.GaussianBlur(seg_prob, (0, 0), smooth)
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    resp = np.sqrt(gx ** 2 + gy ** 2)
    return _norm_p95(resp)


def semantic_edge_boost(boundary_prob, seg_prob, alpha=0.0):
    """语义边缘升权：边界概率 × (1 + α × |∇语义概率|)。

    相界（铁素体↔珠光体交界）处 |∇seg| 高 → 该处边界被放大；
    相内平坦区（|∇seg|≈0）边界保持原值，噪声划痕不被放大。
    归一化：|∇seg| 除以全图最大值映射到 [0,1]。
    """
    if alpha <= 0:
        return boundary_prob
    gx = cv2.Sobel(seg_prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(seg_prob, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    mx = float(grad.max())
    if mx <= 1e-8:
        return boundary_prob
    edge = grad / mx
    return boundary_prob * (1.0 + alpha * edge)


def post_process_prediction_boundary(
    output: torch.Tensor,
    original_size: Tuple[int, int],
    output_dir: str,
    image_basename: str,
    min_instance_area: int = 50,
    max_instance_id: int = 255,
    threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    boundary_logit_scale: float = 1.0,
    sem_edge_boost_alpha: float = 0.0,
    sem_edge_merge_weight: float = 0.0,
    sem_edge_smooth: float = 1.0,
    watershed_dilate_width: int = 2,
    bridge_width: int = 1,
    use_center_seeds: bool = True,
    center_threshold: float = 0.25,
    center_nms_kernel: int = 9,
    save_visualization: bool = True,
) -> Tuple[Dict[str, str], np.ndarray, Dict[int, int]]:
    """
    边界预测版后处理管线。
    """
    os.makedirs(output_dir, exist_ok=True)

    if output.ndim == 3:
        output = output.unsqueeze(0)

    h, w = original_size
    output = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=True)

    seg_logits = output[0, 0].cpu()
    boundary_logits = output[0, 1].cpu()
    center_logits = output[0, 2].cpu() if output.shape[1] >= 3 else None
    if boundary_logit_scale != 1.0:
        boundary_logits = boundary_logits * boundary_logit_scale

    seg_prob = torch.sigmoid(seg_logits).numpy()
    boundary_prob = torch.sigmoid(boundary_logits).numpy()
    center_prob = (
        torch.sigmoid(center_logits).numpy()
        if center_logits is not None and use_center_seeds
        else None
    )

    semantic_mask = (seg_prob > threshold).astype(np.uint8)
    # 语义补缺式融合（可选）：只在边界分支弱处补 |∇seg|，不增厚强边界
    if sem_edge_merge_weight > 0:
        edge = semantic_edge_map(seg_prob, smooth=sem_edge_smooth)
        boundary_prob = (
            boundary_prob
            + sem_edge_merge_weight * edge * (1.0 - boundary_prob)
        )
    if sem_edge_boost_alpha > 0:
        boundary_prob = semantic_edge_boost(
            boundary_prob, seg_prob, alpha=sem_edge_boost_alpha
        )
    # 单阈值二值化
    boundary_mask = (boundary_prob > boundary_threshold).astype(np.uint8)
    inst_map, class_map = boundary_watershed_separation(
        semantic_mask,
        boundary_mask,
        dilate_width=watershed_dilate_width,
        min_area=min_instance_area,
        max_instance_id=max_instance_id,
        bridge_width=bridge_width,
        center_prob=center_prob,
        center_threshold=center_threshold,
        center_nms_kernel=center_nms_kernel,
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
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        color_map[semantic_mask == CLASS_PEARLITE] = [0, 0, 128]
        color_map[semantic_mask == CLASS_FERRITE] = [0, 128, 0]
        cv2.imwrite(mask_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
        output_paths["mask_path"] = mask_path

        boundary_path = os.path.join(output_dir, f"{image_basename}_boundary.png")
        boundary_vis = (boundary_prob * 255).astype(np.uint8)
        cv2.imwrite(boundary_path, boundary_vis)
        output_paths["boundary_path"] = boundary_path

        if center_prob is not None:
            center_path = os.path.join(output_dir, f"{image_basename}_center.png")
            center_vis = (np.clip(center_prob, 0.0, 1.0) * 255).astype(np.uint8)
            cv2.imwrite(center_path, center_vis)
            output_paths["center_path"] = center_path

    return output_paths, inst_map, class_map
