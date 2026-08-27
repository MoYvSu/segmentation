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

from utils.semantic_vote import build_adaptive_lab_prior, instance_semantic_vote


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


def reconstruct_marker_boundary(
    boundary_probability: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    max_steps: int,
) -> np.ndarray:
    """Locally grow strong marker edges through nearby weak responses.

    Unlike unbounded hysteresis, the finite growth radius cannot absorb an
    entire foggy weak-edge component merely because it touches one strong edge.
    The reconstructed mask is used for marker topology only; the watershed
    elevation/barrier continues to use the high-threshold boundary mask.
    """
    probability = np.asarray(boundary_probability, dtype=np.float32)
    strong = probability > float(high_threshold)
    steps = max(0, int(max_steps))
    if steps == 0 or float(low_threshold) >= float(high_threshold):
        return strong.astype(np.uint8)

    weak = probability > float(low_threshold)
    reconstructed = strong.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(steps):
        grown = cv2.dilate(reconstructed, kernel)
        grown = ((grown > 0) & weak).astype(np.uint8)
        if np.array_equal(grown, reconstructed):
            break
        reconstructed = grown
    return reconstructed


_instance_semantic_vote = instance_semantic_vote


def _region_adjacency(region_map: np.ndarray, radius: int = 2):
    """Build a contact-weighted graph, bridging one-pixel watershed seams."""
    labels = [int(v) for v in np.unique(region_map) if int(v) > 0]
    graph = {label: {} for label in labels}
    h, w = region_map.shape
    radius = max(1, int(radius))
    for dy in range(0, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx <= 0:
                continue
            y0a, y1a = 0, h - dy
            y0b, y1b = dy, h
            if dx >= 0:
                x0a, x1a = 0, w - dx
                x0b, x1b = dx, w
            else:
                x0a, x1a = -dx, w
                x0b, x1b = 0, w + dx
            a = region_map[y0a:y1a, x0a:x1a]
            b = region_map[y0b:y1b, x0b:x1b]
            valid = (a > 0) & (b > 0) & (a != b)
            if not np.any(valid):
                continue
            pairs = np.stack((a[valid], b[valid]), axis=1).astype(np.int64)
            pairs.sort(axis=1)
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            for (left, right), count in zip(unique_pairs, counts):
                left, right, count = int(left), int(right), int(count)
                graph[left][right] = graph[left].get(right, 0) + count
                graph[right][left] = graph[right].get(left, 0) + count
    return graph


def _merge_region_map_to_cap(
    region_map: np.ndarray,
    max_regions: int,
) -> Tuple[np.ndarray, list]:
    """Agglomerate smallest adjacent regions without creating a global ID."""
    source_map = np.asarray(region_map, dtype=np.int32)
    labels, counts = np.unique(source_map[source_map > 0], return_counts=True)
    active = {int(label): int(count) for label, count in zip(labels, counts)}
    max_regions = max(1, int(max_regions))
    if len(active) <= max_regions:
        return source_map.copy(), []

    graph = _region_adjacency(source_map, radius=2)
    parent = {label: label for label in active}
    merge_edges = []
    while len(active) > max_regions:
        source = min(active, key=lambda label: (active[label], label))
        neighbours = {
            label: contact for label, contact in graph.get(source, {}).items()
            if label in active and label != source
        }
        if neighbours:
            target = max(
                neighbours,
                key=lambda label: (neighbours[label], active[label], -label),
            )
            parent[source] = target
            merge_edges.append((source, target))
            active[target] += active[source]
            for neighbour, contact in list(graph.get(source, {}).items()):
                if neighbour not in active or neighbour == target:
                    continue
                graph[target][neighbour] = graph[target].get(neighbour, 0) + contact
                graph[neighbour][target] = graph[neighbour].get(target, 0) + contact
                graph[neighbour].pop(source, None)
            graph[target].pop(source, None)
        else:
            parent[source] = 0
        graph.pop(source, None)
        active.pop(source)

    def root(label):
        path = []
        while parent[label] not in (0, label):
            path.append(label)
            label = parent[label]
        resolved = parent[label] if parent[label] == 0 else label
        for item in path:
            parent[item] = resolved
        return resolved

    label_lookup = np.zeros(int(source_map.max()) + 1, dtype=np.int32)
    for label in parent:
        resolved = root(label)
        if resolved > 0:
            label_lookup[label] = resolved
    merged = label_lookup[source_map]

    # Vectorized one-pixel seam fill. A zero pixel is filled only when at least
    # two distinct original regions touch it and all touching regions now share
    # the same agglomerated root. Boundaries between retained roots stay zero.
    zero = source_map == 0
    root_min = np.full(source_map.shape, np.iinfo(np.int32).max, dtype=np.int32)
    root_max = np.zeros(source_map.shape, dtype=np.int32)
    source_min = root_min.copy()
    source_max = np.zeros(source_map.shape, dtype=np.int32)
    merged_pad = np.pad(merged, 1, mode="constant")
    source_pad = np.pad(source_map, 1, mode="constant")
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbour_root = merged_pad[
                1 + dy : 1 + dy + merged.shape[0],
                1 + dx : 1 + dx + merged.shape[1],
            ]
            neighbour_source = source_pad[
                1 + dy : 1 + dy + merged.shape[0],
                1 + dx : 1 + dx + merged.shape[1],
            ]
            valid = zero & (neighbour_root > 0)
            root_min[valid] = np.minimum(root_min[valid], neighbour_root[valid])
            root_max[valid] = np.maximum(root_max[valid], neighbour_root[valid])
            source_min[valid] = np.minimum(
                source_min[valid], neighbour_source[valid]
            )
            source_max[valid] = np.maximum(
                source_max[valid], neighbour_source[valid]
            )
    fill = (
        zero
        & (root_max > 0)
        & (root_min == root_max)
        & (source_min < source_max)
    )
    merged[fill] = root_max[fill]
    return merged, merge_edges


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
    marker_border_seal_width: int = 0,
    semantic_probability: Optional[np.ndarray] = None,
    semantic_vote_mode: str = "hard_majority",
    semantic_vote_erode_width: int = 0,
    semantic_vote_threshold: float = 0.5,
    semantic_vote_options: Optional[Dict] = None,
    semantic_lab_prior: Optional[Dict] = None,
    semantic_vote_audit: Optional[Dict] = None,
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
    border_width = max(0, int(marker_border_seal_width))
    if border_width > 0:
        border_width = min(border_width, max(1, min(h, w) // 2))
        marker_skeleton_belt[:border_width, :] = 255
        marker_skeleton_belt[-border_width:, :] = 255
        marker_skeleton_belt[:, :border_width] = 255
        marker_skeleton_belt[:, -border_width:] = 255
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

    # Step 6: 面积过滤；类别在上限拓扑合并后重新投票。
    candidate_map = np.zeros((h, w), dtype=np.int32)
    unique_labels = np.unique(ws_result)
    unique_labels = unique_labels[unique_labels > 0]

    for label_id in unique_labels:
        inst_mask = ws_result == label_id
        area = int(inst_mask.sum())
        if area < min_area:
            continue
        candidate_map[inst_mask] = int(label_id)

    # Step 7: 超过 8-bit 上限时按局部邻接合并最小区域。严禁把所有溢出
    # 区域写入同一个 ID，否则会制造跨图像的不连通伪实例。
    candidate_map, _ = _merge_region_map_to_cap(candidate_map, max_instance_id)
    final_labels, final_counts = np.unique(
        candidate_map[candidate_map > 0], return_counts=True
    )
    final_labels = [
        int(label)
        for label, _ in sorted(
            zip(final_labels, final_counts), key=lambda item: int(item[1]), reverse=True
        )
    ]
    inst_map = np.zeros((h, w), dtype=np.uint8)
    class_map = {}
    for current_id, label in enumerate(final_labels, start=1):
        instance_mask = candidate_map == label
        vote_result = _instance_semantic_vote(
            instance_mask,
            semantic_mask,
            semantic_probability=semantic_probability,
            mode=semantic_vote_mode,
            erode_width=semantic_vote_erode_width,
            threshold=semantic_vote_threshold,
            lab_prior=semantic_lab_prior,
            return_details=semantic_vote_audit is not None,
            **(semantic_vote_options or {}),
        )
        if semantic_vote_audit is None:
            cls, _ = vote_result
        else:
            cls, score, details = vote_result
            semantic_vote_audit[current_id] = {
                "ferrite_score": float(score),
                **details,
            }
        inst_map[instance_mask] = current_id
        class_map[current_id] = int(cls)

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
    marker_border_seal_width: int = 0,
    marker_boundary_low_threshold: Optional[float] = None,
    marker_boundary_reconstruction_steps: int = 0,
    semantic_vote_mode: str = "hard_majority",
    semantic_vote_erode_width: int = 0,
    semantic_vote_threshold: float = 0.5,
    semantic_vote_options: Optional[Dict] = None,
    original_image_rgb: Optional[np.ndarray] = None,
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
    semantic_lab_prior = None
    if (
        str(semantic_vote_mode).strip().lower() == "adaptive_core_lab"
        and original_image_rgb is not None
    ):
        semantic_lab_prior = build_adaptive_lab_prior(original_image_rgb)
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
    # 高阈值仍控制真实 barrier；低阈值只在强边附近有限步补全 marker 断口。
    boundary_mask = (boundary_prob > boundary_threshold).astype(np.uint8)
    marker_boundary_mask = None
    if (
        marker_boundary_low_threshold is not None
        and int(marker_boundary_reconstruction_steps) > 0
    ):
        marker_boundary_mask = reconstruct_marker_boundary(
            boundary_prob,
            low_threshold=float(marker_boundary_low_threshold),
            high_threshold=float(boundary_threshold),
            max_steps=int(marker_boundary_reconstruction_steps),
        )
    semantic_vote_audit = {}
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
        marker_boundary_mask=marker_boundary_mask,
        marker_border_seal_width=marker_border_seal_width,
        semantic_probability=seg_prob,
        semantic_vote_mode=semantic_vote_mode,
        semantic_vote_erode_width=semantic_vote_erode_width,
        semantic_vote_threshold=semantic_vote_threshold,
        semantic_vote_options=semantic_vote_options,
        semantic_lab_prior=semantic_lab_prior,
        semantic_vote_audit=semantic_vote_audit,
    )

    inst_path = os.path.join(output_dir, f"{image_basename}_inst.png")
    cv2.imwrite(inst_path, inst_map)

    class_json_path = os.path.join(output_dir, f"{image_basename}_class.json")
    class_json = {str(k): v for k, v in class_map.items()}
    with open(class_json_path, "w", encoding="utf-8") as f:
        json.dump(class_json, f, ensure_ascii=False, indent=2)

    output_paths = {"inst_path": inst_path, "class_json_path": class_json_path}

    vote_audit = {}
    for instance_id, cls in class_map.items():
        instance_mask = inst_map == int(instance_id)
        details = semantic_vote_audit[int(instance_id)]
        score = float(details["ferrite_score"])
        vote_audit[str(instance_id)] = {
            "class": int(cls),
            "confidence": float(abs(score - semantic_vote_threshold) * 2.0),
            "area": int(instance_mask.sum()),
            **details,
        }
    vote_audit_path = os.path.join(
        output_dir, f"{image_basename}_class_confidence.json"
    )
    with open(vote_audit_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": semantic_vote_mode,
                "threshold": float(semantic_vote_threshold),
                "erode_width": int(semantic_vote_erode_width),
                "options": semantic_vote_options or {},
                "lab_prior": None if semantic_lab_prior is None else {
                    key: value for key, value in semantic_lab_prior.items()
                    if key != "ferrite_probability"
                },
                "instances": vote_audit,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    output_paths["class_confidence_path"] = vote_audit_path

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

        if marker_boundary_mask is not None:
            marker_boundary_path = os.path.join(
                output_dir, f"{image_basename}_marker_boundary.png"
            )
            cv2.imwrite(marker_boundary_path, marker_boundary_mask * 255)
            output_paths["marker_boundary_path"] = marker_boundary_path

        if center_prob is not None:
            center_path = os.path.join(output_dir, f"{image_basename}_center.png")
            center_vis = (np.clip(center_prob, 0.0, 1.0) * 255).astype(np.uint8)
            cv2.imwrite(center_path, center_vis)
            output_paths["center_path"] = center_path

    return output_paths, inst_map, class_map
