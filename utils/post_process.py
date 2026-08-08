# -*- coding: utf-8 -*-
"""
边界预测版后处理（当前推理唯一路径）
====================================
1. Letterbox 输出上采样回原图尺寸
2. Sigmoid + 阈值 -> 语义掩码 / 边界掩码
3. 边界骨架化 -> 膨胀骨架带 -> 核心剥离 -> 受阻分水岭 -> 实例 ID + 类别映射

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


# ---------------------------------------------------------------------------
# Boundary-based Watershed Instance Separation
# ---------------------------------------------------------------------------

def boundary_watershed_separation(
    semantic_mask: np.ndarray,
    boundary_mask: np.ndarray,
    dilate_width: int = 2,
    min_area: int = 50,
    max_instance_id: int = 255,
    bridge_width: int = 1,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    基于边界预测的受阻分水岭实例分割。

    流程：
    1. 骨架化边界掩码 -> 1px 线条
    2. 膨胀 dilate_width 像素 -> 骨架带（防线）
    3. 全图减去骨架带 -> 独立晶核
    4. 连通域标记 -> 种子
    5. 受阻分水岭缝合 -> 实例图
    6. 每个实例区域语义投票 -> 晶粒类别
    7. 面积过滤 + ID 分配
    """
    h, w = semantic_mask.shape[:2]

    # Step 1: 骨架化边界（可选先膨胀桥接"双峰/双线"输出：
    # 边界概率剖面呈两条脊时，阈值化后是两条分离的细带，骨架化得到粗糙双线；
    # 先膨胀 bridge_width 像素把两条脊合成一条带，骨架取中轴即为平滑单线）
    boundary_binary = (boundary_mask > 0).astype(np.uint8) * 255
    if bridge_width > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * bridge_width + 1, 2 * bridge_width + 1)
        )
        boundary_binary = cv2.dilate(boundary_binary, k)
    try:
        skeleton = cv2.ximgproc.thinning(
            boundary_binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
    except (AttributeError, cv2.error):
        from skimage.morphology import skeletonize as sk_skeletonize
        skeleton = (sk_skeletonize(boundary_binary > 0) * 255).astype(np.uint8)

    # Step 2: 膨胀骨架带
    if dilate_width > 0:
        kernel_size = 2 * dilate_width + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        skeleton_belt = cv2.dilate(skeleton, kernel)
    else:
        skeleton_belt = skeleton

    # Step 3: 空间核心剥离
    cores = cv2.bitwise_not(skeleton_belt)

    # Step 4: 连通域标记 -> 种子
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=8)
    markers = np.zeros((h, w), dtype=np.int32)
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

    # Step 7: 按面积降序分配 ID
    instances.sort(key=lambda x: x[0], reverse=True)
    inst_map = np.zeros((h, w), dtype=np.uint8)
    class_map = {}
    current_id = 1
    for area, cls, inst_mask in instances:
        if current_id > max_instance_id:
            current_id = max_instance_id
        inst_map[inst_mask] = current_id
        class_map[current_id] = int(cls)
        if current_id < max_instance_id:
            current_id += 1

    return inst_map, class_map


def _norm_p95(resp):
    """P95 分位数归一化到 [0,1]（稳健，抗离群）。"""
    p95 = float(np.percentile(resp, 95))
    if p95 <= 1e-8:
        return np.zeros_like(resp, dtype=np.float32)
    return (resp / p95).astype(np.float32)


def semantic_edge_map(seg_prob, mode="gradient", smooth=1.0,
                      valley_weight=1.0, gradient_weight=1.0):
    """语义引导的边界信号图（归一化到 P95 分位数，稳健）。

    mode="gradient": |∇seg|，在相界/凹陷两侧均有响应（双峰）；
    mode="valley":   max(0, -∇²seg)，只在凹陷（晶界线）中心给出单峰响应；
    mode="combined": max(谷响应×w_v, 梯度×w_g)——窄谷（铁素体-铁素体）由谷响应
                     单线覆盖，宽台阶（铁素体-珠光体）由梯度单线覆盖，
                     无需判断"同相/异相"，取响应强的一类。
    smooth: 高斯平滑 sigma（二阶算子对噪声极敏感，valley 模式必须平滑）。
    valley_weight / gradient_weight: combined 模式下两类响应的相对权重。
    """
    src = seg_prob
    if smooth > 0:
        src = cv2.GaussianBlur(seg_prob, (0, 0), smooth)
    if mode == "valley":
        lap = cv2.Laplacian(src, cv2.CV_32F, ksize=3)
        resp = np.clip(-lap, 0, None)
    elif mode == "gradient":
        gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
        resp = np.sqrt(gx ** 2 + gy ** 2)
    elif mode == "combined":
        lap = cv2.Laplacian(src, cv2.CV_32F, ksize=3)
        valley = _norm_p95(np.clip(-lap, 0, None))
        gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
        grad = _norm_p95(np.sqrt(gx ** 2 + gy ** 2))
        resp = np.maximum(valley_weight * valley, gradient_weight * grad)
        return np.clip(resp, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"未知语义边缘模式: {mode}")
    return _norm_p95(resp)


def _skeletonize_center(binary):
    """二值掩码 -> 中轴单线（cv2.ximgproc.thinning，缺 contrib 时回退 skimage）。"""
    b = (binary > 0).astype(np.uint8) * 255
    try:
        return cv2.ximgproc.thinning(b, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except (AttributeError, cv2.error):
        from skimage.morphology import skeletonize as sk_skeletonize
        return (sk_skeletonize(b > 0) * 255).astype(np.uint8)


def semantic_primary_barrier(seg_prob, boundary_prob, percentile=80.0,
                             bridge_dilate=2, band_width=2, line_weight=3.0,
                             smooth=1.0):
    """语义单线为主 + 边界头带内校准的屏障图。

    S = P_b × B + λ × L
      L: |∇seg| 自适应阈值(P80) -> 桥接膨胀 -> 骨架化的单线（位置主源）
      B: 语义带 = 膨胀(L, band_width)（边界"应该在"的区域）
      P_b: 边界头概率（带内强度参考）
      λ × L: 语义线本身始终是屏障（弱边界兜底）

    效果：带外边界头响应（雾状/噪声）被乘法掩掉；位置由语义单线决定；
    强度由边界头在带内校准。返回 (S, L)。
    """
    grad = semantic_edge_map(seg_prob, mode="gradient", smooth=smooth)
    thr = float(np.percentile(grad, percentile))
    bm = (grad > thr).astype(np.uint8) * 255
    if bridge_dilate > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * bridge_dilate + 1, 2 * bridge_dilate + 1))
        bm = cv2.dilate(bm, k)
    sk = _skeletonize_center(bm > 0)
    L = (sk > 0).astype(np.float32)
    if band_width > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * band_width + 1, 2 * band_width + 1))
        B = cv2.dilate(L, k)
    else:
        B = L
    S = boundary_prob * B + line_weight * L
    return S, L


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
    sem_edge_mode: str = "gradient",
    sem_edge_smooth: float = 1.0,
    sem_edge_valley_weight: float = 1.0,
    sem_edge_gradient_weight: float = 1.0,
    semantic_primary: bool = False,
    sem_band_width: int = 2,
    sem_line_weight: float = 3.0,
    sem_percentile: float = 80.0,
    watershed_dilate_width: int = 2,
    bridge_width: int = 1,
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
    if boundary_logit_scale != 1.0:
        boundary_logits = boundary_logits * boundary_logit_scale

    seg_prob = torch.sigmoid(seg_logits).numpy()
    boundary_prob = torch.sigmoid(boundary_logits).numpy()
    boundary_prob_raw = boundary_prob.copy()   # 语义单线模式用作带内参考

    semantic_mask = (seg_prob > threshold).astype(np.uint8)
    if not semantic_primary:
        # 语义引导融合（可选）：
        # 1) 补缺式加性融合：final = bnd + λ·edge·(1−bnd)
        #    只在边界分支弱处补语义梯度，不增厚已有强边界——
        #    加性融合会把相界也放大成厚带，骨架带盖住小晶粒核导致
        #    小铁素体被吞并（实例数 33→25、中位面积 +35% 的失效模式）；
        #    补缺式保持强边界原样、只补漏检的铁素体内部晶界
        # 2) 乘性升权：bnd × (1 + α·edge)（放大相界处已有响应）
        if sem_edge_merge_weight > 0:
            edge = semantic_edge_map(
                seg_prob, mode=sem_edge_mode, smooth=sem_edge_smooth,
                valley_weight=sem_edge_valley_weight,
                gradient_weight=sem_edge_gradient_weight,
            )
            boundary_prob = (
                boundary_prob
                + sem_edge_merge_weight * edge * (1.0 - boundary_prob)
            )
        if sem_edge_boost_alpha > 0:
            boundary_prob = semantic_edge_boost(
                boundary_prob, seg_prob, alpha=sem_edge_boost_alpha
            )
    if semantic_primary:
        # 语义单线为主：S = P_b × B + λ × L，Otsu 自适应阈值
        S, L = semantic_primary_barrier(
            seg_prob, boundary_prob_raw,
            percentile=sem_percentile, band_width=sem_band_width,
            line_weight=sem_line_weight,
        )
        smax = float(S.max())
        if smax > 0:
            s8 = np.clip(S / smax * 255, 0, 255).astype(np.uint8)
            _, bm8 = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            boundary_mask = (bm8 > 0).astype(np.uint8)
        else:
            boundary_mask = np.zeros_like(semantic_mask)
    else:
        # 单阈值二值化（已移除 Canny 式滞后：边界概率在邻域连续，滞后会把强脊
        # 的坡脚也纳入，导致边界带宽度沿脊线变化、轮廓崎岖不平）
        boundary_mask = (boundary_prob > boundary_threshold).astype(np.uint8)
    inst_map, class_map = boundary_watershed_separation(
        semantic_mask,
        boundary_mask,
        dilate_width=watershed_dilate_width,
        min_area=min_instance_area,
        max_instance_id=max_instance_id,
        bridge_width=bridge_width,
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

    return output_paths, inst_map, class_map
