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
from typing import Dict, Tuple

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

    # Step 1: 骨架化边界
    boundary_binary = (boundary_mask > 0).astype(np.uint8) * 255
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


def post_process_prediction_boundary(
    output: torch.Tensor,
    original_size: Tuple[int, int],
    output_dir: str,
    image_basename: str,
    min_instance_area: int = 50,
    max_instance_id: int = 255,
    threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    watershed_dilate_width: int = 2,
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

    seg_prob = torch.sigmoid(seg_logits).numpy()
    boundary_prob = torch.sigmoid(boundary_logits).numpy()

    semantic_mask = (seg_prob > threshold).astype(np.uint8)
    boundary_mask = (boundary_prob > boundary_threshold).astype(np.uint8)

    inst_map, class_map = boundary_watershed_separation(
        semantic_mask,
        boundary_mask,
        dilate_width=watershed_dilate_width,
        min_area=min_instance_area,
        max_instance_id=max_instance_id,
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