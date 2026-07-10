# -*- coding: utf-8 -*-
"""
自适应尺寸还原与拓扑剥离后处理
================================
1. 网络内部动态上采样：推理时记录测试图原始尺寸 (H, W)，通过 F.interpolate
   动态直接对齐原图尺寸，随后进行 torch.argmax 消除过渡带模糊。
2. 拓扑剥离与全局 ID 分配：
   - 对铁素体核（类别1）运行 cv2.connectedComponentsWithStats，切开被晶界阻断的
     粘连晶粒，分派唯一 ID。
   - 对珠光体区域（类别0）独立运行连通域分析，切分独立实例。
   - 过滤面积小于阈值的噪点。
   - 所有实例统一共享 1~255 的整型 ID 编号并按面积降序排列写入单通道 uint8 图像
     `_inst.png`。
   - 同步将 {"实例ID": 类别标签} 写入 `_class.json`。

类别定义：0=珠光体, 1=铁素体核, 2=晶界
"""

import json
import os
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# 类别常量
CLASS_PEARLITE = 0       # 珠光体
CLASS_FERRITE_CORE = 1   # 铁素体核
CLASS_GRAIN_BOUNDARY = 2 # 晶界


def restore_to_original_size(
    logits: torch.Tensor,
    original_size: Tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    网络内部动态上采样：将解码头输出的 logits 动态上采样到原图尺寸。

    推理时记录测试图原始尺寸 (H, W)，全卷积特征流通过解码头后，在最后一层
    使用 F.interpolate(..., size=(H, W), mode='bilinear', align_corners=True)
    动态直接对齐原图尺寸。

    Args:
        logits: [B, C, h, w] 解码头输出 logits
        original_size: (H, W) 原图尺寸
        mode: 插值模式
        align_corners: 是否对齐角点

    Returns:
        logits_full: [B, C, H, W] 原图尺寸 logits
    """
    h, w = original_size
    logits_full = F.interpolate(
        logits,
        size=(h, w),
        mode=mode,
        align_corners=align_corners,
    )
    return logits_full


def argmax_to_mask(
    logits: torch.Tensor,
    original_size: Optional[Tuple[int, int]] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> np.ndarray:
    """
    对 logits 进行动态上采样 + argmax，消除过渡带模糊。

    Args:
        logits: [B, C, h, w] 或 [C, h, w] logits
        original_size: (H, W) 原图尺寸，若为 None 则不上采样
        mode: 插值模式
        align_corners: 是否对齐角点

    Returns:
        mask: [H, W] 或 [B, H, W] 三分类掩码 (uint8)
    """
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if original_size is not None:
        logits = restore_to_original_size(
            logits, original_size, mode=mode, align_corners=align_corners
        )

    # argmax 消除过渡带模糊
    pred = torch.argmax(logits, dim=1)
    pred = pred.cpu().numpy().astype(np.uint8)

    if squeeze:
        pred = pred[0]

    return pred


def topo_instance_separation(
    mask: np.ndarray,
    min_instance_area: int = 50,
    max_instance_id: int = 255,
    connectivity: int = 8,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    拓扑剥离与全局 ID 分配。

    对铁素体核（类别1）运行 cv2.connectedComponentsWithStats，切开被"晶界（类别2）"
    阻断的粘连晶粒，分派唯一 ID。
    对珠光体区域（类别0）独立运行连通域分析，将分离的珠光体团簇切分成独立实例。
    过滤面积小于 min_instance_area 像素的噪点。
    所有实例统一共享 1~255 的整型 ID 编号并按面积降序排列。

    Args:
        mask: [H, W] 三分类掩码 (0=珠光体, 1=铁素体核, 2=晶界)
        min_instance_area: 最小实例面积过滤阈值
        max_instance_id: 最大实例 ID（uint8 上限 255）
        connectivity: 连通域连接方式 (4 或 8)

    Returns:
        inst_map: [H, W] uint8 实例图，1~255 为实例 ID，0 为背景/晶界
        class_map: {实例ID: 类别标签} 字典 (1=铁素体, 0=珠光体)
    """
    h, w = mask.shape[:2]
    inst_map = np.zeros((h, w), dtype=np.uint8)
    class_map = {}

    # 收集所有实例：[(area, class_label, inst_mask), ...]
    instances = []

    # --- 1. 铁素体核（类别1）连通域分析 ---
    ferrite_binary = (mask == CLASS_FERRITE_CORE).astype(np.uint8)
    if ferrite_binary.sum() > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            ferrite_binary, connectivity=connectivity
        )
        # 跳过背景 (label=0)
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_instance_area:
                continue
            inst_mask = (labels == label_id)
            instances.append((area, CLASS_FERRITE_CORE, inst_mask))

    # --- 2. 珠光体（类别0）独立连通域分析 ---
    pearlite_binary = (mask == CLASS_PEARLITE).astype(np.uint8)
    if pearlite_binary.sum() > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            pearlite_binary, connectivity=connectivity
        )
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_instance_area:
                continue
            inst_mask = (labels == label_id)
            instances.append((area, CLASS_PEARLITE, inst_mask))

    # --- 3. 按面积降序排列 ---
    instances.sort(key=lambda x: x[0], reverse=True)

    # --- 4. 分配 1~255 的全局 ID ---
    current_id = 1
    for area, class_label, inst_mask in instances:
        if current_id > max_instance_id:
            print(
                f"警告: 实例数超过 {max_instance_id} 上限，"
                f"剩余实例将被合并到 ID {max_instance_id}"
            )
            current_id = max_instance_id
        inst_map[inst_mask] = current_id
        class_map[current_id] = int(class_label)
        if current_id < max_instance_id:
            current_id += 1

    return inst_map, class_map


def post_process_prediction(
    logits: torch.Tensor,
    original_size: Tuple[int, int],
    output_dir: str,
    image_basename: str,
    min_instance_area: int = 50,
    max_instance_id: int = 255,
    connectivity: int = 8,
    interpolate_mode: str = "bilinear",
    align_corners: bool = True,
    save_visualization: bool = True,
) -> Dict[str, str]:
    """
    完整的后处理流程：
    1. 动态上采样到原图尺寸
    2. argmax 消除过渡带
    3. 拓扑剥离与实例 ID 分配
    4. 保存 _inst.png 和 _class.json

    Args:
        logits: [B, C, h, w] 或 [C, h, w] 模型输出 logits
        original_size: (H, W) 原图尺寸
        output_dir: 输出目录
        image_basename: 图像基名（不含扩展名）
        min_instance_area: 最小实例面积
        max_instance_id: 最大实例 ID
        connectivity: 连通域连接方式
        interpolate_mode: 插值模式
        align_corners: 是否对齐角点
        save_visualization: 是否保存三分类可视化掩码

    Returns:
        output_paths: {
            "inst_path": 实例图路径,
            "class_json_path": 类别映射 JSON 路径,
            "mask_path": 三分类掩码路径 (可选),
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 动态上采样 + argmax
    mask = argmax_to_mask(
        logits,
        original_size=original_size,
        mode=interpolate_mode,
        align_corners=align_corners,
    )

    # Step 2: 拓扑剥离
    inst_map, class_map = topo_instance_separation(
        mask,
        min_instance_area=min_instance_area,
        max_instance_id=max_instance_id,
        connectivity=connectivity,
    )

    # Step 3: 保存实例图 _inst.png
    inst_path = os.path.join(output_dir, f"{image_basename}_inst.png")
    cv2.imwrite(inst_path, inst_map)

    # Step 4: 保存类别映射 _class.json
    class_json_path = os.path.join(output_dir, f"{image_basename}_class.json")
    class_json = {
        str(inst_id): class_label
        for inst_id, class_label in class_map.items()
    }
    with open(class_json_path, "w", encoding="utf-8") as f:
        json.dump(class_json, f, ensure_ascii=False, indent=2)

    output_paths = {
        "inst_path": inst_path,
        "class_json_path": class_json_path,
    }

    # 可选：保存三分类掩码可视化
    if save_visualization:
        mask_path = os.path.join(output_dir, f"{image_basename}_mask.png")
        # 用颜色映射可视化三分类
        color_map = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_map[mask == CLASS_PEARLITE] = [0, 0, 128]       # 红色（珠光体）
        color_map[mask == CLASS_FERRITE_CORE] = [0, 128, 0]   # 绿色（铁素体核）
        color_map[mask == CLASS_GRAIN_BOUNDARY] = [128, 0, 0] # 蓝色（晶界）
        cv2.imwrite(mask_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
        output_paths["mask_path"] = mask_path

    return output_paths